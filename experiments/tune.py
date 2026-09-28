"""Hyperparameter tuning on the validation period (§12).

    python -m experiments.tune --config experiments/configs/paper_baseline.yaml --trials 40

Fairness protocol: both models draw the same number of trials from the same grid of shared
parameters (threshold, leak, new_mean, STDP scale); the improved model additionally draws
γ and tau_z_bars within that budget. The objective is the mean test-day spike accuracy over
all validation folds and tuning seeds; trials whose runs fail the health checks on more than
`--max-unhealthy` of the days are not admissible. Only the signal phase runs (no strategies),
so P&L never influences the choice.

Writes ``results/tune_<model>/trials.csv`` and the best model config (YAML) to ``--out``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from snn_hft.backtest.backtester import Backtester, signal_jobs
from snn_hft.backtest.cache import SignalCache
from snn_hft.backtest.splitter import WalkForwardSplitter
from snn_hft.config import load_config
from snn_hft.config.loader import apply_override
from snn_hft.config.schema import ExperimentConfig

SHARED_GRID: dict[str, list[Any]] = {
    "core.lif.threshold": [2.0, 4.0, 8.0, 16.0, 32.0],
    "core.lif.leak": [0.02, 0.05, 0.1, 0.2],
    "input.new_mean": [0.05, 0.1, 0.2, 0.3],
    "stdp_scale": [0.5, 1.0, 2.0, 4.0],  # multiplies A and B together
}
# Model-specific parameters, tuned within the same trial budget.
PAPER_GRID: dict[str, list[Any]] = {
    "zscore.new_std": [0.05, 0.1, 0.2, 0.4],  # the paper names new_stdev as a hyperparameter
}
IMPROVED_GRID: dict[str, list[Any]] = {
    "rstdp.gamma": [0.1, 0.3, 1.0],
    "rstdp.tau_z_bars": [1.0, 3.0, 10.0],
    "hawkes.event_quantile": [0.8, 0.9, 0.95],  # decision U9
}


def grid_for(model_type: str) -> dict[str, list[Any]]:
    extra = IMPROVED_GRID if model_type == "hawkes_rstdp" else PAPER_GRID
    return {**SHARED_GRID, **extra}


def sample_trials(grid: dict[str, list[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n distinct grid points, drawn uniformly without replacement."""
    points = list(itertools.product(*grid.values()))
    idx = np.random.default_rng(seed).permutation(len(points))[:n]
    return [dict(zip(grid, points[i])) for i in sorted(idx)]


def apply_trial(model: dict, trial: dict[str, Any]) -> dict:
    model = yaml.safe_load(yaml.safe_dump(model))  # deep copy of plain data
    for key, value in trial.items():
        if key == "stdp_scale":
            stdp = model.setdefault("core", {}).setdefault("stdp", {})
            stdp["A"] = round(0.01 * value, 10)
            stdp["B"] = round(-0.0105 * value, 10)
        else:
            apply_override(model, f"{key}={value}")
    return model


def build_tuning_experiment(base: ExperimentConfig, trials: list[dict], seeds: list[int]) -> ExperimentConfig:
    model = base.models[0].model_dump(mode="json")
    models = []
    for i, trial in enumerate(trials):
        m = apply_trial(model, trial)
        m["name"] = f"t{i:03d}"
        models.append(m)
    raw = base.model_dump(mode="json", exclude={"models", "name", "description", "backtest"})
    raw.update(
        name=f"tune_{base.models[0].model}",
        description="validation tuning",
        models=models,
        backtest={**base.backtest.model_dump(mode="json"), "split": "validation", "seeds": seeds,
                  "w_snn": [base.backtest.w_snn[0]], "w_h": [base.backtest.w_h[0]]},
    )
    raw["output"] = {**raw["output"], "save_trades": "none", "cache_diagnostics": False}
    return ExperimentConfig.model_validate(raw)


def collect(cfg: ExperimentConfig, trials: list[dict]) -> pd.DataFrame:
    cache = SignalCache(cfg.output.cache_dir)
    days = cfg.eval_period.days()
    rows = []
    for job in signal_jobs(cfg):
        run = job.run
        for fold in WalkForwardSplitter(run.w_snn).folds(days):
            meta = cache.load_meta(run, fold.test_day)
            m = meta["test_metrics"]
            rows.append({"trial": run.signal.name, "seed": run.seed, "day": fold.test_day, "accuracy": m["accuracy"],
                         "chance": m["base_accuracy"], "signal_rate": m["signal_rate"],
                         "unhealthy": bool(meta["test_health"])})
    per_day = pd.DataFrame(rows)
    table = per_day.groupby("trial").agg(
        accuracy=("accuracy", "mean"), chance=("chance", "mean"), signal_rate=("signal_rate", "mean"),
        unhealthy_share=("unhealthy", "mean"), folds=("day", "count"),
    )
    table["edge_pp"] = (table["accuracy"] - table["chance"]) * 100
    params = pd.DataFrame(trials, index=[f"t{i:03d}" for i in range(len(trials))])
    return params.join(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--grid-seed", type=int, default=0)
    parser.add_argument("--max-unhealthy", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True, help="YAML file for the best model config")
    args = parser.parse_args(argv)

    base = load_config(args.config, args.overrides)
    model_type = base.models[0].model
    trials = sample_trials(grid_for(model_type), args.trials, args.grid_seed)
    cfg = build_tuning_experiment(base, trials, args.seeds)
    print(f"tuning {model_type}: {len(trials)} trials × {len(cfg.eval_period.days())} validation days × {len(args.seeds)} seeds")
    Backtester(cfg, workers=args.workers).compute_signals()

    table = collect(cfg, trials)
    out_dir = cfg.output.results_dir / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "trials.csv", index_label="trial")
    admissible = table[table["unhealthy_share"] <= args.max_unhealthy]
    if admissible.empty:
        print("no admissible trial (all fail the health checks)", file=sys.stderr)
        return 1
    best = admissible["accuracy"].idxmax()
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.sort_values("accuracy", ascending=False).head(15).to_string())
    best_model = next(m for m in cfg.models if m.name == best).model_dump(mode="json")
    best_model["name"] = base.models[0].name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Tuned on the validation period: {cfg.eval_period.start}..{cfg.eval_period.end}, seeds {args.seeds},\n"
        f"# {len(trials)} trials (grid seed {args.grid_seed}); best trial {best}: validation accuracy "
        f"{table.loc[best, 'accuracy']:.4f} (chance {table.loc[best, 'chance']:.4f}).\n"
    )
    args.out.write_text(header + yaml.safe_dump(best_model, sort_keys=False))
    print(f"\nbest: {best} {dict(table.loc[best, list(grid_for(model_type))])}\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
