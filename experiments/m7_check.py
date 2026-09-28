"""M7 check on real validation days: improved preprocessing vs the baseline's.

    python -m experiments.m7_check --config experiments/configs/base.yaml --out docs/reports/m7_improved_preprocessing.md

For each validation fold (W_snn = 1) both pipelines are fitted on the training day and applied to
the test day with the same `new_mean`. Reports realised mean input probabilities (the matched
input rate of §6.4.3), clipping, the reward streams of the R-STDP pools, how strongly each
encoding relates to the next moves, and a causality check of the improved pipeline on real data.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from snn_hft.analysis.compare import markdown
from snn_hft.backtest.cache import BarCache, HawkesParamCache
from snn_hft.config.loader import load_raw
from snn_hft.config.schema import BarsConfig, DataConfig, HawkesRSTDPConfig, OutputConfig, PaperSNNConfig, PeriodsConfig
from snn_hft.data.containers import DayData
from snn_hft.data.store import make_store
from snn_hft.evaluation.spike_metrics import SpikeEvaluator
from snn_hft.models.hawkes.provider import HawkesParamProvider
from snn_hft.preprocessing.bars import aggregate_vwap
from snn_hft.preprocessing.hawkes_steps import candidate_events_source
from snn_hft.preprocessing.pipelines import hawkes_pipeline, paper_pipeline
from snn_hft.testing.causality import check_causal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--new-mean", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--w-h", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    raw = load_raw(args.config)
    data = DataConfig.model_validate(raw.get("data", {}))
    bars = BarsConfig.model_validate(raw.get("bars", {}))
    out_cfg = OutputConfig.model_validate(raw.get("output", {}))
    periods = PeriodsConfig.model_validate(raw.get("periods", {}))
    store = make_store(data)
    bar_cache = BarCache(store, data.symbol, bars.vwap_num, out_cfg.cache_dir)
    params_cache = HawkesParamCache(out_cfg.cache_dir, store.source.venue, data.symbol, data.dataset, bars.vwap_num)
    improved = HawkesRSTDPConfig()
    evaluator = SpikeEvaluator(3)
    t0 = time.time()

    rows, rewards = [], []
    for test in periods.validation.days()[: args.folds]:
        train = test - timedelta(days=1)
        raw_train, raw_test = store.day_data(data.symbol, train), store.day_data(data.symbol, test)
        for new_mean in args.new_mean:
            provider = HawkesParamProvider(
                improved.hawkes, args.w_h, candidate_events_source(bar_cache.get, improved.hawkes.time_axis), params_cache
            )
            cfg = improved.model_copy(update={"input": improved.input.model_copy(update={"new_mean": new_mean})})
            imp = hawkes_pipeline(bars, cfg, provider)
            imp.fit([raw_train])
            imp_day = imp.transform(raw_test)
            base = paper_pipeline(bars, PaperSNNConfig(input={"new_mean": new_mean}))
            base.fit([raw_train])
            base_day = base.transform(raw_test)
            labels = evaluator.labels(imp_day.bars.vwap)
            ok = labels.evaluable
            for name, dd in (("paper (z-score)", base_day), ("improved (Hawkes)", imp_day)):
                prob = dd.channel_prob
                rows.append({
                    "day": test, "new_mean": new_mean, "model": name,
                    "mean_prob_X1": prob[:, 0].mean(), "mean_prob_X2": prob[:, 1].mean(),
                    "clipped_at_1": float((prob >= 1.0).mean()), "at_0": float((prob <= 0.0).mean()),
                    "sp_next_moves": spearmanr(prob[ok].sum(axis=1), labels.strength[ok])[0],
                })
            if new_mean == args.new_mean[0]:
                r = imp_day.reward_mom
                ev = imp_day.branching
                rewards.append({
                    "day": test, "events": len(ev), "event_share": len(ev) / imp_day.n_bars,
                    "threshold": imp_day.extras["hawkes_params"].event_threshold,
                    "rho": imp_day.extras["hawkes_params"].spectral_radius,
                    "p_bg": ev["p_bg"].mean(), "p_same": ev["p_same"].mean(), "p_cross": ev["p_cross"].mean(),
                    "reward_mom_mean": ev["momentum"].mean(), "reward_mom_std": ev["momentum"].std(),
                    "share_positive": float((ev["momentum"] > 0).mean()), "share_negative": float((ev["momentum"] < 0).mean()),
                    "bars_with_reward": float((r != 0).mean()),
                })
    df, rw = pd.DataFrame(rows), pd.DataFrame(rewards)

    # Causality of the improved pipeline on real data (history, test day and a later day).
    test = periods.validation.days()[0] + timedelta(days=args.w_h + 1)
    world_days = [test - timedelta(days=k) for k in range(args.w_h + 1, 0, -1)] + [test, test + timedelta(days=1)]
    world = {d: store.get_day(data.symbol, d) for d in world_days}

    train_day = test - timedelta(days=1)

    def run(w):
        history = candidate_events_source(lambda d: aggregate_vwap(w[d].df, bars.vwap_num), improved.hawkes.time_axis)
        pipe = hawkes_pipeline(bars, improved, HawkesParamProvider(improved.hawkes, args.w_h, history))
        pipe.fit([DayData(day=train_day, trades=w[train_day])])
        return pipe.transform(DayData(day=test, trades=w[test]))

    n_bars = len(world[test]) // bars.vwap_num
    t1 = time.time()
    check_causal(run, world, test, (0, 100, n_bars // 2, n_bars - 2), bars.vwap_num)
    causal_s = time.time() - t1

    summary = df.groupby(["new_mean", "model"]).mean(numeric_only=True).reset_index()
    t_inputs = pd.DataFrame({
        "new_mean": summary["new_mean"], "encoding": summary["model"],
        "mean input prob X1": summary["mean_prob_X1"].map("{:.4f}".format),
        "mean input prob X2": summary["mean_prob_X2"].map("{:.4f}".format),
        "share clipped at 1": summary["clipped_at_1"].map("{:.4f}".format),
        "share at 0": summary["at_0"].map("{:.4f}".format),
        "Spearman(input, next moves)": summary["sp_next_moves"].map("{:.3f}".format),
    })
    rmean = rw.mean(numeric_only=True)
    t_rewards = pd.DataFrame([{k: f"{rmean[k]:.3f}" for k in (
        "event_share", "threshold", "rho", "p_bg", "p_same", "p_cross", "reward_mom_mean", "reward_mom_std",
        "share_positive", "share_negative", "bars_with_reward")}])
    text = f"""# M7 — improved preprocessing on real validation days

Setup: {bars.vwap_num} trades per bar, bar clock, events above the fit-window 0.9 quantile of |d| (U9),
`LinearNormalized` marks, W_h = {args.w_h}, W_snn = 1; first {args.folds} validation folds
({periods.validation.days()[0]} → {periods.validation.days()[args.folds - 1]}). Both pipelines are fitted on the
training day and applied to the test day with the same `new_mean`. Runtime {time.time() - t0:.0f} s.

## Input rates (§6.4.3)

With equal `new_mean`, both encodings deliver the configured mean input probability per channel on
the test days, so input rates match when the models are configured alike (U7 lets each model tune
its own `new_mean`; realised rates are logged per day in every run's `input_stats.csv`).

{markdown(t_inputs)}

*Next moves* = spike strength S_strength (mean absolute return over the next 3 bars).

## Reward streams of the R-STDP pools (new_mean = {args.new_mean[0]})

`reward_mom = M = p_same − p_cross` at event bars, `reward_rev = −reward_mom`, 0 elsewhere.

{markdown(t_rewards)}

## Causality on real data

The full improved pipeline passed the causality harness on {test} (history days, test day and the
following day perturbed after bars 0, 100, {n_bars // 2}, {n_bars - 2}; three perturbation modes each),
in {causal_s:.0f} s.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
