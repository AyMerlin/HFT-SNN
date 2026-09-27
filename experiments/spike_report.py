"""Table-3-style spike report for the signal models of an experiment (M4).

    python -m experiments.spike_report --config experiments/configs/paper_baseline.yaml \
        --set backtest.split=validation --max-folds 10

Walk-forward over the first `--max-folds` days of the experiment's period: fit on the
W_snn previous days, generate signals on the test day, evaluate the paper's spike
definitions on training and test days, and run the health checks. Writes a per-day CSV
to ``results/<experiment>/spike_report.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from snn_hft.backtest.splitter import WalkForwardSplitter
from snn_hft.config import expand_runs, load_config
from snn_hft.data.store import make_store
from snn_hft.evaluation.health import check_health
from snn_hft.evaluation.spike_metrics import SpikeEvaluator
from snn_hft.preprocessing.bars import VWAPBarAggregator
from snn_hft.signals.factory import make_signal_model


class DayCache:
    """Keeps the most recent raw days in memory; consecutive folds share days."""

    def __init__(self, store, symbol: str, size: int = 12):
        self.store, self.symbol, self.size = store, symbol, size
        self._days: OrderedDict = OrderedDict()

    def get(self, day):
        if day not in self._days:
            self._days[day] = self.store.day_data(self.symbol, day)
            while len(self._days) > self.size:
                self._days.popitem(last=False)
        return self._days[day]


def table3(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model", "split"])
    return pd.DataFrame(
        {
            "days": grouped["day"].nunique(),
            "spike accuracy": grouped["accuracy"].mean(),
            "chance level": grouped["base_accuracy"].mean(),
            "momentum spike %": grouped["momentum_pct"].mean(),
            "momentum base %": grouped["base_momentum_pct"].mean(),
            "spikes per day": grouped["n_signals"].mean(),
            "signal rate": grouped["signal_rate"].mean(),
            "health warnings": grouped["health_warnings"].sum(),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    store = make_store(cfg.data)
    days = DayCache(store, cfg.data.symbol)
    bars_step = VWAPBarAggregator(cfg.bars.vwap_num)
    evaluator = SpikeEvaluator(cfg.evaluation.eval_window)
    test_days = cfg.eval_period.days()[: args.max_folds]
    # One run per (model, W_snn, seed); the strategy rule does not affect signals.
    runs = {(r.signal.name, r.w_snn, r.seed): r for r in expand_runs(cfg)}.values()

    rows = []
    t0 = time.time()
    for run in runs:
        for fold in WalkForwardSplitter(run.w_snn).folds(test_days):
            model = make_signal_model(run)
            model.fit([days.get(d) for d in fold.train_days], run.seed)
            signals = model.generate(days.get(fold.test_day), run.seed)
            test_vwap = bars_step.transform(days.get(fold.test_day)).bars.vwap
            evaluated = [("train", s, d.bars.vwap) for s, d in zip(model.train_signals, model.train_days)]
            evaluated.append(("test", signals, test_vwap))
            pools = {p.name: p.size for p in model.network.populations if p.name.startswith("H")}
            for split, series, vwap in evaluated:
                health = check_health(
                    series, pools, model.ticks_per_bar, model.core.lif.t_ref, cfg.evaluation.health
                )
                rows.append(
                    {
                        "model": run.signal.name, "w_snn": run.w_snn, "seed": run.seed, "split": split,
                        "day": series.day, "test_day": fold.test_day,
                        **evaluator.evaluate(vwap, series).as_dict(),
                        "health_warnings": len(health.warnings), "health": "; ".join(health.warnings),
                        **{f"rate_{p}": r for p, r in health.pool_rates.items()},
                    }
                )
            print(f"{run.signal.name} seed {run.seed} {fold.test_day}: {len(signals):,} signals "
                  f"({signals.rate:.1%}), {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = cfg.output.results_dir / cfg.name / "spike_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print("\n" + table3(df).to_string())
    print(f"\nper-day results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
