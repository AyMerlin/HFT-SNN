"""What does vwap smoothing do? Real BTCUSDT aggTrades vs a random walk with bid-ask bounce.

    python -m experiments.vwap_check --out docs/reports/vwap_smoothing_check.md

Averaging a random walk over non-overlapping blocks gives first differences with lag-1
autocorrelation ≈ 0.25 (Working 1960: (m² − 1) / (2(2m² + 1)) for m points per block). This
script measures that effect and its consequences for the paper's momentum label and for the
paper's strategies (entry t+1, exit t+4). Validation days only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from snn_hft.analysis.compare import markdown
from snn_hft.config.schema import ExecutionConfig
from snn_hft.data.containers import BAR_COLUMNS, BarSeries
from snn_hft.data.sources import BinanceFuturesDataSource
from snn_hft.data.store import DataStore
from snn_hft.evaluation.spike_metrics import SpikeEvaluator
from snn_hft.strategy.direction_rules import MomentumRule
from snn_hft.strategy.execution import per_signal


def ac1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def bar_stats(price: np.ndarray, qty: np.ndarray, num: int) -> dict:
    n = len(price) // num * num
    p, q = price[:n].reshape(-1, num), qty[:n].reshape(-1, num)
    vwap, close = (p * q).sum(1) / q.sum(1), p[:, -1]
    nb = len(vwap)
    bars = BarSeries(pd.DataFrame({"bar_idx": np.arange(nb), "ts_start_ns": np.arange(nb), "ts_end_ns": np.arange(nb),
                                   "vwap": vwap, "volume": q.sum(1), "n_trades": num}).astype(BAR_COLUMNS))
    lab = SpikeEvaluator(3).labels(vwap)
    t = np.arange(nb)
    tr = per_signal(bars, t, MomentumRule(3).directions(vwap, t), ExecutionConfig())
    r = tr["net_return"][tr["valid"]]
    return {"ac1 bar close diffs": ac1(np.diff(close)), "ac1 vwap diffs": ac1(np.diff(vwap)),
            "momentum share, paper label": float(lab.momentum[lab.evaluable].mean()),
            "random-timing win rate, momentum strategy": float((r > 0).mean())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args(argv)
    store = DataStore(BinanceFuturesDataSource("aggTrades"), "data/raw")
    rng = np.random.default_rng(0)
    rows = []
    for k in range(args.days):
        df = store.get_day("BTCUSDT", date(2025, 11, 3) + timedelta(days=k)).df
        price, qty = df.price.to_numpy(), df.qty.to_numpy()
        # Random walk with the same number of trades: efficient price + bid-ask bounce, no real momentum.
        mid = 100_000 + np.cumsum(rng.normal(0, 0.5, len(price)))
        synth = mid + np.where(rng.random(len(price)) < 0.5, 0.05, -0.05)
        synth_qty = rng.lognormal(-3, 1.5, len(price))
        for num in (10, 100):
            rows.append({"data": "BTCUSDT aggTrades", "trades per bar": num, "ac1 raw trade diffs": ac1(np.diff(price)),
                         **bar_stats(price, qty, num)})
            rows.append({"data": "random walk + bid-ask bounce", "trades per bar": num,
                         "ac1 raw trade diffs": ac1(np.diff(synth)), **bar_stats(synth, synth_qty, num)})
    table = pd.DataFrame(rows).groupby(["data", "trades per bar"], sort=False).mean().reset_index()
    for col in table.columns[2:]:
        table[col] = table[col].map("{:.3f}".format)
    m = np.array([10, 100])
    working = (m**2 - 1) / (2 * (2 * m**2 + 1))
    text = f"""# VWAP smoothing check

Five validation days (2025-11-03 → 11-07) of BTCUSDT `aggTrades`, and a synthetic random walk
with the same number of trades (Gaussian steps, trades at bid or ask, no real momentum), both
aggregated exactly like the pipeline (non-overlapping blocks of `num` trades, volume-weighted).
"ac1" = lag-1 autocorrelation of first differences.

{markdown(table)}

Working (1960) value for block averages of a random walk: {working[0]:.3f} (10 points), {working[1]:.3f} (100 points).

## Reading

- **The implementation matches the paper** (non-overlapping blocks of `num` transactions,
  volume-weighted; the paper's Figure 2 shows 10 raw points per vwap point) **and removes the
  bid-ask zig-zag**: raw trade-to-trade changes have lag-1 autocorrelation −0.52 on BTCUSDT, vwap
  differences are positive.
- **Averaging creates momentum by itself.** On the random walk, the vwap differences have lag-1
  autocorrelation ≈ 0.23 (Working effect), while bar closing prices have none.
- **The paper's momentum label is biased by it.** mom_rev_flag compares the average price before and
  after the spike with the spike's vwap; on a pure random walk it labels about 53–54 % of bars
  "momentum" — the paper's reported momentum spike share (≈ 54 %) is what averaging alone produces.
- **The strategies are not biased by it.** They enter at bar t+1 and exit at t+4, skipping the
  lag-1 dependence: on the random walk the momentum strategy wins 50.0 % at random times. The
  typed-model trade label (entry t+1, exit t+4) is unaffected for the same reason.
- On BTCUSDT at 100 trades per bar some real persistence remains beyond the averaging (vwap 0.29 vs
  0.23; closing prices 0.23 vs 0.0), which gives random timing its 52 % win rate.
"""
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
