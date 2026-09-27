"""Volume-weighted average price bars (§4.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from snn_hft.data.containers import BAR_COLUMNS, BarSeries, DayData
from snn_hft.preprocessing.base import PreprocessingStep


def aggregate_vwap(trades: pd.DataFrame, num: int) -> BarSeries:
    """Bars of `num` consecutive trades: vwap = Σ p·q / Σ q. An incomplete final group is dropped.

    Bar b depends only on trades [b·num, (b+1)·num), so it is known once its last trade prints.
    """
    n_bars = len(trades) // num
    n = n_bars * num
    price = trades["price"].to_numpy()[:n].reshape(n_bars, num)
    qty = trades["qty"].to_numpy()[:n].reshape(n_bars, num)
    ts = trades["ts_ns"].to_numpy()[:n].reshape(n_bars, num)
    volume = qty.sum(axis=1)
    notional = (price * qty).sum(axis=1)
    vwap = np.divide(notional, volume, out=price.mean(axis=1), where=volume > 0)
    df = pd.DataFrame(
        {
            "bar_idx": np.arange(n_bars, dtype=np.int64),
            "ts_start_ns": ts[:, 0],
            "ts_end_ns": ts[:, -1],
            "vwap": vwap,
            "volume": volume,
            "n_trades": np.full(n_bars, num, dtype=np.int64),
        }
    ).astype(BAR_COLUMNS)
    return BarSeries(df)


class VWAPBarAggregator(PreprocessingStep):
    """Paper Step 1: vwap over windows of `num` transactions (paper: 10)."""

    def __init__(self, num: int = 10, keep_trades: bool = False):
        if num < 1:
            raise ValueError("num must be >= 1")
        self.num = num
        self.keep_trades = keep_trades

    def transform(self, day: DayData) -> DayData:
        if day.trades is None:
            raise ValueError(f"{day.day}: VWAPBarAggregator needs trades")
        bars = aggregate_vwap(day.trades.df, self.num)
        return day.with_fields(bars=bars, trades=day.trades if self.keep_trades else None)

    def __repr__(self) -> str:
        return f"VWAPBarAggregator(num={self.num})"
