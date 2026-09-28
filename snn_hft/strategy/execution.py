"""Execution (§7.2): entry/exit bars and per-trade returns, plus the latency option (U2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from snn_hft.config.schema import ExecutionConfig
from snn_hft.data.containers import BarSeries

__all__ = ["ExecutionConfig", "Trade", "TRADE_FRAME_COLUMNS", "entry_bars", "execute", "per_signal"]

TRADE_FRAME_COLUMNS = [
    "signal_bar", "entry_bar", "exit_bar", "direction", "entry_price", "exit_price",
    "gross_return", "fee", "net_return",
]


@dataclass(frozen=True)
class Trade:
    day: date
    signal_bar: int
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    gross_return: float
    fee: float
    net_return: float


def entry_bars(bars: BarSeries, signal_bars: np.ndarray, execution: ExecutionConfig) -> np.ndarray:
    """Entry bar per signal, −1 if the day ends first.

    Paper rule: t + entry_delay. With latency L > 0, the first bar at or after t + entry_delay
    whose first trade is at least L after the signal bar's last trade.
    """
    n = len(bars)
    entry = signal_bars + execution.entry_delay
    if execution.latency_ms > 0 and len(signal_bars):
        ts_start = bars.df["ts_start_ns"].to_numpy()
        target = bars.df["ts_end_ns"].to_numpy()[signal_bars] + int(round(execution.latency_ms * 1e6))
        entry = np.maximum(entry, np.searchsorted(ts_start, target, side="left"))
    return np.where(entry < n, entry, -1)


def per_signal(bars: BarSeries, signal_bars: np.ndarray, directions: np.ndarray, execution: ExecutionConfig) -> dict:
    """Trade arrays aligned with `signal_bars`; `valid` marks signals that produce a trade.

    Exit at entry + holding, or at the day's last bar if that comes first (end-of-day close).
    r = direction · (P_exit / P_entry − 1) − 2 · fee_rate. Invalid entries hold NaN / −1.
    """
    signal_bars = np.asarray(signal_bars, dtype=np.int64)
    directions = np.asarray(directions, dtype=np.int8)
    entry = entry_bars(bars, signal_bars, execution)
    valid = (directions != 0) & (entry >= 0)
    exit_ = np.where(valid, np.minimum(entry + execution.holding, len(bars) - 1), -1)
    vwap = bars.vwap
    p_in = np.where(valid, vwap[np.maximum(entry, 0)], np.nan)
    p_out = np.where(valid, vwap[np.maximum(exit_, 0)], np.nan)
    gross = directions * (p_out / p_in - 1.0)
    fee = 2.0 * execution.fee_rate
    return {
        "signal_bar": signal_bars, "entry_bar": entry, "exit_bar": exit_, "direction": directions,
        "entry_price": p_in, "exit_price": p_out, "gross_return": gross,
        "fee": np.where(valid, fee, np.nan), "net_return": gross - fee, "valid": valid,
    }


def execute(bars: BarSeries, signal_bars: np.ndarray, directions: np.ndarray, execution: ExecutionConfig) -> pd.DataFrame:
    """Trades for the signals with a non-zero direction; one position per signal, notional 1."""
    arrays = per_signal(bars, signal_bars, directions, execution)
    valid = arrays.pop("valid")
    return pd.DataFrame({k: v[valid] for k, v in arrays.items()})[TRADE_FRAME_COLUMNS]
