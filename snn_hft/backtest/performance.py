"""Table-4 performance metrics (§8.5), computed from per-day aggregates of trades.

Per-day aggregates (pnl, n_trades, n_pos, sum_pos, n_neg, sum_neg) are enough for every
metric, so naive repetitions never need to keep individual trades.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DAILY_COLUMNS = ["pnl", "n_trades", "n_pos", "sum_pos", "n_neg", "sum_neg"]


def daily_aggregate(net_returns: np.ndarray) -> dict[str, float]:
    """Aggregate one day's trade returns (all trades close the same day, notional 1, additive)."""
    r = np.asarray(net_returns, dtype=np.float64)
    pos, neg = r[r > 0], r[r < 0]
    return {
        "pnl": float(r.sum()),
        "n_trades": int(len(r)),
        "n_pos": int(len(pos)),
        "sum_pos": float(pos.sum()),
        "n_neg": int(len(neg)),
        "sum_neg": float(neg.sum()),
    }


class PerformanceCalculator:
    def __init__(self, annualization_days: int = 365):
        self.annualization_days = annualization_days

    def compute(self, daily: pd.DataFrame) -> dict[str, float]:
        """`daily` has one row per evaluated day (days without trades included, pnl 0)."""
        pnl = daily["pnl"].to_numpy(dtype=np.float64)
        n_days = len(pnl)
        std = float(pnl.std(ddof=1)) if n_days > 1 else math.nan
        ann = math.sqrt(self.annualization_days)
        n_trades = int(daily["n_trades"].sum())
        n_pos, n_neg = int(daily["n_pos"].sum()), int(daily["n_neg"].sum())
        mean_win = daily["sum_pos"].sum() / n_pos if n_pos else math.nan
        mean_loss = abs(daily["sum_neg"].sum()) / n_neg if n_neg else math.nan
        return {
            "accumulated_return": float(pnl.sum()),
            "annualized_volatility": std * ann,
            "sharpe": float(pnl.mean()) / std * ann if std and std > 0 else math.nan,
            "win_rate": n_pos / n_trades if n_trades else math.nan,
            "profit_loss_ratio": mean_win / mean_loss if n_pos and n_neg else math.nan,
            "trades_per_day": n_trades / n_days if n_days else math.nan,
            "mean_daily_pnl": float(pnl.mean()) if n_days else math.nan,
            "n_days": n_days,
            "n_trades": n_trades,
        }
