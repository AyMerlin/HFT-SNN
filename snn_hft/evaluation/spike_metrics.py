"""The paper's spike definitions (§8.4), identical for every model.

With X = P = the vwap series of one day (0-indexed bars) and window w:

    r_t          = |X_{t+1} / X_t − 1|
    r_pivot      = median(r_t) over the whole day          (evaluation only, as in the paper)
    S_strength   = (r_{t+1} + … + r_{t+w}) / w
    real spike   ⇔ S_strength > r_pivot
    P_prior_avg  = mean(P_{t−w} … P_{t−1})                 (paper's P_{t+window} typo fixed, P2)
    P_post_avg   = mean(P_{t+1} … P_{t+w})
    mom_rev_flag = (P_prior_avg − P_t) · (P_post_avg − P_t);  reversion ⇔ flag > 0, else momentum

A bar is evaluable if both windows fit inside the day: w ≤ t ≤ n − 2 − w (S_strength needs
X_{t+w+1}). Signals on other bars are excluded and counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from snn_hft.signals.base import SignalSeries

_REL_TOL = 1e-12


def _zero_small(diff: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Treat differences at floating-point noise level as exactly zero (flat prices → momentum)."""
    return np.where(np.abs(diff) <= _REL_TOL * np.abs(ref), 0.0, diff)


@dataclass(frozen=True)
class BarLabels:
    """Per-bar labels of one day, NaN / False where a bar is not evaluable."""

    evaluable: np.ndarray  # bool
    real: np.ndarray  # bool
    momentum: np.ndarray  # bool
    strength: np.ndarray  # S_strength
    mom_rev_flag: np.ndarray
    r_pivot: float


@dataclass(frozen=True)
class DaySpikeMetrics:
    n_signals: int
    n_evaluated: int
    n_excluded: int
    n_real: int
    n_momentum: int
    accuracy: float  # real / evaluated
    momentum_pct: float  # momentum / evaluated
    base_accuracy: float  # share of all evaluable bars that are real: expected accuracy of random timing
    base_momentum_pct: float
    signal_rate: float  # signals per bar

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


class SpikeEvaluator:
    def __init__(self, window: int = 3):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window

    def labels(self, vwap: np.ndarray) -> BarLabels:
        p = np.asarray(vwap, dtype=np.float64)
        n, w = len(p), self.window
        evaluable = np.zeros(n, dtype=bool)
        strength = np.full(n, np.nan)
        flag = np.full(n, np.nan)
        r = np.abs(p[1:] / p[:-1] - 1.0) if n > 1 else np.empty(0)
        r_pivot = float(np.median(r)) if len(r) else np.nan
        lo, hi = w, n - 2 - w
        if hi >= lo:
            t = np.arange(lo, hi + 1)
            # Window means computed per window (no running sums, so no cancellation at price scale).
            r_win = sliding_window_view(r, w).mean(axis=1)  # r_win[i] = mean(r_i … r_{i+w−1})
            p_win = sliding_window_view(p, w).mean(axis=1)
            strength[t] = r_win[t + 1]  # r_{t+1} … r_{t+w}
            prior = _zero_small(p_win[t - w] - p[t], p[t])  # P_{t−w} … P_{t−1}
            post = _zero_small(p_win[t + 1] - p[t], p[t])  # P_{t+1} … P_{t+w}
            flag[t] = prior * post
            evaluable[t] = True
        real = evaluable & (strength > r_pivot)
        momentum = evaluable & ~(flag > 0)
        return BarLabels(evaluable, real, momentum, strength, flag, r_pivot)

    def evaluate(self, vwap: np.ndarray, signals: SignalSeries | np.ndarray) -> DaySpikeMetrics:
        labels = self.labels(vwap)
        bars = signals.bar_idx if isinstance(signals, SignalSeries) else np.asarray(signals, dtype=np.int64)
        return self.metrics(labels, bars)

    @staticmethod
    def metrics(labels: BarLabels, bars: np.ndarray) -> DaySpikeMetrics:
        n_bars = len(labels.evaluable)
        used = bars[labels.evaluable[bars]]
        n_eval = len(used)
        n_real = int(labels.real[used].sum())
        n_mom = int(labels.momentum[used].sum())
        n_evaluable = int(labels.evaluable.sum())
        return DaySpikeMetrics(
            n_signals=len(bars),
            n_evaluated=n_eval,
            n_excluded=len(bars) - n_eval,
            n_real=n_real,
            n_momentum=n_mom,
            accuracy=n_real / n_eval if n_eval else np.nan,
            momentum_pct=n_mom / n_eval if n_eval else np.nan,
            base_accuracy=labels.real.sum() / n_evaluable if n_evaluable else np.nan,
            base_momentum_pct=labels.momentum.sum() / n_evaluable if n_evaluable else np.nan,
            signal_rate=len(bars) / n_bars if n_bars else 0.0,
        )
