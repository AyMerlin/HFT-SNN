"""The paper's direction rules (§7.1, Strategy Logic 1–3), evaluated at a signal bar t on vwap P.

Each rule returns +1 (long), −1 (short) or 0 (no transaction). Differences at floating-point
noise level (1e-12 relative) count as exactly zero, i.e. "no transaction".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_REL_TOL = 1e-12


def _sign(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) <= _REL_TOL * np.abs(scale), 0, np.sign(x)).astype(np.int8)


class DirectionRule(ABC):
    name: str

    @property
    @abstractmethod
    def min_history(self) -> int:
        """Bars needed before t; signals at t < min_history get no transaction."""

    @abstractmethod
    def directions(self, vwap: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Directions for signal bars t (array), int8 in {−1, 0, +1}."""

    def direction(self, vwap: np.ndarray, t: int) -> int:
        return int(self.directions(vwap, np.array([t]))[0])

    def _valid(self, vwap: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (t >= self.min_history) & (t < len(vwap))


class MomentumRule(DirectionRule):
    """position_flag = mean(P_{t−w} … P_{t−1}) − P_t;  > 0 → short, < 0 → long, = 0 → none (P2 typo fixed)."""

    name = "momentum"

    def __init__(self, window: int = 3):
        self.window = window

    @property
    def min_history(self) -> int:
        return self.window

    def directions(self, vwap, t):
        p, t = np.asarray(vwap, dtype=np.float64), np.asarray(t, dtype=np.int64)
        out = np.zeros(len(t), dtype=np.int8)
        ok = self._valid(p, t)
        if len(p) >= self.window and ok.any():
            means = sliding_window_view(p, self.window).mean(axis=1)  # means[i] = mean(P_i … P_{i+w−1})
            flag = means[t[ok] - self.window] - p[t[ok]]
            out[ok] = -_sign(flag, p[t[ok]])
        return out


class AlexandersFilterRule(DirectionRule):
    """ALF = (P_t / P_{t−n} − 1) · 100;  > 0 → long, < 0 → short, = 0 → none."""

    name = "alexanders_filter"

    def __init__(self, n: int = 1):
        self.n = n

    @property
    def min_history(self) -> int:
        return self.n

    def directions(self, vwap, t):
        p, t = np.asarray(vwap, dtype=np.float64), np.asarray(t, dtype=np.int64)
        out = np.zeros(len(t), dtype=np.int8)
        ok = self._valid(p, t)
        alf = (p[t[ok]] / p[t[ok] - self.n] - 1.0) * 100.0
        out[ok] = _sign(alf, np.full(ok.sum(), 100.0))
        return out


class StochasticOscillatorRule(DirectionRule):
    """%K = (P_t − L_n)/(H_n − L_n) · 100 over P_{t−n+1} … P_t;  > 50 → long, < 50 → short,
    = 50 or H_n = L_n → none (P16)."""

    name = "stochastic_oscillator"

    def __init__(self, n: int = 3):
        self.n = n

    @property
    def min_history(self) -> int:
        return self.n - 1

    def directions(self, vwap, t):
        p, t = np.asarray(vwap, dtype=np.float64), np.asarray(t, dtype=np.int64)
        out = np.zeros(len(t), dtype=np.int8)
        ok = self._valid(p, t)
        if len(p) >= self.n and ok.any():
            win = sliding_window_view(p, self.n)  # win[i] = P_i … P_{i+n−1}
            tt = t[ok]
            lo, hi = win[tt - self.n + 1].min(axis=1), win[tt - self.n + 1].max(axis=1)
            flat = hi - lo <= _REL_TOL * np.abs(p[tt])
            k = np.where(flat, 50.0, (p[tt] - lo) / np.where(flat, 1.0, hi - lo) * 100.0)
            out[ok] = np.where(flat, 0, _sign(k - 50.0, np.full(len(tt), 100.0)))
        return out


def make_rule(name: str, momentum_window: int = 3, alf_n: int = 1, stoch_n: int = 3) -> DirectionRule:
    if name == "momentum":
        return MomentumRule(momentum_window)
    if name == "alexanders_filter":
        return AlexandersFilterRule(alf_n)
    if name == "stochastic_oscillator":
        return StochasticOscillatorRule(stoch_n)
    raise ValueError(f"unknown direction rule {name!r}")
