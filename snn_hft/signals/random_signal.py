"""Naive benchmark (§6.6): the same number of signals as a reference, at random bars."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from snn_hft.data.containers import DayData
from snn_hft.signals.base import SignalModel, SignalSeries
from snn_hft.utils.seeding import rng_for


class RandomSignalModel(SignalModel):
    """Samples n distinct bars uniformly from the eligible bars of the day.

    Eligible: enough history for the direction rule (t ≥ min_history) and at least
    `entry_delay` bars left (t ≤ n_bars − 1 − entry_delay). n = number of reference signals.
    `seed` is the repetition index (0..R−1); `run_seed` separates the SNN seeds' references.
    """

    def __init__(self, min_history: int, entry_delay: int, reference: SignalSeries | None = None, run_seed: int = 0):
        self.min_history = min_history
        self.entry_delay = entry_delay
        self.reference = reference
        self.run_seed = run_seed

    @property
    def model_id(self) -> str:
        return f"random-h{self.min_history}-d{self.entry_delay}"

    def fit(self, train_days: Sequence[DayData], seed: int) -> RandomSignalModel:
        return self

    @staticmethod
    def sample(n_bars: int, n: int, min_history: int, entry_delay: int, rng: np.random.Generator) -> np.ndarray:
        lo, hi = min_history, n_bars - 1 - entry_delay
        count = max(hi - lo + 1, 0)
        n = min(n, count)
        return np.sort(rng.choice(count, size=n, replace=False)) + lo if n else np.empty(0, dtype=np.int64)

    def generate(self, day: DayData, seed: int) -> SignalSeries:
        if self.reference is None:
            raise RuntimeError("RandomSignalModel needs a reference SignalSeries")
        rng = rng_for(seed, "naive", self.reference.day, self.run_seed)
        bars = self.sample(self.reference.n_bars, len(self.reference), self.min_history, self.entry_delay, rng)
        return SignalSeries(day=self.reference.day, bar_idx=bars, n_bars=self.reference.n_bars, model_id=self.model_id)
