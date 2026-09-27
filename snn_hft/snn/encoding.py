"""Spike encoders (§6.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SpikeEncoder(ABC):
    @abstractmethod
    def encode(self, channel_prob: np.ndarray, ticks_per_bar: int, rng: np.random.Generator) -> np.ndarray:
        """(N bars, C channels) of values → spike trains of shape (N·T, C), dtype uint8."""


class PoissonEncoder(SpikeEncoder):
    """X_c[b·T + s] ~ Bernoulli(channel_prob[b, c]), independently for every tick s.

    Random numbers are drawn row by row in chunks, so the spikes of the first k bars
    depend only on the seed and on those k bars, not on the day's length (causality).
    """

    def __init__(self, chunk_bars: int = 65_536):
        self.chunk_bars = chunk_bars

    def encode(self, channel_prob: np.ndarray, ticks_per_bar: int, rng: np.random.Generator) -> np.ndarray:
        prob = np.asarray(channel_prob, dtype=np.float64)
        if prob.ndim != 2:
            raise ValueError("channel_prob must have shape (n_bars, n_channels)")
        if np.isnan(prob).any() or (prob < 0).any() or (prob > 1).any():
            raise ValueError("channel_prob must lie in [0, 1]")
        n_bars, n_ch = prob.shape
        out = np.empty((n_bars * ticks_per_bar, n_ch), dtype=np.uint8)
        for start in range(0, n_bars, self.chunk_bars):
            stop = min(start + self.chunk_bars, n_bars)
            p = np.repeat(prob[start:stop], ticks_per_bar, axis=0)
            out[start * ticks_per_bar : stop * ticks_per_bar] = rng.random(p.shape) < p
        return out
