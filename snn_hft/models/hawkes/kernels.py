"""Excitation kernels (§5.2). The numba code in `process.py` is specialised to the exponential
kernel; this class states it explicitly and is used by reference computations and tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Kernel(ABC):
    @abstractmethod
    def value(self, dt: np.ndarray) -> np.ndarray:
        """φ(dt) for dt ≥ 0."""

    @abstractmethod
    def integral(self, dt: np.ndarray) -> np.ndarray:
        """∫_0^dt φ(s) ds."""


@dataclass(frozen=True)
class ExponentialKernel(Kernel):
    """φ(dt) = α · exp(−β · dt); branching ratio α / β."""

    alpha: float
    beta: float

    def value(self, dt):
        return self.alpha * np.exp(-self.beta * np.asarray(dt, dtype=float))

    def integral(self, dt):
        return self.alpha / self.beta * (1.0 - np.exp(-self.beta * np.asarray(dt, dtype=float)))

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta
