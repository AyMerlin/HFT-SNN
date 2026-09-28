"""Mark functions g_k(m) (§5.6), normalised per source type k on the fit window.

Normalisation to mean 1 over the fit window makes the branching matrix A_mk = α_mk / β_mk,
and the fitted normalisation constants are frozen with the parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

N_TYPES = 2  # 0 = up, 1 = down


class MarkFunction(ABC):
    name: str

    @abstractmethod
    def fit(self, marks: np.ndarray, types: np.ndarray) -> MarkFunction:
        """Learn the per-type normalisation from the fit window's events."""

    @abstractmethod
    def __call__(self, marks: np.ndarray, types: np.ndarray) -> np.ndarray:
        """g_{k_i}(m_i) for every event."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @staticmethod
    def _per_type(marks: np.ndarray, types: np.ndarray, stat) -> list[float]:
        out = []
        for k in range(N_TYPES):
            mk = marks[types == k]
            if len(mk) == 0:
                raise ValueError(f"no events of type {k} in the fit window")
            out.append(float(stat(mk)))
        return out


class Unmarked(MarkFunction):
    name = "unmarked"

    def fit(self, marks, types):
        return self

    def __call__(self, marks, types):
        return np.ones(len(marks))

    def to_dict(self):
        return {"name": self.name}


class LinearNormalized(MarkFunction):
    """g_k(m) = m / mean_k(m)."""

    name = "linear_normalized"

    def __init__(self, mean: list[float] | None = None):
        self.mean = mean

    def fit(self, marks, types):
        self.mean = self._per_type(marks, types, np.mean)
        return self

    def __call__(self, marks, types):
        if self.mean is None:
            raise RuntimeError("LinearNormalized used before fit")
        return marks / np.asarray(self.mean)[types]

    def to_dict(self):
        return {"name": self.name, "mean": self.mean}


class Saturating(MarkFunction):
    """g_k(m) = (1 − exp(−m / c_k)) / E_k[1 − exp(−m / c_k)], c_k = median mark of type k."""

    name = "saturating"

    def __init__(self, scale: list[float] | None = None, norm: list[float] | None = None):
        self.scale, self.norm = scale, norm

    def fit(self, marks, types):
        self.scale = self._per_type(marks, types, np.median)
        raw = 1.0 - np.exp(-marks / np.asarray(self.scale)[types])
        self.norm = self._per_type(raw, types, np.mean)
        return self

    def __call__(self, marks, types):
        if self.scale is None:
            raise RuntimeError("Saturating used before fit")
        raw = 1.0 - np.exp(-marks / np.asarray(self.scale)[types])
        return raw / np.asarray(self.norm)[types]

    def to_dict(self):
        return {"name": self.name, "scale": self.scale, "norm": self.norm}


_REGISTRY = {cls.name: cls for cls in (Unmarked, LinearNormalized, Saturating)}


def make_mark_function(name: str) -> MarkFunction:
    if name not in _REGISTRY:
        raise ValueError(f"unknown mark function {name!r}")
    return _REGISTRY[name]()


def mark_function_from_dict(d: dict[str, Any]) -> MarkFunction:
    params = {k: v for k, v in d.items() if k != "name"}
    return _REGISTRY[d["name"]](**params)
