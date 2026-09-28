"""Frozen, JSON-serialisable Hawkes parameters θ_d (§5.2, §5.5).

Index convention: type 0 = up (u), 1 = down (d); `alpha[m, k]` is the excitation of the
target type m by source type k (α_{target,source}), e.g. `alpha[0, 1]` = α_ud.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from snn_hft.models.hawkes.marks import MarkFunction, mark_function_from_dict

TYPE_NAMES = ("u", "d")


def spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(a, dtype=float)))))


@dataclass(frozen=True)
class HawkesParameters:
    mu: tuple[float, float]
    alpha: tuple[tuple[float, float], tuple[float, float]]
    beta: tuple[tuple[float, float], tuple[float, float]]
    marks: dict[str, Any]  # fitted mark function (name and normalisation constants)
    time_axis: str = "bar_index"
    fit: dict[str, Any] = field(default_factory=dict)  # provenance: days, log-likelihood, convergence, …
    event_threshold: float | None = None  # |d| above which a bar is an event (U9); None = every move

    @classmethod
    def from_arrays(
        cls, mu, alpha, beta, mark_fn: MarkFunction, time_axis="bar_index", fit=None, event_threshold=None
    ) -> HawkesParameters:
        return cls(
            mu=tuple(float(x) for x in np.asarray(mu)),
            alpha=tuple(tuple(float(x) for x in row) for row in np.asarray(alpha)),
            beta=tuple(tuple(float(x) for x in row) for row in np.asarray(beta)),
            marks=mark_fn.to_dict(),
            time_axis=time_axis,
            fit=fit or {},
            event_threshold=None if event_threshold is None else float(event_threshold),
        )

    @property
    def mu_arr(self) -> np.ndarray:
        return np.asarray(self.mu, dtype=np.float64)

    @property
    def alpha_arr(self) -> np.ndarray:
        return np.asarray(self.alpha, dtype=np.float64)

    @property
    def beta_arr(self) -> np.ndarray:
        return np.asarray(self.beta, dtype=np.float64)

    @property
    def mark_function(self) -> MarkFunction:
        return mark_function_from_dict(self.marks)

    @property
    def branching_matrix(self) -> np.ndarray:
        """A_mk = α_mk · E[g_k] / β_mk = α_mk / β_mk under marks normalised to mean 1."""
        return self.alpha_arr / self.beta_arr

    @property
    def spectral_radius(self) -> float:
        return spectral_radius(self.branching_matrix)

    def named(self) -> dict[str, float]:
        """θ in the order of §5.2: μ_u, μ_d, α_uu, β_uu, α_ud, β_ud, α_du, β_du, α_dd, β_dd."""
        out = {"mu_u": self.mu[0], "mu_d": self.mu[1]}
        for m, k in ((0, 0), (0, 1), (1, 0), (1, 1)):
            tag = TYPE_NAMES[m] + TYPE_NAMES[k]
            out[f"alpha_{tag}"] = self.alpha[m][k]
            out[f"beta_{tag}"] = self.beta[m][k]
        return out

    def to_json(self) -> str:
        d = asdict(self)
        d["branching_matrix"] = self.branching_matrix.tolist()
        d["spectral_radius"] = self.spectral_radius
        return json.dumps(d, indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> HawkesParameters:
        d = json.loads(text)
        d.pop("branching_matrix", None)
        d.pop("spectral_radius", None)
        d["mu"] = tuple(d["mu"])
        d["alpha"] = tuple(tuple(r) for r in d["alpha"])
        d["beta"] = tuple(tuple(r) for r in d["beta"])
        return cls(**d)
