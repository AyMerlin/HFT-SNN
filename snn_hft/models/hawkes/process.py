"""Bivariate marked Hawkes process with exponential kernels (§5.2–§5.5, §5.7).

    λ_m(t) = μ_m + Σ_k α_mk Σ_{t_i^k < t} g_k(m_i) exp(−β_mk (t − t_i^k)),   m, k ∈ {u, d}

All sums over past events use the O(N) recursion of §5.3
    S_mk(t_n⁻) = exp(−β_mk Δ_n) · [S_mk(t_{n−1}⁻) + 1{k_{n−1} = k} g_{n−1}],
and the log-likelihood gradient uses the companion recursion
    R_mk(t_n⁻) = Σ g_i (t_n − t_i) e^{−β_mk (t_n − t_i)} = e^{−β_mk Δ_n} [R_mk(t_{n−1}⁻) + Δ_n (S_mk(t_{n−1}⁻) + 1{k_{n−1} = k} g_{n−1})].
Events at equal times are strictly ordered: earlier events count as history of later ones.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize

from snn_hft.models.hawkes.marks import MarkFunction, make_mark_function
from snn_hft.models.hawkes.params import HawkesParameters

# θ order of §5.2: μ_u, μ_d, α_uu, β_uu, α_ud, β_ud, α_du, β_du, α_dd, β_dd
THETA_NAMES = ("mu_u", "mu_d", "alpha_uu", "beta_uu", "alpha_ud", "beta_ud", "alpha_du", "beta_du", "alpha_dd", "beta_dd")
_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))


class HawkesFitError(RuntimeError):
    """No restart produced a stationary (ρ(A) < 1) solution."""


@dataclass(frozen=True)
class DayEvents:
    """Up (type 0) and down (type 1) events of one day on the window [0, horizon]."""

    times: np.ndarray  # float64, non-decreasing
    types: np.ndarray  # int64 in {0, 1}
    marks: np.ndarray  # float64 > 0
    horizon: float
    bar_idx: np.ndarray | None = None  # bar of each event, if events come from bars

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", np.asarray(self.times, dtype=np.float64))
        object.__setattr__(self, "types", np.asarray(self.types, dtype=np.int64))
        object.__setattr__(self, "marks", np.asarray(self.marks, dtype=np.float64))
        n = len(self.times)
        if not len(self.types) == len(self.marks) == n:
            raise ValueError("times, types and marks must have the same length")
        if n and (np.any(np.diff(self.times) < 0) or self.times[0] < 0 or self.times[-1] > self.horizon):
            raise ValueError("event times must be sorted and inside [0, horizon]")
        if n and (not np.isin(self.types, (0, 1)).all() or not (self.marks > 0).all()):
            raise ValueError("types must be 0/1 and marks positive")

    def __len__(self) -> int:
        return len(self.times)


# --------------------------------------------------------------------------- numba kernels


@njit(cache=True)
def _loglik_day(times, types, g, horizon, mu, alpha, beta, want_grad):
    """ℓ = Σ_n log λ_{k_n}(t_n⁻) − Σ_m Λ_m(horizon) for one day, and ∂ℓ/∂(μ, α, β)."""
    S = np.zeros((2, 2))
    R = np.zeros((2, 2))
    g_mu = np.zeros(2)
    g_alpha = np.zeros((2, 2))
    g_beta = np.zeros((2, 2))
    ll = 0.0
    prev_t = 0.0
    prev_k = -1
    prev_g = 0.0
    for i in range(times.shape[0]):
        dt = times[i] - prev_t
        for m in range(2):
            for k in range(2):
                e = math.exp(-beta[m, k] * dt)
                add = prev_g if prev_k == k else 0.0
                if want_grad:
                    R[m, k] = e * (R[m, k] + dt * (S[m, k] + add))
                S[m, k] = e * (S[m, k] + add)
        m = types[i]
        lam = mu[m] + alpha[m, 0] * S[m, 0] + alpha[m, 1] * S[m, 1]
        ll += math.log(lam)
        if want_grad:
            g_mu[m] += 1.0 / lam
            for k in range(2):
                g_alpha[m, k] += S[m, k] / lam
                g_beta[m, k] -= alpha[m, k] * R[m, k] / lam
        prev_t = times[i]
        prev_k = types[i]
        prev_g = g[i]
    # Compensator Λ_m(T) = μ_m T + Σ_k α_mk / β_mk Σ_{i ∈ k} g_i (1 − e^{−β_mk (T − t_i)})
    for m in range(2):
        ll -= mu[m] * horizon
        g_mu[m] -= horizon
    for i in range(times.shape[0]):
        k = types[i]
        r = horizon - times[i]
        for m in range(2):
            e = math.exp(-beta[m, k] * r)
            G = g[i] * (1.0 - e)
            ll -= alpha[m, k] / beta[m, k] * G
            if want_grad:
                g_alpha[m, k] -= G / beta[m, k]
                g_beta[m, k] += alpha[m, k] * G / beta[m, k] ** 2 - alpha[m, k] / beta[m, k] * g[i] * r * e
    return ll, g_mu, g_alpha, g_beta


@njit(cache=True)
def _bar_pass(bar_times, ev_bar, types, g, mu, alpha, beta):
    """λ(b⁻) and λ(b⁺) at every bar, and the causal branching split of every event (§5.3, §5.4).

    Events at bar b (in their order) are added after λ(b⁻) is recorded; λ(b⁺) includes them.
    Branching of event j of type m: (μ_m, α_mm S_mm, α_mk S_mk) / λ_m(t_j⁻), k ≠ m.
    """
    nb = bar_times.shape[0]
    ne = types.shape[0]
    lam_minus = np.empty((nb, 2))
    lam_plus = np.empty((nb, 2))
    branching = np.empty((ne, 3))
    S = np.zeros((2, 2))
    t_prev = bar_times[0] if nb else 0.0
    j = 0
    for b in range(nb):
        dt = bar_times[b] - t_prev
        t_prev = bar_times[b]
        for m in range(2):
            for k in range(2):
                S[m, k] *= math.exp(-beta[m, k] * dt)
        for m in range(2):
            lam_minus[b, m] = mu[m] + alpha[m, 0] * S[m, 0] + alpha[m, 1] * S[m, 1]
        while j < ne and ev_bar[j] == b:
            k = types[j]
            m = k
            lam = mu[m] + alpha[m, 0] * S[m, 0] + alpha[m, 1] * S[m, 1]
            branching[j, 0] = mu[m] / lam
            branching[j, 1] = alpha[m, m] * S[m, m] / lam
            branching[j, 2] = alpha[m, 1 - m] * S[m, 1 - m] / lam
            for mm in range(2):
                S[mm, k] += g[j]
            j += 1
        for m in range(2):
            lam_plus[b, m] = mu[m] + alpha[m, 0] * S[m, 0] + alpha[m, 1] * S[m, 1]
    return lam_minus, lam_plus, branching


# --------------------------------------------------------------------------- helpers


def theta_to_arrays(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.array(theta[:2], dtype=np.float64)
    alpha, beta = np.empty((2, 2)), np.empty((2, 2))
    for p, (m, k) in enumerate(_PAIRS):
        alpha[m, k], beta[m, k] = theta[2 + 2 * p], theta[3 + 2 * p]
    return mu, alpha, beta


def arrays_to_theta(mu, alpha, beta) -> np.ndarray:
    theta = [mu[0], mu[1]]
    for m, k in _PAIRS:
        theta += [alpha[m, k], beta[m, k]]
    return np.array(theta, dtype=np.float64)


def spectral_radius_and_grad(a: np.ndarray) -> tuple[float, np.ndarray]:
    """ρ and ∂ρ/∂A of a 2×2 non-negative matrix (closed form)."""
    a11, a12, a21, a22 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
    disc = math.sqrt(max((a11 - a22) ** 2 + 4.0 * a12 * a21, 1e-300))
    rho = 0.5 * (a11 + a22 + disc)
    grad = np.array([[0.5 * (1 + (a11 - a22) / disc), a21 / disc], [a12 / disc, 0.5 * (1 - (a11 - a22) / disc)]])
    return rho, grad


@dataclass
class _PreparedDay:
    times: np.ndarray
    types: np.ndarray
    g: np.ndarray
    horizon: float


class BivariateHawkesProcess:
    """Fitting, likelihood, intensities, branching and simulation of the bivariate marked process."""

    def __init__(self, mark_fn: str = "linear_normalized", time_axis: str = "bar_index"):
        self.mark_fn_name = mark_fn
        self.time_axis = time_axis

    # ------------------------------------------------------------------ likelihood

    @staticmethod
    def _prepare(days: Sequence[DayEvents], mark_fn: MarkFunction) -> list[_PreparedDay]:
        return [_PreparedDay(d.times, d.types, mark_fn(d.marks, d.types), float(d.horizon)) for d in days]

    @staticmethod
    def _loglik(prepared: list[_PreparedDay], mu, alpha, beta, want_grad=True):
        ll, g_mu, g_alpha, g_beta = 0.0, np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2))
        for d in prepared:
            l_d, gm, ga, gb = _loglik_day(d.times, d.types, d.g, d.horizon, mu, alpha, beta, want_grad)
            ll += l_d
            g_mu += gm
            g_alpha += ga
            g_beta += gb
        return ll, g_mu, g_alpha, g_beta

    def loglik(self, params: HawkesParameters, days: Sequence[DayEvents]) -> float:
        prepared = self._prepare(days, params.mark_function)
        return self._loglik(prepared, params.mu_arr, params.alpha_arr, params.beta_arr, want_grad=False)[0]

    def loglik_grad_theta(self, theta: np.ndarray, days: Sequence[DayEvents], mark_fn: MarkFunction) -> tuple[float, np.ndarray]:
        """ℓ(θ) and ∂ℓ/∂θ in the §5.2 order (for tests and diagnostics)."""
        mu, alpha, beta = theta_to_arrays(theta)
        ll, g_mu, g_alpha, g_beta = self._loglik(self._prepare(days, mark_fn), mu, alpha, beta)
        return ll, arrays_to_theta(g_mu, g_alpha, g_beta)

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        days: Sequence[DayEvents],
        restarts: int = 5,
        max_iter: int = 500,
        beta_grid: Sequence[float] = (0.1, 0.5, 1.0, 2.0),
        rng: np.random.Generator | None = None,
        rho_max: float = 0.99,
        penalty: float = 100.0,
        provenance: dict | None = None,
    ) -> HawkesParameters:
        """MLE on log-parameters with L-BFGS-B, ≥ 1 random restarts; solutions with ρ(A) ≥ 1 are rejected (§5.5)."""
        rng = rng or np.random.default_rng(0)
        types = np.concatenate([d.types for d in days])
        marks = np.concatenate([d.marks for d in days])
        mark_fn = make_mark_function(self.mark_fn_name).fit(marks, types)
        prepared = self._prepare(days, mark_fn)
        n_events = len(types)
        if n_events == 0:
            raise ValueError("no events in the fit window")
        total_time = sum(d.horizon for d in days)
        rate = np.array([(types == k).sum() / total_time for k in range(2)])

        def objective(phi: np.ndarray) -> tuple[float, np.ndarray]:
            theta = np.exp(phi)
            mu, alpha, beta = theta_to_arrays(theta)
            ll, g_mu, g_alpha, g_beta = self._loglik(prepared, mu, alpha, beta)
            f = -ll / n_events
            grad_theta = -arrays_to_theta(g_mu, g_alpha, g_beta) / n_events
            rho, d_rho = spectral_radius_and_grad(alpha / beta)
            if rho > rho_max:
                f += penalty * (rho - rho_max) ** 2
                d_a = 2 * penalty * (rho - rho_max) * d_rho  # ∂P/∂A
                grad_theta += arrays_to_theta(np.zeros(2), d_a / beta, -d_a * alpha / beta**2)
            return f, grad_theta * theta  # chain rule for log-parameters

        bounds = [(math.log(1e-8), math.log(1e3))] * 2
        for _ in _PAIRS:
            bounds += [(math.log(1e-8), math.log(1e3)), (math.log(1e-4), math.log(1e3))]
        best, attempts = None, []
        for r in range(max(restarts, 1)):
            beta0 = rng.choice(np.asarray(beta_grid, dtype=float), size=(2, 2))
            alpha0 = 0.2 * beta0 * rng.uniform(0.8, 1.2, size=(2, 2))
            mu0 = np.maximum(0.5 * rate * rng.uniform(0.8, 1.2, size=2), 1e-6)
            phi0 = np.log(arrays_to_theta(mu0, alpha0, beta0))
            res = minimize(objective, phi0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter})
            theta = np.exp(res.x)
            mu, alpha, beta = theta_to_arrays(theta)
            rho = spectral_radius_and_grad(alpha / beta)[0]
            ll = self._loglik(prepared, mu, alpha, beta, want_grad=False)[0]
            attempt = {"restart": r, "loglik": ll, "rho": rho, "converged": bool(res.success), "nit": int(res.nit)}
            attempts.append(attempt)
            if rho < 1.0 and (best is None or ll > best[1]["loglik"]):
                best = (theta, attempt, res)
        if best is None:
            raise HawkesFitError(f"no stationary solution in {len(attempts)} restarts: {attempts}")
        theta, attempt, res = best
        mu, alpha, beta = theta_to_arrays(theta)
        fit = {
            **(provenance or {}),
            "loglik": attempt["loglik"],
            "loglik_per_event": attempt["loglik"] / n_events,
            "converged": attempt["converged"],
            "message": str(res.message),
            "rho": attempt["rho"],
            "n_events": int(n_events),
            "n_days": len(days),
            "restarts": attempts,
        }
        return HawkesParameters.from_arrays(mu, alpha, beta, mark_fn, time_axis=self.time_axis, fit=fit)

    # ------------------------------------------------------------------ transformation

    def bar_pass(self, params: HawkesParameters, events: DayEvents, bar_times: np.ndarray):
        """(λ(b⁻) (N×2), λ(b⁺) (N×2), branching (n_events×3)) with frozen parameters."""
        if events.bar_idx is None:
            raise ValueError("events need bar_idx to be mapped to bars")
        g = params.mark_function(events.marks, events.types)
        return _bar_pass(
            np.asarray(bar_times, dtype=np.float64), np.asarray(events.bar_idx, dtype=np.int64), events.types, g,
            params.mu_arr, params.alpha_arr, params.beta_arr,
        )

    # ------------------------------------------------------------------ simulation

    @staticmethod
    def simulate(
        params: HawkesParameters,
        horizon: float,
        mark_sampler: Callable[[np.random.Generator, int], float],
        rng: np.random.Generator,
    ) -> DayEvents:
        """Ogata thinning (§5.7). Between events the intensity only decays, so λ(t⁺) bounds it."""
        mu, alpha, beta = params.mu_arr, params.alpha_arr, params.beta_arr
        mark_fn = params.mark_function
        S = np.zeros((2, 2))
        t = 0.0
        times, types, marks = [], [], []
        while True:
            lam_bar = float((mu + (alpha * S).sum(axis=1)).sum())
            w = rng.exponential(1.0 / lam_bar)
            if t + w > horizon:
                break
            t += w
            S *= np.exp(-beta * w)
            lam = mu + (alpha * S).sum(axis=1)
            if rng.random() * lam_bar <= lam.sum():
                k = 0 if rng.random() * lam.sum() < lam[0] else 1
                m = float(mark_sampler(rng, k))
                S[:, k] += mark_fn(np.array([m]), np.array([k]))[0]
                times.append(t)
                types.append(k)
                marks.append(m)
        return DayEvents(np.array(times), np.array(types, dtype=np.int64), np.array(marks), horizon)
