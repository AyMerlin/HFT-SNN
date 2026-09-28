from datetime import date, timedelta

import numpy as np
import pytest

from snn_hft.backtest.cache import HawkesParamCache
from snn_hft.config.schema import HawkesConfig
from snn_hft.models.hawkes.events import bar_times, events_from_bars
from snn_hft.models.hawkes.kernels import ExponentialKernel
from snn_hft.models.hawkes.marks import LinearNormalized, Saturating, Unmarked, make_mark_function
from snn_hft.models.hawkes.params import HawkesParameters, spectral_radius
from snn_hft.models.hawkes.process import (
    BivariateHawkesProcess,
    DayEvents,
    arrays_to_theta,
    spectral_radius_and_grad,
    theta_to_arrays,
)
from snn_hft.models.hawkes.provider import HawkesParamProvider

MU = np.array([0.2, 0.25])
ALPHA = np.array([[0.30, 0.10], [0.15, 0.25]])
BETA = np.array([[1.0, 0.5], [0.8, 1.2]])


def params(mu=MU, alpha=ALPHA, beta=BETA, mark=None) -> HawkesParameters:
    mark = mark or LinearNormalized(mean=[1.0, 1.0])
    return HawkesParameters.from_arrays(mu, alpha, beta, mark)


def random_events(n=300, seed=0, horizon=200.0, ties=True) -> DayEvents:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, horizon, n))
    if ties:
        times[10:13] = times[10]  # equal timestamps: strict order
    return DayEvents(times, rng.integers(0, 2, n), rng.exponential(1.0, n), horizon, bar_idx=np.arange(n))


def brute_force_lambda(ev: DayEvents, g, mu, alpha, beta, n, t):
    """λ(t) from the first n events (history = events with index < n)."""
    lam = mu.copy()
    for i in range(n):
        k = ev.types[i]
        lam += alpha[:, k] * g[i] * np.exp(-beta[:, k] * (t - ev.times[i]))
    return lam


def brute_force_loglik(ev: DayEvents, g, mu, alpha, beta):
    ll = 0.0
    for n in range(len(ev)):
        ll += np.log(brute_force_lambda(ev, g, mu, alpha, beta, n, ev.times[n])[ev.types[n]])
    comp = mu.sum() * ev.horizon
    for i in range(len(ev)):
        k = ev.types[i]
        for m in range(2):
            comp += ExponentialKernel(alpha[m, k], beta[m, k]).integral(ev.horizon - ev.times[i]) * g[i]
    return ll - comp


# --------------------------------------------------------------------------- recursion and likelihood


def test_loglik_matches_brute_force():
    ev = random_events()
    p = params()
    g = p.mark_function(ev.marks, ev.types)
    fast = BivariateHawkesProcess().loglik(p, [ev])
    assert fast == pytest.approx(brute_force_loglik(ev, g, MU, ALPHA, BETA), rel=1e-10)


def test_loglik_sums_over_days():
    days = [random_events(seed=s) for s in range(3)]
    proc, p = BivariateHawkesProcess(), params()
    assert proc.loglik(p, days) == pytest.approx(sum(proc.loglik(p, [d]) for d in days), rel=1e-12)


def test_analytic_gradient_matches_finite_differences():
    days = [random_events(seed=1), random_events(seed=2, n=150)]
    mark = LinearNormalized().fit(np.concatenate([d.marks for d in days]), np.concatenate([d.types for d in days]))
    proc = BivariateHawkesProcess()
    theta = arrays_to_theta(MU, ALPHA, BETA)
    _, grad = proc.loglik_grad_theta(theta, days, mark)
    for j in range(len(theta)):
        h = 1e-6 * theta[j]
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        numeric = (proc.loglik_grad_theta(up, days, mark)[0] - proc.loglik_grad_theta(down, days, mark)[0]) / (2 * h)
        assert grad[j] == pytest.approx(numeric, rel=1e-5, abs=1e-6), j


def test_theta_order_follows_the_spec():
    theta = arrays_to_theta(MU, ALPHA, BETA)
    named = params().named()
    assert list(named.values()) == pytest.approx(theta.tolist())
    assert list(named) == ["mu_u", "mu_d", "alpha_uu", "beta_uu", "alpha_ud", "beta_ud", "alpha_du", "beta_du", "alpha_dd", "beta_dd"]
    mu, alpha, beta = theta_to_arrays(theta)
    np.testing.assert_array_equal(alpha, ALPHA)


# --------------------------------------------------------------------------- intensities at bars and branching


def test_bar_intensities_and_branching_match_brute_force():
    rng = np.random.default_rng(3)
    d = rng.normal(0, 1, 400)
    d[0] = np.nan
    d[rng.choice(np.arange(1, 400), 30, replace=False)] = 0.0
    ev = events_from_bars(d, None, 0, "bar_index")
    p = params()
    g = p.mark_function(ev.marks, ev.types)
    lam_minus, lam_plus, br = BivariateHawkesProcess().bar_pass(p, ev, bar_times(len(d), None, 0, "bar_index"))
    for b in (0, 1, 5, 57, 399):
        n_before = int(np.searchsorted(ev.bar_idx, b, side="left"))
        n_upto = int(np.searchsorted(ev.bar_idx, b, side="right"))
        np.testing.assert_allclose(lam_minus[b], brute_force_lambda(ev, g, MU, ALPHA, BETA, n_before, b), rtol=1e-12)
        np.testing.assert_allclose(lam_plus[b], brute_force_lambda(ev, g, MU, ALPHA, BETA, n_upto, b), rtol=1e-12)
    np.testing.assert_allclose(br.sum(axis=1), 1.0, rtol=1e-12)
    assert (br >= 0).all()
    j = 20
    m = ev.types[j]
    lam = brute_force_lambda(ev, g, MU, ALPHA, BETA, j, ev.times[j])[m]
    same = sum(ALPHA[m, m] * g[i] * np.exp(-BETA[m, m] * (ev.times[j] - ev.times[i])) for i in range(j) if ev.types[i] == m)
    assert br[j, 1] == pytest.approx(same / lam, rel=1e-12)  # momentum share: same-direction history


def test_right_limit_includes_the_current_event_only():
    ev = DayEvents(np.array([2.0]), np.array([0]), np.array([1.0]), 5.0, bar_idx=np.array([2]))
    lam_minus, lam_plus, _ = BivariateHawkesProcess().bar_pass(params(), ev, np.arange(5.0))
    np.testing.assert_allclose(lam_minus[2], MU)
    np.testing.assert_allclose(lam_plus[2], MU + ALPHA[:, 0])  # up event excites both targets
    np.testing.assert_allclose(lam_minus[3], MU + ALPHA[:, 0] * np.exp(-BETA[:, 0]))


# --------------------------------------------------------------------------- fitting and simulation


def test_parameter_recovery_from_simulated_data():
    true = params()
    rng = np.random.default_rng(7)
    sampler = lambda r, k: r.exponential(1.0)  # noqa: E731 - mean 1, matches the normalisation
    days = [BivariateHawkesProcess.simulate(true, 8_000.0, sampler, rng) for _ in range(5)]
    assert sum(len(d) for d in days) > 20_000
    fit = BivariateHawkesProcess().fit(days, restarts=5, rng=np.random.default_rng(0))
    np.testing.assert_allclose(fit.mu_arr, MU, rtol=0.15)
    np.testing.assert_allclose(fit.alpha_arr, ALPHA, rtol=0.2)
    np.testing.assert_allclose(fit.beta_arr, BETA, rtol=0.25)
    assert fit.fit["converged"] and fit.fit["rho"] < 1
    assert fit.spectral_radius == pytest.approx(true.spectral_radius, rel=0.15)


def test_fit_is_stationary_and_records_provenance():
    ev = [random_events(seed=s, ties=False) for s in range(2)]
    fit = BivariateHawkesProcess(mark_fn="unmarked").fit(ev, restarts=3, provenance={"for_day": "x"})
    assert fit.spectral_radius < 1 and fit.fit["for_day"] == "x" and len(fit.fit["restarts"]) == 3
    assert fit.marks == {"name": "unmarked"}


def test_spectral_radius_formula():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(0, 1, (2, 2))
        rho, grad = spectral_radius_and_grad(a)
        assert rho == pytest.approx(spectral_radius(a), rel=1e-10)
        h = 1e-7
        for i in range(2):
            for j in range(2):
                b = a.copy()
                b[i, j] += h
                assert grad[i, j] == pytest.approx((spectral_radius_and_grad(b)[0] - rho) / h, rel=1e-4, abs=1e-6)
    assert params().spectral_radius == pytest.approx(spectral_radius(ALPHA / BETA))


# --------------------------------------------------------------------------- marks, parameters, events


def test_mark_functions_normalise_to_mean_one_per_type():
    rng = np.random.default_rng(0)
    marks, types = rng.lognormal(0, 1, 2_000), rng.integers(0, 2, 2_000)
    for mf in (LinearNormalized(), Saturating()):
        g = mf.fit(marks, types)(marks, types)
        for k in (0, 1):
            assert g[types == k].mean() == pytest.approx(1.0)
        again = make_mark_function(mf.name)
        again.__dict__.update({k: v for k, v in mf.to_dict().items() if k != "name"})
        np.testing.assert_allclose(again(marks, types), g)
    np.testing.assert_array_equal(Unmarked()(marks, types), 1.0)


def test_parameters_json_roundtrip():
    p = params()
    q = HawkesParameters.from_json(p.to_json())
    assert q == p
    assert '"branching_matrix"' in p.to_json()


def test_events_from_bars():
    d = np.array([np.nan, 0.5, -0.2, 0.0, 1.5])
    ev = events_from_bars(d, None, 0, "bar_index")
    assert ev.bar_idx.tolist() == [1, 2, 4]  # d = 0 is no event
    assert ev.types.tolist() == [0, 1, 0]  # up, down, up
    assert ev.marks.tolist() == [0.5, 0.2, 1.5]
    assert ev.times.tolist() == [1.0, 2.0, 4.0] and ev.horizon == 5.0
    wall = events_from_bars(d, np.array([0, 1, 2, 3, 4]) * 10**9 + 5 * 10**9, 5 * 10**9, "wallclock")
    assert wall.times.tolist() == [1.0, 2.0, 4.0] and wall.horizon == 86_400.0


# --------------------------------------------------------------------------- freezing (§5.5)


def synthetic_day_events(day: date, perturbed: date | None = None) -> DayEvents:
    rng = np.random.default_rng(day.toordinal() + (999 if day == perturbed else 0))
    return BivariateHawkesProcess.simulate(params(), 1_500.0, lambda r, k: r.exponential(1.0), rng)


def test_parameters_for_day_d_are_never_fit_on_day_d(tmp_path):
    requested = []

    def events_for_day(day):
        requested.append(day)
        return synthetic_day_events(day)

    cfg = HawkesConfig(restarts=2)
    day = date(2025, 11, 10)
    provider = HawkesParamProvider(cfg, w_h=3, events_for_day=events_for_day)
    theta = provider.params_for(day)
    assert requested == [day - timedelta(days=k) for k in (3, 2, 1)]
    assert theta.fit["fit_days"] == ["2025-11-07", "2025-11-08", "2025-11-09"] and theta.fit["for_day"] == "2025-11-10"
    # Changing day d (or any later day) cannot change θ_d.
    other = HawkesParamProvider(cfg, 3, lambda d: synthetic_day_events(d, perturbed=day)).params_for(day)
    assert other == theta
    changed = HawkesParamProvider(cfg, 3, lambda d: synthetic_day_events(d, perturbed=day - timedelta(days=1))).params_for(day)
    assert changed != theta


def test_parameter_cache_is_reused(tmp_path):
    calls = []

    def events_for_day(day):
        calls.append(day)
        return synthetic_day_events(day)

    store = HawkesParamCache(tmp_path, "v", "S", "aggTrades", 100)
    cfg = HawkesConfig(restarts=2)
    day = date(2025, 11, 10)
    first = HawkesParamProvider(cfg, 2, events_for_day, store).params_for(day)
    n_calls = len(calls)
    second = HawkesParamProvider(cfg, 2, events_for_day, store).params_for(day)
    assert len(calls) == n_calls and second == first
    assert HawkesParamProvider(HawkesConfig(restarts=3), 2, events_for_day).key(day) != HawkesParamProvider(cfg, 2, events_for_day).key(day)
