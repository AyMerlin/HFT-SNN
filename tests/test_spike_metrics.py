import numpy as np
import pytest

from snn_hft.config.schema import HealthConfig
from snn_hft.evaluation.health import check_health
from snn_hft.evaluation.spike_metrics import SpikeEvaluator
from snn_hft.signals.base import SignalSeries


def test_hand_example_window_1():
    p = np.array([100.0, 101.0, 103.0, 102.0, 102.5, 104.0])
    ev = SpikeEvaluator(window=1)
    lab = ev.labels(p)
    assert lab.r_pivot == pytest.approx(0.01)  # median of |p[k+1]/p[k] − 1|
    assert lab.evaluable.tolist() == [False, True, True, True, False, False]  # 1 ≤ t ≤ n − 2 − w
    # t=1: S = |102/103 − 1| < pivot → fake;  flag = (100−101)(103−101) < 0 → momentum
    # t=2: S = |102.5/102 − 1| < pivot → fake; flag = (101−103)(102−103) > 0 → reversion
    # t=3: S = |104/102.5 − 1| > pivot → real; flag = (103−102)(102.5−102) > 0 → reversion
    assert lab.real.tolist() == [False, False, False, True, False, False]
    assert lab.momentum.tolist() == [False, True, False, False, False, False]
    m = ev.evaluate(p, np.array([0, 1, 2, 3, 5]))
    assert (m.n_signals, m.n_evaluated, m.n_excluded, m.n_real, m.n_momentum) == (5, 3, 2, 1, 1)
    assert m.accuracy == pytest.approx(1 / 3) and m.momentum_pct == pytest.approx(1 / 3)
    assert m.base_accuracy == pytest.approx(1 / 3) and m.base_momentum_pct == pytest.approx(1 / 3)
    assert m.signal_rate == pytest.approx(5 / 6)


def brute_force(p: np.ndarray, w: int):
    """The paper's formulas transcribed literally with 1-indexed X_1..X_n (P2 typo fixed)."""
    n = len(p)
    X = {i + 1: p[i] for i in range(n)}
    r = {t: abs(X[t + 1] / X[t] - 1) for t in range(1, n)}
    pivot = float(np.median([r[t] for t in range(1, n)]))
    out = {}
    for t in range(1, n + 1):
        if t - w < 1 or t + w + 1 > n:
            continue
        strength = sum(abs(r[t + k]) for k in range(1, w + 1)) / w
        prior = sum(X[t - k] for k in range(1, w + 1)) / w
        post = sum(X[t + k] for k in range(1, w + 1)) / w
        flag = (prior - X[t]) * (post - X[t])
        out[t - 1] = (strength > pivot, not flag > 0)
    return pivot, out


@pytest.mark.parametrize("w", [1, 3, 5])
def test_matches_brute_force_definition(w):
    p = 75_000 + np.cumsum(np.random.default_rng(w).normal(0, 1.5, 400))
    lab = SpikeEvaluator(window=w).labels(p)
    pivot, ref = brute_force(p, w)
    assert lab.r_pivot == pytest.approx(pivot, rel=1e-12)
    assert set(np.flatnonzero(lab.evaluable)) == set(ref)
    for t, (real, momentum) in ref.items():
        assert lab.real[t] == real and lab.momentum[t] == momentum, t


def test_flat_prices_count_as_momentum():
    lab = SpikeEvaluator(window=3).labels(np.array([75600.1] * 12))
    assert lab.momentum[lab.evaluable].all()
    assert not lab.real.any()  # strength 0 is not above a pivot of 0


def test_no_signals_gives_nan_metrics():
    m = SpikeEvaluator().evaluate(np.linspace(100, 110, 50), np.array([], dtype=np.int64))
    assert m.n_signals == 0 and np.isnan(m.accuracy) and np.isnan(m.momentum_pct)


def series(out, pools, n_bars=100):
    diag = {f"spikes_{k}": np.asarray(v) for k, v in pools.items()}
    return SignalSeries(day=None, bar_idx=np.asarray(out), n_bars=n_bars, model_id="m", diagnostics=diag)


def test_health_checks():
    cfg = HealthConfig()
    healthy = check_health(series(range(0, 100, 10), {"H1": np.full(100, 64)}), {"H1": 64}, 10, 2, cfg)
    assert healthy.ok and healthy.output_rate == pytest.approx(0.1) and healthy.pool_rates["H1"] == 1.0
    busy = check_health(series(range(60), {"H1": np.zeros(100)}), {"H1": 64}, 10, 2, cfg)
    assert any("outside" in w for w in busy.warnings) and any("silent" in w for w in busy.warnings)
    saturated = check_health(series([5], {"H1": np.full(100, 64 * 3)}), {"H1": 64}, 10, 2, cfg)
    assert any("saturated" in w for w in saturated.warnings)


def test_signal_series_validation():
    with pytest.raises(ValueError):
        SignalSeries(day=None, bar_idx=np.array([3, 3]), n_bars=10, model_id="m")
    with pytest.raises(ValueError):
        SignalSeries(day=None, bar_idx=np.array([10]), n_bars=10, model_id="m")
