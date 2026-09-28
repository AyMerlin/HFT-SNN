"""Improved signal model (M8): topology, rewards in training, ablations, causality, end-to-end."""

import json
from datetime import date, timedelta

import numpy as np
import pytest

from snn_hft.backtest.backtester import Backtester
from snn_hft.config.schema import BarsConfig, ExperimentConfig, HawkesRSTDPConfig, expand_runs
from snn_hft.data.sources import SyntheticDataSource
from snn_hft.models.hawkes.provider import HawkesParamProvider
from snn_hft.preprocessing.bars import aggregate_vwap
from snn_hft.signals.hawkes_snn import HawkesRSTDPSignalModel
from snn_hft.snn import simulator
from snn_hft.testing.causality import check_causal, day_data
from snn_hft.utils.seeding import rng_for

NUM, W_H = 10, 2
FIRST = date(2025, 10, 15)
DAYS = [FIRST + timedelta(days=i) for i in range(7)]
TRAIN, TEST = DAYS[2:4], DAYS[4]
BASE = {"core": {"lif": {"threshold": 4.0}}, "hawkes": {"restarts": 2, "event_quantile": 0.5}}


@pytest.fixture(scope="module")
def world():
    src = SyntheticDataSource(trades_per_day=3_000, seed=31)
    return {d: src.fetch_day("SYN", d) for d in DAYS}


def model(world, **overrides) -> HawkesRSTDPSignalModel:
    cfg = HawkesRSTDPConfig.model_validate({**BASE, **overrides})
    return HawkesRSTDPSignalModel(cfg, BarsConfig(vwap_num=NUM), "improved-test", W_H,
                                  bars_for_day=lambda d: aggregate_vwap(world[d].df, NUM))


def fit_generate(world, seed=0, **overrides):
    m = model(world, **overrides)
    m.fit([day_data(world, d) for d in TRAIN], seed)
    return m, m.generate(day_data(world, TEST), seed)


def test_topology_rewards_and_frozen_test_weights(world):
    m = model(world)
    m.fit([day_data(world, d) for d in TRAIN], 0)
    net = m.network
    assert net.pop_names == ("X1", "X2", "H1", "H2", "H_mom", "H_rev", "Out")
    rstdp = net.rule == simulator.RULE_RSTDP
    # Initial weights are U(0.2, 0.6); rewards were delivered during training, so R-STDP synapses moved.
    fresh = model(world)
    fresh.network = fresh.build_network()
    fresh.network.init_weights(0.2, 0.6, rng_for(0, "init"))
    assert not np.array_equal(net.flat_weights()[rstdp], fresh.network.flat_weights()[rstdp])
    w = net.flat_weights()
    sig = m.generate(day_data(world, TEST), 0)
    np.testing.assert_array_equal(net.flat_weights(), w)
    assert {"spikes_H_mom", "spikes_H_rev", "spikes_Out"} <= sig.diagnostics.keys()
    assert sig.n_bars == 300 and 0 <= len(sig) < 300


def test_zero_rewards_leave_rstdp_weights_at_their_initial_values(world, monkeypatch):
    m = model(world)
    monkeypatch.setattr(m, "rewards_for", lambda day: {"mom": np.zeros(day.n_bars), "rev": np.zeros(day.n_bars)})
    m.fit([day_data(world, d) for d in TRAIN], 0)
    fresh = model(world)
    fresh.network = fresh.build_network()
    fresh.network.init_weights(0.2, 0.6, rng_for(0, "init"))
    rstdp = m.network.rule == simulator.RULE_RSTDP
    np.testing.assert_array_equal(m.network.flat_weights()[rstdp], fresh.network.flat_weights()[rstdp])


def test_ablations(world):
    a1 = model(world, use_rstdp_pools=False)
    a1.network = a1.build_network()
    assert a1.network.pop_names == ("X1", "X2", "H1", "H2", "Out") and a1.rewards_for(None) is None
    a2 = model(world, keep_direction_pools=False)
    assert a2.build_network().pop_names == ("X1", "X2", "H_mom", "H_rev", "Out")


def test_fold_metadata_reports_theta_of_the_days_used(world):
    m, _ = fit_generate(world)
    meta = m.fold_metadata()["hawkes"]
    assert sorted(meta) == [d.isoformat() for d in (*TRAIN, TEST)]
    assert meta[TEST.isoformat()]["fit_days"] == [d.isoformat() for d in DAYS[2:4]]
    assert m.fold_metadata() == {"hawkes": {}}  # cleared after reading


def test_improved_signal_model_is_causal(world):
    check_causal(lambda w: fit_generate(w)[1], world, TEST, (0, 5, 150, 298), NUM)


def test_harness_flags_theta_fitted_on_the_same_day(world, monkeypatch):
    monkeypatch.setattr(HawkesParamProvider, "fit_days", staticmethod(lambda day, w_h: [day - timedelta(days=1), day]))
    with pytest.raises(AssertionError, match="causality violated"):
        check_causal(lambda w: fit_generate(w)[1], world, TEST, (150,), NUM)


def test_end_to_end_smoke_both_models(tmp_path):
    cfg = ExperimentConfig.model_validate({
        "name": "smoke2",
        "data": {"source": "synthetic", "raw_dir": str(tmp_path / "raw")},
        "bars": {"vwap_num": 10},
        "periods": {"data": {"start": "2025-01-01", "end": "2025-01-08"}, "history_days": 2,
                    "validation": {"start": "2025-01-03", "end": "2025-01-04"},
                    "test": {"start": "2025-01-05", "end": "2025-01-06"}},
        "models": [
            {"model": "paper_snn", "core": {"lif": {"threshold": 8.0}}},
            {"model": "hawkes_rstdp", "name": "improved", **BASE},
            {"model": "hawkes_rstdp", "name": "a1", "use_rstdp_pools": False, **BASE},
        ],
        "backtest": {"split": "test", "seeds": [0], "w_snn": [1], "w_h": [1]},
        "benchmarks": {"naive_reps": 2},
        "strategy": {"rules": ["momentum"]},
        "output": {"results_dir": str(tmp_path / "results"), "cache_dir": str(tmp_path / "cache")},
    })
    result = Backtester(cfg, workers=2, log=lambda _: None).run()
    assert {r["model"] for r in result.table3_rows} == {"paper", "improved", "a1"}
    root = tmp_path / "results" / "smoke2"
    improved = next(r for r in expand_runs(cfg) if r.signal.name == "improved")
    hawkes_files = sorted(p.name for p in (root / improved.run_id / "hawkes").glob("*.json"))
    assert hawkes_files == ["2025-01-04.json", "2025-01-05.json", "2025-01-06.json"]  # train and test days
    theta = json.loads((root / improved.run_id / "hawkes" / "2025-01-05.json").read_text())
    assert theta["fit_days"] == ["2025-01-04"] and theta["rho"] < 1
    paper = next(r for r in expand_runs(cfg) if r.signal.name == "paper")
    assert not (root / paper.run_id / "hawkes").exists()
