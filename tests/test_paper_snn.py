from datetime import date, timedelta

import numpy as np
import pytest

from snn_hft.config.schema import BarsConfig, HawkesRSTDPConfig, PaperSNNConfig
from snn_hft.data.sources import SyntheticDataSource
from snn_hft.signals.paper_snn import PaperSNNSignalModel
from snn_hft.testing.causality import check_causal, day_data

NUM = 10
FIRST = date(2025, 10, 17)
DAYS = [FIRST + timedelta(days=i) for i in range(5)]
TRAIN, TEST = DAYS[:2], DAYS[2]
# A healthier regime than the untuned defaults, so signals are neither absent nor everywhere.
CFG = PaperSNNConfig(core={"lif": {"threshold": 4.0}})


@pytest.fixture(scope="module")
def world():
    src = SyntheticDataSource(trades_per_day=4_000, seed=5)
    return {d: src.fetch_day("SYN", d) for d in DAYS}


def model(cfg=CFG) -> PaperSNNSignalModel:
    return PaperSNNSignalModel(cfg, BarsConfig(vwap_num=NUM), model_id="paper-test")


def fit_generate(w, cfg=CFG, seed=0):
    m = model(cfg).fit([day_data(w, d) for d in TRAIN], seed)
    return m, m.generate(day_data(w, TEST), seed)


def test_generate_returns_one_signal_per_output_bar(world):
    m, sig = fit_generate(world)
    assert sig.n_bars == 400 and sig.day == TEST and sig.model_id == "paper-test"
    out = sig.diagnostics["spikes_Out"]
    np.testing.assert_array_equal(sig.bar_idx, np.flatnonzero(out > 0))
    assert 0 < len(sig) < sig.n_bars
    assert {"spikes_X1", "spikes_X2", "spikes_H1", "spikes_H2", "spikes_Out", "channel_prob"} <= sig.diagnostics.keys()
    assert [s.day for s in m.train_signals] == TRAIN
    assert all(d.trades is None and d.bars is not None for d in m.train_days)


def test_seeded_and_reproducible(world):
    _, a = fit_generate(world, seed=1)
    _, b = fit_generate(world, seed=1)
    _, c = fit_generate(world, seed=2)
    np.testing.assert_array_equal(a.bar_idx, b.bar_idx)
    assert not np.array_equal(a.bar_idx, c.bar_idx)


def test_weights_frozen_on_test_day(world):
    m, _ = fit_generate(world)
    w = m.network.flat_weights()
    m.generate(day_data(world, TEST), 0)
    np.testing.assert_array_equal(m.network.flat_weights(), w)


def test_fresh_initialisation_per_fit_unless_warm_start(world):
    m = model()
    m.fit([day_data(world, TRAIN[0])], 0)
    first = m.network.flat_weights()
    m.fit([day_data(world, TRAIN[0])], 0)
    np.testing.assert_array_equal(m.network.flat_weights(), first)
    warm = model(PaperSNNConfig(core={"lif": {"threshold": 4.0}, "warm_start": True}))
    warm.fit([day_data(world, TRAIN[0])], 0)
    warm.fit([day_data(world, TRAIN[0])], 0)
    assert not np.array_equal(warm.network.flat_weights(), first)


def test_paper_model_is_causal(world):
    def run(w):
        return fit_generate(w)[1]

    check_causal(run, world, TEST, (0, 3, 150, 398), NUM)


def test_paper_model_with_same_day_normalisation_is_flagged(world):
    cfg = PaperSNNConfig(core={"lif": {"threshold": 4.0}}, zscore={"stats_source": "same_day"})
    with pytest.raises(AssertionError, match="causality violated"):
        check_causal(lambda w: fit_generate(w, cfg)[1], world, TEST, (150,), NUM)


def test_rejects_improved_model_config():
    with pytest.raises(TypeError):
        PaperSNNSignalModel(HawkesRSTDPConfig(), BarsConfig(), "x")
