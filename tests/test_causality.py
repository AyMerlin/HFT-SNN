"""Causality harness applied to every preprocessing step (§0 principle 3, §13)."""

from datetime import date, timedelta

import numpy as np
import pytest

from snn_hft.config.schema import BarsConfig, PaperSNNConfig
from snn_hft.data.sources import SyntheticDataSource
from snn_hft.preprocessing.bars import VWAPBarAggregator
from snn_hft.preprocessing.base import Pipeline
from snn_hft.preprocessing.normalization import ShiftedZScoreNormalizer
from snn_hft.preprocessing.pipelines import paper_pipeline
from snn_hft.preprocessing.returns import DirectionalSplitter, PriceDifferencer
from snn_hft.testing.causality import (
    assert_prefix_equal,
    check_causal,
    day_data,
    perturb_tail,
    perturb_world,
)

NUM = 10
FIRST = date(2025, 10, 17)
DAYS = [FIRST + timedelta(days=i) for i in range(6)]
TRAIN, TEST = DAYS[:3], DAYS[3]  # DAYS[4:] are later days, perturbed entirely
BAR_TS = (0, 1, 7, 150, 298)  # 3,000 trades per day -> 300 bars


@pytest.fixture(scope="module")
def world():
    src = SyntheticDataSource(trades_per_day=3_000, seed=11)
    return {d: src.fetch_day("SYN", d) for d in DAYS}


def run_pipeline(make_pipeline):
    def run(w):
        pipe = make_pipeline()
        pipe.fit([day_data(w, d) for d in TRAIN])
        return pipe.transform(day_data(w, TEST))

    return run


STEP_PREFIXES = {
    "bars": lambda: [VWAPBarAggregator(NUM)],
    "differencer": lambda: [VWAPBarAggregator(NUM), PriceDifferencer()],
    "splitter": lambda: [VWAPBarAggregator(NUM), PriceDifferencer(), DirectionalSplitter()],
    "zscore_train_window": lambda: [
        VWAPBarAggregator(NUM), PriceDifferencer(), DirectionalSplitter(), ShiftedZScoreNormalizer()
    ],
}


@pytest.mark.parametrize("name", STEP_PREFIXES)
def test_every_baseline_step_is_causal(world, name):
    make = STEP_PREFIXES[name]
    check_causal(run_pipeline(lambda: Pipeline(make())), world, TEST, BAR_TS, NUM)


def test_paper_pipeline_from_default_config_is_causal(world):
    make = lambda: paper_pipeline(BarsConfig(vwap_num=NUM), PaperSNNConfig())  # noqa: E731
    out = check_causal(run_pipeline(make), world, TEST, BAR_TS, NUM)
    assert out.channel_prob.shape == (300, 2)


def test_harness_detects_same_day_normalisation_leak(world):
    """The paper-literal option uses the whole day's statistics; the harness must flag it."""
    make = lambda: paper_pipeline(  # noqa: E731
        BarsConfig(vwap_num=NUM), PaperSNNConfig(zscore={"stats_source": "same_day"})
    )
    with pytest.raises(AssertionError, match="causality violated"):
        check_causal(run_pipeline(make), world, TEST, BAR_TS, NUM)


def test_harness_detects_a_step_reading_later_days(world):
    """A step that fits on the test day or later days must be flagged."""

    def leaky(w):
        pipe = paper_pipeline(BarsConfig(vwap_num=NUM), PaperSNNConfig())
        pipe.fit([day_data(w, d) for d in DAYS[4:]])  # later days only: prefix of test day untouched
        return pipe.transform(day_data(w, TEST))

    with pytest.raises(AssertionError, match="causality violated"):
        check_causal(leaky, world, TEST, BAR_TS, NUM)


def test_perturbations_change_the_future_but_not_the_past(world):
    rng = np.random.default_rng(0)
    run = run_pipeline(lambda: paper_pipeline(BarsConfig(vwap_num=NUM), PaperSNNConfig()))
    base = run(world)
    t = 150
    for mode in ("values", "truncate", "resample"):
        perturbed = perturb_world(world, TEST, t, NUM, mode, rng)
        assert all(perturbed[d] is world[d] for d in DAYS if d < TEST)
        for d in DAYS[4:]:
            assert not perturbed[d].df.equals(world[d].df)
        out = run(perturbed)
        assert_prefix_equal(base, out, t)
        # Vacuity guard: the perturbation really changes what comes after bar t.
        n = min(base.n_bars, out.n_bars)
        assert base.n_bars != out.n_bars or not np.array_equal(base.channel_prob[t + 1 : n], out.channel_prob[t + 1 : n])


def test_perturb_tail_keeps_head_and_valid_ordering(world):
    frame = world[TEST]
    rng = np.random.default_rng(1)
    for mode in ("values", "truncate", "resample"):
        out = perturb_tail(frame, 505, mode, rng)  # TradeFrame validation checks order and day bounds
        assert out.df.iloc[:505].equals(frame.df.iloc[:505])


def test_harness_rejects_bars_outside_the_day(world):
    with pytest.raises(ValueError, match="outside"):
        check_causal(lambda w: None, world, TEST, [300], NUM)
