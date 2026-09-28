"""Improved-model preprocessing (M7): events, intensity, rate scaling, rewards, causality."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from snn_hft.config.schema import BarsConfig, HawkesConfig, HawkesRSTDPConfig
from snn_hft.data.containers import BAR_COLUMNS, BarSeries, DayData
from snn_hft.data.sources import SyntheticDataSource
from snn_hft.models.hawkes.params import HawkesParameters
from snn_hft.models.hawkes.process import BivariateHawkesProcess
from snn_hft.models.hawkes.provider import HawkesParamProvider
from snn_hft.preprocessing.bars import aggregate_vwap
from snn_hft.preprocessing.base import NotFittedError
from snn_hft.preprocessing.events import EventExtractor
from snn_hft.preprocessing.hawkes_steps import (
    BranchingRewardComputer,
    HawkesIntensityTransformer,
    candidate_events_source,
)
from snn_hft.preprocessing.normalization import IntensityRateScaler
from snn_hft.preprocessing.pipelines import hawkes_pipeline
from snn_hft.preprocessing.returns import PriceDifferencer
from snn_hft.testing.causality import check_causal, day_data

NUM, W_H = 10, 2
FIRST = date(2025, 10, 15)
DAYS = [FIRST + timedelta(days=i) for i in range(7)]  # 2 history days, 2 training days, test day, 2 later days
TRAIN, TEST = DAYS[2:4], DAYS[4]
HAWKES = HawkesConfig(restarts=2, event_quantile=0.5)


@pytest.fixture(scope="module")
def world():
    src = SyntheticDataSource(trades_per_day=3_000, seed=21)
    return {d: src.fetch_day("SYN", d) for d in DAYS}


def provider_for(world, cfg=HAWKES, w_h=W_H, cls=HawkesParamProvider):
    return cls(cfg, w_h, candidate_events_source(lambda d: aggregate_vwap(world[d].df, NUM)))


def day_with_d(vwap, day=TEST) -> DayData:
    n = len(vwap)
    df = pd.DataFrame(
        {"bar_idx": np.arange(n), "ts_start_ns": np.arange(n), "ts_end_ns": np.arange(n), "vwap": vwap,
         "volume": np.ones(n), "n_trades": np.full(n, NUM)}
    ).astype(BAR_COLUMNS)
    return PriceDifferencer().transform(DayData(day=day, bars=BarSeries(df)))


# --------------------------------------------------------------------------- events


def test_event_extraction_sign_rule_marks_and_no_event_on_zero():
    day = EventExtractor().transform(day_with_d([100.0, 100.5, 100.3, 100.3, 101.8]))
    assert day.events.bar_idx.tolist() == [1, 2, 4]
    assert day.events.types.tolist() == [0, 1, 0]
    np.testing.assert_allclose(day.events.marks, [0.5, 0.2, 1.5])
    with pytest.raises(ValueError):
        EventExtractor().transform(DayData(day=TEST))


# --------------------------------------------------------------------------- threshold (U9) and intensity


def test_threshold_is_the_fit_window_quantile_and_frozen(world):
    provider = provider_for(world)
    theta = provider.params_for(TEST)
    moves = np.concatenate([provider.events_for_day(d).marks for d in DAYS[2:4]])
    assert theta.event_threshold == pytest.approx(np.quantile(moves, 0.5))
    assert theta.fit["n_events"] == int((moves > theta.event_threshold).sum())
    assert HawkesParameters.from_json(theta.to_json()).event_threshold == theta.event_threshold
    every = provider_for(world, cfg=HawkesConfig(restarts=2, event_quantile=None)).params_for(TEST)
    assert every.event_threshold is None and every.fit["n_events"] == len(moves)


def test_intensity_transformer_uses_thresholded_events_and_limits(world):
    provider = provider_for(world)
    raw = PriceDifferencer().transform(DayData(day=TEST, bars=aggregate_vwap(world[TEST].df, NUM)))
    day = EventExtractor().transform(raw)
    out = HawkesIntensityTransformer(provider).transform(day)
    theta = provider.params_for(TEST)
    assert (out.events.marks > theta.event_threshold).all()
    assert len(out.events) < len(day.events)
    times = np.arange(out.n_bars, dtype=float)
    lam_minus, lam_plus, br = BivariateHawkesProcess(theta.marks["name"]).bar_pass(theta, out.events, times)
    np.testing.assert_array_equal(out.lambda_u, lam_plus[:, 0])
    np.testing.assert_array_equal(out.lambda_d, lam_plus[:, 1])
    left = HawkesIntensityTransformer(provider, limit="left").transform(day)
    np.testing.assert_array_equal(left.lambda_u, lam_minus[:, 0])
    assert out.branching["bar_idx"].tolist() == out.events.bar_idx.tolist()
    np.testing.assert_allclose(out.branching[["p_bg", "p_same", "p_cross"]].sum(axis=1), 1.0)
    np.testing.assert_allclose(out.branching["momentum"], br[:, 1] - br[:, 2])
    assert out.extras["hawkes_params"] == theta


def test_rate_scaler_matches_new_mean_on_training_days():
    rng = np.random.default_rng(0)
    train = [DayData(day=TEST, lambda_u=rng.uniform(0.1, 0.5, 500), lambda_d=rng.uniform(0.2, 0.4, 500)) for _ in range(2)]
    scaler = IntensityRateScaler(new_mean=0.1).fit(train)
    probs = np.concatenate([scaler.transform(d).channel_prob for d in train])
    np.testing.assert_allclose(probs.mean(axis=0), [0.1, 0.1])  # no clipping at these levels
    spike = DayData(day=TEST, lambda_u=np.array([100.0]), lambda_d=np.array([0.0]))
    assert scaler.transform(spike).channel_prob.tolist() == [[1.0, 0.0]]
    with pytest.raises(NotFittedError):
        IntensityRateScaler().transform(train[0])


def test_rewards_are_the_momentum_score_at_event_bars():
    br = pd.DataFrame({"bar_idx": [2, 5], "p_bg": [0.5, 0.2], "p_same": [0.4, 0.1], "p_cross": [0.1, 0.7],
                       "momentum": [0.3, -0.6]})
    day = BranchingRewardComputer().transform(DayData(day=TEST, bars=day_with_d(np.arange(7.0)).bars, branching=br))
    assert day.reward_mom.tolist() == [0, 0, 0.3, 0, 0, -0.6, 0]
    np.testing.assert_array_equal(day.reward_rev, -day.reward_mom)


def test_hawkes_pipeline_order(world):
    pipe = hawkes_pipeline(BarsConfig(vwap_num=NUM), HawkesRSTDPConfig(hawkes=HAWKES), provider_for(world))
    assert [type(s).__name__ for s in pipe.steps] == [
        "VWAPBarAggregator", "PriceDifferencer", "DirectionalSplitter", "EventExtractor",
        "HawkesIntensityTransformer", "IntensityRateScaler", "BranchingRewardComputer",
    ]


# --------------------------------------------------------------------------- causality


def run_improved(cls=HawkesParamProvider):
    def run(w):
        pipe = hawkes_pipeline(BarsConfig(vwap_num=NUM), HawkesRSTDPConfig(hawkes=HAWKES), provider_for(w, cls=cls))
        pipe.fit([day_data(w, d) for d in TRAIN])
        return pipe.transform(day_data(w, TEST))

    return run


def test_improved_pipeline_is_causal(world):
    out = check_causal(run_improved(), world, TEST, (0, 2, 60, 150, 298), NUM)
    assert out.channel_prob.shape == (300, 2) and out.reward_mom.shape == (300,)
    event_bars = out.events.bar_idx
    np.testing.assert_array_equal(out.reward_mom[event_bars], out.branching["momentum"].to_numpy())
    assert (np.delete(out.reward_mom, event_bars) == 0).all()


class LeakyProvider(HawkesParamProvider):
    """Fits θ_d on a window that includes day d itself."""

    @staticmethod
    def fit_days(day, w_h):
        return [day - timedelta(days=k) for k in range(w_h - 1, -1, -1)]


def test_harness_detects_parameters_fitted_on_the_same_day(world):
    with pytest.raises(AssertionError, match="causality violated"):
        check_causal(run_improved(LeakyProvider), world, TEST, (150,), NUM)
