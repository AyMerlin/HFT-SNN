from datetime import date

import numpy as np
import pandas as pd
import pytest

from snn_hft.config.schema import BarsConfig, PaperSNNConfig
from snn_hft.data.containers import BAR_COLUMNS, TRADE_COLUMNS, BarSeries, DayData, TradeFrame, day_bounds_ns
from snn_hft.preprocessing.bars import VWAPBarAggregator, aggregate_vwap
from snn_hft.preprocessing.base import NotFittedError, Pipeline, PreprocessingStep
from snn_hft.preprocessing.normalization import ShiftedZScoreNormalizer
from snn_hft.preprocessing.pipelines import paper_pipeline
from snn_hft.preprocessing.returns import DirectionalSplitter, PriceDifferencer

DAY = date(2025, 10, 20)
T0 = day_bounds_ns(DAY)[0]


def trades(prices, qtys) -> TradeFrame:
    n = len(prices)
    df = pd.DataFrame(
        {
            "ts_ns": T0 + np.arange(n, dtype=np.int64) * 1_000_000,
            "price": prices,
            "qty": qtys,
            "is_buyer_maker": [False] * n,
            "trade_id": np.arange(n, dtype=np.int64),
        }
    ).astype(TRADE_COLUMNS)
    return TradeFrame(df, "S", "v", DAY)


def day_from_vwap(vwap, day=DAY) -> DayData:
    n = len(vwap)
    df = pd.DataFrame(
        {"bar_idx": np.arange(n), "ts_start_ns": np.arange(n), "ts_end_ns": np.arange(n),
         "vwap": vwap, "volume": np.ones(n), "n_trades": np.full(n, 10)}
    ).astype(BAR_COLUMNS)
    return DayData(day=day, bars=BarSeries(df))


def day_from_d(d, day=DAY) -> DayData:
    d = np.asarray(d, dtype=float)
    return DayData(day=day, d=d, channels_raw=np.column_stack([d, -d]))


# --------------------------------------------------------------------------- bars


def test_vwap_matches_hand_calculation():
    frame = trades([10, 11, 12, 13, 14, 15, 16], [1, 2, 1, 1, 3, 1, 5])
    bars = aggregate_vwap(frame.df, num=3).df
    assert bars["vwap"].tolist() == [44 / 4, 70 / 5]  # (10+22+12)/4, (13+42+15)/5
    assert bars["volume"].tolist() == [4.0, 5.0]
    assert bars["n_trades"].tolist() == [3, 3]
    assert bars["bar_idx"].tolist() == [0, 1]
    # bar timestamps: first and last trade of each group; trade 6 (incomplete group) is dropped
    assert bars["ts_start_ns"].tolist() == [T0, T0 + 3_000_000]
    assert bars["ts_end_ns"].tolist() == [T0 + 2_000_000, T0 + 5_000_000]


def test_vwap_with_fewer_trades_than_one_bar():
    assert len(aggregate_vwap(trades([1.0, 2.0], [1.0, 1.0]).df, num=10)) == 0


def test_bar_step_replaces_trades_by_bars():
    day = DayData(day=DAY, trades=trades([10.0] * 20, [1.0] * 20))
    out = VWAPBarAggregator(num=10).transform(day)
    assert out.trades is None and out.n_bars == 2
    assert VWAPBarAggregator(num=10, keep_trades=True).transform(day).trades is day.trades
    assert day.bars is None  # input not mutated
    with pytest.raises(ValueError):
        VWAPBarAggregator().transform(DayData(day=DAY))


# --------------------------------------------------------------------------- differences


def test_differencer_and_splitter():
    day = DirectionalSplitter().transform(PriceDifferencer().transform(day_from_vwap([100.0, 101.5, 101.0, 103.0])))
    np.testing.assert_array_equal(day.d, [np.nan, 1.5, -0.5, 2.0])
    np.testing.assert_array_equal(day.channels_raw, [[np.nan, np.nan], [1.5, -1.5], [-0.5, 0.5], [2.0, -2.0]])


# --------------------------------------------------------------------------- normalisation


def test_zscore_with_clipping_uses_training_stats():
    norm = ShiftedZScoreNormalizer(new_mean=0.2, new_std=0.1).fit([day_from_d([np.nan, 1, -1, 1, -1])])
    np.testing.assert_allclose(norm.mean_, [0.0, 0.0])
    np.testing.assert_allclose(norm.std_, [1.0, 1.0])
    prob = norm.transform(day_from_d([np.nan, 0, 3, -3, 10, -10])).channel_prob
    expected = np.array(
        [
            [0.0, 0.0],  # bar 0 has no price difference
            [0.2, 0.2],
            [0.5, 0.0],  # 0.2 − 0.3 = −0.1 clipped at 0 (paper's max(·, 0))
            [0.0, 0.5],
            [1.0, 0.0],  # 1.2 clipped to a probability of 1
            [0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(prob, expected)


def test_zscore_training_window_spans_all_training_days():
    norm = ShiftedZScoreNormalizer().fit([day_from_d([np.nan, 2.0, 4.0]), day_from_d([np.nan, 6.0])])
    np.testing.assert_allclose(norm.mean_, [4.0, -4.0])
    np.testing.assert_allclose(norm.std_, [np.std([2.0, 4.0, 6.0])] * 2)


def test_zscore_same_day_uses_the_day_itself():
    norm = ShiftedZScoreNormalizer(stats_source="same_day").fit([])
    prob = norm.transform(day_from_d([np.nan, 1.0, -1.0])).channel_prob
    np.testing.assert_allclose(prob, [[0.0, 0.0], [0.3, 0.1], [0.1, 0.3]])


def test_zscore_errors():
    with pytest.raises(NotFittedError):
        ShiftedZScoreNormalizer().transform(day_from_d([np.nan, 1.0, 2.0]))
    with pytest.raises(ValueError, match="zero standard deviation"):
        ShiftedZScoreNormalizer().fit([day_from_d([np.nan, 1.0, 1.0])])
    with pytest.raises(ValueError):
        ShiftedZScoreNormalizer(stats_source="future")


# --------------------------------------------------------------------------- pipeline


class Recorder(PreprocessingStep):
    def __init__(self, tag, log):
        self.tag, self.log = tag, log

    def fit(self, train_days):
        self.log.append((self.tag, [d.extras.get("seen", ()) for d in train_days]))
        return self

    def transform(self, day):
        return day.with_fields(extras={"seen": (*day.extras.get("seen", ()), self.tag)})


def test_pipeline_fits_each_step_on_days_transformed_by_earlier_steps():
    log = []
    pipe = Pipeline([Recorder("a", log), Recorder("b", log)])
    with pytest.raises(NotFittedError):
        pipe.transform(DayData(day=DAY))
    train = pipe.fit_transform([DayData(day=DAY), DayData(day=date(2025, 10, 21))])
    assert log == [("a", [(), ()]), ("b", [("a",), ("a",)])]
    assert [d.extras["seen"] for d in train] == [("a", "b"), ("a", "b")]
    assert pipe.transform(DayData(day=DAY)).extras["seen"] == ("a", "b")


def test_paper_pipeline_from_config():
    cfg = PaperSNNConfig(input={"new_mean": 0.3}, zscore={"new_std": 0.05, "stats_source": "same_day"})
    pipe = paper_pipeline(BarsConfig(vwap_num=20), cfg)
    assert [type(s).__name__ for s in pipe.steps] == [
        "VWAPBarAggregator", "PriceDifferencer", "DirectionalSplitter", "ShiftedZScoreNormalizer"
    ]
    assert pipe.steps[0].num == 20
    norm = pipe.steps[-1]
    assert (norm.new_mean, norm.new_std, norm.stats_source) == (0.3, 0.05, "same_day")
