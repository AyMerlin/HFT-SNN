import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from snn_hft.backtest.performance import PerformanceCalculator, daily_aggregate
from snn_hft.config.schema import ExecutionConfig
from snn_hft.data.containers import BAR_COLUMNS, BarSeries, DayData
from snn_hft.signals.base import SignalSeries
from snn_hft.signals.big_move import BigMoveSignalModel
from snn_hft.signals.random_signal import RandomSignalModel
from snn_hft.strategy.direction_rules import (
    AlexandersFilterRule,
    MomentumRule,
    StochasticOscillatorRule,
    make_rule,
)
from snn_hft.strategy.execution import entry_bars, execute
from snn_hft.strategy.strategy import Strategy
from snn_hft.utils.seeding import rng_for

DAY = date(2025, 10, 20)


def bars(vwap, ts_start=None, ts_end=None) -> BarSeries:
    n = len(vwap)
    ts_start = np.arange(n) * 10 if ts_start is None else ts_start
    ts_end = np.asarray(ts_start) + 5 if ts_end is None else ts_end
    df = pd.DataFrame(
        {"bar_idx": np.arange(n), "ts_start_ns": ts_start, "ts_end_ns": ts_end, "vwap": vwap,
         "volume": np.ones(n), "n_trades": np.full(n, 10)}
    ).astype(BAR_COLUMNS)
    return BarSeries(df)


# --------------------------------------------------------------------------- direction rules


def test_momentum_rule_hand_examples():
    rule = MomentumRule(3)
    p = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
    # t=3: mean(10, 11, 12) − 11.5 = −0.5 < 0 → long; t=4: mean(11, 12, 11.5) − 13 < 0 → long; t < 3: no history
    assert rule.directions(p, np.array([0, 2, 3, 4])).tolist() == [0, 0, 1, 1]
    assert rule.direction(np.array([12.0, 12.0, 12.0, 11.0]), 3) == -1  # 12 − 11 > 0 → short
    assert rule.direction(np.array([75600.1] * 4), 3) == 0  # flag = 0 → no transaction


def test_alexanders_filter_hand_examples():
    rule = AlexandersFilterRule(1)
    p = np.array([10.0, 11.0, 10.5, 10.5])
    assert rule.directions(p, np.arange(4)).tolist() == [0, 1, -1, 0]


def test_stochastic_oscillator_hand_examples():
    rule = StochasticOscillatorRule(3)
    assert rule.direction(np.array([10.0, 12.0, 11.5]), 2) == 1  # %K = 75
    assert rule.direction(np.array([10.0, 12.0, 10.5]), 2) == -1  # %K = 25
    assert rule.direction(np.array([10.0, 12.0, 11.0]), 2) == 0  # %K = 50
    assert rule.direction(np.array([10.0, 10.0, 10.0]), 2) == 0  # H = L
    assert rule.direction(np.array([10.0, 12.0, 11.5]), 1) == 0  # window includes t: needs t ≥ n − 1


def test_rules_from_config_names():
    assert make_rule("momentum", momentum_window=5).min_history == 5
    assert make_rule("alexanders_filter", alf_n=2).min_history == 2
    assert make_rule("stochastic_oscillator", stoch_n=3).min_history == 2
    with pytest.raises(ValueError):
        make_rule("macd")


# --------------------------------------------------------------------------- execution


def test_entry_exit_and_returns():
    b = bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    trades = execute(b, np.array([1, 2]), np.array([1, -1]), ExecutionConfig())
    assert trades["entry_bar"].tolist() == [2, 3] and trades["exit_bar"].tolist() == [5, 6]
    assert trades["net_return"].tolist() == pytest.approx([105 / 102 - 1, -(106 / 103 - 1)])


def test_end_of_day_truncation_and_no_entry():
    b = bars([100.0, 101.0, 102.0, 103.0, 104.0])
    trades = execute(b, np.array([2, 3, 4]), np.array([1, 1, 1]), ExecutionConfig())
    # signal 2: entry 3, exit capped at the last bar 4; signal 3: entry and exit on the last bar;
    # signal 4: no bar left to enter → no trade
    assert trades["signal_bar"].tolist() == [2, 3]
    assert trades["exit_bar"].tolist() == [4, 4]
    assert trades["net_return"].tolist() == pytest.approx([104 / 103 - 1, 0.0])


def test_zero_direction_means_no_trade_and_fees_are_charged_twice():
    b = bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    trades = execute(b, np.array([0, 1]), np.array([0, 1]), ExecutionConfig(fee_rate=0.0005))
    assert trades["signal_bar"].tolist() == [1]
    assert trades["net_return"].iloc[0] == pytest.approx(105 / 102 - 1 - 0.001)


def test_latency_moves_the_entry_to_the_first_reachable_bar():
    ms = 1_000_000
    b = bars(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        ts_start=np.array([0, 100, 100, 105, 120, 130, 140]) * ms,
        ts_end=np.array([95, 100, 104, 118, 125, 135, 145]) * ms,
    )
    signal = np.array([1])  # ends at 100 ms
    assert entry_bars(b, signal, ExecutionConfig()).tolist() == [2]  # paper rule: next bar
    assert entry_bars(b, signal, ExecutionConfig(latency_ms=0.0)).tolist() == [2]
    assert entry_bars(b, signal, ExecutionConfig(latency_ms=5.0)).tolist() == [3]  # starts at 105 ms
    assert entry_bars(b, signal, ExecutionConfig(latency_ms=10.0)).tolist() == [4]  # first start ≥ 110 ms
    assert entry_bars(b, signal, ExecutionConfig(latency_ms=1_000.0)).tolist() == [-1]  # day over


def test_strategy_trades_for_day_matches_frame():
    b = bars([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 102.0, 105.0])
    day = DayData(day=DAY, bars=b)
    sig = SignalSeries(day=DAY, bar_idx=np.array([3, 4, 6]), n_bars=8, model_id="m")
    strat = Strategy(None, MomentumRule(3), ExecutionConfig())
    frame = strat.trades_frame(day, sig)
    trades = strat.trades_for_day(day, sig)
    assert [t.signal_bar for t in trades] == frame["signal_bar"].tolist()
    assert all(t.day == DAY for t in trades)
    with pytest.raises(ValueError):
        strat.trades_frame(day, SignalSeries(day=DAY, bar_idx=np.array([1]), n_bars=9, model_id="m"))


# --------------------------------------------------------------------------- performance


def test_known_pnl_and_table4_metrics():
    daily = pd.DataFrame([
        {"day": 1, **daily_aggregate([0.01, 0.005, -0.004])},  # pnl 0.011
        {"day": 2, **daily_aggregate([-0.006])},  # pnl −0.006
        {"day": 3, **daily_aggregate([])},  # no trades: pnl 0
        {"day": 4, **daily_aggregate([0.02, -0.001])},  # pnl 0.019
    ])
    m = PerformanceCalculator(365).compute(daily)
    pnl = np.array([0.011, -0.006, 0.0, 0.019])
    assert m["accumulated_return"] == pytest.approx(0.024)
    assert m["annualized_volatility"] == pytest.approx(pnl.std(ddof=1) * math.sqrt(365))
    assert m["sharpe"] == pytest.approx(pnl.mean() / pnl.std(ddof=1) * math.sqrt(365))
    assert m["win_rate"] == pytest.approx(3 / 6)
    assert m["profit_loss_ratio"] == pytest.approx((0.035 / 3) / (0.011 / 3))
    assert m["trades_per_day"] == pytest.approx(6 / 4)


# --------------------------------------------------------------------------- benchmarks


def test_random_signals_are_matched_distinct_eligible_and_seeded():
    ref = SignalSeries(day=DAY, bar_idx=np.arange(0, 1000, 7), n_bars=1000, model_id="m")
    model = RandomSignalModel(min_history=3, entry_delay=1, reference=ref, run_seed=0)
    a, b, c = model.generate(None, 0), model.generate(None, 0), model.generate(None, 1)
    assert len(a) == len(ref) and len(np.unique(a.bar_idx)) == len(a)
    assert a.bar_idx.min() >= 3 and a.bar_idx.max() <= 998
    np.testing.assert_array_equal(a.bar_idx, b.bar_idx)
    assert not np.array_equal(a.bar_idx, c.bar_idx)
    # More reference signals than eligible bars: every eligible bar once.
    assert RandomSignalModel.sample(10, 50, 3, 1, rng_for(0, "naive")).tolist() == list(range(3, 9))


def test_big_move_threshold_matches_training_rate():
    rng = np.random.default_rng(0)
    train = [DayData(day=DAY, bars=bars(100 + np.cumsum(rng.normal(0, 1, 5_001))))]
    model = BigMoveSignalModel(vwap_num=10, target_rate=0.2).fit(train)
    assert model.generate(train[0]).rate == pytest.approx(0.2, abs=0.001)
    test = DayData(day=DAY, bars=bars(100 + np.cumsum(rng.normal(0, 1, 2_000))))
    sig = model.generate(test)
    moves = np.abs(np.diff(test.bars.vwap))
    np.testing.assert_array_equal(sig.bar_idx, np.flatnonzero(moves > model.threshold) + 1)
