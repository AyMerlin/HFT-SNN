"""Strategy = signal model × direction rule × execution (§7.3).

Any SignalModel plugs in unchanged; a naive strategy is the same strategy with a
RandomSignalModel. Direction comes from price only, never from momentum/reversion
information (§0 scope).
"""

from __future__ import annotations

import pandas as pd

from snn_hft.config.schema import RunConfig
from snn_hft.data.containers import DayData
from snn_hft.signals.base import SignalModel, SignalSeries
from snn_hft.signals.factory import make_signal_model
from snn_hft.strategy.direction_rules import DirectionRule, make_rule
from snn_hft.strategy.execution import ExecutionConfig, Trade, execute


class Strategy:
    def __init__(self, signal_model: SignalModel | None, direction_rule: DirectionRule, execution: ExecutionConfig):
        self.signal_model = signal_model
        self.direction_rule = direction_rule
        self.execution = execution

    @classmethod
    def from_config(cls, run: RunConfig) -> Strategy:
        s = run.strategy
        rule = make_rule(run.rule, s.momentum_window, s.alf_n, s.stoch_n)
        return cls(make_signal_model(run), rule, run.execution)

    def trades_frame(self, day: DayData, signals: SignalSeries) -> pd.DataFrame:
        if day.bars is None or len(day.bars) != signals.n_bars:
            raise ValueError(f"{signals.day}: signals were made on a different bar series")
        directions = self.direction_rule.directions(day.bars.vwap, signals.bar_idx)
        return execute(day.bars, signals.bar_idx, directions, self.execution)

    def trades_for_day(self, day: DayData, signals: SignalSeries) -> list[Trade]:
        frame = self.trades_frame(day, signals)
        return [Trade(day=signals.day, **row) for row in frame.to_dict("records")]
