"""Big-move benchmark (decision U5): a signal whenever the current price change is large.

The threshold is the (1 − target_rate) quantile of |d_t| on the training days, where the
target rate is the SNN's signal rate on those training days. The benchmark is therefore
causal (training data plus the current bar) and fires about as often as the SNN. It shows
how much spike accuracy follows from volatility persistence alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from snn_hft.data.containers import DayData
from snn_hft.preprocessing.base import Pipeline
from snn_hft.preprocessing.bars import VWAPBarAggregator
from snn_hft.preprocessing.returns import PriceDifferencer
from snn_hft.signals.base import SignalModel, SignalSeries


class BigMoveSignalModel(SignalModel):
    def __init__(self, vwap_num: int, target_rate: float | None = None):
        self.vwap_num = vwap_num
        self.target_rate = target_rate
        self.threshold: float | None = None
        self._pipeline = Pipeline([VWAPBarAggregator(vwap_num), PriceDifferencer()])

    @property
    def model_id(self) -> str:
        return f"big_move-n{self.vwap_num}"

    def _prepared(self, day: DayData) -> DayData:
        if day.d is not None:
            return day
        if day.bars is not None:
            return PriceDifferencer().transform(day)
        return self._pipeline.transform(day)

    def fit(self, train_days: Sequence[DayData], seed: int = 0) -> BigMoveSignalModel:
        if self.target_rate is None:
            raise RuntimeError("set target_rate (the reference model's training signal rate) before fit")
        self._pipeline.fit([])
        moves = np.concatenate([np.abs(self._prepared(d).d[1:]) for d in train_days])
        rate = min(max(self.target_rate, 0.0), 1.0)
        self.threshold = float(np.quantile(moves, 1.0 - rate)) if rate > 0 else np.inf
        return self

    def generate(self, day: DayData, seed: int = 0) -> SignalSeries:
        if self.threshold is None:
            raise RuntimeError("generate called before fit")
        prepared = self._prepared(day)
        move = np.abs(prepared.d)
        bars = np.flatnonzero(np.nan_to_num(move, nan=-1.0) > self.threshold)
        return SignalSeries(day=day.day, bar_idx=bars, n_bars=prepared.n_bars, model_id=self.model_id)
