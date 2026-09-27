"""Price difference and negative price difference (§4.1, paper Step 2)."""

from __future__ import annotations

import numpy as np

from snn_hft.data.containers import DayData
from snn_hft.preprocessing.base import PreprocessingStep


class PriceDifferencer(PreprocessingStep):
    """d_t = V_t − V_{t−1} for t ≥ 1; bar 0 has no difference (NaN)."""

    def transform(self, day: DayData) -> DayData:
        vwap = day.bars.vwap
        d = np.full(len(vwap), np.nan)
        d[1:] = np.diff(vwap)
        return day.with_fields(d=d)


class DirectionalSplitter(PreprocessingStep):
    """Channel 0 = d_t (X1, price difference), channel 1 = −d_t (X2, negative price difference)."""

    def transform(self, day: DayData) -> DayData:
        if day.d is None:
            raise ValueError(f"{day.day}: DirectionalSplitter needs d (run PriceDifferencer first)")
        return day.with_fields(channels_raw=np.column_stack([day.d, -day.d]))
