"""Event extraction (§4.2): candidate up/down events from price differences."""

from __future__ import annotations

from snn_hft.data.containers import DayData, day_bounds_ns
from snn_hft.models.hawkes.events import events_from_bars
from snn_hft.preprocessing.base import PreprocessingStep


class EventExtractor(PreprocessingStep):
    """Up event if d_t > 0 (mark d_t), down event if d_t < 0 (mark |d_t|), none if d_t = 0.

    Writes every move to `day.events`; the Hawkes step keeps those above θ_d's threshold (U9).
    """

    def __init__(self, time_axis: str = "bar_index"):
        self.time_axis = time_axis

    def transform(self, day: DayData) -> DayData:
        if day.d is None:
            raise ValueError(f"{day.day}: EventExtractor needs d (run PriceDifferencer first)")
        events = events_from_bars(
            day.d, day.bars.df["ts_end_ns"].to_numpy(), day_bounds_ns(day.day)[0], self.time_axis
        )
        return day.with_fields(events=events)

    def __repr__(self) -> str:
        return f"EventExtractor(time_axis={self.time_axis!r})"
