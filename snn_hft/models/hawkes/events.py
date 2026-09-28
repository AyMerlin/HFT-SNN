"""Events from vwap bars (§4.2, §5.1).

Up event at bar t if d_t > 0 (mark d_t), down event if d_t < 0 (mark |d_t|), none if d_t = 0.
Time axis ``bar_index``: the event time is the bar index and the day is the window
[0, N_bars]; ``wallclock``: seconds since the UTC day start at the bar's last trade, window
[0, 86 400]. Each day is an independent realisation (history resets daily).
"""

from __future__ import annotations

import numpy as np

from snn_hft.models.hawkes.process import DayEvents

SECONDS_PER_DAY = 86_400.0


def bar_times(n_bars: int, ts_end_ns: np.ndarray | None, day_start_ns: int, time_axis: str) -> np.ndarray:
    if time_axis == "bar_index":
        return np.arange(n_bars, dtype=np.float64)
    if time_axis == "wallclock":
        return (np.asarray(ts_end_ns, dtype=np.int64) - day_start_ns) / 1e9
    raise ValueError(f"unknown time axis {time_axis!r}")


def events_from_bars(d: np.ndarray, ts_end_ns: np.ndarray | None, day_start_ns: int, time_axis: str) -> DayEvents:
    """Events of one day from its price differences d (NaN at bar 0)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    bars = np.flatnonzero(np.nan_to_num(d, nan=0.0) != 0.0)
    times = bar_times(n, ts_end_ns, day_start_ns, time_axis)[bars]
    horizon = float(n) if time_axis == "bar_index" else SECONDS_PER_DAY
    return DayEvents(
        times=times,
        types=(d[bars] < 0).astype(np.int64),
        marks=np.abs(d[bars]),
        horizon=horizon,
        bar_idx=bars,
    )
