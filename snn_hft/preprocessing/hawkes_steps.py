"""Hawkes preprocessing steps of the improved model (§4.2, §5.3–§5.5, §6.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from snn_hft.data.containers import DayData, day_bounds_ns
from snn_hft.models.hawkes.events import bar_times, threshold_events
from snn_hft.models.hawkes.process import BivariateHawkesProcess
from snn_hft.models.hawkes.provider import HawkesParamProvider
from snn_hft.preprocessing.base import PreprocessingStep


class HawkesIntensityTransformer(PreprocessingStep):
    """λ_u, λ_d at every bar with θ_d = fit(days d − W_h … d − 1) from the provider (§5.5).

    Keeps the events above θ_d's threshold (U9) in `day.events`, writes `lambda_u` / `lambda_d`
    (right limit λ(t⁺) by default, which includes the move of bar t; `limit="left"` excludes it)
    and the causal branching split per event to `day.branching`
    (DataFrame: bar_idx, p_bg, p_same, p_cross, momentum = p_same − p_cross). θ_d goes to
    `day.extras["hawkes_params"]`. The step has no state of its own: θ_d depends on the day,
    not on the training days, so training and test days are transformed identically.
    """

    def __init__(self, provider: HawkesParamProvider, limit: str = "right"):
        if limit not in ("right", "left"):
            raise ValueError(f"unknown limit {limit!r}")
        self.provider = provider
        self.limit = limit

    def transform(self, day: DayData) -> DayData:
        if day.events is None:
            raise ValueError(f"{day.day}: HawkesIntensityTransformer needs events (run EventExtractor first)")
        theta = self.provider.params_for(day.day)
        events = threshold_events(day.events, theta.event_threshold)
        times = bar_times(
            day.n_bars, day.bars.df["ts_end_ns"].to_numpy(), day_bounds_ns(day.day)[0], theta.time_axis
        )
        lam_minus, lam_plus, br = BivariateHawkesProcess(theta.marks["name"], theta.time_axis).bar_pass(theta, events, times)
        lam = lam_plus if self.limit == "right" else lam_minus
        branching = pd.DataFrame(
            {"bar_idx": events.bar_idx, "p_bg": br[:, 0], "p_same": br[:, 1], "p_cross": br[:, 2],
             "momentum": br[:, 1] - br[:, 2]}
        )
        return day.with_fields(
            events=events, lambda_u=lam[:, 0].copy(), lambda_d=lam[:, 1].copy(), branching=branching,
            extras={**day.extras, "hawkes_params": theta},
        )

    def __repr__(self) -> str:
        return f"HawkesIntensityTransformer(W_h={self.provider.w_h}, limit={self.limit!r})"


class BranchingRewardComputer(PreprocessingStep):
    """Pool rewards per bar (§6.5): reward_mom[b] = M(b) = p_same − p_cross at event bars,
    reward_rev = −reward_mom, 0 at bars without an event. M uses only history before bar b."""

    def transform(self, day: DayData) -> DayData:
        if day.branching is None:
            raise ValueError(f"{day.day}: BranchingRewardComputer needs the branching split")
        reward_mom = np.zeros(day.n_bars)
        reward_mom[day.branching["bar_idx"].to_numpy()] = day.branching["momentum"].to_numpy()
        return day.with_fields(reward_mom=reward_mom, reward_rev=-reward_mom)


def candidate_events_source(bars_for_day, time_axis: str = "bar_index"):
    """`events_for_day` for a HawkesParamProvider: every move of a day, from that day's bars.

    `bars_for_day(day) -> BarSeries`; the provider only asks for days before the one it fits for.
    """
    from snn_hft.preprocessing.events import EventExtractor
    from snn_hft.preprocessing.returns import PriceDifferencer

    differencer, extractor = PriceDifferencer(), EventExtractor(time_axis)

    def events_for_day(day):
        return extractor.transform(differencer.transform(DayData(day=day, bars=bars_for_day(day)))).events

    return events_for_day
