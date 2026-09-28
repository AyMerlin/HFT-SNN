"""Frozen parameters per day (§5.5): θ_d = fit(days [d − W_h, d − 1]).

The provider only ever requests the W_h days strictly before the day it returns
parameters for, so no day is encoded or rewarded with parameters fitted on itself.
Training days and test days are treated identically.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from datetime import date
from typing import Protocol

from snn_hft.config.schema import HawkesConfig
from snn_hft.data.store import TradingCalendar
from snn_hft.models.hawkes.events import fit_window_threshold, threshold_events
from snn_hft.models.hawkes.params import HawkesParameters
from snn_hft.models.hawkes.process import BivariateHawkesProcess, DayEvents, HawkesFitError
from snn_hft.utils.repro import stable_hash
from snn_hft.utils.seeding import rng_for

log = logging.getLogger(__name__)


class ParamStore(Protocol):
    def has(self, key: str) -> bool: ...
    def load(self, key: str) -> HawkesParameters: ...
    def save(self, key: str, params: HawkesParameters) -> None: ...


class HawkesParamProvider:
    def __init__(
        self,
        cfg: HawkesConfig,
        w_h: int,
        events_for_day: Callable[[date], DayEvents],
        store: ParamStore | None = None,
    ):
        if w_h < 1:
            raise ValueError("w_h must be >= 1")
        self.cfg = cfg
        self.w_h = w_h
        self.events_for_day = events_for_day
        self.store = store
        self._memo: dict[date, HawkesParameters] = {}

    @property
    def fit_settings_id(self) -> str:
        c = self.cfg
        return stable_hash({"restarts": c.restarts, "grid": list(c.beta_init_grid), "max_iter": c.max_iter}, 8)

    def key(self, day: date) -> str:
        q = "all" if self.cfg.event_quantile is None else f"q{self.cfg.event_quantile:g}"
        return f"{self.cfg.mark_fn}_{self.cfg.time_axis}_{q}_Wh{self.w_h}_{self.fit_settings_id}/{day.isoformat()}"

    @staticmethod
    def fit_days(day: date, w_h: int) -> list[date]:
        return TradingCalendar.previous(day, w_h)

    def params_for(self, day: date) -> HawkesParameters:
        if day in self._memo:
            return self._memo[day]
        key = self.key(day)
        if self.store is not None and self.store.has(key):
            params = self.store.load(key)
        else:
            params = self._fit(day)
            if self.store is not None:
                self.store.save(key, params)
        self._memo[day] = params
        return params

    def _fit(self, day: date) -> HawkesParameters:
        """`events_for_day` returns every move (candidate events); the threshold is learned here."""
        days = self.fit_days(day, self.w_h)
        moves = [self.events_for_day(d) for d in days]
        threshold = fit_window_threshold(moves, self.cfg.event_quantile)
        events = [threshold_events(m, threshold) for m in moves]
        process = BivariateHawkesProcess(self.cfg.mark_fn, self.cfg.time_axis)
        provenance = {
            "for_day": day.isoformat(), "fit_days": [d.isoformat() for d in days], "w_h": self.w_h,
            "event_quantile": self.cfg.event_quantile, "candidate_moves": int(sum(len(m) for m in moves)),
        }
        kwargs = dict(max_iter=self.cfg.max_iter, beta_grid=self.cfg.beta_init_grid, provenance=provenance)
        try:
            params = process.fit(events, restarts=self.cfg.restarts, rng=rng_for(0, "hawkes", day, self.w_h), **kwargs)
        except HawkesFitError:
            log.warning("Hawkes fit for %s: no stationary restart; retrying with more restarts and a stronger penalty", day)
            params = process.fit(
                events, restarts=4 * self.cfg.restarts, penalty=1e4, rng=rng_for(1, "hawkes", day, self.w_h), **kwargs
            )
        return dataclasses.replace(params, event_threshold=threshold)
