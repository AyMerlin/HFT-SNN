"""Improved model (§6.3): Hawkes-intensity input and reward-modulated STDP pools H_mom / H_rev.

Differs from the baseline only by its preprocessing (Hawkes steps instead of the z-score)
and topology (two extra pools trained with R-STDP); the engine, encoder, output layer and
signal extraction are the shared `SNNSignalModel` code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date

import numpy as np

from snn_hft.config.schema import BarsConfig, HawkesRSTDPConfig
from snn_hft.data.containers import BarSeries, DayData
from snn_hft.models.hawkes.provider import HawkesParamProvider, ParamStore
from snn_hft.preprocessing.base import Pipeline
from snn_hft.preprocessing.hawkes_steps import candidate_events_source
from snn_hft.preprocessing.pipelines import hawkes_pipeline
from snn_hft.signals.base import SNNSignalModel
from snn_hft.snn.network import NetworkBuilder, SpikingNetwork


class HawkesRSTDPSignalModel(SNNSignalModel):
    def __init__(
        self,
        cfg: HawkesRSTDPConfig,
        bars: BarsConfig,
        model_id: str,
        w_h: int,
        bars_for_day: Callable[[date], BarSeries],
        params_store: ParamStore | None = None,
    ):
        if not isinstance(cfg, HawkesRSTDPConfig):
            raise TypeError("HawkesRSTDPSignalModel needs a HawkesRSTDPConfig")
        super().__init__(cfg.core, cfg.input.ticks_per_bar, model_id)
        self.cfg = cfg
        self.bars = bars
        # One provider for the model's lifetime: θ_d is shared by every fold, seed and strategy.
        self.provider = HawkesParamProvider(
            cfg.hawkes, w_h, candidate_events_source(bars_for_day, cfg.hawkes.time_axis), params_store
        )
        self._days_used: set[date] = set()

    def build_pipeline(self) -> Pipeline:
        return hawkes_pipeline(self.bars, self.cfg, self.provider)

    def build_network(self) -> SpikingNetwork:
        return NetworkBuilder(self.cfg.core).hawkes_rstdp(
            self.cfg.rstdp,
            self.ticks_per_bar,
            use_rstdp_pools=self.cfg.use_rstdp_pools,
            keep_direction_pools=self.cfg.keep_direction_pools,
        )

    def rewards_for(self, day: DayData) -> Mapping[str, np.ndarray] | None:
        if not self.network.reward_streams:
            return None
        return {"mom": day.reward_mom, "rev": day.reward_rev}

    def transform(self, day: DayData) -> DayData:
        out = super().transform(day)
        self._days_used.add(out.day)
        return out

    def _simulate(self, day, seed, purpose, learn, epoch=0):
        self._days_used.add(day.day)
        return super()._simulate(day, seed, purpose, learn, epoch)

    def fold_metadata(self) -> dict:
        """θ_d of every day this model transformed since the last call (stored with the fold, §8.6)."""
        out = {}
        for d in sorted(self._days_used):
            theta = self.provider.params_for(d)
            out[d.isoformat()] = {
                **theta.named(),
                "event_threshold": theta.event_threshold,
                "rho": theta.spectral_radius,
                "branching_matrix": theta.branching_matrix.tolist(),
                "loglik_per_event": theta.fit.get("loglik_per_event"),
                "converged": theta.fit.get("converged"),
                "n_events": theta.fit.get("n_events"),
                "fit_days": theta.fit.get("fit_days"),
                "marks": theta.marks,
            }
        self._days_used.clear()
        return {"hawkes": out}
