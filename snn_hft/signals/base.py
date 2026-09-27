"""Signal model interface and the SNN plumbing shared by both SNN models (§2.4, §6.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from snn_hft.config.schema import SNNCoreConfig
from snn_hft.data.containers import DayData
from snn_hft.preprocessing.base import Pipeline
from snn_hft.snn.encoding import PoissonEncoder, SpikeEncoder
from snn_hft.snn.network import DayRun, SpikingNetwork
from snn_hft.utils.seeding import rng_for


@dataclass
class SignalSeries:
    """Bars of one day at which a model emitted a price-spike signal."""

    day: date
    bar_idx: np.ndarray  # sorted, unique bar indices
    n_bars: int
    model_id: str
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)  # per-bar arrays, e.g. pool spike counts

    def __post_init__(self) -> None:
        self.bar_idx = np.asarray(self.bar_idx, dtype=np.int64)
        if len(self.bar_idx) and (np.any(np.diff(self.bar_idx) <= 0) or self.bar_idx[0] < 0 or self.bar_idx[-1] >= self.n_bars):
            raise ValueError("bar_idx must be sorted, unique and within the day")

    def __len__(self) -> int:
        return len(self.bar_idx)

    @property
    def rate(self) -> float:
        """Fraction of bars with a signal."""
        return len(self.bar_idx) / self.n_bars if self.n_bars else 0.0


class SignalModel(ABC):
    """Produces signals for a test day after fitting on training days (causality: §0 principle 3)."""

    @abstractmethod
    def fit(self, train_days: Sequence[DayData], seed: int) -> SignalModel: ...

    @abstractmethod
    def generate(self, day: DayData, seed: int) -> SignalSeries: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...


class SNNSignalModel(SignalModel):
    """Pipeline → Poisson encoding → network simulation → one signal per bar with an output spike.

    Subclasses provide the pipeline, the network and (optionally) reward streams.
    Weights are initialised fresh per fit unless `warm_start`; membrane potentials and
    traces reset at the start of every day unless `reset_state_per_day` is false.
    """

    OUTPUT = "Out"

    def __init__(self, core: SNNCoreConfig, ticks_per_bar: int, model_id: str, encoder: SpikeEncoder | None = None):
        self.core = core
        self.ticks_per_bar = ticks_per_bar
        self._model_id = model_id
        self.encoder = encoder or PoissonEncoder()
        self.pipeline: Pipeline | None = None
        self.network: SpikingNetwork | None = None
        self.train_signals: list[SignalSeries] = []
        self.train_days: list[DayData] = []  # training days after preprocessing (bars etc.)

    @property
    def model_id(self) -> str:
        return self._model_id

    @abstractmethod
    def build_pipeline(self) -> Pipeline: ...

    @abstractmethod
    def build_network(self) -> SpikingNetwork: ...

    def rewards_for(self, day: DayData) -> Mapping[str, np.ndarray] | None:
        """Reward streams for a training day (none for plain STDP models)."""
        return None

    def fit(self, train_days: Sequence[DayData], seed: int) -> SNNSignalModel:
        if not train_days:
            raise ValueError("fit needs at least one training day")
        days = sorted(train_days, key=lambda d: d.day)
        self.pipeline = self.build_pipeline()
        transformed = self.pipeline.fit_transform(days)
        if self.network is None or not self.core.warm_start:
            self.network = self.build_network()
            self.network.init_weights(
                self.core.synapse.w_init_low, self.core.synapse.w_init_high, rng_for(seed, "init")
            )
        self.network.reset_state()
        self.train_signals = []
        for epoch in range(self.core.epochs):
            for day in transformed:
                run = self._simulate(day, seed, "train", learn=True, epoch=epoch)
                if epoch == self.core.epochs - 1:
                    self.train_signals.append(self._series(day, run))
        self.train_days = [d.with_fields(trades=None) for d in transformed]
        return self

    def generate(self, day: DayData, seed: int) -> SignalSeries:
        if self.pipeline is None or self.network is None:
            raise RuntimeError("generate called before fit")
        transformed = self.pipeline.transform(day)
        run = self._simulate(transformed, seed, "test", learn=self.core.learn_during_test)
        return self._series(transformed, run)

    def transform(self, day: DayData) -> DayData:
        """The model's preprocessing of a day (fitted state only)."""
        return self.pipeline.transform(day)

    # ------------------------------------------------------------------ internals

    def _simulate(self, day: DayData, seed: int, purpose: str, learn: bool, epoch: int = 0) -> DayRun:
        if self.core.reset_state_per_day:
            self.network.reset_state()
        spikes = self.encoder.encode(day.channel_prob, self.ticks_per_bar, rng_for(seed, purpose, day.day, epoch))
        rewards = self.rewards_for(day) if learn else None
        return self.network.run_day(spikes, self.ticks_per_bar, learn=learn, rewards=rewards)

    def _series(self, day: DayData, run: DayRun) -> SignalSeries:
        out = run.pop_counts(self.OUTPUT)
        diagnostics = {f"spikes_{name}": run.pop_counts(name).copy() for name in run.pop_names}
        diagnostics["channel_prob"] = day.channel_prob
        return SignalSeries(
            day=day.day,
            bar_idx=np.flatnonzero(out > 0),  # at most one signal per bar (P11)
            n_bars=day.n_bars,
            model_id=self.model_id,
            diagnostics=diagnostics,
        )
