"""Preprocessing step interface and pipeline (§4).

Causality contract (§0 principle 3): `fit` learns state only from the training
days it receives; `transform` uses that fitted state plus data of the same day
up to the current bar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from snn_hft.data.containers import DayData


class NotFittedError(RuntimeError):
    pass


class PreprocessingStep(ABC):
    def fit(self, train_days: Sequence[DayData]) -> PreprocessingStep:
        """Learn state from training days only. Stateless steps keep this default."""
        return self

    @abstractmethod
    def transform(self, day: DayData) -> DayData:
        """Return a copy of `day` with this step's fields filled."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Pipeline:
    """Fits steps in order: step i is fit on training days transformed by steps < i."""

    def __init__(self, steps: Sequence[PreprocessingStep]):
        self.steps = list(steps)
        self._fitted = False

    def fit_transform(self, train_days: Sequence[DayData]) -> list[DayData]:
        """Fit every step and return the training days transformed by the fitted pipeline."""
        days = list(train_days)
        for step in self.steps:
            step.fit(days)
            days = [step.transform(d) for d in days]
        self._fitted = True
        return days

    def fit(self, train_days: Sequence[DayData]) -> Pipeline:
        self.fit_transform(train_days)
        return self

    def transform(self, day: DayData) -> DayData:
        if not self._fitted:
            raise NotFittedError("Pipeline.transform called before fit")
        for step in self.steps:
            day = step.transform(day)
        return day

    def __repr__(self) -> str:
        return " → ".join(repr(s) for s in self.steps)
