"""Normalisation of the input channels to per-tick spike probabilities (§4.1, §4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from snn_hft.data.containers import DayData
from snn_hft.preprocessing.base import NotFittedError, PreprocessingStep


def _channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and population standard deviation over rows without NaN."""
    rows = x[~np.isnan(x).any(axis=1)]
    if len(rows) < 2:
        raise ValueError("need at least two bars with a price difference to normalise")
    mean, std = rows.mean(axis=0), rows.std(axis=0)
    if not (std > 0).all():
        raise ValueError("a channel has zero standard deviation; cannot normalise")
    return mean, std


class ShiftedZScoreNormalizer(PreprocessingStep):
    """Paper Step 3: z = max((x − mean(x)) / stdev(x) · new_std + new_mean, 0).

    The result is clipped to [0, 1] and used as the per-tick spike probability.
    `stats_source="train_window"` (default) takes mean/stdev from the training days;
    `"same_day"` takes them from the transformed day itself, as the paper literally does,
    which uses future bars of that day (see DESIGN_DECISIONS P4).
    Bar 0 has no price difference and gets probability 0.
    """

    def __init__(
        self,
        new_mean: float = 0.2,
        new_std: float = 0.1,
        stats_source: Literal["train_window", "same_day"] = "train_window",
    ):
        if stats_source not in ("train_window", "same_day"):
            raise ValueError(f"unknown stats_source {stats_source!r}")
        self.new_mean = new_mean
        self.new_std = new_std
        self.stats_source = stats_source
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, train_days: Sequence[DayData]) -> ShiftedZScoreNormalizer:
        if self.stats_source == "train_window":
            if not train_days:
                raise ValueError("ShiftedZScoreNormalizer needs at least one training day")
            self.mean_, self.std_ = _channel_stats(np.concatenate([d.channels_raw for d in train_days]))
        return self

    def transform(self, day: DayData) -> DayData:
        x = day.channels_raw
        if self.stats_source == "same_day":
            mean, std = _channel_stats(x)
        elif self.mean_ is None:
            raise NotFittedError("ShiftedZScoreNormalizer.transform called before fit")
        else:
            mean, std = self.mean_, self.std_
        z = (x - mean) / std * self.new_std + self.new_mean
        prob = np.clip(z, 0.0, 1.0)
        prob[np.isnan(prob)] = 0.0
        return day.with_fields(channel_prob=prob)

    def __repr__(self) -> str:
        return (
            f"ShiftedZScoreNormalizer(new_mean={self.new_mean}, new_std={self.new_std}, "
            f"stats_source={self.stats_source!r})"
        )
