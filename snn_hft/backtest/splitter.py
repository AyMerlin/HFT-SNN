"""Walk-forward folds (§8.1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from collections.abc import Iterable


@dataclass(frozen=True)
class FoldSpec:
    train_days: tuple[date, ...]
    test_day: date


class WalkForwardSplitter:
    """Train on the W_snn days before each test day, oldest first; test on the day itself."""

    def __init__(self, w_snn: int):
        if w_snn < 1:
            raise ValueError("w_snn must be >= 1")
        self.w_snn = w_snn

    def folds(self, test_days: Iterable[date]) -> list[FoldSpec]:
        return [
            FoldSpec(tuple(d - timedelta(days=k) for k in range(self.w_snn, 0, -1)), d)
            for d in sorted(test_days)
        ]
