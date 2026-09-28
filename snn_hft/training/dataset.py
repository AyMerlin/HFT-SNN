"""Input windows of the last T limit-order-book states (DeepLOB §III-C).

Sample i ends at row `end[i]` and covers rows `end[i]−T+1 … end[i]`; its target is the
label of row `end[i]`. With `cross_segments=True` every row from T−1 on is a sample and
windows may span two stocks or days, as in the authors' code (`dataX[i] = X[i:i+T]`,
`dataY = Y[T−1:]`). With `cross_segments=False` only windows inside one (stock, day)
segment are kept.

Windows are gathered per batch with one fancy-indexing call instead of being stored.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch


class WindowDataset:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        T: int = 100,
        segments: Sequence | None = None,
        cross_segments: bool = True,
    ):
        features = np.ascontiguousarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError(f"features {features.shape} and targets {targets.shape} do not match")
        if T < 1 or len(features) < T:
            raise ValueError(f"need at least T = {T} rows, got {len(features)}")
        self.features, self.targets, self.T = features, targets, T
        if cross_segments:
            self.end = np.arange(T - 1, len(features), dtype=np.int64)
        else:
            if segments is None:
                raise ValueError("cross_segments=False needs the segments")
            self.end = np.concatenate(
                [np.arange(s.start + T - 1, s.stop, dtype=np.int64) for s in segments if s.stop - s.start >= T]
                or [np.empty(0, np.int64)]
            )
        self._offsets = np.arange(-T + 1, 1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.end)

    @property
    def y(self) -> np.ndarray:
        """Targets of all samples, in sample order."""
        return self.targets[self.end]

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows (B, 1, T, F) and targets (B,) of the given sample indices."""
        rows = self.end[idx]
        x = self.features[rows[:, None] + self._offsets]
        return torch.from_numpy(x).unsqueeze(1), torch.from_numpy(self.targets[rows])

    def batches(
        self, batch_size: int, shuffle: bool = False, generator: np.random.Generator | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """All samples once; the last batch may be smaller (as in Keras `fit`)."""
        order = np.arange(len(self))
        if shuffle:
            if generator is None:
                raise ValueError("shuffle needs a generator")
            order = generator.permutation(order)
        for i in range(0, len(order), batch_size):
            yield self.batch(order[i : i + batch_size])
