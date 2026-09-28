"""Read-only access to converted FI-2010 files and the DeepLOB paper's two setups (§V-B).

- Setup 1: nine anchored folds; fold i trains on days 1…i (`train_cf<i>`) and tests on day
  i+1 (`test_cf<i>`). Paper horizons k = 10, 50, 100.
- Setup 2: trains on days 1–7 (`train_cf7`) and tests on days 8–10 (`test_cf7`, `test_cf8`,
  `test_cf9` concatenated, as in the authors' code). Paper horizons k = 10, 20, 50.

The validation part is the last 20 % of the training file (the paper does not specify one;
authors' code and LOBCAST). No network access, no conversion: files come from
`scripts.data.fi2010_prepare`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snn_hft.data.fi2010 import format as fmt

SETUP_HORIZONS: dict[int, tuple[int, ...]] = {1: (10, 50, 100), 2: (10, 20, 50)}
TRAIN_FRACTION = 0.8  # of each training file; the rest is the validation part


@dataclass(frozen=True)
class FI2010Part:
    """Consecutive samples used as one split part (train, validation or test)."""

    name: str
    lob: np.ndarray  # (N, 40) float32, dataset column order
    handcrafted: np.ndarray  # (N, 104) float32
    labels: np.ndarray  # (N, 5) int8, dataset values 1/2/3
    segments: tuple[fmt.Segment, ...]  # (stock, day) runs, positions relative to this part

    def __len__(self) -> int:
        return len(self.labels)

    def targets(self, k: int) -> np.ndarray:
        """Class indices for horizon k: 0 = up, 1 = stationary, 2 = down."""
        return self.labels[:, fmt.horizon_row(k)].astype(np.int64) - 1

    def class_counts(self, k: int) -> np.ndarray:
        return np.bincount(self.targets(k), minlength=3)

    def slice(self, start: int, stop: int, name: str | None = None) -> FI2010Part:
        segs = []
        for s in self.segments:
            a, b = max(s.start, start), min(s.stop, stop)
            if b > a:
                segs.append(fmt.Segment(s.stock, s.day, a - start, b - start))
        return FI2010Part(name or self.name, self.lob[start:stop], self.handcrafted[start:stop],
                          self.labels[start:stop], tuple(segs))

    @staticmethod
    def concat(parts: list[FI2010Part], name: str) -> FI2010Part:
        segs, offset = [], 0
        for p in parts:
            segs += [fmt.Segment(s.stock, s.day, s.start + offset, s.stop + offset) for s in p.segments]
            offset += len(p)
        return FI2010Part(name, np.concatenate([p.lob for p in parts]), np.concatenate([p.handcrafted for p in parts]),
                          np.concatenate([p.labels for p in parts]), tuple(segs))


@dataclass(frozen=True)
class FI2010Split:
    setup: int
    fold: int | None  # Setup 1 only
    train: FI2010Part
    val: FI2010Part
    test: FI2010Part

    @property
    def horizons(self) -> tuple[int, ...]:
        return SETUP_HORIZONS[self.setup]


class FI2010DataSource:
    def __init__(self, root: str | Path = "data/standardized", auction: str = "NoAuction", normalization: str = "Zscore"):
        self.directory = fmt.dataset_dir(root, auction, normalization)
        self.manifest = fmt.Manifest.load(self.directory)
        if (self.manifest.auction, self.manifest.normalization) != (auction, normalization):
            raise fmt.FI2010FormatError(f"{self.directory} holds {self.manifest.auction}/{self.manifest.normalization}")

    def file(self, kind: fmt.Kind, fold: int) -> FI2010Part:
        name = fmt.stem(kind, fold)
        entry = self.manifest.files.get(name)
        if entry is None:
            raise fmt.FI2010FormatError(f"{name} missing from {self.directory / fmt.MANIFEST_NAME}")
        arrays = fmt.read_arrays(self.directory, name)
        fmt.validate_arrays(arrays, entry)
        return FI2010Part(name, arrays["lob"], arrays["handcrafted"], arrays["labels"], tuple(entry.segments))

    @staticmethod
    def train_val(train_file: FI2010Part, train_fraction: float = TRAIN_FRACTION) -> tuple[FI2010Part, FI2010Part]:
        """First ⌊fraction·N⌋ samples for training, the rest for validation (authors' code)."""
        n_train = int(np.floor(len(train_file) * train_fraction))
        return (train_file.slice(0, n_train, f"{train_file.name}[train]"),
                train_file.slice(n_train, len(train_file), f"{train_file.name}[val]"))

    def setup1(self, fold: int, train_fraction: float = TRAIN_FRACTION) -> FI2010Split:
        train, val = self.train_val(self.file("train", fold), train_fraction)
        return FI2010Split(1, fold, train, val, self.file("test", fold))

    def setup1_folds(self, train_fraction: float = TRAIN_FRACTION) -> list[FI2010Split]:
        return [self.setup1(i, train_fraction) for i in range(1, fmt.N_FOLDS + 1)]

    def setup2(self, train_fraction: float = TRAIN_FRACTION) -> FI2010Split:
        train, val = self.train_val(self.file("train", 7), train_fraction)
        test = FI2010Part.concat([self.file("test", i) for i in (7, 8, 9)], "test_cf7+8+9")
        return FI2010Split(2, None, train, val, test)
