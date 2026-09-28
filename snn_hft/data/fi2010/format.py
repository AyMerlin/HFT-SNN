"""Converted FI-2010 format: the single definition shared by the converter and the framework.

FI-2010 (Ntakaris et al., 2018) ships 149 × N text matrices, one column per sample:
rows 1–40 the limit order book (per level 1…10: ask price, ask volume, bid price, bid
volume), rows 41–144 hand-crafted features, rows 145–149 labels for the horizons
k = 10, 20, 30, 50, 100 (1 = up, 2 = stationary, 3 = down). The files come in nine anchored
folds: `Train_…_CF_i` holds days 1…i, `Test_…_CF_i` day i+1; each fold is normalized with
the statistics of its own training days. Within a file, samples are ordered stock by stock
(stock 1 all days, then stock 2, …).

A converted file `<stem>` (`train_cf<i>` / `test_cf<i>`) is three `.npy` arrays with one
row per sample — `<stem>.lob.npy` (N, 40) float32, `<stem>.handcrafted.npy` (N, 104)
float32, `<stem>.labels.npy` (N, 5) int8 with the dataset's values 1/2/3 — plus its entry in
`manifest.json`, which lists the (stock, day) segments of the file.

Must not import any other `snn_hft` module (tests/lob/test_architecture.py).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

SCHEMA_VERSION = 1

N_DAYS = 10
N_STOCKS = 5
N_FOLDS = 9
N_ROWS = 149
N_LOB = 40
N_HANDCRAFTED = 104
HORIZONS: tuple[int, ...] = (10, 20, 30, 50, 100)  # label rows 145…149, in events

LABEL_UP, LABEL_STATIONARY, LABEL_DOWN = 1, 2, 3  # values as stored in the dataset
CLASS_NAMES: tuple[str, ...] = ("up", "stationary", "down")  # class index = label − 1

AUCTIONS: tuple[str, ...] = ("NoAuction", "Auction")
NORMALIZATIONS: tuple[str, ...] = ("Zscore", "MinMax", "DecPre")

ARRAYS: dict[str, tuple[str, int]] = {  # name -> (dtype, columns)
    "lob": ("float32", N_LOB),
    "handcrafted": ("float32", N_HANDCRAFTED),
    "labels": ("int8", len(HORIZONS)),
}
MANIFEST_NAME = "manifest.json"

Kind = Literal["train", "test"]


class FI2010FormatError(ValueError):
    """Converted FI-2010 files do not follow this format."""


def horizon_row(k: int) -> int:
    """Index of horizon `k` (events) in the label array."""
    if k not in HORIZONS:
        raise ValueError(f"FI-2010 has no labels for k = {k}; available: {HORIZONS}")
    return HORIZONS.index(k)


def lob_column_names() -> list[str]:
    """Names of the 40 LOB columns in dataset order (level-major: ask px, ask vol, bid px, bid vol)."""
    return [f"{name}_{lvl}" for lvl in range(1, 11) for name in ("ask_px", "ask_qty", "bid_px", "bid_qty")]


def stem(kind: Kind, fold: int) -> str:
    if kind not in ("train", "test") or not 1 <= fold <= N_FOLDS:
        raise ValueError(f"no FI-2010 file {kind} fold {fold}")
    return f"{kind}_cf{fold}"


def days_of(kind: Kind, fold: int) -> list[int]:
    """Days (1-based) contained in a fold file."""
    return list(range(1, fold + 1)) if kind == "train" else [fold + 1]


def dataset_dir(root: str | Path, auction: str = "NoAuction", normalization: str = "Zscore") -> Path:
    if auction not in AUCTIONS or normalization not in NORMALIZATIONS:
        raise ValueError(f"unknown FI-2010 variant {auction}/{normalization}")
    return Path(root) / "fi2010" / f"{auction}_{normalization}"


# --------------------------------------------------------------------------- manifest


@dataclass
class Segment:
    """Samples [start, stop) of one stock on one day inside a file."""

    stock: int  # 1…5 in dataset order
    day: int  # 1…10
    start: int
    stop: int

    @property
    def n(self) -> int:
        return self.stop - self.start


@dataclass
class FileEntry:
    kind: Kind
    fold: int
    n: int
    days: list[int]
    segments: list[Segment]
    label_counts: dict[str, list[int]]  # horizon -> [up, stationary, down]
    source_member: str

    def __post_init__(self) -> None:
        self.segments = [s if isinstance(s, Segment) else Segment(**s) for s in self.segments]


@dataclass
class Manifest:
    dataset_id: str
    archive_sha256: str
    auction: str
    normalization: str
    files: dict[str, FileEntry] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    code_git_hash: str | None = None
    code_git_dirty: bool | None = None
    created_utc: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["files"] = {k: asdict(v) for k, v in sorted(self.files.items())}
        return out

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Manifest:
        obj = dict(obj)
        if obj.get("schema_version") != SCHEMA_VERSION:
            raise FI2010FormatError(f"manifest schema_version {obj.get('schema_version')} != {SCHEMA_VERSION}")
        files = {k: FileEntry(**v) for k, v in obj.pop("files", {}).items()}
        try:
            return cls(**obj, files=files)
        except TypeError as exc:
            raise FI2010FormatError(f"invalid manifest: {exc}") from None

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / MANIFEST_NAME
        _atomic(path, lambda tmp: tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n"))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> Manifest:
        path = Path(directory) / MANIFEST_NAME
        if not path.exists():
            raise FI2010FormatError(f"no manifest at {path}; run scripts.data.fi2010_prepare")
        return cls.from_dict(json.loads(path.read_text()))


# --------------------------------------------------------------------------- arrays


def array_path(directory: str | Path, file_stem: str, name: str) -> Path:
    if name not in ARRAYS:
        raise ValueError(f"unknown array {name!r}")
    return Path(directory) / f"{file_stem}.{name}.npy"


def split_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Dataset matrix (149, N) -> the three stored arrays, one row per sample."""
    if matrix.ndim != 2 or matrix.shape[0] != N_ROWS:
        raise FI2010FormatError(f"expected a ({N_ROWS}, N) matrix, got {matrix.shape}")
    labels = matrix[N_LOB + N_HANDCRAFTED :]
    if not np.all(np.isin(labels, (LABEL_UP, LABEL_STATIONARY, LABEL_DOWN))):
        raise FI2010FormatError("labels outside {1, 2, 3}")
    return {
        "lob": np.ascontiguousarray(matrix[:N_LOB].T, dtype=np.float32),
        "handcrafted": np.ascontiguousarray(matrix[N_LOB : N_LOB + N_HANDCRAFTED].T, dtype=np.float32),
        "labels": np.ascontiguousarray(labels.T, dtype=np.int8),
    }


def write_arrays(directory: str | Path, file_stem: str, arrays: dict[str, np.ndarray]) -> None:
    validate_arrays(arrays)
    Path(directory).mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        _atomic(array_path(directory, file_stem, name), lambda tmp, a=arr: _save_npy(tmp, a))


def read_arrays(directory: str | Path, file_stem: str, mmap: bool = False) -> dict[str, np.ndarray]:
    arrays = {
        name: np.load(array_path(directory, file_stem, name), mmap_mode="r" if mmap else None, allow_pickle=False)
        for name in ARRAYS
    }
    validate_arrays(arrays)
    return arrays


def validate_arrays(arrays: dict[str, np.ndarray], entry: FileEntry | None = None) -> None:
    if set(arrays) != set(ARRAYS):
        raise FI2010FormatError(f"arrays {sorted(arrays)} != {sorted(ARRAYS)}")
    n = len(arrays["labels"])
    for name, (dtype, cols) in ARRAYS.items():
        a = arrays[name]
        if a.dtype != np.dtype(dtype) or a.ndim != 2 or a.shape != (n, cols):
            raise FI2010FormatError(f"{name}: {a.dtype} {a.shape}, expected {dtype} ({n}, {cols})")
    if entry is not None:
        if n != entry.n:
            raise FI2010FormatError(f"{n} samples, manifest says {entry.n}")
        segs = entry.segments
        if not segs or segs[0].start != 0 or segs[-1].stop != n or any(a.stop != b.start for a, b in zip(segs, segs[1:])):
            raise FI2010FormatError("segments do not tile the file")


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as fh:  # a file handle keeps np.save from appending ".npy"
        np.save(fh, arr, allow_pickle=False)


def _atomic(path: Path, write) -> None:
    """Write via a temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        write(tmp_path)
        os.chmod(tmp_path, 0o644)  # mkstemp creates 0600
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
