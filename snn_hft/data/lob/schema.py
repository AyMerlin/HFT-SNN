"""Standardized limit-order-book format (plan §4.5).

The single definition of the files the retrieval layer writes and the framework reads:

- `LOBFrame`: one venue, one symbol, one UTC day, one parquet file. Columns `ts_ns`,
  `recv_ts_ns`, then `ask_px_1..N`, `ask_qty_1..N`, `bid_px_1..N`, `bid_qty_1..N`.
- `Manifest`: one JSON file per dataset with provenance and the status of every day.

This module is shared by `retrieval/` (writing) and `snn_hft` (reading). It must not import
any other `snn_hft` module (enforced by tests/lob/test_architecture.py).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
NS_PER_DAY = 86_400 * 1_000_000_000

DayStatus = Literal["ok", "flagged", "excluded"]
DAY_STATUSES: tuple[str, ...] = ("ok", "flagged", "excluded")
SERVED_STATUSES: tuple[str, ...] = ("ok", "flagged")  # the framework never serves "excluded"

# Day-status thresholds on the fraction of the UTC day without a valid book (plan §4.2).
GAP_OK_MAX = 0.02
GAP_FLAGGED_MAX = 0.05

MANIFEST_NAME = "manifest.json"
QUALITY_REPORT_NAME = "quality_report.csv"


class LOBSchemaError(ValueError):
    """A frame, file or manifest does not follow the standardized format."""


# --------------------------------------------------------------------------- columns


def side_columns(side: Literal["ask", "bid"], kind: Literal["px", "qty"], depth: int) -> list[str]:
    return [f"{side}_{kind}_{i}" for i in range(1, depth + 1)]


def lob_columns(depth: int) -> list[str]:
    """All columns of a LOBFrame of the given depth, in file order."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    return [
        "ts_ns",
        "recv_ts_ns",
        *side_columns("ask", "px", depth),
        *side_columns("ask", "qty", depth),
        *side_columns("bid", "px", depth),
        *side_columns("bid", "qty", depth),
    ]


def arrow_schema(depth: int) -> pa.Schema:
    fields = [pa.field("ts_ns", pa.int64(), nullable=False), pa.field("recv_ts_ns", pa.int64(), nullable=True)]
    for side in ("ask", "bid"):
        fields += [pa.field(c, pa.float64(), nullable=False) for c in side_columns(side, "px", depth)]
        fields += [pa.field(c, pa.float32(), nullable=False) for c in side_columns(side, "qty", depth)]
    return pa.schema(fields)


def depth_of(columns: list[str]) -> int:
    """Depth N implied by a column list (number of ask price columns)."""
    return sum(1 for c in columns if c.startswith("ask_px_"))


def day_bounds_ns(day: date) -> tuple[int, int]:
    """[start, end) of a UTC day in epoch nanoseconds."""
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) * 1_000_000_000
    return start, start + NS_PER_DAY


def day_file_name(day: date) -> str:
    return f"{day.isoformat()}.parquet"


def dataset_dir(root: str | Path, venue: str, symbol: str) -> Path:
    """Directory of one standardized dataset, e.g. data/standardized/lob/bybit/BTCUSDT."""
    return Path(root) / venue / symbol


# --------------------------------------------------------------------------- frame validation


def validate_frame(df: pd.DataFrame, depth: int, day: date | None = None) -> None:
    """Raise LOBSchemaError unless `df` is a valid LOBFrame of the given depth.

    Checks: exact columns and dtypes; `ts_ns` non-decreasing (and inside `day` if given);
    every level present (no NaN) with positive finite prices and quantities; asks strictly
    increasing and bids strictly decreasing across levels; book neither crossed nor locked.
    """
    expected = lob_columns(depth)
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        raise LOBSchemaError(f"columns differ from depth-{depth} schema (missing {missing[:4]}, extra {extra[:4]})")
    dtypes = {"ts_ns": "int64", "recv_ts_ns": "Int64"}
    for c in expected[2:]:
        dtypes[c] = "float64" if "_px_" in c else "float32"
    for c, want in dtypes.items():
        got = str(df[c].dtype)
        if c == "recv_ts_ns" and got in ("int64", "Int64"):
            continue
        if got != want:
            raise LOBSchemaError(f"column {c} has dtype {got}, expected {want}")
    if len(df) == 0:
        return
    ts = df["ts_ns"].to_numpy()
    if np.any(np.diff(ts) < 0):
        raise LOBSchemaError("ts_ns is not sorted")
    if day is not None:
        lo, hi = day_bounds_ns(day)
        if ts[0] < lo or ts[-1] >= hi:
            raise LOBSchemaError(f"ts_ns outside UTC day {day}")
    ask_px = df[side_columns("ask", "px", depth)].to_numpy()
    bid_px = df[side_columns("bid", "px", depth)].to_numpy()
    qty = df[side_columns("ask", "qty", depth) + side_columns("bid", "qty", depth)].to_numpy()
    for name, arr in (("price", np.concatenate([ask_px, bid_px], axis=1)), ("quantity", qty)):
        if not np.all(np.isfinite(arr)):
            raise LOBSchemaError(f"non-finite {name} (missing level?)")
        if np.any(arr <= 0):
            raise LOBSchemaError(f"non-positive {name}")
    if depth > 1 and (np.any(np.diff(ask_px, axis=1) <= 0) or np.any(np.diff(bid_px, axis=1) >= 0)):
        raise LOBSchemaError("price levels are not strictly ordered away from the touch")
    if np.any(ask_px[:, 0] <= bid_px[:, 0]):
        raise LOBSchemaError("crossed or locked book (ask_px_1 <= bid_px_1)")


def frame_from_arrays(
    ts_ns: np.ndarray,
    recv_ts_ns: np.ndarray | None,
    ask_px: np.ndarray,
    ask_qty: np.ndarray,
    bid_px: np.ndarray,
    bid_qty: np.ndarray,
) -> pd.DataFrame:
    """Assemble a LOBFrame from (rows, depth) level arrays; does not validate."""
    n, depth = ask_px.shape
    data: dict[str, Any] = {"ts_ns": np.asarray(ts_ns, dtype=np.int64)}
    data["recv_ts_ns"] = (
        pd.array(np.asarray(recv_ts_ns, dtype=np.int64), dtype="Int64")
        if recv_ts_ns is not None
        else pd.arrays.IntegerArray(np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool))  # all null
    )
    for side, px, qty in (("ask", ask_px, ask_qty), ("bid", bid_px, bid_qty)):
        for i in range(depth):
            data[f"{side}_px_{i + 1}"] = np.asarray(px[:, i], dtype=np.float64)
        for i in range(depth):
            data[f"{side}_qty_{i + 1}"] = np.asarray(qty[:, i], dtype=np.float32)
    return pd.DataFrame(data, columns=lob_columns(depth))


# --------------------------------------------------------------------------- files


FILE_META_KEY = b"snn_hft.lob"


def write_frame(df: pd.DataFrame, path: str | Path, meta: dict[str, Any]) -> Path:
    """Write a validated LOBFrame atomically (temporary file, then rename) with zstd."""
    depth = depth_of(list(df.columns))
    table = pa.Table.from_pandas(df, schema=arrow_schema(depth), preserve_index=False)
    file_meta = {**meta, "schema_version": SCHEMA_VERSION, "depth": depth}
    table = table.replace_schema_metadata({FILE_META_KEY: json.dumps(file_meta, default=str).encode()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.chmod(tmp, 0o644)  # mkstemp creates 0600
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_file_meta(path: str | Path) -> dict[str, Any]:
    raw = (pq.read_schema(path).metadata or {}).get(FILE_META_KEY)
    if raw is None:
        raise LOBSchemaError(f"{path}: no {FILE_META_KEY.decode()} metadata")
    return json.loads(raw)


def read_frame(path: str | Path, day: date | None = None, validate: bool = True) -> pd.DataFrame:
    """Read one standardized day file; validates schema and content unless `validate=False`."""
    table = pq.read_table(path)
    depth = depth_of(table.column_names)
    if not table.schema.remove_metadata().equals(arrow_schema(depth)):
        raise LOBSchemaError(f"{path}: arrow schema differs from the depth-{depth} LOBFrame schema")
    df = table.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
    df["ts_ns"] = df["ts_ns"].astype("int64")
    if validate:
        validate_frame(df, depth, day)
    return df


# --------------------------------------------------------------------------- manifest


def day_status(gap_fraction: float, ok_max: float = GAP_OK_MAX, flagged_max: float = GAP_FLAGGED_MAX) -> DayStatus:
    """`ok` if the invalid-book share of the day is <= 2 %, `flagged` if <= 5 %, else `excluded`."""
    if gap_fraction <= ok_max:
        return "ok"
    if gap_fraction <= flagged_max:
        return "flagged"
    return "excluded"


@dataclass
class DayEntry:
    """Manifest record of one day."""

    status: DayStatus
    file: str | None  # file name relative to the dataset directory; None if nothing was written
    n_rows: int
    gap_fraction: float
    raw_file: str | None = None
    raw_sha256: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in DAY_STATUSES:
            raise LOBSchemaError(f"unknown day status {self.status!r}")


@dataclass
class Manifest:
    """Provenance and per-day status of one standardized dataset."""

    source: str
    venue: str
    symbol: str
    depth: int
    code_git_hash: str | None = None
    code_git_dirty: bool | None = None
    script: str | None = None
    script_args: dict[str, Any] = field(default_factory=dict)
    created_utc: str | None = None
    updated_utc: str | None = None
    quality_report: str = QUALITY_REPORT_NAME
    schema_version: int = SCHEMA_VERSION
    days: dict[str, DayEntry] = field(default_factory=dict)  # ISO date -> entry

    @property
    def date_range(self) -> tuple[date, date] | None:
        if not self.days:
            return None
        ds = sorted(self.days)
        return date.fromisoformat(ds[0]), date.fromisoformat(ds[-1])

    def served_days(self) -> list[date]:
        """Days the framework may use: status ok or flagged, with a file."""
        return sorted(date.fromisoformat(d) for d, e in self.days.items() if e.status in SERVED_STATUSES and e.file)

    def days_with_status(self, status: DayStatus) -> list[date]:
        return sorted(date.fromisoformat(d) for d, e in self.days.items() if e.status == status)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        rng = self.date_range
        out["date_range"] = None if rng is None else [rng[0].isoformat(), rng[1].isoformat()]
        out["days"] = {d: asdict(self.days[d]) for d in sorted(self.days)}
        return out

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Manifest:
        obj = dict(obj)
        obj.pop("date_range", None)
        if obj.get("schema_version") != SCHEMA_VERSION:
            raise LOBSchemaError(f"manifest schema_version {obj.get('schema_version')} != {SCHEMA_VERSION}")
        days = {d: DayEntry(**e) for d, e in obj.pop("days", {}).items()}
        try:
            m = cls(**obj, days=days)
        except TypeError as exc:
            raise LOBSchemaError(f"invalid manifest: {exc}") from None
        if m.depth < 1:
            raise LOBSchemaError("manifest depth must be >= 1")
        return m

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / MANIFEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=MANIFEST_NAME, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
            fh.write("\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> Manifest:
        path = Path(directory) / MANIFEST_NAME
        if not path.exists():
            raise LOBSchemaError(f"no manifest at {path}")
        return cls.from_dict(json.loads(path.read_text()))
