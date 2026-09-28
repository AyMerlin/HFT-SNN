"""Per-day and per-dataset quality reports (plan §4.2 "Quality control", Table 0 of §10)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path

import pandas as pd

from snn_hft.data.lob.schema import NS_PER_DAY, DayStatus, day_status

NS_PER_S = 1_000_000_000


@dataclass
class DayQualityReport:
    """Everything the retrieval layer knows about the quality of one standardized day."""

    day: date
    raw_file: str | None = None
    # messages
    n_messages: int = 0
    n_snapshots: int = 0
    n_outside_day: int = 0  # messages stamped outside the UTC day (includes Bybit's closing snapshot)
    n_ts_backwards: int = 0
    n_seq_gaps: int = 0
    n_deltas_dropped: int = 0
    n_invalid_levels: int = 0
    n_snapshot_checks: int = 0
    n_snapshot_mismatches: int = 0
    # rows
    n_rows_emitted: int = 0
    n_rows_crossed_or_locked: int = 0
    n_rows_shallow: int = 0  # fewer than N levels on a side
    n_rows_nonpositive: int = 0
    n_rows_written: int = 0
    # time
    first_ts_ns: int | None = None
    last_ts_ns: int | None = None
    invalid_book_s: float = 0.0
    silence_s: float = 0.0  # message-free stretches longer than the silence threshold
    max_interval_s: float = 0.0
    n_intervals_over_1s: int = 0
    # verdict
    status: DayStatus = "excluded"
    notes: list[str] = field(default_factory=list)

    @property
    def gap_s(self) -> float:
        return self.invalid_book_s + self.silence_s

    @property
    def gap_fraction(self) -> float:
        return self.gap_s * NS_PER_S / NS_PER_DAY

    @property
    def n_rows_removed(self) -> int:
        return self.n_rows_emitted - self.n_rows_written

    def decide_status(self) -> DayStatus:
        """Gap thresholds of plan §4.2, then downgrades for reconstruction problems."""
        status = day_status(self.gap_fraction)
        if self.n_rows_written == 0:
            status = "excluded"
            self.notes.append("no valid rows")
        elif self.n_snapshot_mismatches and status == "ok":
            status = "flagged"
            self.notes.append(f"{self.n_snapshot_mismatches} snapshot mismatch(es)")
        self.status = status
        return status

    def to_row(self) -> dict:
        row = {f.name: getattr(self, f.name) for f in fields(self)}
        row["day"] = self.day.isoformat()
        row["notes"] = "; ".join(self.notes)
        row["gap_s"] = self.gap_s
        row["gap_fraction"] = self.gap_fraction
        row["n_rows_removed"] = self.n_rows_removed
        return row


GAP_COLUMNS = ["day", "start_ns", "end_ns", "reason", "duration_s"]


@dataclass
class GapRecord:
    day: date
    start_ns: int
    end_ns: int | None
    reason: str

    def to_row(self) -> dict:
        row = asdict(self)
        row["day"] = self.day.isoformat()
        row["duration_s"] = None if self.end_ns is None else (self.end_ns - self.start_ns) / NS_PER_S
        return row


def upsert_csv(
    path: str | Path,
    rows: list[dict],
    key: str = "day",
    replace_keys: set[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Replace the rows of `path` whose `key` is in `replace_keys` (default: the new rows' keys).

    `columns` gives the header to write when there are no rows at all.
    """
    path = Path(path)
    new = pd.DataFrame(rows, columns=columns if not rows else None)
    keys = replace_keys if replace_keys is not None else set(new[key]) if len(new) else set()
    old = None
    if path.exists():
        try:
            old = pd.read_csv(path, dtype={key: str})
        except pd.errors.EmptyDataError:
            old = None
    if old is not None:
        old = old[~old[key].isin(keys)]
        new = pd.concat([old, new], ignore_index=True) if len(new) else old
    if len(new):
        new = new.sort_values(key, kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def summarize(quality: pd.DataFrame) -> dict:
    """Dataset-level summary of a quality_report.csv (Table 0 inputs)."""
    q = quality
    status_counts = q["status"].value_counts().reindex(["ok", "flagged", "excluded"], fill_value=0)
    served = q[q["status"].isin(["ok", "flagged"])]
    return {
        "days": int(len(q)),
        "ok": int(status_counts["ok"]),
        "flagged": int(status_counts["flagged"]),
        "excluded": int(status_counts["excluded"]),
        "first_day": q["day"].min() if len(q) else None,
        "last_day": q["day"].max() if len(q) else None,
        "seq_gaps_total": int(q["n_seq_gaps"].sum()),
        "gap_s_total": float(q["gap_s"].sum()),
        "gap_fraction_max": float(q["gap_fraction"].max()) if len(q) else 0.0,
        "snapshot_checks": int(q["n_snapshot_checks"].sum()),
        "snapshot_mismatches": int(q["n_snapshot_mismatches"].sum()),
        "rows_removed_total": int(q["n_rows_removed"].sum()),
        "rows_per_day_median": float(served["n_rows_written"].median()) if len(served) else 0.0,
        "rows_per_day_min": int(served["n_rows_written"].min()) if len(served) else 0,
        "rows_per_day_max": int(served["n_rows_written"].max()) if len(served) else 0,
    }
