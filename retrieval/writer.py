"""Writes standardized day files, the manifest and the quality report (plan §4.2, §4.5)."""

from __future__ import annotations

import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from retrieval.quality import GAP_COLUMNS, upsert_csv
from retrieval.standardize import DayResult
from snn_hft.data.lob.schema import (
    QUALITY_REPORT_NAME,
    DayEntry,
    LOBSchemaError,
    Manifest,
    dataset_dir,
    day_file_name,
    depth_of,
    validate_frame,
    write_frame,
)

GAPS_NAME = "gaps.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]


def git_state() -> tuple[str | None, bool | None]:
    """(commit, dirty) of the retrieval code's repository, or (None, None) outside git."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return commit, None if status is None else bool(status)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StandardizedDatasetWriter:
    """One standardized dataset: `<root>/<venue>/<symbol>/` with day files and a manifest.

    Days with status `excluded` get a manifest entry and a quality row but no file.
    """

    def __init__(
        self,
        root: str | Path,
        venue: str,
        symbol: str,
        depth: int,
        source: str,
        script: str | None = None,
        script_args: dict[str, Any] | None = None,
    ):
        self.directory = dataset_dir(root, venue, symbol)
        commit, dirty = git_state()
        try:
            self.manifest = Manifest.load(self.directory)
        except LOBSchemaError:
            if (self.directory / "manifest.json").exists():
                raise
            self.manifest = Manifest(source=source, venue=venue, symbol=symbol, depth=depth, created_utc=_now())
        m = self.manifest
        if (m.source, m.venue, m.symbol, m.depth) != (source, venue, symbol, depth):
            raise LOBSchemaError(
                f"{self.directory} holds {m.source}/{m.venue}/{m.symbol} depth {m.depth}; "
                f"refusing to add {source}/{venue}/{symbol} depth {depth}"
            )
        m.code_git_hash, m.code_git_dirty = commit, dirty
        m.script = script
        m.script_args = dict(script_args or {})

    def file_meta(self) -> dict[str, Any]:
        """Dataset-level parquet metadata; `write_day_file` adds the per-day fields."""
        m = self.manifest
        return {"venue": m.venue, "symbol": m.symbol, "source": m.source, "code_git_hash": m.code_git_hash}

    def write_day(self, result: DayResult, raw_sha256: str | None = None) -> DayEntry:
        """Write the day file (unless excluded) and record the day (single-process use)."""
        file = write_day_file(self.directory, self.manifest.depth, result, self.file_meta(), raw_sha256)
        return self.record_day(result, file, raw_sha256)

    def record_day(self, result: DayResult, file: str | None, raw_sha256: str | None = None) -> DayEntry:
        """Update manifest, quality report and gap list for a day whose file is written."""
        rep = result.report
        day = rep.day.isoformat()
        entry = DayEntry(
            status=rep.status,
            file=file,
            n_rows=rep.n_rows_written if file else 0,
            gap_fraction=rep.gap_fraction,
            raw_file=rep.raw_file,
            raw_sha256=raw_sha256,
            notes=list(rep.notes),
        )
        self.manifest.days[day] = entry
        upsert_csv(self.directory / QUALITY_REPORT_NAME, [rep.to_row()])
        upsert_csv(
            self.directory / GAPS_NAME, [g.to_row() for g in result.gaps], replace_keys={day}, columns=GAP_COLUMNS
        )
        self.save()
        return entry

    def save(self) -> Path:
        self.manifest.updated_utc = _now()
        return self.manifest.save(self.directory)


def write_day_file(
    directory: Path, depth: int, result: DayResult, meta: dict[str, Any], raw_sha256: str | None = None
) -> str | None:
    """Validate and write one day's LOBFrame; remove a stale file for an excluded day.

    Safe to call from worker processes (touches only the day's own file).
    """
    rep, frame = result.report, result.frame
    day: date = rep.day
    path = directory / day_file_name(day)
    if rep.status == "excluded":
        if path.exists():
            path.unlink()
        return None
    if depth_of(list(frame.columns)) != depth:
        raise LOBSchemaError(f"{day}: frame depth differs from the dataset depth {depth}")
    validate_frame(frame, depth, day)
    write_frame(frame, path, {**meta, "day": day.isoformat(), "raw_file": rep.raw_file, "raw_sha256": raw_sha256})
    return path.name
