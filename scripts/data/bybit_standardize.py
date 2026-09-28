"""Reconstruct, quality-check and write standardized Bybit order-book days.

    python -m scripts.data.bybit_standardize --symbol BTCUSDT --start 2026-06-29 --end 2026-09-26 --workers 4

Reads `<raw-root>/bybit/<symbol>/*.zip` and writes `<out-root>/bybit/<symbol>/<day>.parquet`,
`manifest.json`, `quality_report.csv` and `gaps.csv`.

- `--workers 1` processes days in order with one book, so a day file that does not start
  with a snapshot continues from the previous day's book.
- `--workers N` processes days independently; each day then needs its own opening
  snapshot (Bybit files have one). Frames are identical; the quality report differs only
  in the opening snapshot check and the ~0.1 s before it.
- A day without a downloaded file is recorded as `excluded` and resets the book.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from retrieval.book import OrderBookBuilder
from retrieval.bybit import SOURCE, VENUE, BybitMessageParser, BybitOrderBookDownloader
from retrieval.http import recorded_sha256
from retrieval.quality import DayQualityReport
from retrieval.standardize import DEFAULT_DEPTH, DEFAULT_SILENCE_GAP_S, DayResult, DayStandardizer
from retrieval.writer import StandardizedDatasetWriter, write_day_file
from scripts.data._common import args_dict, day_range, iso_date
from snn_hft.data.lob.schema import frame_from_arrays


def _empty_frame(depth: int):
    z = np.empty((0, depth))
    return frame_from_arrays(np.empty(0, np.int64), None, z, z, z, z)


def _missing(day: date, depth: int) -> DayResult:
    rep = DayQualityReport(day=day, notes=["no raw file"], silence_s=86_400.0)  # no data all day
    rep.decide_status()
    return DayResult(_empty_frame(depth), rep, [])


def _standardize(day: date, raw: Path, symbol: str, depth: int, silence: float, builder: OrderBookBuilder) -> DayResult:
    parser = BybitMessageParser(symbol)
    return DayStandardizer(depth, silence).run(parser.iter_file(raw), day, builder, raw_file=raw.name)


def _worker(job: tuple) -> tuple[DayResult, str | None, str | None, float]:
    """Standardize and write one day in a worker process; returns the result without its frame."""
    day, raw, symbol, depth, silence, directory, meta = job
    t0 = time.time()
    sha = recorded_sha256(raw)
    result = _standardize(day, raw, symbol, depth, silence, OrderBookBuilder())
    file = write_day_file(directory, depth, result, meta, sha)
    return replace(result, frame=_empty_frame(depth)), file, sha, time.time() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", type=iso_date, required=True)
    ap.add_argument("--end", type=iso_date, required=True)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--raw-root", default="data/raw/lob")
    ap.add_argument("--out-root", default="data/standardized/lob")
    ap.add_argument("--silence-gap-s", type=float, default=DEFAULT_SILENCE_GAP_S)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true", help="skip days already in the manifest")
    ap.add_argument("--delete-raw", action="store_true", help="delete each raw zip after its day is written")
    args = ap.parse_args(argv)

    days = day_range(args.start, args.end)
    raw_dl = BybitOrderBookDownloader(args.raw_root, args.symbol)
    writer = StandardizedDatasetWriter(
        args.out_root, VENUE, args.symbol, args.depth, SOURCE, "scripts.data.bybit_standardize", args_dict(args)
    )
    if args.skip_existing:
        days = [d for d in days if d.isoformat() not in writer.manifest.days]
    raws = {d: raw_dl.local_path(d) for d in days}

    def done(result: DayResult, file: str | None, sha: str | None, secs: float) -> None:
        entry = writer.record_day(result, file, sha)
        rep = result.report
        print(f"{rep.day} {entry.status:8s} rows={rep.n_rows_written:>8,d} gaps={rep.n_seq_gaps} "
              f"gap={100 * rep.gap_fraction:.2f}% snapshot_checks={rep.n_snapshot_checks}"
              f"/{rep.n_snapshot_mismatches} mismatched  {secs:.0f}s", flush=True)
        raw = raws.get(rep.day)
        if args.delete_raw and file and raw is not None:
            raw.unlink()

    if args.workers <= 1:
        builder = OrderBookBuilder()
        prev_day = None
        for d in days:
            raw = raws[d]
            t0 = time.time()
            if prev_day is None or (d - prev_day).days != 1:
                builder.reset()  # carry the book only from the previous calendar day
            prev_day = d
            if raw is None:
                builder.reset()
                result = _missing(d, args.depth)
                done(result, write_day_file(writer.directory, args.depth, result, writer.file_meta()), None, 0.0)
                continue
            sha = recorded_sha256(raw)
            result = _standardize(d, raw, args.symbol, args.depth, args.silence_gap_s, builder)
            file = write_day_file(writer.directory, args.depth, result, writer.file_meta(), sha)
            done(replace(result, frame=_empty_frame(args.depth)), file, sha, time.time() - t0)
    else:
        for d in days:
            if raws[d] is None:
                result = _missing(d, args.depth)
                done(result, write_day_file(writer.directory, args.depth, result, writer.file_meta()), None, 0.0)
        jobs = [
            (d, raws[d], args.symbol, args.depth, args.silence_gap_s, writer.directory, writer.file_meta())
            for d in days
            if raws[d] is not None
        ]
        with ProcessPoolExecutor(args.workers) as pool:
            for out in pool.map(_worker, jobs):
                done(*out)
    served = writer.manifest.served_days()
    print(f"manifest: {len(writer.manifest.days)} days, {len(served)} served (ok/flagged) -> {writer.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
