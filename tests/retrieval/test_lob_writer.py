"""Writer (plan §4.5, §11): files validate against the schema; manifest is complete."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from retrieval.book import BookMessage, OrderBookBuilder
from retrieval.standardize import DayStandardizer
from retrieval.writer import GAPS_NAME, StandardizedDatasetWriter
from snn_hft.data.lob.schema import (
    QUALITY_REPORT_NAME,
    LOBSchemaError,
    Manifest,
    day_bounds_ns,
    read_file_meta,
    read_frame,
)

S = 1_000_000_000


def day_result(day: date, n: int = 50, gap: bool = False):
    t0, t1 = day_bounds_ns(day)
    msgs = [BookMessage(t0, True, [(99.0, 1.0), (98.0, 1.0)], [(101.0, 1.0), (102.0, 1.0)], seq=0)]
    step = (t1 - t0) // n
    for i in range(1, n):
        prev = i - 1 if not (gap and i == 2) else -5
        msgs.append(BookMessage(t0 + i * step, False, [(99.0, float(i))], [], seq=i, prev_seq=prev))
    return DayStandardizer(depth=2, silence_gap_s=86_400).run(msgs, day, OrderBookBuilder(), raw_file=f"{day}.zip")


def writer(root):
    return StandardizedDatasetWriter(root, "bybit", "BTCUSDT", 2, "test-source", "tests", {"depth": 2})


def test_written_day_validates_and_manifest_is_complete(tmp_path):
    w = writer(tmp_path)
    d = date(2026, 7, 1)
    entry = w.write_day(day_result(d), raw_sha256="abc")
    assert entry.status == "ok" and entry.file == "2026-07-01.parquet"
    df = read_frame(w.directory / entry.file, day=d)
    assert len(df) == 50 and df["recv_ts_ns"].isna().all()
    assert read_file_meta(w.directory / entry.file)["raw_sha256"] == "abc"

    m = Manifest.load(w.directory)
    assert (m.source, m.venue, m.symbol, m.depth) == ("test-source", "bybit", "BTCUSDT", 2)
    assert m.script == "tests" and m.script_args == {"depth": 2}
    assert m.created_utc and m.updated_utc and m.code_git_hash is not None
    assert m.quality_report == QUALITY_REPORT_NAME and (w.directory / QUALITY_REPORT_NAME).exists()
    assert m.days["2026-07-01"].raw_file == "2026-07-01.zip"
    assert m.to_dict()["date_range"] == ["2026-07-01", "2026-07-01"]


def test_excluded_day_has_no_file_and_is_not_served(tmp_path):
    w = writer(tmp_path)
    good, bad = date(2026, 7, 1), date(2026, 7, 2)
    w.write_day(day_result(good))
    res = day_result(bad, gap=True)
    assert res.report.status == "excluded"
    w.write_day(res)
    m = Manifest.load(w.directory)
    assert m.days[bad.isoformat()].file is None
    assert m.served_days() == [good]
    assert not (w.directory / "2026-07-02.parquet").exists()
    gaps = pd.read_csv(w.directory / GAPS_NAME)
    assert list(gaps["day"]) == ["2026-07-02"] and list(gaps["reason"]) == ["sequence"]


def test_rewriting_a_day_replaces_its_rows(tmp_path):
    w = writer(tmp_path)
    d = date(2026, 7, 1)
    w.write_day(day_result(d, gap=True))
    w.write_day(day_result(d))
    q = pd.read_csv(w.directory / QUALITY_REPORT_NAME)
    assert len(q) == 1 and q["status"].iloc[0] == "ok"
    assert len(pd.read_csv(w.directory / GAPS_NAME)) == 0
    assert (w.directory / "2026-07-01.parquet").exists()


def test_writer_refuses_a_different_dataset_in_the_same_directory(tmp_path):
    writer(tmp_path).write_day(day_result(date(2026, 7, 1)))
    with pytest.raises(LOBSchemaError):
        StandardizedDatasetWriter(tmp_path, "bybit", "BTCUSDT", 20, "test-source")
