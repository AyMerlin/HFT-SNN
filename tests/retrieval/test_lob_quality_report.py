"""Quality report script: Table-0 summary and content statistics on a small dataset."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from retrieval.standardize import DayResult
from retrieval.quality import DayQualityReport
from retrieval.writer import StandardizedDatasetWriter
from scripts.data.lob_quality_report import day_content, report
from snn_hft.data.lob.schema import day_bounds_ns, frame_from_arrays

DAY = date(2026, 7, 1)
S = 1_000_000_000


def dataset(tmp_path):
    """One day, depth 10: 6 rows, one exact repeat of the top 10, mid moves twice."""
    t0 = day_bounds_ns(DAY)[0]
    bid1 = np.array([100.0, 100.0, 100.0, 100.5, 100.5, 101.0])
    qty1 = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0])  # row 2 repeats row 1
    ap = bid1[:, None] + 1.0 + np.arange(10)[None, :]
    bp = bid1[:, None] - np.arange(10)[None, :]
    aq = np.ones((6, 10))
    bq = np.ones((6, 10))
    bq[:, 0] = qty1
    frame = frame_from_arrays(t0 + np.arange(6) * S, None, ap, aq, bp, bq)
    rep = DayQualityReport(day=DAY, n_messages=6, n_rows_emitted=6, n_rows_written=6)
    rep.decide_status()
    w = StandardizedDatasetWriter(tmp_path, "bybit", "BTCUSDT", 10, "test")
    w.write_day(DayResult(frame, rep, []))
    return w.directory


def test_day_content_counts_top10_events_and_mid_moves(tmp_path):
    c = day_content(dataset(tmp_path) / "2026-07-01.parquet")
    assert (c["rows"], c["events"]) == (6, 5)
    assert c["mid_change_share"] == pytest.approx(2 / 4)
    assert c["s_per_event_median"] == pytest.approx(1.0)
    assert c["spread_one_tick_share"] == 1.0


def test_report_renders_status_table_and_content(tmp_path):
    text = report(dataset(tmp_path), content=True)
    assert "| 1 | 1 | 0 | 0 |" in text  # days / ok / flagged / excluded
    assert "## Content (served days)" in text and "k = 100" in text
