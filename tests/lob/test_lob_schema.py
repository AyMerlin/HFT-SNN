"""Standardized LOB format (plan §4.5): columns, validation, file round trip, manifest."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from snn_hft.data.lob.schema import (
    DayEntry,
    LOBSchemaError,
    Manifest,
    day_bounds_ns,
    frame_from_arrays,
    lob_columns,
    read_frame,
    validate_frame,
    write_frame,
)

DAY = date(2026, 7, 1)


def frame(n=4, depth=2, recv=None):
    t0 = day_bounds_ns(DAY)[0]
    ap = np.tile([101.0 + i for i in range(depth)], (n, 1))
    bp = np.tile([99.0 - i for i in range(depth)], (n, 1))
    q = np.ones((n, depth))
    return frame_from_arrays(t0 + np.arange(n) * 1000, recv, ap, q, bp, q)


def test_columns_in_plan_order():
    assert lob_columns(2) == [
        "ts_ns", "recv_ts_ns", "ask_px_1", "ask_px_2", "ask_qty_1", "ask_qty_2",
        "bid_px_1", "bid_px_2", "bid_qty_1", "bid_qty_2",
    ]


def test_valid_frame_passes_and_dtypes_follow_plan():
    df = frame()
    validate_frame(df, 2, DAY)
    assert df["ts_ns"].dtype == np.int64 and df["ask_px_1"].dtype == np.float64 and df["ask_qty_1"].dtype == np.float32


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.drop(columns="bid_qty_2"),
        lambda df: df.assign(ask_qty_1=df["ask_qty_1"].astype("float64")),
        lambda df: df.assign(ts_ns=df["ts_ns"][::-1].to_numpy()),
        lambda df: df.assign(ts_ns=df["ts_ns"] - 10**12),  # before the day
        lambda df: df.assign(ask_px_1=df["bid_px_1"]),  # locked
        lambda df: df.assign(ask_px_2=df["ask_px_1"]),  # levels not strictly ordered
        lambda df: df.assign(bid_qty_1=np.float32(np.nan)),
        lambda df: df.assign(bid_qty_1=np.float32(0.0)),
    ],
)
def test_invalid_frames_are_rejected(mutate):
    with pytest.raises(LOBSchemaError):
        validate_frame(mutate(frame()), 2, DAY)


def test_file_round_trip_keeps_dtypes_and_nulls(tmp_path):
    for recv in (None, np.arange(4) + 5):
        df = frame(recv=recv)
        path = write_frame(df, tmp_path / "d.parquet", {"venue": "x"})
        back = read_frame(path, day=DAY)
        pd.testing.assert_frame_equal(back, df)
    assert oct(path.stat().st_mode & 0o777) == "0o644"


def test_manifest_round_trip_and_served_days(tmp_path):
    m = Manifest(source="s", venue="v", symbol="X", depth=20)
    m.days["2026-07-02"] = DayEntry("excluded", None, 0, 0.2)
    m.days["2026-07-01"] = DayEntry("flagged", "2026-07-01.parquet", 10, 0.03)
    m.days["2026-07-03"] = DayEntry("ok", "2026-07-03.parquet", 10, 0.0)
    m.save(tmp_path)
    back = Manifest.load(tmp_path)
    assert back.to_dict() == m.to_dict()
    assert back.served_days() == [date(2026, 7, 1), date(2026, 7, 3)]
    assert back.date_range == (date(2026, 7, 1), date(2026, 7, 3))


def test_manifest_rejects_unknown_status_and_schema_version(tmp_path):
    with pytest.raises(LOBSchemaError):
        DayEntry("bad", None, 0, 0.0)
    m = Manifest(source="s", venue="v", symbol="X", depth=20).to_dict()
    m["schema_version"] = 99
    with pytest.raises(LOBSchemaError):
        Manifest.from_dict(m)
