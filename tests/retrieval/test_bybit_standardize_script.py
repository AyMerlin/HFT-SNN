"""End-to-end run of scripts.data.bybit_standardize on synthetic Bybit day files."""

from __future__ import annotations

import json
import zipfile
from datetime import date

import pandas as pd

from retrieval.http import sha256_of
from scripts.data import bybit_standardize
from snn_hft.data.lob.schema import Manifest, day_bounds_ns, read_frame

SYM = "BTCUSDT"
DAYS = [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)]


def line(kind, ts_ms, u, bids, asks):
    data = {"s": SYM, "b": [[str(p), str(q)] for p, q in bids], "a": [[str(p), str(q)] for p, q in asks], "u": u, "seq": u}
    return json.dumps({"topic": "orderbook.200.BTCUSDT", "type": kind, "ts": ts_ms, "data": data, "cts": ts_ms})


def write_day(raw_dir, day, u0, bids, asks):
    """A Bybit-like day: opening snapshot, a delta every 30 s, closing snapshot after midnight.

    `bids`/`asks` carry the book from the previous day, as in real consecutive files.
    """
    t0 = day_bounds_ns(day)[0] // 1_000_000
    lines, u = [line("snapshot", t0 + 129, u0, bids.items(), asks.items())], u0
    for k in range(1, 2880):
        u += 1
        size = float(1 + k % 5)
        bids[99.0] = size
        lines.append(line("delta", t0 + 129 + 30_000 * k, u, [(99.0, size)], []))
    u += 1
    lines.append(line("snapshot", t0 + 86_400_000 + 129, u, bids.items(), asks.items()))
    name = f"{day.isoformat()}_{SYM}_ob200.data.zip"
    path = raw_dir / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name[:-4], "\n".join(lines) + "\n")
    (raw_dir / (name + ".sha256")).write_text(f"{sha256_of(path)}  {name}\n")
    return u


def make_raw(tmp_path, days):
    raw_dir = tmp_path / "raw" / "bybit" / SYM
    raw_dir.mkdir(parents=True)
    u = 1
    bids = {100.0 - i: 1.0 for i in range(1, 4)}
    asks = {100.0 + i: 1.0 for i in range(1, 4)}
    for d in days:
        u = write_day(raw_dir, d, u, bids, asks) + 1
    return tmp_path / "raw"


def run(raw_root, out_root, *extra):
    args = ["--start", DAYS[0].isoformat(), "--end", DAYS[-1].isoformat(), "--depth", "3",
            "--raw-root", str(raw_root), "--out-root", str(out_root), *extra]
    assert bybit_standardize.main(args) == 0
    return out_root / "bybit" / SYM


def test_sequential_and_parallel_runs_write_identical_days(tmp_path):
    raw = make_raw(tmp_path, DAYS)
    seq_dir = run(raw, tmp_path / "seq", "--workers", "1")
    par_dir = run(raw, tmp_path / "par", "--workers", "2")
    m_seq, m_par = Manifest.load(seq_dir), Manifest.load(par_dir)
    assert [e.status for e in m_seq.days.values()] == ["ok"] * 3 == [e.status for e in m_par.days.values()]
    for d in DAYS:
        a, b = read_frame(seq_dir / f"{d}.parquet", day=d), read_frame(par_dir / f"{d}.parquet", day=d)
        pd.testing.assert_frame_equal(a, b)
        assert len(a) == 2880
    q = pd.read_csv(seq_dir / "quality_report.csv")
    # sequential: days 2 and 3 open with a valid book, so their opening snapshot is checked too
    assert list(q["n_snapshot_checks"]) == [1, 2, 2] and q["n_snapshot_mismatches"].sum() == 0
    assert m_seq.days["2026-07-01"].raw_sha256 is not None


def test_missing_raw_day_is_excluded_and_resets_the_book(tmp_path):
    raw = make_raw(tmp_path, [DAYS[0], DAYS[2]])
    out = run(raw, tmp_path / "out", "--workers", "1")
    m = Manifest.load(out)
    assert m.days["2026-07-02"].status == "excluded" and "no raw file" in m.days["2026-07-02"].notes
    assert m.days["2026-07-02"].gap_fraction == 1.0
    assert m.served_days() == [DAYS[0], DAYS[2]]
    q = pd.read_csv(out / "quality_report.csv").set_index("day")
    assert q.loc["2026-07-03", "n_snapshot_mismatches"] == 0


def test_skip_existing_does_not_carry_the_book_over_a_skipped_day(tmp_path):
    raw = make_raw(tmp_path, DAYS)
    out = run(raw, tmp_path / "out", "--workers", "1")
    m = Manifest.load(out)
    del m.days["2026-07-01"], m.days["2026-07-03"]  # pretend days 1 and 3 are missing
    m.save(out)
    run(raw, tmp_path / "out", "--workers", "1", "--skip-existing")
    q = pd.read_csv(out / "quality_report.csv").set_index("day")
    assert q.loc["2026-07-03", "n_snapshot_mismatches"] == 0
    assert q.loc["2026-07-03", "n_snapshot_checks"] == 1  # fresh book: only the closing check
    assert set(Manifest.load(out).days) == {d.isoformat() for d in DAYS}


def test_delete_raw_removes_zips_after_writing(tmp_path):
    raw = make_raw(tmp_path, DAYS[:1])
    bybit_standardize.main(["--start", "2026-07-01", "--end", "2026-07-01", "--depth", "3", "--raw-root", str(raw),
                            "--out-root", str(tmp_path / "o"), "--delete-raw"])
    assert not list((raw / "bybit" / SYM).glob("*.zip"))
    assert (tmp_path / "o" / "bybit" / SYM / "2026-07-01.parquet").exists()
