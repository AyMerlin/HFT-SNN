"""Bybit and Tardis parsers (plan §11): message parsing and equivalence on one synthetic stream."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from retrieval.book import OrderBookBuilder
from retrieval.bybit import BybitFormatError, BybitMessageParser
from retrieval.tardis import COLUMNS, TardisL2Parser

SYM = "BTCUSDT"
MS = 1_000_000


def bybit_line(kind, ts_ms, bids, asks, u):
    data = {"s": SYM, "b": [[f"{p:.1f}", f"{q:.3f}"] for p, q in bids], "a": [[f"{p:.1f}", f"{q:.3f}"] for p, q in asks],
            "u": u, "seq": 1000 + u}
    return json.dumps({"topic": f"orderbook.200.{SYM}", "type": kind, "ts": ts_ms, "data": data, "cts": ts_ms - 1})


def tardis_rows(kind, ts_ms, bids, asks):
    snap = "true" if kind == "snapshot" else "false"
    rows = []
    for side, levels in (("bid", bids), ("ask", asks)):
        for p, q in levels:
            rows.append(f"binance-futures,{SYM},{ts_ms * 1000},{ts_ms * 1000 + 7},{snap},{side},{p:.1f},{q:.3f}")
    return rows


def synthetic_stream(n_deltas=300, seed=0):
    """(kind, ts_ms, bids, asks) messages: a snapshot, then random inserts/updates/deletes."""
    rng = np.random.default_rng(seed)
    mid = 1000.0
    bids = {round(mid - 0.1 * i, 1): round(float(rng.uniform(0.1, 5)), 3) for i in range(1, 31)}
    asks = {round(mid + 0.1 * i, 1): round(float(rng.uniform(0.1, 5)), 3) for i in range(0, 30)}
    msgs = [("snapshot", 1_000, sorted(bids.items(), reverse=True), sorted(asks.items()))]
    for k in range(n_deltas):
        db, da = [], []
        for book, out, sign in ((bids, db, -1), (asks, da, 1)):
            for _ in range(int(rng.integers(1, 4))):
                best = max(bids) if sign < 0 else min(asks)
                p = round(best + sign * 0.1 * int(rng.integers(0, 25)), 1)
                if sign < 0 and p >= min(asks) or sign > 0 and p <= max(bids):
                    continue  # never cross the book
                q = 0.0 if (p in book and rng.random() < 0.3 and len(book) > 25) else round(float(rng.uniform(0.1, 5)), 3)
                if q == 0.0:
                    book.pop(p)
                else:
                    book[p] = q
                out.append((p, q))
        if not db and not da:
            db = [(max(bids), bids[max(bids)])]
        msgs.append(("delta", 1_000 + 100 * (k + 1), db, da))
    return msgs


def test_bybit_parser_fields():
    p = BybitMessageParser(SYM)
    m = p.parse(bybit_line("snapshot", 5, [(99.0, 1.0)], [(101.0, 2.0)], u=40))
    assert (m.ts_ns, m.is_snapshot, m.seq, m.prev_seq, m.recv_ts_ns) == (5 * MS, True, 40, None, None)
    assert m.bids == [(99.0, 1.0)] and m.asks == [(101.0, 2.0)]
    d = p.parse(bybit_line("delta", 6, [(99.0, 0.0)], [], u=41))
    assert (d.is_snapshot, d.seq, d.prev_seq, d.bids) == (False, 41, 40, [(99.0, 0.0)])


def test_bybit_parser_rejects_wrong_symbol_and_type():
    p = BybitMessageParser("ETHUSDT")
    with pytest.raises(BybitFormatError):
        p.parse(bybit_line("snapshot", 1, [], [], u=1))
    bad = json.loads(bybit_line("snapshot", 1, [], [], u=1))
    bad["type"] = "trade"
    with pytest.raises(BybitFormatError):
        BybitMessageParser(SYM).parse(json.dumps(bad))


def test_bybit_iter_file_reads_zip(tmp_path):
    lines = [bybit_line("snapshot", 1, [(99.0, 1.0)], [(101.0, 1.0)], 1), bybit_line("delta", 2, [(98.0, 2.0)], [], 2)]
    path = tmp_path / f"2026-01-01_{SYM}_ob200.data.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"2026-01-01_{SYM}_ob200.data", "\n".join(lines) + "\n")
    msgs = list(BybitMessageParser(SYM).iter_file(path))
    assert [m.seq for m in msgs] == [1, 2]


def test_tardis_groups_rows_into_messages_and_resnapshots():
    rows = (
        ["exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"]
        + tardis_rows("snapshot", 1, [(99.0, 1.0), (98.0, 1.0)], [(101.0, 1.0)])
        + tardis_rows("delta", 2, [(99.0, 2.0)], [(101.0, 0.0)])
        + tardis_rows("delta", 3, [(97.0, 1.0)], [])
        + tardis_rows("snapshot", 4, [(90.0, 1.0)], [(110.0, 1.0)])
    )
    msgs = list(TardisL2Parser(SYM).iter_text(io.StringIO("\n".join(rows) + "\n")))
    assert [m.is_snapshot for m in msgs] == [True, False, False, True]
    assert msgs[0].bids == [(99.0, 1.0), (98.0, 1.0)] and msgs[1].asks == [(101.0, 0.0)]
    assert (msgs[1].ts_ns, msgs[1].recv_ts_ns, msgs[1].seq) == (2 * MS, 2 * MS + 7_000, None)


def test_tardis_rejects_unexpected_columns():
    with pytest.raises(ValueError):
        list(TardisL2Parser(SYM).iter_text(io.StringIO("a,b\n1,2\n")))


def test_bybit_and_tardis_parsers_build_identical_books():
    stream = synthetic_stream()
    by_lines = [bybit_line(k, t, b, a, u=i + 1) for i, (k, t, b, a) in enumerate(stream)]
    ta_text = "\n".join([",".join(COLUMNS)] + [r for k, t, b, a in stream for r in tardis_rows(k, t, b, a)]) + "\n"
    by_msgs = [BybitMessageParser(SYM).parse(line) for line in by_lines]
    ta_msgs = list(TardisL2Parser(SYM).iter_text(io.StringIO(ta_text)))
    assert len(by_msgs) == len(ta_msgs) == len(stream)
    b1, b2 = OrderBookBuilder(), OrderBookBuilder()
    for m1, m2 in zip(by_msgs, ta_msgs):
        assert m1.ts_ns == m2.ts_ns
        assert b1.apply(m1) and b2.apply(m2)
        assert b1.top(20) == b2.top(20)
    assert b1.stats.n_seq_gaps == 0
    assert len(b1.bids) > 20 and len(b1.asks) > 20  # the stream exercises more than the output depth
