"""Tardis.dev normalized `incremental_book_L2` CSV files (plan §4.4).

Columns: exchange, symbol, timestamp, local_timestamp, is_snapshot, side, price, amount
(timestamps in microseconds). Each row is one price level; amount 0 deletes the level.
Consecutive snapshot rows form one snapshot, and a snapshot that follows updates resets the
book (Tardis inserts one after every reconnection). Rows sharing (timestamp,
local_timestamp) come from one exchange message. The files carry no update ids, so the
sequence check does not apply; gaps show up only as re-snapshots.

Only the parser is used at L0 (parser-equivalence tests). Download and standardization of
the Binance free days follow at L7.
"""

from __future__ import annotations

import csv
import gzip
import io
from collections.abc import Iterable, Iterator
from pathlib import Path

from retrieval.book import BookMessage

US_TO_NS = 1_000
COLUMNS = ["exchange", "symbol", "timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"]


class TardisFormatError(ValueError):
    pass


class TardisL2Parser:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def iter_rows(self, rows: Iterable[dict[str, str]]) -> Iterator[BookMessage]:
        key = None
        is_snap = False
        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []
        ts = recv = 0

        def flush() -> BookMessage:
            return BookMessage(ts_ns=ts, is_snapshot=is_snap, bids=bids, asks=asks, recv_ts_ns=recv)

        for row in rows:
            if row["symbol"] != self.symbol:
                raise TardisFormatError(f"symbol {row['symbol']!r} in a {self.symbol} file")
            snap = row["is_snapshot"] == "true"
            this_ts, this_recv = int(row["timestamp"]), int(row["local_timestamp"])
            new_key = ("snap",) if snap else (this_ts, this_recv)
            if key is None or new_key != key or snap != is_snap:
                if key is not None:
                    yield flush()
                key, is_snap, bids, asks = new_key, snap, [], []
                ts, recv = this_ts * US_TO_NS, this_recv * US_TO_NS
            level = (float(row["price"]), float(row["amount"]))
            side = row["side"]
            if side == "bid":
                bids.append(level)
            elif side == "ask":
                asks.append(level)
            else:
                raise TardisFormatError(f"unknown side {side!r}")
        if key is not None:
            yield flush()

    def iter_text(self, text: io.TextIOBase) -> Iterator[BookMessage]:
        reader = csv.DictReader(text)
        if reader.fieldnames != COLUMNS:
            raise TardisFormatError(f"unexpected columns {reader.fieldnames}")
        yield from self.iter_rows(reader)

    def iter_file(self, path: str | Path) -> Iterator[BookMessage]:
        with gzip.open(path, "rt", newline="") as fh:
            yield from self.iter_text(fh)
