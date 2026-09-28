"""Bybit historical order-book files (plan §4.1, §4.2; format in docs/DATA_RETRIEVAL.md).

Files are served from a public directory listing, one zip per UTC day, e.g.
`https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/2026-09-26_BTCUSDT_ob200.data.zip`.
Each zip holds one JSON-lines file: a snapshot at 00:00 UTC, deltas about every 100 ms, and a
closing snapshot stamped just after the next midnight. Deltas carry an update id `u` that
increases by one per message. The depth changed from 500 to 200 levels on 2025-08-21.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

from retrieval.book import BookMessage
from retrieval.http import DownloadError, HttpDownloader

try:  # optional speed-up; the standard library parser gives identical results
    import orjson

    _loads: Callable[[bytes], Any] = orjson.loads
except ImportError:  # pragma: no cover - depends on the environment
    _loads = json.loads

BASE_URL = "https://quote-saver.bycsi.com/orderbook"
VENUE = "bybit"
SOURCE = "bybit-history-data"
MS_TO_NS = 1_000_000

_FILE_RE = r'href="(?P<name>(?P<day>\d{{4}}-\d{{2}}-\d{{2}})_{symbol}_ob(?P<depth>\d+)\.data\.zip)"'


class BybitFormatError(ValueError):
    pass


class BybitMessageParser:
    """One JSON line of a Bybit order-book file -> `BookMessage`.

    `ts` (system time the message was generated, ms) is the exchange timestamp; the
    matching-engine time `cts` is not kept. Deltas must follow the previous message's `u`
    by exactly one; a snapshot starts a new sequence.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

    def parse(self, line: bytes | str) -> BookMessage:
        m = _loads(line)
        kind = m.get("type")
        data = m.get("data") or {}
        if kind not in ("snapshot", "delta"):
            raise BybitFormatError(f"unknown message type {kind!r}")
        if data.get("s") != self.symbol:
            raise BybitFormatError(f"symbol {data.get('s')!r} in a {self.symbol} file")
        u = int(data["u"])
        is_snapshot = kind == "snapshot"
        return BookMessage(
            ts_ns=int(m["ts"]) * MS_TO_NS,
            is_snapshot=is_snapshot,
            bids=[(float(p), float(q)) for p, q in data.get("b", ())],
            asks=[(float(p), float(q)) for p, q in data.get("a", ())],
            recv_ts_ns=None,
            seq=u,
            prev_seq=None if is_snapshot else u - 1,
        )

    def iter_file(self, path: str | Path) -> Iterator[BookMessage]:
        """Stream the messages of a downloaded day zip."""
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".data")]
            if len(names) != 1:
                raise BybitFormatError(f"{path}: expected one .data member, found {zf.namelist()}")
            with io.BufferedReader(zf.open(names[0]), 1 << 20) as fh:
                for line in fh:
                    if line.strip():
                        yield self.parse(line)


class BybitOrderBookDownloader:
    """Lists and downloads Bybit daily order-book zips into `raw_root/bybit/<symbol>/`."""

    def __init__(
        self,
        raw_root: str | Path,
        symbol: str,
        category: str = "linear",
        base_url: str = BASE_URL,
        http: HttpDownloader | None = None,
    ):
        self.raw_root = Path(raw_root)
        self.symbol = symbol
        self.category = category
        self.base_url = base_url.rstrip("/")
        self.http = http or HttpDownloader()
        self._listing: dict[date, str] | None = None

    @property
    def listing_url(self) -> str:
        return f"{self.base_url}/{self.category}/{self.symbol}/"

    @property
    def directory(self) -> Path:
        return self.raw_root / VENUE / self.symbol

    def available(self, refresh: bool = False) -> dict[date, str]:
        """Day -> file name, parsed from the public directory listing."""
        if self._listing is None or refresh:
            html = self.http.get_text(self.listing_url)
            pattern = re.compile(_FILE_RE.format(symbol=re.escape(self.symbol)))
            self._listing = {date.fromisoformat(m["day"]): m["name"] for m in pattern.finditer(html)}
            if not self._listing:
                raise DownloadError(f"no {self.symbol} files in the listing at {self.listing_url}")
        return self._listing

    def local_path(self, day: date) -> Path | None:
        """The downloaded zip of a day, if complete (independent of the remote listing)."""
        for p in sorted(self.directory.glob(f"{day.isoformat()}_{self.symbol}_ob*.data.zip")):
            if Path(str(p) + ".sha256").exists():
                return p
        return None

    def download(self, day: date) -> tuple[Path, bool]:
        name = self.available().get(day)
        if name is None:
            raise DownloadError(f"{day} is not in the Bybit listing for {self.symbol}")
        return self.http.fetch(f"{self.listing_url}{name}", self.directory / name, validate=_check_zip)

    def download_range(self, days: list[date], log: Callable[[str], None] = print) -> list[Path]:
        missing = [d for d in days if d not in self.available()]
        if missing:
            raise DownloadError(f"not in the listing: {[d.isoformat() for d in missing]}")
        paths = []
        for d in days:
            path, fresh = self.download(d)
            log(f"{d} {'downloaded' if fresh else 'present'} {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
            paths.append(path)
        return paths


def depth_of_file(path: str | Path) -> int:
    m = re.search(r"_ob(\d+)\.data\.zip$", str(path))
    if m is None:
        raise BybitFormatError(f"cannot read the book depth from {path}")
    return int(m.group(1))


def _check_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            if not any(n.endswith(".data") for n in zf.namelist()):
                raise DownloadError(f"{path.name}: no .data member")
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"{path.name}: {exc}") from None
