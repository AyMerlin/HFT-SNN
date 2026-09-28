"""Bybit downloader with mocked HTTP (plan §11): listing, resumability, validation."""

from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest

from retrieval.bybit import BybitOrderBookDownloader, depth_of_file
from retrieval.http import DownloadError, HttpDownloader, is_complete, sha256_of

LISTING = """<html><body><ul>
<li><a href="2025-08-20_BTCUSDT_ob500.data.zip">2025-08-20_BTCUSDT_ob500.data.zip</a></li>
<li><a href="2025-08-21_BTCUSDT_ob200.data.zip">2025-08-21_BTCUSDT_ob200.data.zip</a></li>
<li><a href="2025-08-21_BTCUSDTX_ob200.data.zip">other symbol</a></li>
</ul></body></html>"""


def zip_bytes(name="x.data", text='{"a": 1}\n') -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, length: int | None = None):
        self.body, self.status_code = body, status
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    @property
    def text(self) -> str:
        return self.body.decode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk):
        for i in range(0, len(self.body), chunk):
            yield self.body[i : i + chunk]

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, routes):
        self.routes = routes  # url -> list of responses, served in order (last one repeats)
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        queue = self.routes[url]
        return queue.pop(0) if len(queue) > 1 else queue[0]


BASE = "https://example.test/orderbook/linear/BTCUSDT/"


def downloader(tmp_path, routes):
    session = FakeSession(routes)
    http = HttpDownloader(session=session, sleep=lambda s: None, log=lambda m: None)
    return BybitOrderBookDownloader(tmp_path, "BTCUSDT", base_url="https://example.test/orderbook", http=http), session


def test_listing_maps_days_to_files_for_the_symbol_only(tmp_path):
    dl, _ = downloader(tmp_path, {BASE: [FakeResponse(LISTING.encode())]})
    assert dl.available() == {
        date(2025, 8, 20): "2025-08-20_BTCUSDT_ob500.data.zip",
        date(2025, 8, 21): "2025-08-21_BTCUSDT_ob200.data.zip",
    }
    assert depth_of_file("2025-08-21_BTCUSDT_ob200.data.zip") == 200


def test_download_writes_file_and_sidecar_then_skips(tmp_path):
    body = zip_bytes()
    url = BASE + "2025-08-21_BTCUSDT_ob200.data.zip"
    dl, session = downloader(tmp_path, {BASE: [FakeResponse(LISTING.encode())], url: [FakeResponse(body)]})
    path, fresh = dl.download(date(2025, 8, 21))
    assert fresh and path.read_bytes() == body and is_complete(path)
    assert path.with_name(path.name + ".sha256").read_text().split()[0] == sha256_of(path)
    assert dl.local_path(date(2025, 8, 21)) == path
    n_calls = len(session.calls)
    assert dl.download(date(2025, 8, 21)) == (path, False)
    assert len(session.calls) == n_calls  # skipped without any request


def test_incomplete_download_is_redone(tmp_path):
    body = zip_bytes()
    url = BASE + "2025-08-21_BTCUSDT_ob200.data.zip"
    dl, _ = downloader(tmp_path, {BASE: [FakeResponse(LISTING.encode())], url: [FakeResponse(body)]})
    target = dl.directory / "2025-08-21_BTCUSDT_ob200.data.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"partial")  # no sidecar: an interrupted earlier run
    path, fresh = dl.download(date(2025, 8, 21))
    assert fresh and path.read_bytes() == body


def test_truncated_or_corrupt_downloads_are_retried_then_fail(tmp_path):
    good = zip_bytes()
    url = BASE + "2025-08-21_BTCUSDT_ob200.data.zip"
    truncated = FakeResponse(good[:10], length=len(good))
    dl, session = downloader(tmp_path, {BASE: [FakeResponse(LISTING.encode())], url: [truncated, FakeResponse(good)]})
    path, _ = dl.download(date(2025, 8, 21))
    assert path.read_bytes() == good and session.calls.count(url) == 2

    corrupt = FakeResponse(b"not a zip")
    dl2, _ = downloader(tmp_path / "b", {BASE: [FakeResponse(LISTING.encode())], url: [corrupt]})
    with pytest.raises(DownloadError):
        dl2.download(date(2025, 8, 21))
    assert not any((tmp_path / "b").rglob("*.zip*"))  # no partial or unverified file left


def test_days_missing_from_listing_raise(tmp_path):
    dl, _ = downloader(tmp_path, {BASE: [FakeResponse(LISTING.encode())]})
    with pytest.raises(DownloadError):
        dl.download_range([date(2025, 8, 21), date(2025, 8, 22)], log=lambda m: None)
