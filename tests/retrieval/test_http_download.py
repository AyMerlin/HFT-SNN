"""Resumable downloads with mocked HTTP (plan §11): skip completed files, retry, validate."""

from __future__ import annotations

import requests
import pytest

from retrieval.http import DownloadError, HttpDownloader, is_complete, sha256_of

URL = "https://example.test/data.zip"


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, length: int | None = None):
        self.body, self.status_code = body, status
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    @property
    def text(self) -> str:
        return self.body.decode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk):
        for i in range(0, len(self.body), chunk):
            yield self.body[i : i + chunk]

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)  # served in order; the last one repeats
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def http(*responses):
    session = FakeSession(responses)
    return HttpDownloader(session=session, sleep=lambda s: None, log=lambda m: None), session


def test_download_writes_file_and_sidecar_then_skips(tmp_path):
    dl, session = http(FakeResponse(b"payload"))
    path, fresh = dl.fetch(URL, tmp_path / "d" / "data.zip")
    assert fresh and path.read_bytes() == b"payload" and is_complete(path)
    assert path.with_name("data.zip.sha256").read_text().split()[0] == sha256_of(path)
    assert dl.fetch(URL, path) == (path, False)
    assert session.calls == 1  # the second call made no request


def test_incomplete_download_is_redone(tmp_path):
    target = tmp_path / "data.zip"
    target.write_bytes(b"partial")  # no sidecar: an interrupted earlier run
    dl, _ = http(FakeResponse(b"payload"))
    path, fresh = dl.fetch(URL, target)
    assert fresh and path.read_bytes() == b"payload"


def test_truncated_download_is_retried(tmp_path):
    dl, session = http(FakeResponse(b"pay", length=7), FakeResponse(b"payload"))
    path, _ = dl.fetch(URL, tmp_path / "data.zip")
    assert path.read_bytes() == b"payload" and session.calls == 2


def test_failed_validation_retries_then_fails_without_leaving_files(tmp_path):
    def validate(p):
        raise DownloadError("bad content")

    dl, session = http(FakeResponse(b"payload"))
    with pytest.raises(DownloadError):
        dl.fetch(URL, tmp_path / "data.zip", validate=validate)
    assert session.calls == dl.retries + 1
    assert not list(tmp_path.iterdir())


def test_get_text_raises_after_retries_on_http_errors():
    dl, session = http(FakeResponse(b"", status=503))
    with pytest.raises(DownloadError):
        dl.get_text(URL)
    assert session.calls == dl.retries + 1
