"""Resumable file downloads shared by the venue downloaders."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

CHUNK = 1 << 20
SHA_SUFFIX = ".sha256"


class DownloadError(RuntimeError):
    pass


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def recorded_sha256(path: str | Path) -> str | None:
    """sha256 written next to a completed download, or None if the download is incomplete."""
    side = Path(str(path) + SHA_SUFFIX)
    return side.read_text().split()[0] if side.exists() and Path(path).exists() else None


def is_complete(path: str | Path) -> bool:
    return recorded_sha256(path) is not None


class HttpDownloader:
    """GET with retries and exponential backoff; files land atomically with a sha256 sidecar.

    A file counts as downloaded once its `.sha256` sidecar exists (the sidecar is written after
    the atomic rename), so interrupted downloads are redone and completed ones skipped.
    """

    def __init__(
        self,
        session: Any | None = None,
        retries: int = 4,
        backoff_s: float = 2.0,
        timeout_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ):
        self.session = session or requests.Session()
        self.retries = retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self.sleep = sleep
        self.log = log

    def _with_retries(self, what: str, fn: Callable[[], Any]) -> Any:
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except (requests.RequestException, OSError, DownloadError) as exc:
                if attempt == self.retries:
                    raise DownloadError(f"{what}: {exc}") from exc
                wait = self.backoff_s * 2**attempt
                self.log(f"{what}: {exc}; retrying in {wait:.0f}s")
                self.sleep(wait)
        raise AssertionError("unreachable")

    def get_text(self, url: str) -> str:
        def fetch() -> str:
            resp = self.session.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
            return resp.text

        return self._with_retries(url, fetch)

    def fetch(self, url: str, path: str | Path, validate: Callable[[Path], None] | None = None) -> tuple[Path, bool]:
        """Download `url` to `path` unless already complete; return (path, downloaded_now)."""
        path = Path(path)
        if is_complete(path):
            return path, False
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")

        def fetch_once() -> None:
            resp = self.session.get(url, stream=True, timeout=self.timeout_s)
            try:
                resp.raise_for_status()
                expected = resp.headers.get("Content-Length")
                n = 0
                with open(part, "wb") as fh:
                    for block in resp.iter_content(CHUNK):
                        fh.write(block)
                        n += len(block)
                if expected is not None and int(expected) != n:
                    raise DownloadError(f"truncated: {n} of {expected} bytes")
                if validate is not None:
                    validate(part)
            finally:
                resp.close()

        try:
            self._with_retries(url, fetch_once)
            os.replace(part, path)
        finally:
            if part.exists():
                part.unlink()
        Path(str(path) + SHA_SUFFIX).write_text(f"{sha256_of(path)}  {path.name}\n")
        return path, True
