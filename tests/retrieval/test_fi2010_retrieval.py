"""FI-2010 retrieval: conversion of an archive with FI-2010's structure, and download with mocked HTTP."""

from __future__ import annotations

import numpy as np
import pytest

import snn_hft.data.fi2010.format as fmt
from retrieval.fi2010 import (
    ARCHIVE_PATH,
    ETSIN_AUTHORIZE_URL,
    METAX_FILES_URL,
    FI2010Converter,
    FI2010Downloader,
    stock_boundaries,
)
from retrieval.http import DownloadError, HttpDownloader, sha256_of


def test_conversion_reproduces_every_file_with_its_segments(synthetic_fi2010, converted_fi2010):
    out = fmt.dataset_dir(converted_fi2010)
    m = fmt.Manifest.load(out)
    assert set(m.files) == {fmt.stem(k, i) for k in ("train", "test") for i in range(1, 10)}
    assert all(m.checks["labels_match_day_files"].values())
    for fold in range(1, 10):
        for kind, source in (("train", synthetic_fi2010.train), ("test", synthetic_fi2010.test)):
            name = fmt.stem(kind, fold)
            arrays = fmt.read_arrays(out, name)
            expected = fmt.split_matrix(source[fold])
            for key in fmt.ARRAYS:
                np.testing.assert_array_equal(arrays[key], expected[key])
            entry = m.files[name]
            fmt.validate_arrays(arrays, entry)
            assert entry.days == fmt.days_of(kind, fold)
    # train_cf3: stock 1 days 1-3, stock 2 days 1-3, ...
    L = synthetic_fi2010.lengths
    segs = m.files["train_cf3"].segments
    assert [(s.stock, s.day) for s in segs[:4]] == [(1, 1), (1, 2), (1, 3), (2, 1)]
    assert [s.n for s in segs[:4]] == [L[0, 0], L[1, 0], L[2, 0], L[0, 1]]
    counts = m.files["test_cf1"].label_counts["50"]
    labels = synthetic_fi2010.test[1][144 + 3]
    assert counts == [int((labels == v).sum()) for v in (1, 2, 3)]


def test_conversion_rejects_files_not_ordered_stock_by_stock(tmp_path, fi2010_builder):
    bad = fi2010_builder(tmp_path / "bad.zip", seed=1, stock_major=False)
    with pytest.raises(fmt.FI2010FormatError, match="train_cf2"):
        FI2010Converter(bad.archive, log=lambda m: None).convert(tmp_path / "out", archive_sha256="x")
    assert not (fmt.dataset_dir(tmp_path / "out") / fmt.MANIFEST_NAME).exists()


def test_stock_boundaries_found_from_price_jumps(synthetic_fi2010):
    bounds, separation = stock_boundaries(synthetic_fi2010.test[4])
    assert list(np.diff(bounds)) == list(synthetic_fi2010.lengths[4])
    assert separation > 10


class Resp:
    def __init__(self, body=b"", json_obj=None, status=200):
        self.body, self._json, self.status_code = body, json_obj, status
        self.headers = {"Content-Length": str(len(body))}
        self.text = body.decode(errors="replace")

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk):
        yield self.body

    def close(self):
        pass


class Session:
    def __init__(self, payload: bytes, published_sha: str):
        self.payload, self.sha = payload, published_sha
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if url == METAX_FILES_URL:
            return Resp(json_obj=[{"pathname": ARCHIVE_PATH, "checksum": f"sha256:{self.sha}"}])
        assert url == "https://download.test/file?token=t"
        return Resp(self.payload)

    def post(self, url, json=None, **kw):
        self.calls.append(url)
        assert url == ETSIN_AUTHORIZE_URL and json["file"] == ARCHIVE_PATH
        return Resp(json_obj={"url": "https://download.test/file?token=t"})


def downloader(tmp_path, payload, sha):
    session = Session(payload, sha)
    http = HttpDownloader(session=session, sleep=lambda s: None, log=lambda m: None)
    return FI2010Downloader(tmp_path, http=http, log=lambda m: None), session


def test_download_verifies_published_checksum_and_is_skipped_when_present(tmp_path):
    payload = b"archive bytes"
    (tmp_path / "ref").write_bytes(payload)
    sha = sha256_of(tmp_path / "ref")
    dl, session = downloader(tmp_path, payload, sha)
    path = dl.download()
    assert path.read_bytes() == payload
    n = len(session.calls)
    assert dl.download() == path
    assert session.calls[n:] == [METAX_FILES_URL]  # only the checksum lookup


def test_download_with_wrong_checksum_fails(tmp_path):
    dl, _ = downloader(tmp_path, b"tampered", "0" * 64)
    with pytest.raises(DownloadError, match="differs from the published"):
        dl.download()
