"""FI-2010 benchmark dataset (Ntakaris et al., 2018): download from Fairdata and conversion.

The dataset is open (CC BY 4.0) and published as one archive,
`/published/BenchmarkDatasets/BenchmarkDatasets.zip` (1.86 GB), in the Fairdata dataset
73eb48d7-4dbc-4a10-a52a-da745b47a649. Metax publishes its sha256; Etsin issues a short-lived
download URL for it (`POST /api/download/authorize` with the dataset id and file path).
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from retrieval.http import DownloadError, HttpDownloader, recorded_sha256
from retrieval.provenance import git_state, utc_now
import snn_hft.data.fi2010.format as fmt

DATASET_ID = "73eb48d7-4dbc-4a10-a52a-da745b47a649"
ARCHIVE_PATH = "/published/BenchmarkDatasets/BenchmarkDatasets.zip"
ARCHIVE_NAME = "BenchmarkDatasets.zip"
METAX_FILES_URL = f"https://metax.fairdata.fi/v3/datasets/{DATASET_ID}/files?pagination=false"
ETSIN_AUTHORIZE_URL = "https://etsin.fairdata.fi/api/download/authorize"


class FI2010Downloader:
    """Downloads the FI-2010 archive into `raw_root/fi2010/` and verifies its sha256."""

    def __init__(self, raw_root: str | Path, http: HttpDownloader | None = None, log: Callable[[str], None] = print):
        self.directory = Path(raw_root) / "fi2010"
        self.http = http or HttpDownloader()
        self.log = log

    @property
    def archive(self) -> Path:
        return self.directory / ARCHIVE_NAME

    def published_sha256(self) -> str:
        """sha256 of the archive as published in Metax."""
        resp = self.http.session.get(METAX_FILES_URL, timeout=self.http.timeout_s)
        resp.raise_for_status()
        for f in resp.json():
            if f.get("pathname") == ARCHIVE_PATH:
                algo, _, value = f["checksum"].partition(":")
                if algo != "sha256":
                    raise DownloadError(f"unexpected checksum algorithm {algo!r}")
                return value
        raise DownloadError(f"{ARCHIVE_PATH} not listed in the Metax dataset {DATASET_ID}")

    def download_url(self) -> str:
        resp = self.http.session.post(
            ETSIN_AUTHORIZE_URL, json={"cr_id": DATASET_ID, "file": ARCHIVE_PATH}, timeout=self.http.timeout_s
        )
        resp.raise_for_status()
        url = resp.json().get("url")
        if not url:
            raise DownloadError(f"Etsin returned no download URL: {resp.text[:200]}")
        return url

    def download(self) -> Path:
        """Fetch the archive unless a verified copy exists; raise if the checksum differs."""
        expected = self.published_sha256()
        if recorded_sha256(self.archive) == expected:
            self.log(f"{self.archive} present and verified")
            return self.archive
        path, _ = self.http.fetch(self.download_url(), self.archive)
        got = recorded_sha256(path)
        if got != expected:
            raise DownloadError(f"{path.name}: sha256 {got} differs from the published {expected}")
        self.log(f"downloaded {path} ({path.stat().st_size / 1e9:.2f} GB), sha256 verified")
        return path


# --------------------------------------------------------------------------- conversion

MEMBER_RE = re.compile(
    r"^BenchmarkDatasets/(?P<auction>Auction|NoAuction)/\d\.(?P=auction)_(?P<norm>Zscore|MinMax|DecPre)/"
    r"(?P=auction)_(?P=norm)_(?:Training|Testing)/(?P<kind>Train|Test)_Dst_(?P=auction)_(?:ZScore|MinMax|DecPre)"
    r"_CF_(?P<fold>\d)\.txt$"
)
PRICE_ROWS = [4 * lvl + side for lvl in range(10) for side in (0, 2)]  # ask and bid prices, levels 1-10


def member_names(names: list[str], auction: str, normalization: str) -> dict[tuple[str, int], str]:
    """(kind, fold) -> archive member for one dataset variant."""
    out = {}
    for n in names:
        m = MEMBER_RE.match(n)
        if m and m["auction"] == auction and m["norm"] == normalization:
            out[("train" if m["kind"] == "Train" else "test", int(m["fold"]))] = n
    expected = {(k, i) for k in ("train", "test") for i in range(1, fmt.N_FOLDS + 1)}
    if set(out) != expected:
        raise ValueError(f"{auction}/{normalization}: missing members {sorted(expected - set(out))}")
    return out


def read_matrix(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """One FI-2010 text file -> (149, N) float64 matrix."""
    with io.TextIOWrapper(zf.open(member), encoding="ascii") as fh:
        rows = [np.array(line.split(), dtype=np.float64) for line in fh if line.strip()]
    if len(rows) != fmt.N_ROWS or len({len(r) for r in rows}) != 1:
        raise fmt.FI2010FormatError(f"{member}: {len(rows)} rows of lengths {sorted({len(r) for r in rows})[:3]}")
    return np.vstack(rows)


def stock_boundaries(day: np.ndarray) -> tuple[np.ndarray, float]:
    """Boundaries [0, b1, …, b4, N] of the five stocks in a single-day matrix.

    A change of stock shifts every price level at once, so the four largest jumps of the
    summed absolute change of the 20 price rows mark the boundaries. Returns the boundaries
    and the ratio between the 4th and 5th largest jump (the separation of the decision).
    """
    score = np.abs(np.diff(day[PRICE_ROWS], axis=1)).sum(axis=0)
    order = np.argsort(score)[::-1]
    cuts = np.sort(order[: fmt.N_STOCKS - 1]) + 1
    separation = float(score[order[fmt.N_STOCKS - 2]] / max(score[order[fmt.N_STOCKS - 1]], 1e-12))
    return np.concatenate([[0], cuts, [day.shape[1]]]), separation


def label_counts(labels: np.ndarray) -> dict[str, list[int]]:
    return {str(k): [int((labels[:, j] == v).sum()) for v in (1, 2, 3)] for j, k in enumerate(fmt.HORIZONS)}


def max_affine_residual(x: np.ndarray, y: np.ndarray) -> float:
    """Largest residual of the per-row least-squares fit x_r ≈ a_r·y_r + b_r (rows = features)."""
    ym, xm = y.mean(axis=1, keepdims=True), x.mean(axis=1, keepdims=True)
    yc, xc = y - ym, x - xm
    var = (yc**2).sum(axis=1, keepdims=True)
    a = np.where(var > 0, (xc * yc).sum(axis=1, keepdims=True) / np.where(var > 0, var, 1), 0.0)
    return float(np.abs(xc - a * yc).max())


class FI2010Converter:
    """Converts one variant (e.g. NoAuction/Zscore) of the archive into the format of
    `snn_hft.data.fi2010.format`, with every file's (stock, day) segments.

    Segments come from the single-day files (day 1 = `Train_CF_1`, day d = `Test_CF_{d-1}`),
    whose stock boundaries are found from price jumps. Every file is verified before its
    arrays are written: its labels must equal the stock-by-stock concatenation of the day
    files' labels, and its LOB rows must be an affine re-normalization of the day files' rows
    on every segment. The manifest is written only after all files passed, and the framework
    reads nothing without it.
    """

    AFFINE_TOL = 1e-4  # z-scores are printed with ~7 significant digits

    def __init__(self, archive: str | Path, auction: str = "NoAuction", normalization: str = "Zscore",
                 log: Callable[[str], None] = print):
        self.archive = Path(archive)
        self.auction, self.normalization = auction, normalization
        self.log = log

    def convert(self, out_root: str | Path, archive_sha256: str) -> fmt.Manifest:
        out = fmt.dataset_dir(out_root, self.auction, self.normalization)
        commit, dirty = git_state()
        manifest = fmt.Manifest(DATASET_ID, archive_sha256, self.auction, self.normalization,
                                code_git_hash=commit, code_git_dirty=dirty, created_utc=utc_now())
        with zipfile.ZipFile(self.archive) as zf:
            members = member_names(zf.namelist(), self.auction, self.normalization)
            days: dict[int, np.ndarray] = {1: read_matrix(zf, members[("train", 1)])}
            for d in range(2, fmt.N_DAYS + 1):
                days[d] = read_matrix(zf, members[("test", d - 1)])
            bounds, separations = {}, {}
            for d, m in days.items():
                bounds[d], separations[d] = stock_boundaries(m)
            manifest.checks = {"stock_boundary_separation_min": min(separations.values()),
                               "labels_match_day_files": {}, "lob_affine_max_residual": {}}
            for fold in range(1, fmt.N_FOLDS + 1):
                for kind in ("train", "test"):
                    self._convert_file(zf, members[(kind, fold)], kind, fold, days, bounds, out, manifest)
        manifest.save(out)
        return manifest

    def _convert_file(self, zf, member, kind, fold, days, bounds, out, manifest) -> None:
        name = fmt.stem(kind, fold)
        matrix = read_matrix(zf, member)
        file_days = fmt.days_of(kind, fold)
        segments, pieces = [], []
        pos = 0
        for s in range(fmt.N_STOCKS):  # files are ordered stock by stock, then by day
            for d in file_days:
                a, b = int(bounds[d][s]), int(bounds[d][s + 1])
                segments.append(fmt.Segment(stock=s + 1, day=d, start=pos, stop=pos + b - a))
                pieces.append((d, a, b))
                pos += b - a
        if pos != matrix.shape[1]:
            raise fmt.FI2010FormatError(f"{name}: segments cover {pos} samples, file has {matrix.shape[1]}")
        lab = fmt.N_LOB + fmt.N_HANDCRAFTED
        rebuilt = np.hstack([days[d][lab:, a:b] for d, a, b in pieces])
        labels_ok = bool(np.array_equal(rebuilt, matrix[lab:]))
        resid = max(
            max_affine_residual(matrix[: fmt.N_LOB, seg.start : seg.stop], days[d][: fmt.N_LOB, a:b])
            for seg, (d, a, b) in zip(segments, pieces)
        )
        manifest.checks["labels_match_day_files"][name] = labels_ok
        manifest.checks["lob_affine_max_residual"][name] = resid
        if not labels_ok or resid > self.AFFINE_TOL:
            raise fmt.FI2010FormatError(f"{name}: segments do not match the day files (labels {labels_ok}, residual {resid:.2e})")
        arrays = fmt.split_matrix(matrix)
        entry = fmt.FileEntry(kind=kind, fold=fold, n=matrix.shape[1], days=file_days, segments=segments,
                              label_counts=label_counts(arrays["labels"]), source_member=member)
        fmt.validate_arrays(arrays, entry)
        fmt.write_arrays(out, name, arrays)
        manifest.files[name] = entry
        self.log(f"{name}: {entry.n:>7,d} samples, days {file_days[0]}-{file_days[-1]}, affine residual {resid:.1e}")
