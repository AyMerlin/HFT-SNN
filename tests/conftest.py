"""Shared fixtures."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

N_DAYS, N_STOCKS, N_ROWS = 10, 5, 149


@dataclass
class SyntheticFI2010:
    """A small archive with FI-2010's structure and the data it was built from."""

    archive: Path
    lengths: np.ndarray  # (day, stock) number of samples
    train: dict[int, np.ndarray]  # fold -> (149, N) matrix as written
    test: dict[int, np.ndarray]


def _member(kind: str, fold: int, auction="NoAuction") -> str:
    folder = "Training" if kind == "Train" else "Testing"
    return (f"BenchmarkDatasets/{auction}/1.{auction}_Zscore/{auction}_Zscore_{folder}/"
            f"{kind}_Dst_{auction}_ZScore_CF_{fold}.txt")


def build_synthetic_fi2010(path: Path, seed: int = 0, stock_major: bool = True) -> SyntheticFI2010:
    """Raw LOB per (day, stock) with stock-specific price levels, random features and labels;
    fold i z-scored with the statistics of days 1…i; training files ordered stock by stock
    (or day by day with `stock_major=False`, which the converter must reject)."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, 40, size=(N_DAYS, N_STOCKS))
    raw = {}
    for d in range(N_DAYS):
        for s in range(N_STOCKS):
            n = lengths[d, s]
            mid = 10.0 * (s + 1) + np.cumsum(rng.normal(0, 0.001, n))
            m = np.empty((N_ROWS, n))
            for lvl in range(10):
                m[4 * lvl] = mid + 0.01 * (lvl + 1)
                m[4 * lvl + 1] = rng.uniform(1, 100, n)
                m[4 * lvl + 2] = mid - 0.01 * (lvl + 1)
                m[4 * lvl + 3] = rng.uniform(1, 100, n)
            m[40:144] = rng.normal(size=(104, n))
            m[144:] = rng.integers(1, 4, size=(5, n))
            raw[d, s] = m

    def order(days):
        keys = [(d, s) for s in range(N_STOCKS) for d in days] if stock_major else [(d, s) for d in days for s in range(N_STOCKS)]
        return np.hstack([raw[k] for k in keys])

    train, test = {}, {}
    for fold in range(1, N_DAYS):
        tr = order(range(fold))
        te = np.hstack([raw[fold, s] for s in range(N_STOCKS)])
        mean, std = tr[:144].mean(axis=1, keepdims=True), tr[:144].std(axis=1, keepdims=True)
        for mat in (tr, te):
            mat[:144] = (mat[:144] - mean) / std
        train[fold], test[fold] = tr, te

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fold in range(1, N_DAYS):
            for kind, mat in (("Train", train[fold]), ("Test", test[fold])):
                text = "\n".join(" ".join(f"{v:.8e}" for v in row) for row in mat) + "\n"
                zf.writestr(_member(kind, fold), text)
    # the text round trip limits precision: return what a reader of the archive sees
    train = {k: np.array([[float(f"{v:.8e}") for v in row] for row in m]) for k, m in train.items()}
    test = {k: np.array([[float(f"{v:.8e}") for v in row] for row in m]) for k, m in test.items()}
    return SyntheticFI2010(path, lengths, train, test)


@pytest.fixture(scope="session")
def fi2010_builder():
    """The archive builder, for tests that need a variant of the synthetic data."""
    return build_synthetic_fi2010


@pytest.fixture(scope="session")
def synthetic_fi2010(tmp_path_factory) -> SyntheticFI2010:
    return build_synthetic_fi2010(tmp_path_factory.mktemp("fi2010") / "BenchmarkDatasets.zip")


@pytest.fixture(scope="session")
def converted_fi2010(tmp_path_factory, synthetic_fi2010) -> Path:
    """Root directory holding the converted synthetic archive (NoAuction/Zscore)."""
    from retrieval.fi2010 import FI2010Converter

    root = tmp_path_factory.mktemp("standardized")
    FI2010Converter(synthetic_fi2010.archive, log=lambda m: None).convert(root, archive_sha256="test")
    return root
