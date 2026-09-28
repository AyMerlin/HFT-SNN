"""FI-2010 format and data source: splits of the paper's Setup 1 and Setup 2 (§V-B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import snn_hft.data.fi2010.format as fmt
from snn_hft.data.fi2010.source import SETUP_HORIZONS, FI2010DataSource, FI2010Part

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "standardized"
HAS_REAL = (fmt.dataset_dir(REAL_ROOT) / fmt.MANIFEST_NAME).exists()


def test_format_constants_and_helpers():
    assert fmt.HORIZONS == (10, 20, 30, 50, 100) and fmt.horizon_row(50) == 3
    with pytest.raises(ValueError):
        fmt.horizon_row(40)
    assert fmt.lob_column_names()[:5] == ["ask_px_1", "ask_qty_1", "bid_px_1", "bid_qty_1", "ask_px_2"]
    assert fmt.days_of("train", 3) == [1, 2, 3] and fmt.days_of("test", 3) == [4]
    with pytest.raises(ValueError):
        fmt.stem("train", 10)


def test_split_matrix_and_array_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    m = rng.normal(size=(149, 7))
    m[144:] = rng.integers(1, 4, size=(5, 7))
    arrays = fmt.split_matrix(m)
    assert arrays["lob"].shape == (7, 40) and arrays["labels"].dtype == np.int8
    np.testing.assert_allclose(arrays["lob"][:, 0], m[0], rtol=1e-6)
    fmt.write_arrays(tmp_path, "train_cf1", arrays)
    back = fmt.read_arrays(tmp_path, "train_cf1")
    for k in fmt.ARRAYS:
        np.testing.assert_array_equal(back[k], arrays[k])
    assert oct((tmp_path / "train_cf1.lob.npy").stat().st_mode & 0o777) == "0o644"
    bad = m.copy()
    bad[146, 0] = 4
    with pytest.raises(fmt.FI2010FormatError):
        fmt.split_matrix(bad)


def test_segments_must_tile_the_file():
    arrays = fmt.split_matrix(np.vstack([np.zeros((144, 4)), np.ones((5, 4))]))
    entry = fmt.FileEntry("test", 1, 4, [2], [fmt.Segment(1, 2, 0, 2), fmt.Segment(2, 2, 3, 4)], {}, "m")
    with pytest.raises(fmt.FI2010FormatError, match="tile"):
        fmt.validate_arrays(arrays, entry)


def test_setup2_uses_days_1_to_7_and_tests_on_days_8_to_10(converted_fi2010, synthetic_fi2010):
    s = FI2010DataSource(converted_fi2010).setup2()
    n7 = synthetic_fi2010.train[7].shape[1]
    assert (len(s.train), len(s.val)) == (int(np.floor(0.8 * n7)), n7 - int(np.floor(0.8 * n7)))
    assert len(s.test) == sum(synthetic_fi2010.test[i].shape[1] for i in (7, 8, 9))
    assert {seg.day for seg in s.test.segments} == {8, 9, 10}
    assert {seg.day for seg in s.train.segments} | {seg.day for seg in s.val.segments} == set(range(1, 8))
    assert s.horizons == SETUP_HORIZONS[2] == (10, 20, 50)
    np.testing.assert_array_equal(s.test.targets(10), np.concatenate(
        [synthetic_fi2010.test[i][144] for i in (7, 8, 9)]).astype(int) - 1)


def test_setup1_fold_i_trains_on_days_1_to_i_and_tests_on_day_i_plus_1(converted_fi2010):
    src = FI2010DataSource(converted_fi2010)
    folds = src.setup1_folds()
    assert len(folds) == 9 and folds[0].horizons == (10, 50, 100)
    for i, f in enumerate(folds, start=1):
        assert {s.day for s in f.train.segments} | {s.day for s in f.val.segments} == set(range(1, i + 1))
        assert {s.day for s in f.test.segments} == {i + 1}


def merged(segments):
    """Join adjacent runs of the same (stock, day)."""
    out = []
    for s in segments:
        if out and (out[-1].stock, out[-1].day, out[-1].stop) == (s.stock, s.day, s.start):
            out[-1] = fmt.Segment(s.stock, s.day, out[-1].start, s.stop)
        else:
            out.append(s)
    return out


def test_part_slicing_and_concatenation_keep_segments_consistent(converted_fi2010):
    part = FI2010DataSource(converted_fi2010).file("train", 3)
    cut = part.segments[1].start + 5  # inside the second segment
    a, b = part.slice(0, cut), part.slice(cut, len(part))
    assert a.segments[-1].stop == cut and b.segments[0].start == 0
    assert len(a.segments) + len(b.segments) == len(part.segments) + 1
    both = FI2010Part.concat([a, b], "x")
    assert merged(both.segments) == list(part.segments)
    np.testing.assert_array_equal(both.lob, part.lob)
    np.testing.assert_array_equal(both.labels, part.labels)


def test_missing_manifest_is_reported(tmp_path):
    with pytest.raises(fmt.FI2010FormatError, match="fi2010_prepare"):
        FI2010DataSource(tmp_path)


@pytest.mark.skipif(not HAS_REAL, reason="real FI-2010 not converted (python -m scripts.data.fi2010_prepare)")
def test_real_fi2010_matches_the_authors_setup2_and_day_sizes():
    src = FI2010DataSource(REAL_ROOT)
    s2 = src.setup2()
    assert (len(s2.train), len(s2.val), len(s2.test)) == (203_800, 50_950, 139_587)  # authors' notebook
    days = [len(src.file("train", 1))] + [len(src.file("test", i)) for i in range(1, 10)]
    assert days == [39512, 38397, 28535, 37023, 34785, 39152, 37346, 55478, 52172, 31937]
    assert sum(days) == 394_337
    m = src.manifest
    assert all(m.checks["labels_match_day_files"].values()) and max(m.checks["lob_affine_max_residual"].values()) < 1e-4
    assert list(s2.test.class_counts(10)) == [21167, 98638, 19782]
