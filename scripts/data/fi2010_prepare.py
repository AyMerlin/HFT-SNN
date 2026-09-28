"""Download FI-2010 from Fairdata, verify it and convert one variant for the framework.

    python -m scripts.data.fi2010_prepare                       # NoAuction / Zscore (the paper's data)
    python -m scripts.data.fi2010_prepare --normalization DecPre

Writes `<raw-root>/fi2010/BenchmarkDatasets.zip` (+ .sha256) and
`<out-root>/fi2010/<auction>_<normalization>/` with one set of arrays per fold file and
`manifest.json`. A verified archive is not downloaded again.
"""

from __future__ import annotations

import argparse
import time

from retrieval.fi2010 import FI2010Converter, FI2010Downloader
from retrieval.http import recorded_sha256
from snn_hft.data.fi2010.format import AUCTIONS, NORMALIZATIONS, dataset_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--out-root", default="data/standardized")
    ap.add_argument("--auction", choices=AUCTIONS, default="NoAuction")
    ap.add_argument("--normalization", choices=NORMALIZATIONS, default="Zscore")
    args = ap.parse_args(argv)

    t0 = time.time()
    archive = FI2010Downloader(args.raw_root).download()
    manifest = FI2010Converter(archive, args.auction, args.normalization).convert(args.out_root, recorded_sha256(archive))
    c = manifest.checks
    print(f"checks: labels match day files {all(c['labels_match_day_files'].values())}, "
          f"max affine residual {max(c['lob_affine_max_residual'].values()):.1e}, "
          f"stock-boundary separation >= {c['stock_boundary_separation_min']:.1f}")
    print(f"wrote {dataset_dir(args.out_root, args.auction, args.normalization)} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
