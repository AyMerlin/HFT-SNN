"""Download raw Bybit order-book day files for a symbol and date range.

    python -m scripts.data.bybit_download --symbol BTCUSDT --start 2026-06-29 --end 2026-09-26

Files land untouched in `<raw-root>/bybit/<symbol>/` with a `.sha256` sidecar; completed
files are skipped, so the command can be re-run after an interruption. `--list` prints the
available range instead of downloading.
"""

from __future__ import annotations

import argparse

from retrieval.bybit import BybitOrderBookDownloader
from scripts.data._common import day_range, iso_date


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--category", default="linear")
    ap.add_argument("--start", type=iso_date)
    ap.add_argument("--end", type=iso_date)
    ap.add_argument("--raw-root", default="data/raw/lob")
    ap.add_argument("--list", action="store_true", help="print the available days and exit")
    args = ap.parse_args(argv)

    dl = BybitOrderBookDownloader(args.raw_root, args.symbol, args.category)
    avail = dl.available()
    if args.list or args.start is None:
        days = sorted(avail)
        print(f"{len(days)} days available: {days[0]} .. {days[-1]}")
        depths: dict[str, list] = {}
        for d in days:
            depths.setdefault(avail[d].rsplit("_", 1)[1].split(".")[0], []).append(d)
        for k, ds in depths.items():
            print(f"  {k}: {ds[0]} .. {ds[-1]} ({len(ds)} days)")
        return 0
    if args.end is None:
        ap.error("--end is required with --start")
    dl.download_range(day_range(args.start, args.end))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
