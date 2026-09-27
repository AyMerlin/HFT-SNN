"""Download and cache all days of the study period, then write a coverage report.

    python -m experiments.download_data --config experiments/configs/base.yaml
    python -m experiments.download_data --config ... --start 2025-10-17 --end 2025-10-20

Uses the `data` and `periods.data` sections of the config. The report goes to
``{raw_dir}/coverage_{venue}_{symbol}_{dataset}.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from snn_hft.config.loader import apply_override, load_raw
from snn_hft.config.schema import BarsConfig, DataConfig, PeriodsConfig
from snn_hft.data.store import TradingCalendar, make_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE")
    parser.add_argument("--start", type=date.fromisoformat, help="first day (default: periods.data.start)")
    parser.add_argument("--end", type=date.fromisoformat, help="last day (default: periods.data.end)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    raw = load_raw(args.config)
    for assignment in args.overrides:
        apply_override(raw, assignment)
    data_cfg = DataConfig.model_validate(raw.get("data", {}))
    periods = PeriodsConfig.model_validate(raw.get("periods", {}))
    vwap_num = BarsConfig.model_validate(raw.get("bars", {})).vwap_num
    start = args.start or periods.data.start
    end = args.end or periods.data.end

    store = make_store(data_cfg)
    days = TradingCalendar.days(start, end)
    todo = sum(not store.has_day(data_cfg.symbol, d) for d in days)
    print(f"{data_cfg.symbol} {data_cfg.dataset} {start}..{end}: {len(days)} days, {todo} to download")

    t0 = time.time()
    done = 0

    def progress(day, rec, exc):
        nonlocal done
        done += 1
        status = f"FAILED: {exc}" if exc else f"{rec.n_trades:>10,} trades"
        print(f"[{done:>3}/{todo}] {day} {status}  ({time.time() - t0:.0f}s)", flush=True)

    failures = store.ensure_days(data_cfg.symbol, days, workers=args.workers, on_done=progress)

    report = store.coverage(data_cfg.symbol, start, end, vwap_num=vwap_num)
    out = data_cfg.raw_dir / f"coverage_{store.source.venue}_{data_cfg.symbol}_{data_cfg.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    cached = report[report["cached"]]
    print(f"\ncached {len(cached)}/{len(report)} days -> {out}")
    if len(cached):
        print(cached[["n_trades", "n_bars", "hours_covered", "max_gap_s"]].describe().round(1).to_string())
    for day, exc in sorted(failures.items()):
        print(f"missing {day}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
