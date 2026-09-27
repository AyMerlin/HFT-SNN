"""Data coverage report for the study period (M1 check-in).

    python -m experiments.data_report --config experiments/configs/base.yaml

Reads the coverage CSV written by ``experiments.download_data`` and writes a
markdown summary plus a chart of daily activity to ``docs/reports/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from snn_hft.config.loader import load_raw  # noqa: E402
from snn_hft.config.schema import BarsConfig, DataConfig, InputConfig, PeriodsConfig  # noqa: E402

GAP_WARN_S = 60.0
INK, INK_2, GRID, SURFACE, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
BAND = "#f0efeb"


def period_of(day, periods: PeriodsConfig) -> str:
    if day in periods.test:
        return "test"
    if day in periods.validation:
        return "validation"
    return "history"


def fmt_range(s: pd.Series, scale: float = 1.0, digits: int = 0) -> str:
    q = (s / scale).quantile([0.0, 0.5, 1.0])
    return f"{q.iloc[1]:,.{digits}f} ({q.iloc[0]:,.{digits}f} – {q.iloc[2]:,.{digits}f})"


def plot(cov: pd.DataFrame, periods: PeriodsConfig, symbol: str, dataset: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    days = pd.to_datetime(cov["day"])
    top = cov["n_trades"].max() / 1e6 * 1.12
    for name, rng, shade in (("history", None, True), ("validation", periods.validation, False), ("test", periods.test, True)):
        start = pd.Timestamp(rng.start if rng else periods.data.start)
        end = pd.Timestamp(rng.end if rng else periods.validation.start - pd.Timedelta(days=1)) + pd.Timedelta(days=1)
        if shade:
            ax.axvspan(start, end, color=BAND, lw=0, zorder=0)
        ax.text(start + (end - start) / 2, top * 0.97, name, ha="center", va="top", color=INK_2, fontsize=9)
    ax.plot(days, cov["n_trades"] / 1e6, color=SERIES, lw=1.5, zorder=3)
    ax.set_ylim(0, top)
    ax.set_ylabel("aggTrades per day (millions)", color=INK_2, fontsize=9)
    ax.set_title(f"{symbol} {dataset}: trades per UTC day", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/reports"))
    args = parser.parse_args(argv)

    raw = load_raw(args.config)
    data_cfg = DataConfig.model_validate(raw.get("data", {}))
    periods = PeriodsConfig.model_validate(raw.get("periods", {}))
    vwap_num = BarsConfig.model_validate(raw.get("bars", {})).vwap_num
    ticks = InputConfig().ticks_per_bar
    path = data_cfg.raw_dir / f"coverage_{data_cfg.venue}_{data_cfg.symbol}_{data_cfg.dataset}.csv"
    cov = pd.read_csv(path, parse_dates=["day"])
    cov["day"] = cov["day"].dt.date
    cov["period"] = [period_of(d, periods) for d in cov["day"]]
    ok = cov[cov["cached"]].copy()
    ok["n_bars"] = ok["n_trades"] // vwap_num

    size_gb = sum(p.stat().st_size for p in (data_cfg.raw_dir / data_cfg.venue / data_cfg.symbol / data_cfg.dataset).glob("*.parquet")) / 1e9
    missing = cov.loc[~cov["cached"], "day"].tolist()
    gaps = ok[(ok["max_gap_s"] > GAP_WARN_S) | (ok["hours_covered"] < 23.9)]

    lines = [
        f"# M1 data coverage — {data_cfg.symbol} {data_cfg.dataset}",
        "",
        f"Source: Binance public archive (USDⓈ-M futures), every file verified against its `.CHECKSUM`.  ",
        f"Range {periods.data.start} → {periods.data.end}: **{len(ok)}/{len(cov)} days cached**, "
        f"{ok['n_trades'].sum() / 1e6:,.0f}M trades, {size_gb:.1f} GB parquet.  ",
        f"Bars at `vwap_num = {vwap_num}`; simulation ticks at `T = {ticks}` per bar.",
        "",
        "| Period | Days | Trades per day, median (min – max) | Bars per day | Ticks per day (millions) |",
        "|---|---|---|---|---|",
    ]
    for name in ("history", "validation", "test"):
        part = ok[ok["period"] == name]
        lines.append(
            f"| {name} | {len(part)} | {fmt_range(part['n_trades'])} | {fmt_range(part['n_bars'])} | "
            f"{fmt_range(part['n_bars'] * ticks, 1e6, 2)} |"
        )
    lines.append(
        f"| **all** | {len(ok)} | {fmt_range(ok['n_trades'])} | {fmt_range(ok['n_bars'])} | "
        f"{fmt_range(ok['n_bars'] * ticks, 1e6, 2)} |"
    )
    weekend = ok[pd.to_datetime(ok["day"]).dt.dayofweek >= 5]
    weekday = ok[pd.to_datetime(ok["day"]).dt.dayofweek < 5]
    busiest = ok.nlargest(3, "n_trades")[["day", "n_trades"]]
    quietest = ok.nsmallest(3, "n_trades")[["day", "n_trades"]]
    lines += [
        "",
        f"- Weekday median {weekday['n_trades'].median():,.0f} trades vs weekend median {weekend['n_trades'].median():,.0f}.",
        "- Busiest days: " + ", ".join(f"{r.day} ({r.n_trades:,})" for r in busiest.itertuples()) + ".",
        "- Quietest days: " + ", ".join(f"{r.day} ({r.n_trades:,})" for r in quietest.itertuples()) + ".",
        f"- Trades outside the UTC day in the files: {int(ok['n_outside_day'].sum())}.",
        f"- Missing days: {', '.join(map(str, missing)) if missing else 'none'}.",
        f"- Days with a gap between trades above {GAP_WARN_S:.0f} s or less than 23.9 h covered: "
        + (", ".join(f"{r.day} (max gap {r.max_gap_s:,.0f} s)" for r in gaps.itertuples()) if len(gaps) else "none")
        + f". Longest gap on any day: {ok['max_gap_s'].max():,.1f} s.",
        "",
        "![Trades per day](m1_data_coverage.png)",
        "",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "m1_data_coverage.md").write_text("\n".join(lines))
    plot(ok, periods, data_cfg.symbol, data_cfg.dataset, args.out_dir / "m1_data_coverage.png")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
