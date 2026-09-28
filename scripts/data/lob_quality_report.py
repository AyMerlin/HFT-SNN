"""Quality report (Table 0 of plan §10) for any standardized LOB dataset.

    python -m scripts.data.lob_quality_report data/standardized/lob/bybit/BTCUSDT --out docs/reports/lob/l0_bybit_quality.md

Reads `manifest.json`, `quality_report.csv` and `gaps.csv`; writes Markdown. Class
distributions per split and horizon are added once labels exist (L3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval.quality import summarize
from retrieval.writer import GAPS_NAME
from snn_hft.data.lob.schema import QUALITY_REPORT_NAME, Manifest, read_frame, side_columns

HORIZONS = (10, 20, 50, 100)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def day_content(path: Path, levels: int = 10) -> dict:
    """Descriptive statistics of one standardized day (reads the top `levels` only).

    "Events" are rows where any of the top-`levels` prices or sizes changed, i.e. the event
    clock of the plan's LOBEventSampler; they set the wall-clock span of a k-event horizon.
    """
    df = read_frame(path, validate=False)
    cols = [c for side in ("ask", "bid") for kind in ("px", "qty") for c in side_columns(side, kind, levels)]
    top = df[cols].to_numpy(dtype=np.float64)
    changed = np.ones(len(top), dtype=bool)
    changed[1:] = (top[1:] != top[:-1]).any(axis=1)
    ts = df["ts_ns"].to_numpy()[changed]
    ask, bid = df["ask_px_1"].to_numpy(), df["bid_px_1"].to_numpy()
    mid = (ask + bid) / 2
    spread_bps = (ask - bid) / mid * 1e4
    gaps_s = np.diff(ts) / 1e9
    out = {
        "rows": len(df),
        "events": int(changed.sum()),
        "event_share": float(changed.mean()),
        "mid_change_share": float(np.mean(np.diff(mid[changed]) != 0)) if changed.sum() > 1 else float("nan"),
        "mid_min": float(mid.min()),
        "mid_max": float(mid.max()),
        "spread_bps_median": float(np.median(spread_bps)),
        "spread_one_tick_share": float(np.mean(np.isclose(ask - bid, np.min(ask - bid)))),
        "s_per_event_median": float(np.median(gaps_s)) if len(gaps_s) else float("nan"),
    }
    for k in HORIZONS:  # median wall-clock span of k consecutive events
        out[f"s_per_{k}_events"] = float(np.median((ts[k:] - ts[:-k]) / 1e9)) if len(ts) > k else float("nan")
    return out


def content_section(directory: Path, m: Manifest) -> list[str]:
    rows = []
    for d in m.served_days():
        rows.append({"day": d.isoformat(), **day_content(directory / m.days[d.isoformat()].file)})
    c = pd.DataFrame(rows)
    if c.empty:
        return []
    summary = pd.DataFrame([{
        "top-10 events per day (median)": f"{c['events'].median():,.0f}",
        "share of rows that change the top 10": f"{100 * c['event_share'].median():.1f} %",
        "share of events that move the mid": f"{100 * c['mid_change_share'].median():.1f} %",
        "mid price range": f"{c['mid_min'].min():,.1f} .. {c['mid_max'].max():,.1f}",
        "median spread": f"{c['spread_bps_median'].median():.3f} bps",
        "one-tick spread share": f"{100 * c['spread_one_tick_share'].median():.1f} %",
    }])
    spans = pd.DataFrame([{f"k = {k}": f"{c[f's_per_{k}_events'].median():.1f} s" for k in HORIZONS}])
    per_day = c[["day", "rows", "events", "mid_min", "mid_max", "spread_bps_median", "s_per_100_events"]].round(3)
    return [
        "## Content (served days)",
        "",
        _md(summary),
        "",
        "Wall-clock span of k top-10 events (median over days of the per-day median):",
        "",
        _md(spans),
        "",
        "<details><summary>Per day</summary>",
        "",
        _md(per_day),
        "",
        "</details>",
        "",
    ]


def report(directory: Path, content: bool = False) -> str:
    m = Manifest.load(directory)
    q = pd.read_csv(directory / QUALITY_REPORT_NAME, dtype={"day": str})
    gaps_path = directory / GAPS_NAME
    try:
        gaps = pd.read_csv(gaps_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        gaps = pd.DataFrame()
    s = summarize(q)
    rng = m.date_range
    out = [
        f"# Data quality — {m.venue} {m.symbol} (depth {m.depth})",
        "",
        f"Source `{m.source}`; days {rng[0]} .. {rng[1]}; retrieval code `{m.code_git_hash}`"
        + (" (dirty tree)" if m.code_git_dirty else "")
        + f"; manifest updated {m.updated_utc}.",
        "",
        "## Days per status",
        "",
        _md(pd.DataFrame([{k: s[k] for k in ("days", "ok", "flagged", "excluded")}])),
        "",
        "Status rule (plan §4.2): gap time ≤ 2 % of the day → ok, ≤ 5 % → flagged, else excluded. "
        "Gap time = invalid book (sequence gap until the next snapshot) + message-free stretches "
        "longer than the silence threshold. A day with a snapshot mismatch is at best flagged.",
        "",
        "## Reconstruction and gaps",
        "",
        _md(pd.DataFrame([{
            "sequence gaps": s["seq_gaps_total"],
            "gap time (s)": round(s["gap_s_total"], 1),
            "max gap share": f"{100 * s['gap_fraction_max']:.3f} %",
            "snapshot checks": s["snapshot_checks"],
            "snapshot mismatches": s["snapshot_mismatches"],
            "rows removed": s["rows_removed_total"],
        }])),
        "",
        "## Rows per served day",
        "",
        _md(pd.DataFrame([{
            "median": f"{s['rows_per_day_median']:,.0f}",
            "min": f"{s['rows_per_day_min']:,}",
            "max": f"{s['rows_per_day_max']:,}",
        }])),
        "",
    ]
    flagged = q[q["status"] != "ok"]
    if len(flagged):
        out += ["## Days not ok", "", _md(flagged[["day", "status", "gap_fraction", "n_seq_gaps", "notes"]]), ""]
    if len(gaps):
        out += ["## Gaps", "", _md(gaps), ""]
    if content:
        out += content_section(directory, m)
    cols = [
        "day", "status", "n_messages", "n_snapshots", "n_seq_gaps", "n_snapshot_checks", "n_snapshot_mismatches",
        "n_rows_written", "n_rows_removed", "gap_s", "max_interval_s", "n_intervals_over_1s",
    ]
    per_day = q[cols].copy()
    per_day["gap_s"] = per_day["gap_s"].round(2)
    per_day["max_interval_s"] = per_day["max_interval_s"].round(2)
    out += ["## Per day", "", _md(per_day), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", type=Path, help="dataset directory holding manifest.json")
    ap.add_argument("--out", type=Path, help="write Markdown here instead of stdout")
    ap.add_argument("--content", action="store_true", help="add price, spread and event-clock statistics (reads every day file)")
    args = ap.parse_args(argv)
    text = report(args.dataset, content=args.content)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
