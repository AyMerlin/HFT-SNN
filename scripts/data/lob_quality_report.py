"""Quality report (Table 0 of plan §10) for any standardized LOB dataset.

    python -m scripts.data.lob_quality_report data/standardized/lob/bybit/BTCUSDT --out docs/reports/lob/l0_bybit_quality.md

Reads `manifest.json`, `quality_report.csv` and `gaps.csv`; writes Markdown. Class
distributions per split and horizon are added once labels exist (L3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retrieval.quality import summarize
from retrieval.writer import GAPS_NAME
from snn_hft.data.lob.schema import QUALITY_REPORT_NAME, Manifest


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def report(directory: Path) -> str:
    m = Manifest.load(directory)
    q = pd.read_csv(directory / QUALITY_REPORT_NAME, dtype={"day": str})
    gaps_path = directory / GAPS_NAME
    gaps = pd.read_csv(gaps_path) if gaps_path.exists() else pd.DataFrame()
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
    args = ap.parse_args(argv)
    text = report(args.dataset)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
