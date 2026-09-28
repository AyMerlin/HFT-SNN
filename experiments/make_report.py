"""Markdown tables (paper Table 3 / Table 4 style) from an experiment's saved results.

    python -m experiments.make_report --results results/e1_paper_baseline --out docs/reports/e1_tables.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from snn_hft.analysis.compare import table3_markdown, table4_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results", required=True, type=Path, help="results/<experiment>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    t3 = pd.read_csv(args.results / "table3.csv")
    t4 = pd.read_csv(args.results / "table4.csv")
    order = {"model": 0, "naive": 1, "big_move": 2}
    t3 = t3.sort_values(["model", "split", "source"], key=lambda s: s.map(order) if s.name == "source" else s)
    t4 = t4.sort_values(["model", "rule", "latency_ms", "source"], key=lambda s: s.map(order) if s.name == "source" else s)
    lines = [f"# {args.title or args.results.name}", "", "## Table 3 — spikes", "", table3_markdown(t3), ""]
    for latency in sorted(t4["latency_ms"].unique()):
        label = "paper execution (next bar)" if latency == 0 else f"{latency:g} ms latency"
        lines += [f"## Table 4 — strategies, {label}", "", table4_markdown(t4, latency), ""]
    lines += ["Mean over seeds; ± is the standard deviation across seeds (random timing: across its seeds' "
              "repetition means). Returns are fee-free (decision U3).", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
