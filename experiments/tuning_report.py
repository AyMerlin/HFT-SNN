"""Markdown report of a tuning run (all trials, admissibility, chosen trial).

    python -m experiments.tuning_report --trials results/tune_paper_snn/trials.csv \
        --out docs/reports/m5_tuning_paper.md --title "M5 tuning — paper SNN (100 trades per bar)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from snn_hft.analysis.compare import markdown

LABELS = {
    "core.lif.threshold": "Threshold",
    "core.lif.leak": "Leak",
    "input.new_mean": "new_mean",
    "zscore.new_std": "new_std",
    "stdp_scale": "STDP scale",
    "rstdp.gamma": "γ",
    "rstdp.tau_z_bars": "τ_z (bars)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--max-unhealthy", type=float, default=0.05)
    parser.add_argument("--notes", default="", help="extra markdown paragraph")
    args = parser.parse_args(argv)

    t = pd.read_csv(args.trials)
    params = [c for c in LABELS if c in t.columns]
    t["admissible"] = t["unhealthy_share"] <= args.max_unhealthy
    t = t.sort_values(["admissible", "accuracy"], ascending=[False, False])
    best = t.iloc[0]
    table = pd.DataFrame({"Trial": t["trial"], **{LABELS[p]: t[p] for p in params}})
    table["Accuracy"] = (t["accuracy"] * 100).map("{:.2f} %".format)
    table["Edge over chance"] = t["edge_pp"].map("{:+.2f} pp".format)
    table["Signal rate"] = (t["signal_rate"] * 100).map("{:.1f} %".format)
    table["Unhealthy days"] = (t["unhealthy_share"] * 100).map("{:.0f} %".format)
    table["Admissible"] = t["admissible"].map({True: "yes", False: "no"})
    chosen = ", ".join(f"{LABELS[p]} {best[p]:g}" for p in params)
    text = f"""# {args.title}

Protocol: [DESIGN_DECISIONS I39, I41](../DESIGN_DECISIONS.md). {len(t)} distinct grid points, walk-forward
`W_snn = 1` on all validation days, {int(best['folds'])} folds per trial. Objective: mean test-day spike
accuracy; admissible if at most {args.max_unhealthy:.0%} of the days fail a health check.
Chance level (random timing) on these days: {t['chance'].mean() * 100:.2f} %.
Admissible trials: {int(t['admissible'].sum())} of {len(t)}.

**Chosen: {best['trial']}** — {chosen}. Validation accuracy {best['accuracy'] * 100:.2f} %
({best['edge_pp']:+.2f} pp over chance), signal rate {best['signal_rate'] * 100:.1f} %.

{args.notes}

{markdown(table)}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
