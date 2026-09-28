"""Head-to-head report: the paper's strategies with the paper SNN vs with the improved model.

    python -m experiments.head_to_head_report --results results/head_to_head --out docs/reports/head_to_head.md
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from snn_hft.analysis.compare import markdown  # noqa: E402

RULES = {"momentum": "Momentum", "alexanders_filter": "Alexander's filter", "stochastic_oscillator": "Stochastic oscillator"}
LAT = {0.0: "next bar", 10.0: "10 ms", 100.0: "100 ms", 1000.0: "1 s"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"paper": "#2a78d6", "improved": "#eb6834"}
LABELS = {"paper": "paper SNN", "improved": "improved model"}


def daily(root: Path, model: str, rule: str) -> pd.DataFrame:
    """Seed-mean daily P&L and trade counts per (day, latency, source)."""
    frames = [pd.read_parquet(f) for f in glob.glob(str(root / f"{model}__{rule}__*/daily_pnl.parquet"))]
    d = pd.concat(frames)
    return d.groupby(["latency_ms", "source", "day"])[["pnl", "n_trades"]].mean()


def plot(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, facecolor=SURFACE, sharey=False)
    for ax, lat in zip(axes, (0.0, 1000.0)):
        ax.set_facecolor(SURFACE)
        for model in ("paper", "improved"):
            d = daily(root, model, "momentum").loc[lat]
            excess = (d.loc["model", "pnl"] - d.loc["naive_mean", "pnl"]).sort_index()
            days = pd.to_datetime(excess.index)
            y = excess.cumsum().to_numpy() * 100
            ax.plot(days, y, color=COLORS[model], lw=2, label=LABELS[model], zorder=3)
            ax.annotate(f"{LABELS[model]} {y[-1]:+.0f} pp", (days[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.axhline(0, color=INK_2, lw=0.8, zorder=2)
        ax.set_title(f"{LAT[lat]} execution", loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_ylabel("cumulative P&L above random timing (pp)", color=INK_2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(0.93, 0.99))
    fig.suptitle("Momentum strategy: does spike timing beat random timing? (fee-free, mean of 5 seeds)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.results

    t4 = pd.read_csv(root / "table4.csv")
    t3 = pd.read_csv(root / "table3.csv")
    t3 = t3[t3.split == "test"].set_index(["model", "source"])

    spike = pd.DataFrame([{
        "Model": LABELS[m],
        "Spike accuracy": f"{t3.loc[(m, 'model'), 'spike_accuracy'] * 100:.2f} %",
        "Random timing": f"{t3.loc[(m, 'naive'), 'spike_accuracy'] * 100:.2f} %",
        "Big-move (same rate)": f"{t3.loc[(m, 'big_move'), 'spike_accuracy'] * 100:.2f} %",
        "Signals per day": f"{t3.loc[(m, 'model'), 'spikes_per_day']:,.0f}",
    } for m in ("paper", "improved")])

    perf_rows, test_rows = [], []
    for rule, rule_label in RULES.items():
        dp, di = daily(root, "paper", rule), daily(root, "improved", rule)
        for lat in sorted(t4.latency_ms.unique()):
            row = {"Strategy": rule_label, "Execution": LAT[lat]}
            for m, d in (("paper", dp), ("improved", di)):
                r = t4[(t4.model == m) & (t4.rule == rule) & (t4.latency_ms == lat)].set_index("source")
                x = d.loc[lat]
                per_trade = x.loc["model", "pnl"].sum() / x.loc["model", "n_trades"].sum() * 1e4
                row[f"{LABELS[m]}: return"] = f"{r.loc['model', 'accumulated_return'] * 100:,.0f} %"
                row[f"{LABELS[m]}: Sharpe (random)"] = f"{r.loc['model', 'sharpe']:.2f} ({r.loc['naive', 'sharpe']:.2f})"
                row[f"{LABELS[m]}: win rate"] = f"{r.loc['model', 'win_rate'] * 100:.2f} %"
                row[f"{LABELS[m]}: bp/trade"] = f"{per_trade:.3f}"
            perf_rows.append(row)
            # Paired daily comparisons over the test days.
            diff_models = dp.loc[lat].loc["model", "pnl"] - di.loc[lat].loc["model", "pnl"]
            ex_p = dp.loc[lat].loc["model", "pnl"] - dp.loc[lat].loc["naive_mean", "pnl"]
            ex_i = di.loc[lat].loc["model", "pnl"] - di.loc[lat].loc["naive_mean", "pnl"]
            test_rows.append({
                "Strategy": rule_label, "Execution": LAT[lat],
                "paper SNN − random: days ahead, p": f"{int((ex_p > 0).sum())}/{len(ex_p)}, {wilcoxon(ex_p).pvalue:.2g}",
                "improved − random: days ahead, p": f"{int((ex_i > 0).sum())}/{len(ex_i)}, {wilcoxon(ex_i).pvalue:.2g}",
                "improved − paper SNN (P&L/day)": f"{-diff_models.mean() * 100:+.3f} pp",
                "days improved ahead, p": f"{int((diff_models < 0).sum())}/{len(diff_models)}, {wilcoxon(diff_models).pvalue:.2g}",
            })
    chart = args.out.with_suffix(".png")
    plot(root, chart)

    text = f"""# Head-to-head strategy backtests: paper SNN vs improved model

The paper's three strategies (§7), each driven once by the paper SNN (E1 settings) and once by the
improved model (E2 settings), both tuned on the validation period only. Identical test days
(2025-12-01 → 2026-09-26, 300 days), seeds 0–4, 100 aggTrades per bar, fee-free (U3). Every trade
goes through the `Strategy` class (§7.3) with the model's cached signals. Each model's random-timing
benchmark fires as often as that model, so "above random timing" isolates the value of the timing.

```bash
.venv/bin/python -m experiments.run_experiment --config experiments/configs/head_to_head.yaml
.venv/bin/python -m experiments.head_to_head_report --results results/head_to_head --out docs/reports/head_to_head.md
```

## Spikes

{markdown(spike)}

## Strategy performance

Accumulated returns add up trades of notional 1, so they grow with the number of trades (the
improved model trades about twice as often); per-trade return (bp) and Sharpe compare better.

{markdown(pd.DataFrame(perf_rows))}

## Does the timing add value? (paired over the 300 test days)

{markdown(pd.DataFrame(test_rows))}

![Cumulative P&L above random timing]({chart.name})

## Reading

- **Improved vs paper SNN.** The improved model's strategies earn more per day than the paper
  SNN's for every rule with next-bar, 10 ms and 100 ms execution (paired Wilcoxon p ≤ 0.009; not
  significant at 1 s), and have a higher Sharpe ratio at every latency. Most of the extra P&L comes
  from trading about twice as often: per trade they earn the same with next-bar execution
  (0.066–0.077 bp) and somewhat more with latency (e.g. 0.019–0.020 vs 0.012–0.013 bp at 100 ms).
- **Spike timing vs random timing.** Neither model makes the strategies more profitable than
  random timing with the same number of trades. The paper SNN is never significantly different;
  the improved model is slightly behind with next-bar execution (ahead on 129–134 of 300 days,
  p ≈ 0.04–0.13) and level with latency.
- **Costs.** Both earn well under 0.1 bp per trade against a taker round trip of about 10 bp.
"""
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
