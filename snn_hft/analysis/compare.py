"""Comparison tables in the style of the paper's Table 3 (spikes) and Table 4 (strategies).

Rows come from the backtester (one per seed); tables report the mean over seeds and,
where it matters, the standard deviation across seeds.
"""

from __future__ import annotations

import pandas as pd

TABLE4_METRICS = [
    "accumulated_return", "annualized_volatility", "sharpe", "win_rate", "profit_loss_ratio", "trades_per_day",
]


def table3(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["model", "w_snn", "w_h", "split", "source"]
    grouped = df.groupby(keys, dropna=False, sort=True)
    return grouped.agg(
        seeds=("seed", "nunique"),
        spike_accuracy=("accuracy", "mean"),
        spike_accuracy_std=("accuracy", "std"),
        chance_level=("base_accuracy", "mean"),
        momentum_pct=("momentum_pct", "mean"),
        momentum_base_pct=("base_momentum_pct", "mean"),
        spikes_per_day=("n_signals", "mean"),
        signal_rate=("signal_rate", "mean"),
    ).reset_index()


def table4(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["model", "w_snn", "w_h", "rule", "latency_ms", "source"]
    grouped = df.groupby(keys, dropna=False, sort=True)[TABLE4_METRICS]
    mean = grouped.mean().add_suffix("")
    std = grouped.std(ddof=1).add_suffix("_std")
    seeds = df.groupby(keys, dropna=False, sort=True)["seed"].nunique().rename("seeds")
    return pd.concat([seeds, mean, std], axis=1).reset_index()


def markdown(df: pd.DataFrame) -> str:
    """A plain GitHub-markdown table (no extra dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def table3_markdown(t3: pd.DataFrame) -> str:
    out = pd.DataFrame(
        {
            "Model": t3["model"],
            "Split": t3["split"],
            "Signals": t3["source"].map({"model": "model", "naive": "random timing", "big_move": "big-move"}),
            "Seeds": t3["seeds"],
            "Spike accuracy": (t3["spike_accuracy"] * 100).map("{:.2f} %".format),
            "± seeds": (t3["spike_accuracy_std"] * 100).map(lambda v: "" if pd.isna(v) else f"{v:.2f} pp"),
            "Chance level": (t3["chance_level"] * 100).map("{:.2f} %".format),
            "Momentum %": (t3["momentum_pct"] * 100).map("{:.2f} %".format),
            "Momentum base %": (t3["momentum_base_pct"] * 100).map("{:.2f} %".format),
            "Spikes per day": t3["spikes_per_day"].map("{:,.0f}".format),
        }
    )
    return markdown(out)


def table4_markdown(t4: pd.DataFrame, latency: float) -> str:
    rows = t4[t4["latency_ms"] == latency]
    pct = {"accumulated_return", "annualized_volatility", "win_rate"}
    names = {"accumulated_return": "Accumulated return", "annualized_volatility": "Annualized volatility",
             "sharpe": "Sharpe", "win_rate": "Win rate", "profit_loss_ratio": "Profit/loss", "trades_per_day": "Trades per day"}
    out = pd.DataFrame(
        {
            "Strategy": rows["rule"].str.replace("_", " "),
            "Signals": rows["source"].map({"model": "model", "naive": "random timing", "big_move": "big-move"}),
        }
    )
    for m, label in names.items():
        scale, unit = (100, " %") if m in pct else (1, "")
        fmt = "{:,.0f}" if m == "trades_per_day" else "{:,.2f}"
        out[label] = [
            fmt.format(a * scale) + unit + ("" if pd.isna(b) or m == "trades_per_day" else f" ± {b * scale:,.2f}")
            for a, b in zip(rows[m], rows[f"{m}_std"])
        ]
    return markdown(out)


def format_table3(t3: pd.DataFrame) -> str:
    out = t3.copy()
    for col in ("spike_accuracy", "chance_level", "momentum_pct", "momentum_base_pct", "signal_rate"):
        out[col] = (out[col] * 100).map("{:.2f} %".format)
    out["spike_accuracy_std"] = (out["spike_accuracy_std"] * 100).map("{:.2f} pp".format)
    out["spikes_per_day"] = out["spikes_per_day"].map("{:,.0f}".format)
    return out.to_string(index=False)


def format_table4(t4: pd.DataFrame) -> str:
    out = t4[["model", "w_snn", "w_h", "rule", "latency_ms", "source", "seeds"]].copy()
    pct = {"accumulated_return", "annualized_volatility", "win_rate"}
    for m in TABLE4_METRICS:
        scale, unit = (100, " %") if m in pct else (1, "")
        mean, std = t4[m] * scale, t4[f"{m}_std"] * scale
        out[m] = [
            f"{a:,.2f}{unit}" + ("" if pd.isna(b) else f" ± {b:,.2f}") for a, b in zip(mean, std)
        ]
    return out.to_string(index=False)
