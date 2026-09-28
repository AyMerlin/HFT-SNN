"""Premise checks for a momentum/reversion ("typed") model — U10, Phase A.

    python -m experiments.typed_premise_checks --out docs/reports/typed_premise_checks.md

Trade label at bar t (paper execution, entry t+1, exit t+4):
    dir_t = MomentumRule(3) direction at t,  follow_ret_t = dir_t · (vwap[t+4] / vwap[t+1] − 1),
    "momentum" = following wins (follow_ret > 0), "reversion" = fading wins (follow_ret < 0).
The paper's mom_rev_flag label is reported next to it.

A1–A3 use validation days only (they inform the design). A4 analyses the existing E1/E2 test
results for the critique and is never used for design choices.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from snn_hft.analysis.compare import markdown  # noqa: E402
from snn_hft.backtest.cache import BarCache, HawkesParamCache  # noqa: E402
from snn_hft.config import expand_runs, load_config  # noqa: E402
from snn_hft.config.loader import load_raw  # noqa: E402
from snn_hft.config.schema import BarsConfig, DataConfig, ExecutionConfig, HawkesConfig, OutputConfig, PeriodsConfig  # noqa: E402
from snn_hft.data.containers import DayData  # noqa: E402
from snn_hft.data.store import make_store  # noqa: E402
from snn_hft.evaluation.spike_metrics import SpikeEvaluator  # noqa: E402
from snn_hft.models.hawkes.events import events_from_bars, threshold_events  # noqa: E402
from snn_hft.models.hawkes.process import BivariateHawkesProcess  # noqa: E402
from snn_hft.models.hawkes.provider import HawkesParamProvider  # noqa: E402
from snn_hft.preprocessing.hawkes_steps import candidate_events_source  # noqa: E402
from snn_hft.signals.big_move import BigMoveSignalModel  # noqa: E402
from snn_hft.signals.factory import make_signal_model  # noqa: E402
from snn_hft.strategy.direction_rules import MomentumRule  # noqa: E402
from snn_hft.strategy.execution import per_signal  # noqa: E402

BASE_CONFIG = Path("experiments/configs/base.yaml")
PAPER_CONFIG = Path("experiments/configs/paper_baseline.yaml")
IMPROVED_CONFIG = Path("experiments/configs/improved.yaml")
QUANTILES = (0.8, 0.9, 0.95)
LAG_BUCKETS = ("0", "1", "2", "3–5", ">5")
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# --------------------------------------------------------------------------- shared helpers


@dataclass
class Ctx:
    data: DataConfig
    bars: BarsConfig
    periods: PeriodsConfig
    bar_cache: BarCache
    params_cache: HawkesParamCache


def ctx() -> Ctx:
    raw = load_raw(BASE_CONFIG)
    data = DataConfig.model_validate(raw.get("data", {}))
    bars = BarsConfig.model_validate(raw.get("bars", {}))
    out = OutputConfig.model_validate(raw.get("output", {}))
    store = make_store(data)
    return Ctx(
        data, bars, PeriodsConfig.model_validate(raw.get("periods", {})),
        BarCache(store, data.symbol, bars.vwap_num, out.cache_dir),
        HawkesParamCache(out.cache_dir, store.source.venue, data.symbol, data.dataset, bars.vwap_num),
    )


def diffs(vwap: np.ndarray) -> np.ndarray:
    d = np.full(len(vwap), np.nan)
    d[1:] = np.diff(vwap)
    return d


def trade_label(vwap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dir_t, follow_ret_t); follow_ret is NaN where dir_t = 0 or t + 4 is outside the day."""
    n = len(vwap)
    t = np.arange(n)
    dirs = MomentumRule(3).directions(vwap, t).astype(np.int64)
    ok = (dirs != 0) & (t + 4 <= n - 1)
    ret = np.full(n, np.nan)
    ret[ok] = dirs[ok] * (vwap[t[ok] + 4] / vwap[t[ok] + 1] - 1.0)
    return dirs, ret


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties counted half."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    ok = ~np.isnan(score)
    score, positive = score[ok], positive[ok]
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fmt(x, digits=3):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


# --------------------------------------------------------------------------- A1: teacher check


def a1_day(day: date, q: float) -> tuple[pd.DataFrame, dict]:
    c = ctx()
    hcfg = HawkesConfig(event_quantile=q)
    provider = HawkesParamProvider(hcfg, 1, candidate_events_source(c.bar_cache.get, hcfg.time_axis), c.params_cache)
    theta = provider.params_for(day)
    bars = c.bar_cache.get(day)
    vwap = bars.vwap
    n = len(vwap)
    events = threshold_events(events_from_bars(diffs(vwap), None, 0, "bar_index"), theta.event_threshold)
    _, lam_plus, br = BivariateHawkesProcess(hcfg.mark_fn, hcfg.time_axis).bar_pass(theta, events, np.arange(n, dtype=float))
    b, m = events.bar_idx, events.types
    lam_same, lam_opp = lam_plus[b, m], lam_plus[b, 1 - m]
    dirs, ret = trade_label(vwap)
    lab = SpikeEvaluator(3).labels(vwap)
    df = pd.DataFrame({
        "day": day, "q": q, "bar": b, "M": br[:, 1] - br[:, 2], "p_same": br[:, 1], "p_cross": br[:, 2],
        "asym": (lam_same - lam_opp) / (lam_plus[b, 0] + lam_plus[b, 1]), "dir": dirs[b],
        "event_dir": np.where(m == 0, 1, -1), "ret": ret[b], "paper_eval": lab.evaluable[b], "paper_mom": lab.momentum[b],
    })
    decided = ~np.isnan(ret) & (ret != 0)
    base = {"day": day, "q": q, "decided_bars": int(decided.sum()), "reversion_bars": int((ret[decided] < 0).sum()),
            "paper_eval_bars": int(lab.evaluable.sum()), "paper_reversion_bars": int((lab.evaluable & ~lab.momentum).sum())}
    return df, base


def a1b_day(day: date, q: float = 0.9) -> pd.DataFrame:
    """Causal features the typed network could see at bar t, oriented by the trade direction dir_t."""
    c = ctx()
    hcfg = HawkesConfig(event_quantile=q)
    provider = HawkesParamProvider(hcfg, 1, candidate_events_source(c.bar_cache.get, hcfg.time_axis), c.params_cache)
    theta = provider.params_for(day)
    bars, train_bars = c.bar_cache.get(day), c.bar_cache.get(day - timedelta(days=1))
    vwap = bars.vwap
    n = len(vwap)
    d = np.nan_to_num(diffs(vwap))
    events = threshold_events(events_from_bars(diffs(vwap), None, 0, "bar_index"), theta.event_threshold)
    _, lam, _ = BivariateHawkesProcess(hcfg.mark_fn, hcfg.time_axis).bar_pass(theta, events, np.arange(n, dtype=float))
    dirs, ret = trade_label(vwap)
    scale = float(np.median(np.abs(np.diff(train_bars.vwap)))) or 1.0
    ok = ~np.isnan(ret) & (ret != 0) & (np.arange(n) >= 3)
    t = np.flatnonzero(ok)
    sgn = dirs[t]
    lam_dir = np.where(sgn > 0, lam[t, 0], lam[t, 1])
    lam_opp = np.where(sgn > 0, lam[t, 1], lam[t, 0])
    feats = {"y": (ret[t] > 0).astype(float), "ret": ret[t], "day": day}
    for k in range(3):
        feats[f"move_{k}"] = sgn * d[t - k] / scale  # signed move k bars back, in the trade direction
        feats[f"size_{k}"] = np.log1p(np.abs(d[t - k]) / scale)
        lam_dir_k = np.where(sgn > 0, lam[t - k, 0], lam[t - k, 1])
        lam_opp_k = np.where(sgn > 0, lam[t - k, 1], lam[t - k, 0])
        feats[f"asym_{k}"] = (lam_dir_k - lam_opp_k) / (lam_dir_k + lam_opp_k)
    feats["log_lam_total"] = np.log(lam_dir + lam_opp)
    return pd.DataFrame(feats)


# --------------------------------------------------------------------------- A2 / A3: pools and firing


def fold(day: date, which: str, seed: int = 0) -> dict:
    cfg = load_config(IMPROVED_CONFIG if which == "improved" else PAPER_CONFIG)
    run = next(r for r in expand_runs(cfg) if r.signal.name == which and r.rule == "momentum" and r.seed == seed)
    store = make_store(run.data)
    model = make_signal_model(run)
    evaluator = SpikeEvaluator(3)
    train_day = day - timedelta(days=1)
    model.fit([store.day_data(run.data.symbol, train_day)], seed)
    test_raw = store.day_data(run.data.symbol, day)
    sig = model.generate(test_raw, seed)
    train_t = model.train_days[0]
    learning = model.train_signals[0]
    frozen_run = model._simulate(train_t, seed, "train", learn=False)
    frozen_bars = np.flatnonzero(frozen_run.pop_counts("Out") > 0)
    out = {
        "day": day, "model": which,
        "rate_learning": learning.rate, "rate_frozen": len(frozen_bars) / train_t.n_bars, "rate_test": sig.rate,
        "acc_learning": evaluator.evaluate(train_t.bars.vwap, learning).accuracy,
        "acc_frozen": evaluator.evaluate(train_t.bars.vwap, frozen_bars).accuracy,
        "acc_test": evaluator.evaluate(model.transform(test_raw).bars.vwap, sig).accuracy,
    }
    if which != "improved":
        return {"rates": out}
    test_t = model.transform(test_raw)
    tau = max(1, int(round(model.cfg.rstdp.tau_z_bars)))
    diag = sig.diagnostics
    raw_d = diag["spikes_H_mom"].astype(np.int64) - diag["spikes_H_rev"].astype(np.int64)
    D = np.convolve(raw_d, np.ones(tau, dtype=np.int64))[: len(raw_d)]  # sum over bars t − τ + 1 … t
    _, ret = trade_label(test_t.bars.vwap)
    lab = evaluator.labels(test_t.bars.vwap)
    br = test_t.branching
    n_sizes = {p.name: p.size for p in model.network.populations}
    pools = {p: diag[f"spikes_{p}"] / n_sizes[p] for p in ("H1", "H2", "H_mom", "H_rev") if f"spikes_{p}" in diag}
    return {
        "rates": out,
        "signals": pd.DataFrame({"D": D[sig.bar_idx], "ret": ret[sig.bar_idx], "paper_eval": lab.evaluable[sig.bar_idx],
                                 "paper_mom": lab.momentum[sig.bar_idx]}),
        "all_bars": pd.DataFrame({"D": D, "ret": ret, "paper_eval": lab.evaluable, "paper_mom": lab.momentum}),
        "events": pd.DataFrame({"D": D[br["bar_idx"].to_numpy()], "M": br["momentum"].to_numpy()}),
        "grid": pd.DataFrame({"x1": test_t.channel_prob[:, 0], "x2": test_t.channel_prob[:, 1], **pools}),
    }


# --------------------------------------------------------------------------- A4: critique on E1/E2 test results


def lag_bucket(lag: np.ndarray) -> np.ndarray:
    return np.select([lag == 0, lag == 1, lag == 2, (lag >= 3) & (lag <= 5)], ["0", "1", "2", "3–5"], ">5")


def a4_day(day: date, signals: dict[str, np.ndarray], rates: dict[str, float], big_rates: tuple[float, ...], keep_rows: bool) -> dict:
    c = ctx()
    bars, train_bars = c.bar_cache.get(day), c.bar_cache.get(day - timedelta(days=1))
    vwap = bars.vwap
    n = len(vwap)
    d = diffs(vwap)
    ad = np.abs(np.nan_to_num(d))
    thr = float(np.quantile(np.abs(np.diff(train_bars.vwap)), 0.9))
    idx = np.arange(n)
    last_big = np.maximum.accumulate(np.where(ad > thr, idx, -1))
    lag = np.where(last_big >= 0, idx - last_big, 10**9)
    lab = SpikeEvaluator(3).labels(vwap)
    trades = per_signal(bars, idx, MomentumRule(3).directions(vwap, idx), ExecutionConfig())
    ret = np.where(trades["valid"], trades["net_return"], np.nan)
    sources = dict(signals)
    for name, rate in rates.items():
        big = BigMoveSignalModel(c.bars.vwap_num, target_rate=rate).fit([DayData(day=day, bars=train_bars)])
        sources[f"big-move at {name}'s rate"] = big.generate(DayData(day=day, bars=bars)).bar_idx
    lag_rows = []
    for name, b in sources.items():
        lag_rows.append(pd.DataFrame({"source": name, "bucket": lag_bucket(lag[b]), "eval": lab.evaluable[b],
                                      "real": lab.real[b], "ret": ret[b]}))
    decile = np.full(n, -1)
    decile[1:] = np.minimum((rankdata(ad[1:], method="ordinal") - 1) * 10 // (n - 1), 9)
    dec_rows = [pd.DataFrame({"source": "all bars", "decile": decile[lab.evaluable], "real": lab.real[lab.evaluable]})]
    for name in signals:
        b = signals[name]
        ok = lab.evaluable[b]
        dec_rows.append(pd.DataFrame({"source": name, "decile": decile[b][ok], "real": lab.real[b][ok]}))
    s = np.sign(np.nan_to_num(d[1:]))
    a = ad[1:]
    ac = {}
    for L in (1, 2, 3, 5, 10, 20, 50):
        ac[("sign", L)] = np.corrcoef(s[:-L], s[L:])[0, 1]
        ac[("abs", L)] = np.corrcoef(a[:-L], a[L:])[0, 1]
    curve = []
    for r in big_rates:
        big = BigMoveSignalModel(c.bars.vwap_num, target_rate=r).fit([DayData(day=day, bars=train_bars)])
        b = big.generate(DayData(day=day, bars=bars)).bar_idx
        m = SpikeEvaluator.metrics(lab, b)
        curve.append({"target_rate": r, "rate": m.signal_rate, "accuracy": m.accuracy, "chance": m.base_accuracy})
    out = {"lags": pd.concat(lag_rows), "deciles": pd.concat(dec_rows), "autocorr": ac, "curve": pd.DataFrame(curve)}
    if keep_rows:
        scale = float(np.median(np.abs(np.diff(train_bars.vwap)))) or 1.0
        f = np.log1p(ad / scale)
        roll = pd.Series(ad).rolling(20, min_periods=1).mean().to_numpy()
        ok = lab.evaluable & (idx >= 2)
        rows = {"y": lab.real[ok].astype(float), "d0": f[ok], "d1": f[np.maximum(idx - 1, 0)][ok],
                "d2": f[np.maximum(idx - 2, 0)][ok], "vol": np.log1p(roll / scale)[ok]}
        for name, b in signals.items():
            ind = np.zeros(n)
            ind[b] = 1.0
            rows[f"sig_{name}"] = ind[ok]
        out["reg"] = pd.DataFrame(rows)
    return out


def logit(X: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray, float]:
    """Logistic regression by Newton–Raphson: (coefficients, standard errors, log-likelihood)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    return beta, se, ll


# --------------------------------------------------------------------------- figures


def heatmaps(grid: pd.DataFrame, out: Path) -> pd.DataFrame:
    edges = {}
    for col in ("x1", "x2"):
        qs = np.unique(np.quantile(grid[col], [0, 0.5, 0.75, 0.9, 0.97, 1.0]))
        edges[col] = qs
    g = grid.assign(b1=pd.cut(grid.x1, edges["x1"], include_lowest=True), b2=pd.cut(grid.x2, edges["x2"], include_lowest=True))
    pools = [p for p in ("H_mom", "H_rev", "H1", "H2") if p in g]
    fig, axes = plt.subplots(1, len(pools), figsize=(3.6 * len(pools), 3.4), dpi=150, facecolor=SURFACE)
    tables = []
    vmax = max(g.groupby(["b1", "b2"], observed=True)[p].mean().max() for p in pools)
    for ax, pool in zip(axes, pools):
        mat = g.groupby(["b2", "b1"], observed=True)[pool].mean().unstack()
        ax.imshow(mat.to_numpy(), origin="lower", cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(mat.to_numpy()):
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK if v < 0.6 * vmax else "white")
        ax.set_xticks(range(mat.shape[1]), [f"{iv.left:.2f}–{iv.right:.2f}" for iv in mat.columns], rotation=45, fontsize=6, color=INK_2)
        ax.set_yticks(range(mat.shape[0]), [f"{iv.left:.2f}–{iv.right:.2f}" for iv in mat.index], fontsize=6, color=INK_2)
        ax.set_xlabel("X1 input prob (λ_u)", fontsize=7, color=INK_2)
        ax.set_ylabel("X2 input prob (λ_d)", fontsize=7, color=INK_2)
        ax.set_title(pool, loc="left", fontsize=9, color=INK)
        label = lambda iv: f"{iv.left:.2f}–{iv.right:.2f}"  # noqa: E731
        tables.append(mat.rename(index=label, columns=label).rename_axis(index=f"{pool}: X2 \\ X1"))
    fig.suptitle("Mean spikes per neuron per bar by input level (validation days, tuned improved model)", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return tables


def curve_figure(curve: pd.DataFrame, points: pd.DataFrame, chance: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(curve["rate"] * 100, curve["accuracy"] * 100, color="#2a78d6", lw=2, marker="o", ms=4, label="big-move filter", zorder=3)
    ax.axhline(chance * 100, color=INK_2, lw=0.8, ls="--", zorder=2)
    ax.text(curve["rate"].max() * 100, chance * 100 + 0.3, "random timing", ha="right", fontsize=7, color=INK_2)
    for _, p in points.iterrows():
        ax.scatter(p["rate"] * 100, p["accuracy"] * 100, s=40, color="#eb6834", zorder=4, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(p["label"], (p["rate"] * 100, p["accuracy"] * 100), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("signals as share of bars (%, log scale)", fontsize=8, color=INK_2)
    ax.set_ylabel("spike accuracy (%)", fontsize=8, color=INK_2)
    ax.set_title("Spike accuracy vs signal rate on the E1/E2 test days", loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8, zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--csv-dir", type=Path, default=Path("results/typed_premise_checks"))
    args = parser.parse_args(argv)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    c = ctx()
    val_days = c.periods.validation.days()
    test_days = c.periods.test.days()
    t0 = time.time()
    pool = ProcessPoolExecutor(args.workers)

    # ---------------- A1
    a1 = list(pool.map(a1_day, *zip(*[(d, q) for q in QUANTILES for d in val_days])))
    ev = pd.concat([x[0] for x in a1], ignore_index=True)
    base = pd.DataFrame([x[1] for x in a1])
    ev.to_parquet(args.csv_dir / "a1_events.parquet", index=False)
    decided = ev[~ev["ret"].isna() & (ev["ret"] != 0)]
    auc_rows, dec_tables, best_auc, best_feat, fade_positive, inverted = [], {}, 0.0, "", False, []
    for q in QUANTILES:
        e, pe = decided[decided.q == q], ev[(ev.q == q) & ev.paper_eval]
        row = {"event quantile": q, "events per day": f"{(ev.q == q).sum() / len(val_days):,.0f}",
               "reversion share (trade label)": fmt(float((e.ret < 0).mean()))}
        for feat, name in (("M", "M"), ("p_same", "p_same"), ("p_cross", "p_cross"), ("asym", "λ asymmetry")):
            a_trade, a_paper = auc(e[feat].to_numpy(), (e.ret > 0).to_numpy()), auc(pe[feat].to_numpy(), pe.paper_mom.to_numpy())
            row[f"{name}: trade"] = fmt(a_trade)
            row[f"{name}: paper"] = fmt(a_paper)
            expected = 1 - a_trade if feat == "p_cross" else a_trade  # p_cross should be higher for reversion
            if expected > best_auc:
                best_auc, best_feat = expected, f"{name} (quantile {q})"
            if expected < 0.47:
                inverted.append(f"{name} at quantile {q}: {a_trade:.3f}")
        auc_rows.append(row)
        dec = e.assign(decile=pd.qcut(e.M.rank(method="first"), 10, labels=False))
        g = dec.groupby("decile")
        table = pd.DataFrame({
            "M decile": g.size().index + 1, "events": g.size().values,
            "M range": [f"{a:+.2f} … {b:+.2f}" for a, b in zip(g.M.min(), g.M.max())],
            "P(follow wins)": g.ret.apply(lambda r: (r > 0).mean()).map("{:.3f}".format).values,
            "follow bp": (g.ret.mean() * 1e4).map("{:+.3f}".format).values,
            "fade bp": (-g.ret.mean() * 1e4).map("{:+.3f}".format).values,
        })
        fade_positive |= bool((g.ret.mean() < 0).any())
        dec_tables[q] = table
    all_rev = base.groupby("q")[["decided_bars", "reversion_bars", "paper_eval_bars", "paper_reversion_bars"]].sum()

    # ---------------- A1b: is the trade label predictable from causal inputs at all?
    a1b = pd.concat(list(pool.map(a1b_day, val_days)), ignore_index=True)
    feat_cols = [col for col in a1b.columns if col not in ("y", "ret", "day")]
    first = a1b.day < val_days[len(val_days) // 2]
    mu, sd = a1b.loc[first, feat_cols].mean(), a1b.loc[first, feat_cols].std().replace(0, 1)
    Z = ((a1b[feat_cols] - mu) / sd).clip(-10, 10).to_numpy()
    Xall = np.column_stack([np.ones(len(a1b)), Z])
    b_fit, _, _ = logit(Xall[first.to_numpy()], a1b.y.to_numpy()[first.to_numpy()])
    score = Xall @ b_fit
    a1b_rows = [{"sample": "fit half (first 22 days)", "AUC": fmt(auc(score[first.to_numpy()], a1b.y.to_numpy()[first.to_numpy()] > 0.5)),
                 "bars": f"{int(first.sum()):,}"},
                {"sample": "held-out half (last 23 days)", "AUC": fmt(auc(score[~first.to_numpy()], a1b.y.to_numpy()[~first.to_numpy()] > 0.5)),
                 "bars": f"{int((~first).sum()):,}"}]
    single = [{"feature": col, "AUC (all days)": fmt(auc(a1b[col].to_numpy(), a1b.y.to_numpy() > 0.5))} for col in feat_cols]
    # Value of typing on the held-out half: follow or fade by the fitted classifier vs always follow.
    held = ~first.to_numpy()
    r_h, p_h = a1b.ret.to_numpy()[held] * 1e4, (1 / (1 + np.exp(-score)))[held]
    cut10 = np.quantile(p_h, 0.10)
    typing_rows = [
        {"momentum strategy, held-out validation bars": "always follow (untyped)", "bp per trade": f"{r_h.mean():+.3f}", "trades faded": "0 %"},
        {"momentum strategy, held-out validation bars": "fade when the classifier says reversion (p < 0.5)",
         "bp per trade": f"{np.where(p_h >= 0.5, r_h, -r_h).mean():+.3f}", "trades faded": f"{np.mean(p_h < 0.5):.0%}"},
        {"momentum strategy, held-out validation bars": "fade the 10 % most reversion-like bars",
         "bp per trade": f"{np.where(p_h >= cut10, r_h, -r_h).mean():+.3f}", "trades faded": "10 %"},
        {"momentum strategy, held-out validation bars": "oracle: true type known (not causal, upper bound)",
         "bp per trade": f"{np.abs(r_h).mean():+.3f}", "trades faded": f"{np.mean(r_h < 0):.0%}"},
    ]

    # ---------------- A2 / A3
    folds = list(pool.map(fold, *zip(*[(d, m) for m in ("paper", "improved") for d in val_days])))
    rates = pd.DataFrame([f["rates"] for f in folds])
    imp = [f for f in folds if "signals" in f]
    sigs = pd.concat([f["signals"] for f in imp], ignore_index=True)
    allb = pd.concat([f["all_bars"] for f in imp], ignore_index=True)
    evs = pd.concat([f["events"] for f in imp], ignore_index=True)
    grid = pd.concat([f["grid"] for f in imp], ignore_index=True)
    a2 = []
    for where, df in (("at output signals", sigs), ("at all bars", allb)):
        dd = df[~df.ret.isna() & (df.ret != 0)]
        pe = df[df.paper_eval]
        a2.append({"where": where, "AUC vs trade label": fmt(auc(dd.D.to_numpy(), (dd.ret > 0).to_numpy())),
                   "AUC vs paper label": fmt(auc(pe.D.to_numpy(), pe.paper_mom.to_numpy())), "n": f"{len(df):,}"})
    ne = evs[evs.M != 0]
    a2.append({"where": "at Hawkes events", "AUC vs sign(M)": fmt(auc(ne.D.to_numpy(), (ne.M > 0).to_numpy())),
               "Spearman(D, M)": fmt(spearmanr(evs.D, evs.M)[0]), "n": f"{len(evs):,}"})
    heat_png = args.out.with_name(args.out.stem + "_pools.png")
    heat_tables = heatmaps(grid, heat_png)
    rate_tab = rates.groupby("model")[["rate_learning", "rate_frozen", "rate_test", "acc_learning", "acc_frozen", "acc_test"]].mean()

    # ---------------- A4 (test period, critique only)
    sig_files = {
        "paper SNN": "results/e1_paper_baseline/paper__momentum__Wsnn1__seed0/signals.parquet",
        "improved": "results/e2_improved/improved__momentum__Wsnn1__Wh1__seed0/signals.parquet",
    }
    by_day = {name: {k: g.bar_idx.to_numpy() for k, g in pd.read_parquet(f).groupby("day")} for name, f in sig_files.items()}
    n_bars = {d: len(c.bar_cache.get(d)) for d in test_days}
    big_rates = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.25, 0.4)
    tasks = []
    for i, d in enumerate(test_days):
        sig = {name: by_day[name].get(pd.Timestamp(d).date(), by_day[name].get(d, np.empty(0, np.int64))) for name in sig_files}
        tasks.append((d, sig, {n: len(s) / n_bars[d] for n, s in sig.items()}, big_rates, i % 5 == 0))
    a4 = list(pool.map(a4_day, *zip(*tasks)))
    pool.shutdown()
    lags = pd.concat([x["lags"] for x in a4], ignore_index=True)
    lag_tab = []
    for name, g in lags.groupby("source", sort=False):
        for bucket in LAG_BUCKETS:
            b = g[g.bucket == bucket]
            lag_tab.append({"signals": name, "bars since big move": bucket, "share": f"{len(b) / len(g):.3f}",
                            "real-spike rate": fmt(float(b[b["eval"]].real.mean()) if b["eval"].any() else math.nan),
                            "bp per trade": fmt(float(b.ret.mean() * 1e4) if b.ret.notna().any() else math.nan)})
    decs = pd.concat([x["deciles"] for x in a4], ignore_index=True)
    dec_acc = decs.groupby(["decile", "source"]).real.mean().unstack()
    dec_share = decs[decs.source != "all bars"].groupby(["source", "decile"]).size().unstack(0)
    dec_share = dec_share / dec_share.sum()
    ac = pd.DataFrame([x["autocorr"] for x in a4]).mean()
    curve = pd.concat([x["curve"] for x in a4]).groupby("target_rate").mean()
    reg = pd.concat([x["reg"] for x in a4 if "reg" in x], ignore_index=True)
    feats = ["d0", "d1", "d2", "vol"]
    Xb = np.column_stack([np.ones(len(reg)), reg[feats].to_numpy()])
    y = reg.y.to_numpy()
    beta0, _, ll0 = logit(Xb, y)
    p0 = 1 / (1 + np.exp(-(Xb @ beta0)))
    reg_rows = []
    for name in sig_files:
        X = np.column_stack([Xb, reg[f"sig_{name}"].to_numpy()])
        beta, se, ll1 = logit(X, y)
        p1 = 1 / (1 + np.exp(-(X @ beta)))
        reg_rows.append({"model": name, "signal coefficient": f"{beta[-1]:+.4f}", "std. error": f"{se[-1]:.4f}",
                         "z": f"{beta[-1] / se[-1]:+.1f}", "LR statistic": f"{2 * (ll1 - ll0):.1f}",
                         "AUC without → with signal": f"{auc(p0, y > 0.5):.4f} → {auc(p1, y > 0.5):.4f}"})
    t3 = {e: pd.read_csv(f"results/{e}/table3.csv") for e in ("e1_paper_baseline", "e2_improved")}
    pts = []
    for e, m, label in (("e1_paper_baseline", "paper", "paper SNN"), ("e2_improved", "improved", "improved"),
                        ("e2_improved", "a1_no_rstdp", "A1"), ("e2_improved", "improved_rate005", "improved @0.05"),
                        ("e2_improved", "paper_rate03", "paper @0.3")):
        r = t3[e][(t3[e].model == m) & (t3[e].split == "test") & (t3[e].source == "model")].iloc[0]
        pts.append({"label": label, "rate": r.signal_rate, "accuracy": r.spike_accuracy})
    points = pd.DataFrame(pts)
    curve_png = args.out.with_name(args.out.stem + "_rate_curve.png")
    curve_figure(curve, points, float(curve.chance.mean()), curve_png)

    # ---------------- gate and report
    gate_ok = best_auc >= 0.53 and fade_positive
    ac_tab = pd.DataFrame({"lag (bars)": [1, 2, 3, 5, 10, 20, 50],
                           "autocorr sign(d)": [fmt(ac[("sign", L)]) for L in (1, 2, 3, 5, 10, 20, 50)],
                           "autocorr |d|": [fmt(ac[("abs", L)]) for L in (1, 2, 3, 5, 10, 20, 50)]})
    dec_tab = pd.DataFrame({"|d_t| decile": dec_acc.index + 1,
                            **{f"accuracy: {c_}": dec_acc[c_].map("{:.3f}".format) for c_ in dec_acc.columns},
                            **{f"share of signals: {c_}": dec_share[c_].map("{:.3f}".format).values for c_ in dec_share.columns}})
    curve_tab = pd.DataFrame({"target rate": curve.index, "realised rate": curve.rate.map("{:.4f}".format),
                              "big-move accuracy": curve.accuracy.map("{:.4f}".format)})
    rate_rows = pd.DataFrame({"model": rate_tab.index,
                              **{col: rate_tab[col].map("{:.4f}".format).values for col in rate_tab.columns}})
    text = f"""# Premise checks for a momentum/reversion model (U10, Phase A)

100 aggTrades per bar. A1–A3: validation days {val_days[0]} → {val_days[-1]} ({len(val_days)} days), W_h = 1,
W_snn = 1, seed 0 — these inform the design. A4: the existing E1/E2 test results (seed 0 signals),
for the critique only. Runtime {time.time() - t0:.0f} s. Raw data: `{args.csv_dir}/`.

**Trade label** at bar t (paper execution): dir_t = MomentumRule(3) direction,
follow_ret_t = dir_t · (vwap[t+4] / vwap[t+1] − 1); *momentum* = following wins (> 0),
*reversion* = fading wins (< 0); bars with dir_t = 0, no exit bar or a zero return are excluded.
**Paper label**: mom_rev_flag (§8.4), momentum ⇔ flag ≤ 0.

## Gate

- Best Hawkes feature AUC on the trade label, in its expected direction: **{best_auc:.3f}**, {best_feat} (rule: ≥ 0.53).
- Features clearly in the opposite direction (AUC below 0.47 in the expected orientation): {"; ".join(inverted) if inverted else "none"}.
- Fading has a positive mean return in at least one M decile: **{"yes" if fade_positive else "no"}**.
- Suggested rule {"**passed**" if gate_ok else "**not passed**"}.

## A1 — Hawkes teacher check (at Hawkes event bars)

AUC > 0.5 means the feature is higher for momentum. For p_cross the expected direction is
reversed (higher for reversion), so its AUC should lie below 0.5.

{markdown(pd.DataFrame(auc_rows))}

Reversion base rate under the trade label over all decided bars:
{", ".join(f"q = {q}: {all_rev.loc[q, 'reversion_bars'] / all_rev.loc[q, 'decided_bars']:.3f}" for q in QUANTILES[:1])}
(paper label: {all_rev.iloc[0]['paper_reversion_bars'] / all_rev.iloc[0]['paper_eval_bars']:.3f}).

""" + "\n".join(f"### M deciles, event quantile {q}\n\n{markdown(dec_tables[q])}\n" for q in QUANTILES) + f"""
## A1b — is the trade label predictable from causal inputs at all?

The typed network would see λ_u, λ_d (Hawkes intensities at t⁺) and, through them, recent moves.
Logistic regression of "following wins" on features oriented by the trade direction dir_t: signed
moves and move sizes at t, t−1, t−2, the intensity asymmetry at t, t−1, t−2 and the log total
intensity (all bars with a decided trade label, event quantile 0.9). Fitted on the first half of
the validation days, evaluated on the second half. An AUC near 0.5 means no causal feature set of
this kind separates momentum from reversion, whatever teacher trains the network.

{markdown(pd.DataFrame(a1b_rows))}

Single features:

{markdown(pd.DataFrame(single))}

Value of typing (fee-free, paper execution): how much a typed strategy could earn over always
following if its type came from this classifier, compared with a perfect (non-causal) type.

{markdown(pd.DataFrame(typing_rows))}

## A2 — pool check (tuned improved model, validation folds)

D = spikes(H_mom) − spikes(H_rev) summed over the last τ_z bars (τ_z = 1 bar for the tuned model).

{markdown(pd.DataFrame(a2))}

Mean spikes per neuron per bar by input level (rows: X2 = λ_d-encoded input, columns: X1 = λ_u-encoded input):

![pool firing by input level]({heat_png.name})

""" + "\n".join(f"{markdown(t.map(lambda v: '' if math.isnan(v) else f'{v:.3f}').reset_index().rename(columns=str))}\n" for t in heat_tables) + f"""
## A3 — firing rate during learning vs with the final weights

Each training day re-simulated with the final (frozen) weights and the same input spikes.

{markdown(rate_rows)}

## A4 — critique on the E1/E2 test results (not used for design)

### Bars since the last big move (|d| above the training day's 0.9 quantile)

{markdown(pd.DataFrame(lag_tab))}

### Autocorrelation by lag (100 trades per bar)

{markdown(ac_tab)}

### Accuracy within |d_t| deciles (a signal adds information if it beats "all bars" in its own decile)

{markdown(dec_tab)}

### Logistic regression of the real-spike label (every 5th test day, evaluable bars)

Baseline features: log(1 + |d_t|/s), log(1 + |d_t−1|/s), log(1 + |d_t−2|/s), log(1 + 20-bar mean |d|/s), s = training-day
median |d|. The signal indicator is added on top; a positive, significant coefficient means the signal
carries information beyond these features.

{markdown(pd.DataFrame(reg_rows))}

### Accuracy vs signal rate

{markdown(curve_tab)}

![accuracy vs signal rate]({curve_png.name})
"""
    args.out.write_text(text)
    print(text[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
