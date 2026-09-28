"""Hawkes fits on real validation days (decision U6 review after M6).

    python -m experiments.hawkes_review --config experiments/configs/base.yaml --out docs/reports/m6_hawkes_review.md

For every validation day d and window W_h, θ_d = fit(days [d − W_h, d − 1]) on the bar clock,
then applied to day d. Reports event occupancy, parameters, ρ(A), the causal branching split
(background / same direction / opposite direction), the spread of the momentum score
M = p_same − p_cross that drives the R-STDP rewards, next-day out-of-sample log-likelihood,
and whether the intensity carries information about the next moves beyond |d_t|.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from snn_hft.analysis.compare import markdown
from snn_hft.backtest.cache import BarCache, HawkesParamCache
from snn_hft.config.loader import load_raw
from snn_hft.config.schema import BarsConfig, DataConfig, HawkesConfig, OutputConfig, PeriodsConfig
from snn_hft.data.containers import day_bounds_ns
from snn_hft.data.store import make_store
from snn_hft.evaluation.spike_metrics import SpikeEvaluator
from snn_hft.models.hawkes.events import bar_times, events_from_bars
from snn_hft.models.hawkes.process import BivariateHawkesProcess
from snn_hft.models.hawkes.provider import HawkesParamProvider


def _setup(raw: dict):
    data = DataConfig.model_validate(raw.get("data", {}))
    bars = BarsConfig.model_validate(raw.get("bars", {}))
    out = OutputConfig.model_validate(raw.get("output", {}))
    store = make_store(data)
    bar_cache = BarCache(store, data.symbol, bars.vwap_num, out.cache_dir)
    params_cache = HawkesParamCache(out.cache_dir, store.source.venue, data.symbol, data.dataset, bars.vwap_num)
    return data, bars, bar_cache, params_cache


def _day_events(bar_cache: BarCache, day: date, time_axis: str):
    bars = bar_cache.get(day)
    vwap = bars.vwap
    d = np.full(len(vwap), np.nan)
    d[1:] = np.diff(vwap)
    ts_end = bars.df["ts_end_ns"].to_numpy()
    start = day_bounds_ns(day)[0]
    return bars, d, events_from_bars(d, ts_end, start, time_axis), bar_times(len(vwap), ts_end, start, time_axis)


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman correlation of x and y after removing the rank-linear effect of `control` from both."""
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(control)
    design = np.column_stack([np.ones_like(rc), rc])
    res_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def review_day(raw: dict, day: date, w_h: int, hawkes: dict, eval_window: int) -> dict:
    data, bars_cfg, bar_cache, params_cache = _setup(raw)
    cfg = HawkesConfig.model_validate(hawkes)
    provider = HawkesParamProvider(cfg, w_h, lambda d: _day_events(bar_cache, d, cfg.time_axis)[2], params_cache)
    t0 = time.perf_counter()
    theta = provider.params_for(day)
    fit_s = time.perf_counter() - t0
    bars, d, events, times = _day_events(bar_cache, day, cfg.time_axis)
    proc = BivariateHawkesProcess(cfg.mark_fn, cfg.time_axis)
    lam_minus, lam_plus, br = proc.bar_pass(theta, events, times)
    momentum = br[:, 1] - br[:, 2]
    oos = proc.loglik(theta, [events]) / max(len(events), 1)
    labels = SpikeEvaluator(eval_window).labels(bars.vwap)
    ok = labels.evaluable
    strength = labels.strength[ok]
    lam_total = lam_plus[ok].sum(axis=1)
    abs_d = np.abs(np.nan_to_num(d))[ok]
    p = theta.named()
    return {
        "day": day, "w_h": w_h, "fit_s": fit_s, "bars": len(d), "events": len(events),
        "occupancy": len(events) / len(d), "up_share": float((events.types == 0).mean()),
        **p, "rho": theta.spectral_radius, "converged": theta.fit.get("converged"),
        "A_uu": theta.branching_matrix[0, 0], "A_ud": theta.branching_matrix[0, 1],
        "A_du": theta.branching_matrix[1, 0], "A_dd": theta.branching_matrix[1, 1],
        "p_bg": br[:, 0].mean(), "p_same": br[:, 1].mean(), "p_cross": br[:, 2].mean(),
        "M_mean": momentum.mean(), "M_std": momentum.std(), "M_p10": np.quantile(momentum, 0.1),
        "M_p90": np.quantile(momentum, 0.9), "M_abs_gt_0.2": float((np.abs(momentum) > 0.2).mean()),
        "oos_loglik_per_event": oos, "insample_loglik_per_event": theta.fit.get("loglik_per_event"),
        "rho_s_lambda_strength": spearmanr(lam_total, strength)[0],
        "rho_s_absd_strength": spearmanr(abs_d, strength)[0],
        "partial_lambda_strength_given_absd": partial_spearman(lam_total, strength, abs_d),
        "lambda_cv": float(lam_plus.sum(axis=1).std() / lam_plus.sum(axis=1).mean()),
    }


def variant_day(raw: dict, day: date, time_axis: str, quantile: float | None, eval_window: int) -> dict:
    """Exploration of event definitions not in the spec (W_h = 1): a causal threshold on |d_t|
    taken as a quantile of the fit day's |d|, applied to fit and test day alike."""
    from snn_hft.models.hawkes.process import DayEvents

    _, _, bar_cache, _ = _setup(raw)
    fit_day = day - pd.Timedelta(days=1).to_pytimedelta()

    def events(dd, thr):
        bars, d, ev, times = _day_events(bar_cache, dd, time_axis)
        if thr is not None:
            keep = ev.marks > thr
            ev = DayEvents(ev.times[keep], ev.types[keep], ev.marks[keep], ev.horizon, ev.bar_idx[keep])
        return bars, d, ev, times

    _, d_fit, _, _ = _day_events(bar_cache, fit_day, time_axis)
    thr = None if quantile is None else float(np.quantile(np.abs(d_fit[1:]), quantile))
    proc = BivariateHawkesProcess("linear_normalized", time_axis)
    theta = proc.fit([events(fit_day, thr)[2]], restarts=5, rng=np.random.default_rng(0))
    bars, d, ev, times = events(day, thr)
    _, lam_plus, br = proc.bar_pass(theta, ev, times)
    momentum = br[:, 1] - br[:, 2]
    labels = SpikeEvaluator(eval_window).labels(bars.vwap)
    ok = labels.evaluable
    lam = lam_plus[ok].sum(axis=1)
    abs_d = np.abs(np.nan_to_num(d))[ok]
    return {
        "time_axis": time_axis, "events": "every move" if quantile is None else f"abs d above q{int(quantile * 100)}",
        "event_share": len(ev) / len(d), "rho": theta.spectral_radius, "p_bg": br[:, 0].mean(),
        "p_same": br[:, 1].mean(), "p_cross": br[:, 2].mean(), "M_std": momentum.std(),
        "M_abs_gt_0.2": float((np.abs(momentum) > 0.2).mean()),
        "lambda_cv": float(lam_plus.sum(axis=1).std() / lam_plus.sum(axis=1).mean()),
        "sp_lambda": spearmanr(lam, labels.strength[ok])[0], "sp_absd": spearmanr(abs_d, labels.strength[ok])[0],
        "partial": partial_spearman(lam, labels.strength[ok], abs_d),
    }


def fmt(df: pd.DataFrame, cols: dict[str, str], digits: int = 3) -> pd.DataFrame:
    out = pd.DataFrame({"W_h": df.index})
    for col, label in cols.items():
        out[label] = [f"{v:.{digits}f}" for v in df[col]]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("results/m6_hawkes_review.csv"))
    parser.add_argument("--variant-days", type=int, default=10, help="validation days for the event-definition comparison")
    args = parser.parse_args(argv)

    raw = load_raw(args.config)
    periods = PeriodsConfig.model_validate(raw.get("periods", {}))
    hawkes = HawkesConfig().model_dump(mode="json")
    days = periods.validation.days()
    tasks = [(raw, d, w, hawkes, 3) for w in args.windows for d in days]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(review_day, *zip(*tasks)))
    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    variants = [("bar_index", None), ("wallclock", None), ("bar_index", 0.8), ("bar_index", 0.9), ("wallclock", 0.9)]
    vtasks = [(raw, d, axis, q, 3) for axis, q in variants for d in days[: args.variant_days]]
    with ProcessPoolExecutor(args.workers) as pool:
        vdf = pd.DataFrame(list(pool.map(variant_day, *zip(*vtasks))))
    vdf.to_csv(args.csv.with_name(args.csv.stem + "_variants.csv"), index=False)
    vby = vdf.groupby(["time_axis", "events"], sort=False).mean(numeric_only=True).reset_index()
    vtable = pd.DataFrame({"time axis": vby["time_axis"], "events": vby["events"]})
    for col, label in {"event_share": "event share of bars", "rho": "ρ(A)", "p_bg": "p_bg", "p_same": "p_same",
                       "p_cross": "p_cross", "M_std": "M std", "M_abs_gt_0.2": "share abs(M) > 0.2",
                       "lambda_cv": "CV of λ", "sp_lambda": "Spearman(λ⁺, next moves)",
                       "sp_absd": "Spearman(abs d_t, next moves)", "partial": "partial (λ⁺ given abs d_t)"}.items():
        vtable[label] = vby[col].map("{:.3f}".format)

    by = df.groupby("w_h").mean(numeric_only=True)
    by_std = df.groupby("w_h").std(numeric_only=True)
    cv = (by_std / by.abs())
    bars_cfg = BarsConfig.model_validate(raw.get("bars", {}))

    params_cols = {c: c for c in ("mu_u", "mu_d", "alpha_uu", "beta_uu", "alpha_ud", "beta_ud", "alpha_du", "beta_du", "alpha_dd", "beta_dd")}
    text = f"""# M6 — Hawkes fits on real validation days (U6 review)

Setup: BTCUSDT `aggTrades`, {bars_cfg.vwap_num} trades per bar, bar clock, `LinearNormalized` marks,
5 restarts; θ_d = fit(days d − W_h … d − 1), applied to day d. Validation days
{days[0]} → {days[-1]} ({len(days)} days × {len(args.windows)} windows, {time.time() - t0:.0f} s on {args.workers} workers).
Means over days; per-day values in `{args.csv}`.

## Events and fit

{markdown(fmt(by, {"bars": "bars/day", "events": "events/day", "occupancy": "event share of bars", "up_share": "up share", "rho": "ρ(A)", "fit_s": "fit seconds"}, 3))}

All fits converged: {bool(df["converged"].all())}.

## Parameters (mean over days) and branching matrix A = α/β

{markdown(fmt(by, params_cols, 3))}

{markdown(fmt(by, {"A_uu": "A_uu (up→up)", "A_ud": "A_ud (down→up)", "A_du": "A_du (up→down)", "A_dd": "A_dd (down→down)"}, 3))}

Parameter stability across consecutive days (coefficient of variation, std / mean):

{markdown(fmt(cv, {**params_cols, "rho": "ρ(A)"}, 2))}

## Branching split and momentum score (drives the R-STDP rewards)

{markdown(fmt(by, {"p_bg": "background p_bg", "p_same": "same direction p_same", "p_cross": "opposite p_cross", "M_mean": "M mean", "M_std": "M std", "M_p10": "M 10 %", "M_p90": "M 90 %", "M_abs_gt_0.2": "share abs(M) > 0.2"}, 3))}

## Next-day fit and information content

{markdown(fmt(by, {"insample_loglik_per_event": "in-sample loglik/event", "oos_loglik_per_event": "next-day loglik/event", "lambda_cv": "CV of λ_u+λ_d over bars", "rho_s_lambda_strength": "Spearman(λ⁺, next moves)", "rho_s_absd_strength": "Spearman(abs d_t, next moves)", "partial_lambda_strength_given_absd": "partial Spearman(λ⁺ given abs d_t)"}, 3))}

*Next moves* = the paper's spike strength S_strength (mean absolute return over the next 3 bars).
The partial correlation measures what the intensity adds beyond the current price change,
i.e. the memory Problem 1 is about.

## Alternative event definitions (first {args.variant_days} validation days, W_h = 1)

Not in the spec except `wallclock`. The threshold variants count a bar as an event only if
abs(d_t) exceeds the given quantile of abs(d) on the fit day (causal; frozen with θ_d).

{markdown(vtable)}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
