"""Does the price difference remove the day trend (paper Step 2)? And what about intraday scale?

    python -m experiments.differencing_check --out docs/reports/differencing_check.md

Validation days, 100 trades per bar. Measures the trend left in d_t = V_t − V_{t−1} relative to its
noise, and the hour-of-day profile of |d_t|, of the paper SNN's input and of its signal rate
(tuned E1 model, fitted on the previous day, seed 0).
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from snn_hft.analysis.compare import markdown
from snn_hft.backtest.cache import BarCache
from snn_hft.config import expand_runs, load_config
from snn_hft.data.containers import day_bounds_ns
from snn_hft.data.store import make_store
from snn_hft.signals.factory import make_signal_model

HOUR_NS = 3_600_000_000_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = load_config("experiments/configs/paper_baseline.yaml")
    run = next(r for r in expand_runs(cfg) if r.rule == "momentum" and r.seed == 0)
    store = make_store(cfg.data)
    bar_cache = BarCache(store, cfg.data.symbol, cfg.bars.vwap_num, cfg.output.cache_dir)
    days = cfg.periods.validation.days()
    trend_rows, hourly = [], []
    for day in days:
        bars = bar_cache.get(day)
        v = bars.vwap
        d = np.diff(v)
        t = np.arange(len(d))
        line = np.linspace(v[0], v[-1], len(v))
        trend_rows.append({
            "daily move (%)": (v[-1] / v[0] - 1) * 100,
            "drift per bar = mean d (USD)": d.mean(),
            "std of d (USD)": d.std(),
            "abs(mean d) / std d": abs(d.mean()) / d.std(),
            "share of d variance due to drift": d.mean() ** 2 / (d.mean() ** 2 + d.var()),
            "R² of d on time": np.corrcoef(t, d)[0, 1] ** 2,
            "R² of the level on a straight trend": 1 - ((v - line) ** 2).sum() / ((v - v.mean()) ** 2).sum(),
            "lag-1 autocorr of the level": np.corrcoef(v[:-1], v[1:])[0, 1],
            "lag-1 autocorr of d": np.corrcoef(d[:-1], d[1:])[0, 1],
        })
        model = make_signal_model(run)
        model.fit([store.day_data(cfg.data.symbol, day - timedelta(days=1))], run.seed)
        raw = store.day_data(cfg.data.symbol, day)
        sig = model.generate(raw, run.seed)
        tday = model.transform(raw)
        hours = (tday.bars.df["ts_end_ns"].to_numpy() - day_bounds_ns(day)[0]) // HOUR_NS
        fired = np.zeros(tday.n_bars)
        fired[sig.bar_idx] = 1
        absd = np.abs(np.nan_to_num(tday.d))
        hourly.append(pd.DataFrame({"hour": hours[1:], "abs_d": absd[1:] / absd[1:].mean(), "signal": fired[1:],
                                    "input": tday.channel_prob[1:].sum(axis=1)}))
    trend = pd.DataFrame(trend_rows).describe().loc[["mean", "min", "max"]].T
    trend_tab = pd.DataFrame({"measure (per day)": trend.index, **{c: trend[c].map("{:.4f}".format).values for c in trend.columns}})
    h = pd.concat(hourly)
    g = h.groupby("hour")
    prof = pd.DataFrame({"UTC hour": g.size().index, "bars per day": (g.size() / len(days)).map("{:,.0f}".format).values,
                         "mean abs(d) / day mean": g.abs_d.mean().map("{:.3f}".format).values,
                         "input prob X1 + X2": g.input.mean().map("{:.3f}".format).values,
                         "SNN signal rate": g.signal.mean().map("{:.3f}".format).values})
    ratio_d, ratio_s = g.abs_d.mean().max() / g.abs_d.mean().min(), g.signal.mean().max() / g.signal.mean().min()
    bars_ratio = g.size().max() / g.size().min()
    corr = np.corrcoef(g.abs_d.mean(), g.signal.mean())[0, 1]
    text = f"""# Price-difference check (paper Step 2)

Validation days {days[0]} → {days[-1]} ({len(days)} days), 100 aggTrades per bar. The paper differences
the vwap to (1) remove the intraday trend and (2) remove a "magnitude bias" between trading hours.

## 1. Trend

{markdown(trend_tab)}

- The drift that a day's trend leaves in d_t is below 2 % of the standard deviation of d_t on every day
  and explains at most 0.03 % of its variance; d_t has no time trend (R² ≤ 0.0007). The level is
  non-stationary (lag-1 autocorrelation ≈ 1), d_t is not. **Differencing removes the trend as intended.**
- Intraday price paths are not straight-line trends: a line from the day's first to its last price
  explains less than the day's mean on average (negative R²). The "trend" the paper removes is a
  random-walk level, which differencing handles regardless of its shape.
- The remaining lag-1 autocorrelation of d_t (≈ 0.30) is mostly the averaging effect of the vwap
  ([vwap_smoothing_check.md](vwap_smoothing_check.md)), not trend.

## 2. Scale across hours

{markdown(prof)}

- Busy hours produce up to {bars_ratio:.1f}× more bars than quiet hours, but each bar's move changes little:
  mean |d| per hour varies by only {ratio_d:.2f}× (max / min). With bars of a fixed number of trades, the
  bar clock itself equalises most intraday volatility ("business time").
- Differencing removes level effects: within a day the price level changes by a few percent, so d_t
  in USD changes scale by the same few percent. It does not remove volatility differences between
  hours; those are small here because of the trade-count bars.
- The SNN's signal rate follows the hourly |d| closely (correlation {corr:.2f}) and amplifies it
  ({ratio_s:.2f}× max / min), which is the intended behaviour (it fires on larger moves), not a bias
  from price levels.
- d_t is in USD, not in returns. Over one training day this does not matter; for multi-day training
  windows (E4, W_snn up to 10 days) price-level changes between days (typically a few percent, more in
  volatile weeks) rescale d_t by the same amount relative to the z-score statistics.
"""
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
