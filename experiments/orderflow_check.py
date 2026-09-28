"""Appendix check (U11): does order flow make the trade outcome predictable?

    python -m experiments.orderflow_check --out docs/reports/orderflow_check.md

Validation days only. Order-flow imbalance of bar t: (taker-buy volume − taker-sell volume) /
volume of the bar's 100 aggTrades (`is_buyer_maker` = the taker sold). Trade label as in the
premise checks (follow the momentum rule at t, entry t+1, exit t+4). Features are oriented by the
trade direction dir_t. A logistic regression is fitted on the first half of the validation days
and evaluated on the second half, with price features only and with order flow added.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.typed_premise_checks import auc, logit, trade_label
from snn_hft.analysis.compare import markdown
from snn_hft.config.loader import load_raw
from snn_hft.config.schema import BarsConfig, DataConfig, PeriodsConfig
from snn_hft.data.store import make_store

BASE = Path("experiments/configs/base.yaml")


def day_features(day: date) -> pd.DataFrame:
    raw = load_raw(BASE)
    data, bars_cfg = DataConfig.model_validate(raw["data"]), BarsConfig.model_validate(raw.get("bars", {}))
    store = make_store(data)
    trades = store.get_day(data.symbol, day).df
    train = store.get_day(data.symbol, day - timedelta(days=1)).df
    num = bars_cfg.vwap_num

    def bars(df):
        n = len(df) // num * num
        p = df.price.to_numpy()[:n].reshape(-1, num)
        q = df.qty.to_numpy()[:n].reshape(-1, num)
        sell = df.is_buyer_maker.to_numpy()[:n].reshape(-1, num)
        vwap = (p * q).sum(1) / q.sum(1)
        ofi = (q * np.where(sell, -1.0, 1.0)).sum(1) / q.sum(1)
        count_imb = np.where(sell, -1.0, 1.0).mean(1)
        return vwap, ofi, count_imb

    vwap, ofi, cimb = bars(trades)
    scale = float(np.median(np.abs(np.diff(bars(train)[0])))) or 1.0
    dirs, ret = trade_label(vwap)
    n = len(vwap)
    d = np.zeros(n)
    d[1:] = np.diff(vwap)
    t = np.flatnonzero(~np.isnan(ret) & (ret != 0) & (np.arange(n) >= 20))
    s = dirs[t]
    f = {"y": (ret[t] > 0).astype(float), "ret": ret[t], "day": day}
    for k in range(3):
        f[f"move_{k}"] = s * d[t - k] / scale
        f[f"size_{k}"] = np.log1p(np.abs(d[t - k]) / scale)
    cum = np.cumsum(ofi)
    for k in range(5):
        f[f"ofi_{k}"] = s * ofi[t - k]
    f["ofi_sum5"] = s * (cum[t] - cum[t - 5])
    f["ofi_sum20"] = s * (cum[t] - cum[t - 20])
    f["count_imb_0"] = s * cimb[t]
    return pd.DataFrame(f)


def fit_eval(df: pd.DataFrame, cols: list[str], first: np.ndarray) -> tuple[float, float, np.ndarray]:
    mu, sd = df.loc[first, cols].mean(), df.loc[first, cols].std().replace(0, 1)
    X = np.column_stack([np.ones(len(df)), ((df[cols] - mu) / sd).clip(-10, 10).to_numpy()])
    beta, _, _ = logit(X[first], df.y.to_numpy()[first])
    p = 1 / (1 + np.exp(-(X @ beta)))
    y = df.y.to_numpy() > 0.5
    return auc(p[first], y[first]), auc(p[~first], y[~first]), p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    days = PeriodsConfig.model_validate(load_raw(BASE).get("periods", {})).validation.days()
    with ProcessPoolExecutor(8) as pool:
        df = pd.concat(list(pool.map(day_features, days)), ignore_index=True)
    first = (df.day < days[len(days) // 2]).to_numpy()
    price = [c for c in df.columns if c.startswith(("move_", "size_"))]
    flow = [c for c in df.columns if c.startswith(("ofi", "count_imb"))]
    rows = []
    for name, cols in (("price features", price), ("order flow only", flow), ("price + order flow", price + flow)):
        a_fit, a_held, p = fit_eval(df, cols, first)
        rows.append({"features": name, "AUC fit half": f"{a_fit:.3f}", "AUC held-out half": f"{a_held:.3f}"})
        if name == "price + order flow":
            p_all = p
    single = pd.DataFrame([{"feature": c, "AUC (all days)": f"{auc(df[c].to_numpy(), df.y.to_numpy() > 0.5):.3f}"} for c in flow])
    r = df.ret.to_numpy()[~first] * 1e4
    ph = p_all[~first]
    cut = np.quantile(ph, 0.10)
    typing = pd.DataFrame([
        {"momentum strategy, held-out validation bars": "always follow", "bp per trade": f"{r.mean():+.3f}"},
        {"momentum strategy, held-out validation bars": "fade when classifier says reversion (p < 0.5)",
         "bp per trade": f"{np.where(ph >= 0.5, r, -r).mean():+.3f}", "trades faded": f"{np.mean(ph < 0.5):.0%}"},
        {"momentum strategy, held-out validation bars": "fade the 10 % most reversion-like bars",
         "bp per trade": f"{np.where(ph >= cut, r, -r).mean():+.3f}", "trades faded": "10 %"},
        {"momentum strategy, held-out validation bars": "oracle (not causal)", "bp per trade": f"{np.abs(r).mean():+.3f}"},
    ])
    text = f"""# Appendix — does order flow make the trade outcome predictable? (U11)

Validation days {days[0]} → {days[-1]}, 100 aggTrades per bar, {len(df):,} bars with a decided trade label.
Order-flow imbalance (OFI) of a bar = (taker-buy − taker-sell volume) / bar volume; count imbalance
= the same for trade counts. All features are oriented by the trade direction dir_t (positive = in the
direction the momentum rule would trade). Fit on the first {int(np.sum(np.array(days) < days[len(days) // 2]))} days, evaluated on the rest.

{markdown(pd.DataFrame(rows))}

Single order-flow features:

{markdown(single)}

Value of typing with price + order flow (fee-free, paper execution):

{markdown(typing)}
"""
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
