"""Causality harness (plan §0 principle 3, §13).

A *world* maps each day to its trades. `check_causal` runs a closure on the world,
then, for each bar t, perturbs every trade after bar t of the evaluated day and every
later day, runs the closure again and asserts that all outputs up to bar t are
unchanged. Three perturbations are used, so leaks through values, through the day's
length, and through the number of later trades are all caught:

- ``values``   — same timestamps, different prices, sizes and sides;
- ``truncate`` — the day ends early (fewer bars);
- ``resample`` — the remainder of the day is replaced by new trades of a different count.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from snn_hft.data.containers import TRADE_COLUMNS, BarSeries, DayData, TradeFrame, day_bounds_ns
from snn_hft.models.hawkes.process import DayEvents
from snn_hft.signals.base import SignalSeries

World = dict[date, TradeFrame]
MODES = ("values", "truncate", "resample")


def perturb_tail(frame: TradeFrame, first_idx: int, mode: str, rng: np.random.Generator) -> TradeFrame:
    """Replace trades from index `first_idx` on; earlier trades are untouched."""
    df = frame.df
    head, tail = df.iloc[:first_idx], df.iloc[first_idx:]
    k = len(tail)
    if mode == "values":
        new_tail = tail.copy()
        drift = np.exp(np.cumsum(rng.normal(0.0, 2e-3, k)) + rng.normal(0.0, 0.02))
        new_tail["price"] = tail["price"].to_numpy() * drift
        new_tail["qty"] = rng.lognormal(-3.0, 1.5, k)
        new_tail["is_buyer_maker"] = rng.random(k) < 0.5
    elif mode == "truncate":
        new_tail = tail.iloc[: k // 3]
    elif mode == "resample":
        start_ns, end_ns = day_bounds_ns(frame.day)
        lo = int(head["ts_ns"].iloc[-1]) if len(head) else start_ns
        n_new = int(rng.integers(max(1, k // 2), 2 * k + 50))
        ref = float(df["price"].iloc[max(first_idx - 1, 0)])
        first_id = int(df["trade_id"].max()) + 1
        new_tail = pd.DataFrame(
            {
                "ts_ns": np.sort(rng.integers(lo, end_ns, size=n_new)),
                "price": ref * np.exp(np.cumsum(rng.normal(0.0, 1e-3, n_new))),
                "qty": rng.lognormal(-3.0, 1.5, n_new),
                "is_buyer_maker": rng.random(n_new) < 0.5,
                "trade_id": first_id + np.arange(n_new),
            }
        )
    else:
        raise ValueError(f"unknown perturbation mode {mode!r}")
    merged = pd.concat([head, new_tail], ignore_index=True).astype(TRADE_COLUMNS)
    return TradeFrame(merged, symbol=frame.symbol, venue=frame.venue, day=frame.day, dataset=frame.dataset)


def perturb_world(
    world: World, day: date, bar_t: int, trades_per_bar: int, mode: str, rng: np.random.Generator
) -> World:
    """Perturb `day` after bar `bar_t` and every later day entirely; earlier days stay as they are."""
    out: World = {}
    for d, frame in world.items():
        if d < day:
            out[d] = frame
        elif d == day:
            out[d] = perturb_tail(frame, (bar_t + 1) * trades_per_bar, mode, rng)
        else:
            out[d] = perturb_tail(frame, 0, mode, rng)
    return out


def assert_prefix_equal(a: Any, b: Any, t: int, path: str = "") -> None:
    """Assert that two outputs agree on everything that belongs to bars 0..t.

    Arrays are indexed by bar; DataFrames with a ``bar_idx`` column are filtered by it;
    a SignalSeries is compared on its signals at bars ≤ t and its per-bar diagnostics; DayEvents
    on its events at bars ≤ t (the window length may differ);
    dicts, dataclasses, BarSeries and DayData are compared field by field.
    """
    where = path or "output"
    if a is None or b is None:
        assert a is None and b is None, f"{where}: one side is None"
        return
    if isinstance(a, TradeFrame):
        return  # raw inputs are not outputs
    if isinstance(a, DayEvents):  # indexed by event: compare the events at bars ≤ t
        ka, kb = a.bar_idx <= t, b.bar_idx <= t
        for name in ("bar_idx", "times", "types", "marks"):
            np.testing.assert_array_equal(getattr(a, name)[ka], getattr(b, name)[kb], err_msg=f"{where}.{name}")
        return
    if isinstance(a, SignalSeries):
        assert (a.day, a.model_id) == (b.day, b.model_id), f"{where}: different day or model"
        np.testing.assert_array_equal(a.bar_idx[a.bar_idx <= t], b.bar_idx[b.bar_idx <= t], err_msg=f"{where}.bar_idx")
        assert_prefix_equal(a.diagnostics, b.diagnostics, t, f"{path}.diagnostics" if path else "diagnostics")
        return
    if isinstance(a, BarSeries):
        assert_prefix_equal(a.df, b.df, t, path)
        return
    if isinstance(a, np.ndarray):
        assert len(a) > t and len(b) > t, f"{where}: shorter than bar {t}"
        np.testing.assert_array_equal(a[: t + 1], b[: t + 1], err_msg=where)
        return
    if isinstance(a, pd.DataFrame):
        if "bar_idx" in a.columns:
            a, b = a[a["bar_idx"] <= t], b[b["bar_idx"] <= t]
        else:
            a, b = a.iloc[: t + 1], b.iloc[: t + 1]
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), obj=where)
        return
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{where}: keys differ"
        for key in a:
            assert_prefix_equal(a[key], b[key], t, f"{path}.{key}" if path else str(key))
        return
    if dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            name = field.name
            assert_prefix_equal(getattr(a, name), getattr(b, name), t, f"{path}.{name}" if path else name)
        return
    assert a == b, f"{where}: {a!r} != {b!r}"


def check_causal(
    run: Callable[[World], Any],
    world: World,
    day: date,
    bar_ts: Iterable[int],
    trades_per_bar: int,
    compare: Callable[[Any, Any, int], None] = assert_prefix_equal,
    modes: Iterable[str] = MODES,
    seed: int = 0,
) -> Any:
    """Run `run(world)` under perturbations after each bar t of `day`; outputs up to t must not change.

    Returns the unperturbed output.
    """
    bar_ts = list(bar_ts)
    n_bars = len(world[day]) // trades_per_bar
    bad = [t for t in bar_ts if not 0 <= t < n_bars]
    if bad:
        raise ValueError(f"bars {bad} outside 0..{n_bars - 1} of {day}")
    base = run(world)
    rng = np.random.default_rng(seed)
    for t in bar_ts:
        for mode in modes:
            out = run(perturb_world(world, day, t, trades_per_bar, mode, rng))
            try:
                compare(base, out, t)
            except AssertionError as exc:
                raise AssertionError(f"causality violated up to bar {t} ({mode} perturbation): {exc}") from exc
    return base


def day_data(world: World, day: date) -> DayData:
    return DayData(day=day, trades=world[day])
