"""Backtest tests (§13): splitter, identical test days, signal-cache reuse, end-to-end smoke test."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from snn_hft.backtest.backtester import Backtester, compute_folds, signal_jobs
from snn_hft.backtest.cache import SignalCache
from snn_hft.backtest.splitter import WalkForwardSplitter
from snn_hft.config.schema import ExperimentConfig, expand_runs


def synthetic_cfg(tmp_path, **overrides) -> ExperimentConfig:
    raw = {
        "name": "smoke",
        "data": {"source": "synthetic", "raw_dir": str(tmp_path / "raw")},
        "periods": {
            "data": {"start": "2025-01-01", "end": "2025-01-08"},
            "history_days": 2,
            "validation": {"start": "2025-01-03", "end": "2025-01-04"},
            "test": {"start": "2025-01-05", "end": "2025-01-07"},
        },
        "models": [{"model": "paper_snn", "core": {"lif": {"threshold": 8.0}}}],
        "backtest": {"split": "test", "seeds": [0, 1], "w_snn": [1], "latencies_ms": [0, 50]},
        "benchmarks": {"naive_reps": 4},
        "output": {"results_dir": str(tmp_path / "results"), "cache_dir": str(tmp_path / "cache"), "save_trades": "all"},
    }
    for key, value in overrides.items():
        raw[key] = {**raw.get(key, {}), **value} if isinstance(value, dict) else value
    return ExperimentConfig.model_validate(raw)


def test_walk_forward_folds():
    folds = WalkForwardSplitter(3).folds([date(2025, 1, 10), date(2025, 1, 5)])
    assert folds[0].test_day == date(2025, 1, 5)
    assert folds[0].train_days == (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4))
    assert folds[1].train_days[-1] == date(2025, 1, 9)


def test_test_days_identical_across_window_configurations(tmp_path):
    a = synthetic_cfg(tmp_path, backtest={"w_snn": [1]})
    b = synthetic_cfg(tmp_path, backtest={"w_snn": [2]})
    assert a.eval_period.days() == b.eval_period.days()
    for cfg in (a, b):
        job = signal_jobs(cfg)[0]
        folds = WalkForwardSplitter(job.run.w_snn).folds(cfg.eval_period.days())
        assert [f.test_day for f in folds] == a.eval_period.days()


def test_signal_jobs_share_signals_across_rules(tmp_path):
    cfg = synthetic_cfg(tmp_path)
    jobs = signal_jobs(cfg)
    assert len(jobs) == 2  # two seeds; the three rules share signals
    assert all(len(j.runs) == 3 for j in jobs)
    assert len({(r.model_id, r.seed) for j in jobs for r in j.runs}) == 2


def test_signal_cache_roundtrip_and_reuse(tmp_path):
    cfg = synthetic_cfg(tmp_path)
    run = expand_runs(cfg)[0]
    folds = WalkForwardSplitter(1).folds(cfg.eval_period.days()[:1])
    compute_folds(run, folds)
    cache = SignalCache(cfg.output.cache_dir)
    assert cache.has(run, folds[0].test_day)
    first = cache.load(run, folds[0].test_day)
    assert first.test.n_bars == 2_000 and len(first.train_metrics) == 1
    assert {"spikes_Out", "spikes_H1", "channel_prob"} <= first.test.diagnostics.keys()
    # A second backtester run needs no fold that is already cached.
    bt = Backtester(cfg, workers=1)
    tasks = bt._signal_tasks(signal_jobs(cfg), cfg.eval_period.days())
    assert all(folds[0].test_day not in [f.test_day for f in fs] for r, fs in tasks if r.seed == run.seed)


def test_trades_saved_for_first_seed_only_by_default(tmp_path):
    cfg = synthetic_cfg(tmp_path, output={"save_trades": "first_seed"}, backtest={"latencies_ms": [0]},
                        benchmarks={"naive_reps": 1})
    Backtester(cfg, workers=2, log=lambda _: None).run()
    root = tmp_path / "results" / "smoke"
    assert (root / "paper__momentum__Wsnn1__seed0" / "trades.parquet").is_file()
    assert not (root / "paper__momentum__Wsnn1__seed1" / "trades.parquet").exists()


def test_cache_without_diagnostics_keeps_metrics(tmp_path):
    cfg = synthetic_cfg(tmp_path, output={"cache_diagnostics": False})
    run = expand_runs(cfg)[0]
    folds = WalkForwardSplitter(1).folds(cfg.eval_period.days()[:1])
    compute_folds(run, folds)
    meta = SignalCache(cfg.output.cache_dir).load_meta(run, folds[0].test_day)
    assert 0 <= meta["test_metrics"]["accuracy"] <= 1
    assert SignalCache(cfg.output.cache_dir).load(run, folds[0].test_day).test.diagnostics == {}


def test_end_to_end_smoke(tmp_path):
    cfg = synthetic_cfg(tmp_path)
    result = Backtester(cfg, workers=2, log=lambda _: None).run()
    root = tmp_path / "results" / "smoke"
    runs = expand_runs(cfg)
    for run in runs:
        d = root / run.run_id
        for name in ("config.yaml", "meta.json", "performance.json", "daily_pnl.parquet", "spike_metrics.csv",
                     "input_stats.csv", "signals.parquet", "trades.parquet", "health.csv"):
            assert (d / name).is_file(), f"{run.run_id}/{name}"
        perf = json.loads((d / "performance.json").read_text())
        assert set(perf["latencies"]) == {"0.0", "50.0"}
        assert {"model", "naive", "big_move"} <= perf["latencies"]["0.0"].keys()
        assert perf["latencies"]["0.0"]["naive"]["reps"] == 4
        trades = pd.read_parquet(d / "trades.parquet")
        assert set(trades["latency_ms"]) == {0.0, 50.0}
        daily = pd.read_parquet(d / "daily_pnl.parquet")
        model_daily = daily[(daily.source == "model") & (daily.latency_ms == 0.0)]
        # Daily P&L is the sum of that day's trades.
        by_day = trades[trades.latency_ms == 0.0].groupby("day")["net_return"].sum()
        np.testing.assert_allclose(model_daily.set_index("day")["pnl"].loc[by_day.index], by_day.values)
        assert perf["latencies"]["0.0"]["model"]["n_days"] == 3
    # Tables: 3 test days × (model, naive, big_move) sources, both splits for the model.
    assert len(result.table4_rows) == 2 * 3 * 2 * 3  # seeds × rules × latencies × sources
    test_rows = {(r["seed"], r["source"]): r for r in result.table3_rows if r["split"] == "test"}
    for seed in (0, 1):  # the big-move benchmark fires about as often as the model on test days
        ratio = test_rows[seed, "big_move"]["signal_rate"] / test_rows[seed, "model"]["signal_rate"]
        assert 0.5 < ratio < 2.0
    sources = {(r["split"], r["source"]) for r in result.table3_rows}
    assert sources == {("train", "model"), ("test", "model"), ("test", "naive"), ("test", "big_move")}
    # Signals are shared: one parquet hard-linked into every rule's directory.
    first = [r for r in runs if r.seed == 0]
    inodes = {(root / r.run_id / "signals.parquet").stat().st_ino for r in first}
    assert len(inodes) == 1


def test_tuning_grid_and_trials():
    from experiments.tune import SHARED_GRID, apply_trial, grid_for, sample_trials

    assert set(grid_for("paper_snn")) == set(SHARED_GRID) | {"zscore.new_std"}
    assert set(grid_for("hawkes_rstdp")) == set(SHARED_GRID) | {"rstdp.gamma", "rstdp.tau_z_bars", "hawkes.event_quantile"}
    assert all(grid_for(m)[k] == v for m in ("paper_snn", "hawkes_rstdp") for k, v in SHARED_GRID.items())
    trials = sample_trials(SHARED_GRID, 10, seed=0)
    assert len({tuple(t.values()) for t in trials}) == 10
    assert trials == sample_trials(SHARED_GRID, 10, seed=0)
    model = apply_trial({"model": "paper_snn"}, {"core.lif.threshold": 8.0, "stdp_scale": 2.0, "input.new_mean": 0.3})
    assert model["core"]["lif"]["threshold"] == 8.0
    assert model["core"]["stdp"] == {"A": 0.02, "B": -0.021}
    assert model["input"]["new_mean"] == 0.3


def test_tables_aggregate_over_seeds_and_render_markdown():
    from snn_hft.analysis.compare import table3, table3_markdown, table4, table4_markdown

    base = {"model": "paper", "w_snn": 1, "w_h": None, "split": "test", "source": "model", "base_accuracy": 0.5,
            "momentum_pct": 0.8, "base_momentum_pct": 0.8, "n_signals": 100, "signal_rate": 0.1}
    t3 = table3([{**base, "seed": 0, "accuracy": 0.52}, {**base, "seed": 1, "accuracy": 0.54}])
    assert t3.loc[0, "spike_accuracy"] == pytest.approx(0.53) and t3.loc[0, "seeds"] == 2
    assert "53.00 %" in table3_markdown(t3)
    perf = {"accumulated_return": 1.0, "annualized_volatility": 0.5, "sharpe": 20.0, "win_rate": 0.6,
            "profit_loss_ratio": 1.1, "trades_per_day": 1000.0}
    rows = [{"model": "paper", "w_snn": 1, "w_h": None, "rule": "momentum", "latency_ms": 0.0, "source": "model",
             "seed": s, **{k: v * (1 + s) for k, v in perf.items()}} for s in (0, 1)]
    t4 = table4(rows)
    assert t4.loc[0, "sharpe"] == pytest.approx(30.0) and t4.loc[0, "sharpe_std"] == pytest.approx(np.std([20, 40], ddof=1))
    assert "| momentum | model |" in table4_markdown(t4, 0.0)
