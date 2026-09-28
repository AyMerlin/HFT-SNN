"""Walk-forward backtester (§8.3).

Two phases:

1. **Signals.** For every signal job (model × W_snn × W_h × seed) and every test day, fit the
   model on the fold's training days and generate the test day's signals. Folds run in
   parallel worker processes and each result is cached (`SignalCache`); a job with
   `warm_start` runs its folds in order in one process.
2. **Evaluation.** Per job, walk through the test days once: spike metrics (model, naive
   benchmark, big-move benchmark), trades of every direction rule and latency, naive
   repetitions, then per-run result files (§8.6). Signals do not depend on the direction
   rule, so all three strategies share one signal job.
"""

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from snn_hft.backtest.cache import BarCache, FoldResult, SignalCache
from snn_hft.backtest.performance import DAILY_COLUMNS, PerformanceCalculator, daily_aggregate
from snn_hft.backtest.splitter import FoldSpec, WalkForwardSplitter
from snn_hft.backtest.store import ResultStore
from snn_hft.config.schema import ExecutionConfig, ExperimentConfig, RunConfig, expand_runs
from snn_hft.data.containers import BarSeries, DayData
from snn_hft.data.store import make_store
from snn_hft.evaluation.health import check_health
from snn_hft.evaluation.spike_metrics import DaySpikeMetrics, SpikeEvaluator
from snn_hft.signals.base import SignalSeries
from snn_hft.signals.big_move import BigMoveSignalModel
from snn_hft.signals.factory import make_signal_model
from snn_hft.signals.random_signal import RandomSignalModel
from snn_hft.strategy.direction_rules import DirectionRule, make_rule
from snn_hft.strategy.execution import TRADE_FRAME_COLUMNS, per_signal
from snn_hft.utils.seeding import rng_for

SPIKE_FIELDS = list(DaySpikeMetrics.__dataclass_fields__)


@dataclass(frozen=True)
class SignalJob:
    """Runs that share signals: one per direction rule, same model, windows and seed."""

    runs: tuple[RunConfig, ...]

    @property
    def run(self) -> RunConfig:
        return self.runs[0]


def signal_jobs(cfg: ExperimentConfig) -> list[SignalJob]:
    groups: dict[tuple, list[RunConfig]] = defaultdict(list)
    for run in expand_runs(cfg):
        groups[(run.signal.name, run.w_snn, run.w_h, run.seed)].append(run)
    return [SignalJob(tuple(runs)) for runs in groups.values()]


# --------------------------------------------------------------------------- phase 1: signals


def _hidden_pools(model) -> dict[str, int]:
    return {p.name: p.size for p in model.network.populations if p.name.startswith("H")}


def _input_stat(sig: SignalSeries, split: str, ticks_per_bar: int) -> dict:
    n = max(sig.n_bars, 1) * ticks_per_bar
    prob = sig.diagnostics["channel_prob"]
    return {
        "day": sig.day.isoformat(), "split": split,
        "rate_X1": float(sig.diagnostics["spikes_X1"].sum() / n), "rate_X2": float(sig.diagnostics["spikes_X2"].sum() / n),
        "prob_X1": float(prob[:, 0].mean()), "prob_X2": float(prob[:, 1].mean()),
    }


class _DayLoader:
    def __init__(self, run: RunConfig, size: int = 12):
        self.store, self.symbol = make_store(run.data), run.data.symbol
        self.size, self.days = size, OrderedDict()

    def get(self, day: date) -> DayData:
        if day not in self.days:
            self.days[day] = self.store.day_data(self.symbol, day)
            while len(self.days) > self.size:
                self.days.popitem(last=False)
        return self.days[day]


def compute_folds(run: RunConfig, folds: Sequence[FoldSpec]) -> list[dict]:
    """Fit + generate for consecutive folds of one job and cache each result (worker entry point)."""
    cache = SignalCache(run.output.cache_dir)
    loader = _DayLoader(run, size=run.w_snn + 2)
    evaluator = SpikeEvaluator(run.evaluation.eval_window)
    model = make_signal_model(run)
    summaries = []
    for fold in folds:
        t0 = time.perf_counter()
        model.fit([loader.get(d) for d in fold.train_days], run.seed)
        test = model.generate(loader.get(fold.test_day), run.seed)
        pools, t_ref, T = _hidden_pools(model), model.core.lif.t_ref, model.ticks_per_bar
        train_metrics, input_stats = [], []
        for sig, day in zip(model.train_signals, model.train_days):
            health = check_health(sig, pools, T, t_ref, run.evaluation.health)
            train_metrics.append(
                {"day": sig.day.isoformat(), **evaluator.evaluate(day.bars.vwap, sig).as_dict(),
                 "health": health.warnings, "pool_rates": health.pool_rates}
            )
            input_stats.append(_input_stat(sig, "train", T))
        input_stats.append(_input_stat(test, "test", T))
        health = check_health(test, pools, T, t_ref, run.evaluation.health)
        test_vwap = model.transform(loader.get(fold.test_day)).bars.vwap
        result = FoldResult(
            test=test,
            test_metrics=evaluator.evaluate(test_vwap, test).as_dict(),
            train_metrics=train_metrics,
            train_signal_rate=float(np.mean([s.rate for s in model.train_signals])),
            test_health=health.warnings,
            input_stats=input_stats,
            runtime_s=time.perf_counter() - t0,
            extra={"test_pool_rates": health.pool_rates, **getattr(model, "fold_metadata", lambda: {})()},
        )
        cache.save(run, result, diagnostics=run.output.cache_diagnostics)
        summaries.append({"test_day": fold.test_day, "n_signals": len(test), "rate": test.rate, "runtime_s": result.runtime_s})
    return summaries


def compute_hawkes_params(run: RunConfig, days: Sequence[date]) -> int:
    """Fit θ_d for the given days into the parameter cache (phase 0, worker entry point)."""
    from snn_hft.models.hawkes.provider import HawkesParamProvider
    from snn_hft.preprocessing.hawkes_steps import candidate_events_source
    from snn_hft.signals.factory import hawkes_sources

    bars_for_day, store = hawkes_sources(run)
    provider = HawkesParamProvider(
        run.signal.hawkes, run.w_h, candidate_events_source(bars_for_day, run.signal.hawkes.time_axis), store
    )
    for day in days:
        provider.params_for(day)
    return len(days)


# --------------------------------------------------------------------------- phase 2: evaluation


class DayEvaluator:
    """Per-bar trade returns of every (rule, latency) and spike labels for one day, so the model,
    the naive repetitions and the big-move benchmark are all evaluated by indexing."""

    def __init__(self, bars: BarSeries, rules: dict[str, DirectionRule], executions: dict[float, ExecutionConfig],
                 evaluator: SpikeEvaluator):
        self.bars = bars
        vwap = bars.vwap
        all_t = np.arange(len(bars))
        self.labels = evaluator.labels(vwap)
        self.directions = {name: rule.directions(vwap, all_t) for name, rule in rules.items()}
        self.executions = executions
        self._valid, self._net = {}, {}
        for name, dirs in self.directions.items():
            for lat, ex in executions.items():
                arrays = per_signal(bars, all_t, dirs, ex)
                self._valid[name, lat] = arrays["valid"]
                self._net[name, lat] = arrays["net_return"]

    def net_returns(self, rule: str, latency: float, bar_idx: np.ndarray) -> np.ndarray:
        valid = self._valid[rule, latency][bar_idx]
        return self._net[rule, latency][bar_idx][valid]

    def trades(self, rule: str, latency: float, bar_idx: np.ndarray) -> pd.DataFrame:
        arrays = per_signal(self.bars, bar_idx, self.directions[rule][bar_idx], self.executions[latency])
        valid = arrays.pop("valid")
        return pd.DataFrame({k: v[valid] for k, v in arrays.items()})[TRADE_FRAME_COLUMNS]

    def spike(self, bar_idx: np.ndarray) -> DaySpikeMetrics:
        return SpikeEvaluator.metrics(self.labels, bar_idx)


def _trade_schema() -> pa.Schema:
    return pa.schema(
        [("day", pa.date32()), ("latency_ms", pa.float64()), ("signal_bar", pa.int64()), ("entry_bar", pa.int64()),
         ("exit_bar", pa.int64()), ("direction", pa.int8()), ("entry_price", pa.float64()), ("exit_price", pa.float64()),
         ("gross_return", pa.float64()), ("fee", pa.float64()), ("net_return", pa.float64())]
    )


def _mean_std(dicts: list[dict]) -> dict:
    df = pd.DataFrame(dicts)
    return {"mean": df.mean(numeric_only=True).to_dict(), "std": df.std(ddof=1, numeric_only=True).to_dict(), "reps": len(df)}


def evaluate_job(job: SignalJob, test_days: Sequence[date], save_trades: bool = False) -> dict:
    """Evaluate one signal job on all test days and write one result directory per rule."""
    run0 = job.run
    cache = SignalCache(run0.output.cache_dir)
    store = make_store(run0.data)
    bar_cache = BarCache(store, run0.data.symbol, run0.bars.vwap_num, run0.output.cache_dir)
    results = ResultStore(run0.output.results_dir, run0.experiment)
    evaluator = SpikeEvaluator(run0.evaluation.eval_window)
    perf = PerformanceCalculator(run0.evaluation.annualization_days)
    s = run0.strategy
    rules = {r.rule: make_rule(r.rule, s.momentum_window, s.alf_n, s.stoch_n) for r in job.runs}
    latencies = [float(x) for x in run0.latencies_ms]
    executions = {lat: run0.execution.model_copy(update={"latency_ms": lat}) for lat in latencies}
    reps = run0.benchmarks.naive_reps
    splitter = WalkForwardSplitter(run0.w_snn)
    n_days = len(test_days)

    daily: dict[tuple, list[dict]] = defaultdict(list)
    naive_daily = {(r, lat): np.zeros((reps, n_days, len(DAILY_COLUMNS))) for r in rules for lat in latencies}
    spike_rows, signal_frames, input_rows, health_rows = [], [], [], []
    hawkes_meta: dict[str, dict] = {}
    writers = {}
    if save_trades:
        for run in job.runs:
            writers[run.rule] = pq.ParquetWriter(results.run_dir(run.run_id) / "trades.parquet", _trade_schema(), compression="zstd")

    try:
        for di, day in enumerate(test_days):
            fold = cache.load(run0, day)
            sig = fold.test
            bars = bar_cache.get(day)
            if len(bars) != sig.n_bars:
                raise RuntimeError(f"{day}: cached signals have {sig.n_bars} bars, bar cache {len(bars)}")
            de = DayEvaluator(bars, rules, executions, evaluator)

            spike_rows.append({"day": day, "split": "test", "source": "model", "rule": "", **de.spike(sig.bar_idx).as_dict()})
            for m in fold.train_metrics:
                spike_rows.append({"day": m["day"], "test_day": day, "split": "train", "source": "model", "rule": "",
                                   **{k: m[k] for k in SPIKE_FIELDS}})

            fold_spec = splitter.folds([day])[0]
            # Fires about as often as the model on this test day (like the naive benchmark); the
            # threshold itself comes from the training days, so the signals stay causal.
            big = BigMoveSignalModel(run0.bars.vwap_num, target_rate=sig.rate)
            big.fit([DayData(day=d, bars=bar_cache.get(d)) for d in fold_spec.train_days])
            big_sig = big.generate(DayData(day=day, bars=bars))
            if run0.benchmarks.big_move:
                spike_rows.append({"day": day, "split": "test", "source": "big_move", "rule": "", **de.spike(big_sig.bar_idx).as_dict()})

            for name, rule in rules.items():
                samples = [
                    RandomSignalModel.sample(sig.n_bars, len(sig), rule.min_history, run0.execution.entry_delay,
                                             rng_for(rep, "naive", day, run0.seed))
                    for rep in range(reps)
                ]
                if reps:
                    naive_spikes = pd.DataFrame([de.spike(b).as_dict() for b in samples]).mean().to_dict()
                    spike_rows.append({"day": day, "split": "test", "source": "naive", "rule": name, **naive_spikes})
                for lat in latencies:
                    daily[name, lat, "model"].append({"day": day, **daily_aggregate(de.net_returns(name, lat, sig.bar_idx))})
                    if run0.benchmarks.big_move:
                        daily[name, lat, "big_move"].append({"day": day, **daily_aggregate(de.net_returns(name, lat, big_sig.bar_idx))})
                    for rep, b in enumerate(samples):
                        agg = daily_aggregate(de.net_returns(name, lat, b))
                        naive_daily[name, lat][rep, di] = [agg[c] for c in DAILY_COLUMNS]
                    if name in writers:
                        frame = de.trades(name, lat, sig.bar_idx)
                        frame.insert(0, "latency_ms", lat)
                        frame.insert(0, "day", day)
                        writers[name].write_table(pa.Table.from_pandas(frame, schema=_trade_schema(), preserve_index=False))

            pools = [k for k in sig.diagnostics if k.startswith("spikes_") and not k.startswith("spikes_X")]
            if not pools:
                raise RuntimeError("evaluation needs per-bar diagnostics; set output.cache_diagnostics=true")
            signal_frames.append(pd.DataFrame(
                {"day": day, "bar_idx": sig.bar_idx, **{k: sig.diagnostics[k][sig.bar_idx] for k in pools}}
            ))
            input_rows += fold.input_stats
            hawkes_meta.update(fold.extra.get("hawkes", {}))
            health_rows.append({"day": day, "split": "test", "warnings": "; ".join(fold.test_health)})
            health_rows += [{"day": m["day"], "split": "train", "warnings": "; ".join(m["health"])} for m in fold.train_metrics]
    finally:
        for w in writers.values():
            w.close()

    spikes = pd.DataFrame(spike_rows)
    health = pd.DataFrame(health_rows)
    inputs = pd.DataFrame(input_rows)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    signals_path = None
    summary = {"table3": [], "table4": []}
    ident = {"model": run0.signal.name, "model_type": run0.signal.model, "w_snn": run0.w_snn, "w_h": run0.w_h, "seed": run0.seed}

    for split in ("train", "test"):
        for source in ("model", "naive", "big_move"):
            rows = spikes[(spikes["split"] == split) & (spikes["source"] == source)]
            if source == "naive":
                rows = rows[rows["rule"] == next(iter(rules))]  # chance level from the first rule's naive samples
            if len(rows):
                summary["table3"].append({**ident, "split": split, "source": source,
                                          **rows[SPIKE_FIELDS].mean(numeric_only=True).to_dict()})

    for run in job.runs:
        name = run.rule
        results.write_run_header(run, test_days=[test_days[0], test_days[-1]], n_test_days=n_days)
        performance = {"rule": name, "latencies": {}}
        daily_frames = []
        for lat in latencies:
            entry = {}
            for source in ("model", "big_move"):
                if (name, lat, source) in daily:
                    frame = pd.DataFrame(daily[name, lat, source])
                    entry[source] = perf.compute(frame)
                    daily_frames.append(frame.assign(latency_ms=lat, source=source))
            if reps:
                per_rep = [perf.compute(pd.DataFrame(naive_daily[name, lat][rep], columns=DAILY_COLUMNS)) for rep in range(reps)]
                entry["naive"] = _mean_std(per_rep)
                mean_daily = pd.DataFrame(naive_daily[name, lat].mean(axis=0), columns=DAILY_COLUMNS).assign(day=list(test_days))
                daily_frames.append(mean_daily.assign(latency_ms=lat, source="naive_mean"))
            performance["latencies"][str(lat)] = entry
            for source, metrics in entry.items():
                values = metrics["mean"] if source == "naive" else metrics
                summary["table4"].append({**ident, "rule": name, "latency_ms": lat, "source": source, **values})
        performance["spike_test"] = spikes[(spikes.split == "test") & (spikes.source == "model")][SPIKE_FIELDS].mean(numeric_only=True).to_dict()
        performance["spike_train"] = spikes[(spikes.split == "train") & (spikes.source == "model")][SPIKE_FIELDS].mean(numeric_only=True).to_dict()
        performance["health"] = {
            "test_days_with_warnings": int((health[health.split == "test"]["warnings"] != "").sum()),
            "train_days_with_warnings": int((health[health.split == "train"]["warnings"] != "").sum()),
        }
        results.write_json(run.run_id, "performance.json", performance)
        results.write_frame(run.run_id, "daily_pnl.parquet", pd.concat(daily_frames, ignore_index=True))
        results.write_frame(run.run_id, "spike_metrics.csv", spikes[spikes["rule"].isin(["", name])])
        results.write_frame(run.run_id, "input_stats.csv", inputs)
        results.write_frame(run.run_id, "health.csv", health)
        if signals_path is None:
            signals_path = results.write_frame(run.run_id, "signals.parquet", signals)
            hawkes_paths = {
                day: results.write_json(run.run_id, f"hawkes/{day}.json", theta) for day, theta in sorted(hawkes_meta.items())
            }
        else:
            results.link(signals_path, run.run_id, "signals.parquet")
            for day, path in hawkes_paths.items():
                results.link(path, run.run_id, f"hawkes/{day}.json")
    return summary


# --------------------------------------------------------------------------- orchestration


@dataclass
class BacktestResult:
    table3_rows: list[dict]
    table4_rows: list[dict]


class Backtester:
    def __init__(self, cfg: ExperimentConfig, workers: int | None = None, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.workers = workers or os.cpu_count() or 1
        self.log = log

    def _signal_tasks(self, jobs: list[SignalJob], test_days: list[date]) -> list[tuple[RunConfig, list[FoldSpec]]]:
        cache = SignalCache(self.cfg.output.cache_dir)
        tasks = []
        for job in jobs:
            folds = WalkForwardSplitter(job.run.w_snn).folds(test_days)
            missing = [f for f in folds if not cache.has(job.run, f.test_day)]
            if not missing:
                continue
            if job.run.signal.core.warm_start:
                tasks.append((job.run, folds))  # state flows from fold to fold: recompute the sequence
                continue
            size = max(1, math.ceil(len(missing) / (self.workers * 3)))
            tasks += [(job.run, missing[i : i + size]) for i in range(0, len(missing), size)]
        return tasks

    def _hawkes_tasks(self, jobs: list[SignalJob], test_days: list[date]) -> list[tuple[RunConfig, list[date]]]:
        """Days whose θ_d is needed and not cached, one list per distinct Hawkes setting."""
        from snn_hft.models.hawkes.provider import HawkesParamProvider
        from snn_hft.signals.factory import hawkes_sources

        needed: dict[tuple, tuple[RunConfig, set[date]]] = {}
        for job in jobs:
            run = job.run
            if not run.signal.uses_hawkes:
                continue
            key = (run.signal.hawkes.model_dump_json(), run.w_h)
            days = needed.setdefault(key, (run, set()))[1]
            for fold in WalkForwardSplitter(run.w_snn).folds(test_days):
                days.update(fold.train_days)
                days.add(fold.test_day)
        tasks = []
        for run, days in needed.values():
            _, store = hawkes_sources(run)
            probe = HawkesParamProvider(run.signal.hawkes, run.w_h, events_for_day=None, store=store)
            missing = sorted(d for d in days if not store.has(probe.key(d)))
            size = max(1, math.ceil(len(missing) / (self.workers * 2)))
            tasks += [(run, missing[i : i + size]) for i in range(0, len(missing), size)]
        return tasks

    def compute_hawkes(self, jobs: list[SignalJob]) -> None:
        """Phase 0: fit every needed θ_d once, in parallel, so signal workers only read the cache."""
        tasks = self._hawkes_tasks(jobs, self.cfg.eval_period.days())
        if not tasks:
            return
        n = sum(len(d) for _, d in tasks)
        self.log(f"hawkes: fitting θ_d for {n} (day, setting) pairs")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(self.workers, len(tasks))) as pool:
            done = 0
            for fut in as_completed([pool.submit(compute_hawkes_params, run, days) for run, days in tasks]):
                done += fut.result()
                self.log(f"  hawkes {done}/{n}, {time.time() - t0:.0f}s")

    def compute_signals(self) -> list[SignalJob]:
        """Phase 1: fit + generate every missing fold, in parallel, into the signal cache."""
        jobs = signal_jobs(self.cfg)
        test_days = self.cfg.eval_period.days()
        self.compute_hawkes(jobs)
        tasks = self._signal_tasks(jobs, test_days)
        n_folds = sum(len(f) for _, f in tasks)
        self.log(f"{self.cfg.name}: {len(jobs)} signal jobs × {len(test_days)} days; {n_folds} folds to compute")
        if not tasks:
            return jobs
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(self.workers, len(tasks))) as pool:
            done = 0
            for fut in as_completed([pool.submit(compute_folds, run, folds) for run, folds in tasks]):
                done += len(fut.result())
                elapsed = time.time() - t0
                self.log(f"  signals {done}/{n_folds} folds, {elapsed:.0f}s elapsed, ~{elapsed / done * (n_folds - done):.0f}s left")
        return jobs

    def evaluate(self, jobs: list[SignalJob]) -> BacktestResult:
        """Phase 2: strategies, benchmarks and result files for every job (signals must be cached)."""
        test_days = self.cfg.eval_period.days()
        t0 = time.time()
        table3, table4 = [], []
        with ProcessPoolExecutor(max_workers=min(self.workers, len(jobs))) as pool:
            mode, first_seed = self.cfg.output.save_trades, min(self.cfg.backtest.seeds)
            futures = [
                pool.submit(evaluate_job, job, test_days, mode == "all" or (mode == "first_seed" and job.run.seed == first_seed))
                for job in jobs
            ]
            for done, fut in enumerate(as_completed(futures), start=1):
                summary = fut.result()
                table3 += summary["table3"]
                table4 += summary["table4"]
                self.log(f"  evaluated {done}/{len(jobs)} jobs, {time.time() - t0:.0f}s")
        return BacktestResult(table3, table4)

    def run(self) -> BacktestResult:
        t0 = time.time()
        result = self.evaluate(self.compute_signals())
        self.log(f"done in {time.time() - t0:.0f}s")
        return result
