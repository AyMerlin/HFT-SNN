"""On-disk caches: vwap bars per day and signal-generation results per fold (§8.3).

Signals do not depend on the direction rule, the execution or the benchmarks, so one
cached fold result serves every strategy, latency and naive repetition.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from snn_hft.config.schema import RunConfig
from snn_hft.data.containers import BAR_COLUMNS, BarSeries
from snn_hft.data.store import DataStore
from snn_hft.preprocessing.bars import VWAPBarAggregator
from snn_hft.signals.base import SignalSeries


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    write(tmp)
    tmp.replace(path)


class BarCache:
    """vwap bars per day, shared by both models, the evaluator and the strategies."""

    def __init__(self, store: DataStore, symbol: str, vwap_num: int, cache_dir: str | Path):
        self.store, self.symbol, self.vwap_num = store, symbol, vwap_num
        src = store.source
        self.root = Path(cache_dir) / "bars" / src.venue / symbol / src.dataset / f"num{vwap_num}"

    def get(self, day: date) -> BarSeries:
        path = self.root / f"{day.isoformat()}.parquet"
        if path.is_file():
            return BarSeries(pd.read_parquet(path).astype(BAR_COLUMNS))
        bars = VWAPBarAggregator(self.vwap_num).transform(self.store.day_data(self.symbol, day)).bars
        _atomic_write(path, lambda p: bars.df.to_parquet(p, compression="zstd", index=False))
        return bars


@dataclass
class FoldResult:
    """Everything the evaluation needs from one fit + generate."""

    test: SignalSeries
    test_metrics: dict  # spike metrics of the test day
    train_metrics: list[dict]  # spike metrics and health per training day
    train_signal_rate: float  # mean signal rate on the training days (big-move benchmark target)
    test_health: list[str]
    input_stats: list[dict]  # realised input rate per day and channel (train and test)
    runtime_s: float
    extra: dict = field(default_factory=dict)  # model-specific metadata (e.g. Hawkes fits)


class HawkesParamCache:
    """θ_d as JSON per (data, bar size, code version, mark function, time axis, W_h, fit settings, day).

    Reused across folds, seeds, strategies and experiments (§5.5)."""

    def __init__(self, cache_dir: str | Path, venue: str, symbol: str, dataset: str, vwap_num: int):
        from snn_hft.config.schema import SIGNAL_CODE_VERSION

        self.root = Path(cache_dir) / "hawkes" / venue / symbol / dataset / f"num{vwap_num}" / f"code{SIGNAL_CODE_VERSION}"

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def load(self, key: str):
        from snn_hft.models.hawkes.params import HawkesParameters

        return HawkesParameters.from_json(self.path(key).read_text())

    def save(self, key: str, params) -> None:
        _atomic_write(self.path(key), lambda p: p.write_text(params.to_json()))


class SignalCache:
    """``{cache_dir}/signals/{model_id}/Wsnn{w}_Wh{h}/seed{s}/{test_day}.npz``."""

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir) / "signals"

    def path(self, run: RunConfig, test_day: date) -> Path:
        wh = run.w_h if run.w_h is not None else "-"
        return self.root / run.model_id / f"Wsnn{run.w_snn}_Wh{wh}" / f"seed{run.seed}" / f"{test_day.isoformat()}.npz"

    def has(self, run: RunConfig, test_day: date) -> bool:
        return self.path(run, test_day).is_file()

    def save(self, run: RunConfig, result: FoldResult, diagnostics: bool = True) -> None:
        sig = result.test
        arrays = {"bar_idx": sig.bar_idx}
        for key, value in (sig.diagnostics.items() if diagnostics else ()):
            if key.startswith("spikes_"):
                arrays[key] = value.astype(np.uint16)
            else:
                arrays[key] = value.astype(np.float32)
        meta = {
            "day": sig.day.isoformat(),
            "n_bars": sig.n_bars,
            "model_id": sig.model_id,
            "test_metrics": result.test_metrics,
            "train_metrics": result.train_metrics,
            "train_signal_rate": result.train_signal_rate,
            "test_health": result.test_health,
            "input_stats": result.input_stats,
            "runtime_s": result.runtime_s,
            "extra": result.extra,
        }
        arrays["meta"] = np.array(json.dumps(meta, default=str))

        def write(tmp: Path) -> None:
            with open(tmp, "wb") as fh:  # a file handle keeps numpy from appending ".npz" to the temp name
                np.savez_compressed(fh, **arrays)

        _atomic_write(self.path(run, sig.day), write)

    def load_meta(self, run: RunConfig, test_day: date) -> dict:
        """Only the small metadata of a fold (metrics, health, input stats), without arrays."""
        with np.load(self.path(run, test_day)) as z:
            return json.loads(str(z["meta"]))

    def load(self, run: RunConfig, test_day: date) -> FoldResult:
        with np.load(self.path(run, test_day)) as z:
            meta = json.loads(str(z["meta"]))
            diagnostics = {k: z[k] for k in z.files if k not in ("bar_idx", "meta")}
            test = SignalSeries(
                day=date.fromisoformat(meta["day"]),
                bar_idx=z["bar_idx"],
                n_bars=meta["n_bars"],
                model_id=meta["model_id"],
                diagnostics=diagnostics,
            )
        return FoldResult(
            test=test,
            test_metrics=meta["test_metrics"],
            train_metrics=meta["train_metrics"],
            train_signal_rate=meta["train_signal_rate"],
            test_health=meta["test_health"],
            input_stats=meta["input_stats"],
            runtime_s=meta["runtime_s"],
            extra=meta.get("extra", {}),
        )
