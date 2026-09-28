"""Result layout (§8.6): ``results/<experiment>/<run_id>/...``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from snn_hft.config.loader import dump_config
from snn_hft.config.schema import RunConfig
from snn_hft.utils.repro import collect_meta


class ResultStore:
    def __init__(self, results_dir: str | Path, experiment: str):
        self.root = Path(results_dir) / experiment

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_run_header(self, run: RunConfig, **meta: Any) -> Path:
        path = self.run_dir(run.run_id)
        dump_config(run, path / "config.yaml")
        (path / "meta.json").write_text(json.dumps(collect_meta(seed=run.seed, model_id=run.model_id, **meta), indent=2, default=str))
        return path

    def write_json(self, run_id: str, name: str, obj: Any) -> None:
        (self.run_dir(run_id) / name).write_text(json.dumps(obj, indent=2, default=_json_default))

    def write_frame(self, run_id: str, name: str, df: pd.DataFrame) -> Path:
        path = self.run_dir(run_id) / name
        if name.endswith(".parquet"):
            df.to_parquet(path, compression="zstd", index=False)
        else:
            df.to_csv(path, index=False)
        return path

    def link(self, source: Path, run_id: str, name: str) -> None:
        """Hard-link a strategy-independent file into another run directory (no extra disk)."""
        target = self.run_dir(run_id) / name
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            target.write_bytes(source.read_bytes())

    def write_summary(self, name: str, df: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / name
        df.to_csv(path, index=False)
        return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)
