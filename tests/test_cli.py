import json
from pathlib import Path

import yaml

from experiments.run_experiment import main
from snn_hft.config import RunConfig

CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "paper_baseline.yaml"


def test_resolve_only_writes_resolved_configs_and_meta(tmp_path):
    code = main(
        [
            "--config",
            str(CONFIG),
            "--resolve-only",
            "--set",
            f"output.results_dir={tmp_path}",
            "--set",
            "backtest.seeds=[0, 1]",
        ]
    )
    assert code == 0

    exp_dir = tmp_path / "e1_paper_baseline"
    assert (exp_dir / "experiment.yaml").is_file()
    run_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    assert len(run_dirs) == 3 * 2  # three strategies × two seeds

    run_dir = exp_dir / "paper__momentum__Wsnn1__seed1"
    run = RunConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text()))
    assert run.seed == 1 and run.rule == "momentum" and run.split == "test"

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seed"] == 1
    assert meta["model_id"] == run.model_id
    assert {"commit", "dirty"} <= meta["git"].keys()
    assert meta["packages"]["numba"] is not None
