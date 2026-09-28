"""Run an experiment described by one YAML config.

    python -m experiments.run_experiment --config experiments/configs/paper_baseline.yaml
    python -m experiments.run_experiment --config ... --set 'backtest.seeds=[0]' --resolve-only

Every run gets ``results/<experiment>/<run_id>/`` with its resolved ``config.yaml`` and
``meta.json`` (git commit, package versions, machine, seed). A full run adds the result
files of plan §8.6, plus ``table3.csv`` / ``table4.csv`` for the experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from snn_hft.analysis.compare import format_table3, format_table4, table3, table4
from snn_hft.backtest.backtester import Backtester
from snn_hft.backtest.store import ResultStore
from snn_hft.config import dump_config, expand_runs, load_config
from snn_hft.utils.repro import collect_meta


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="experiment YAML file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override a config value (repeatable), e.g. models.0.core.lif.threshold=1.2",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="validate the config and write resolved run configs without running",
    )
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: all cores)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    runs = expand_runs(cfg)

    exp_dir = cfg.output.results_dir / cfg.name
    dump_config(cfg, exp_dir / "experiment.yaml")
    period = cfg.eval_period
    print(
        f"{cfg.name}: {len(runs)} runs, {len(cfg.models)} model(s), "
        f"{cfg.backtest.split} period {period.start}..{period.end} ({len(period.days())} days)"
    )
    if args.resolve_only:
        for run in runs:
            run_dir = exp_dir / run.run_id
            dump_config(run, run_dir / "config.yaml")
            meta = collect_meta(seed=run.seed, model_id=run.model_id, config_file=str(args.config))
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"resolved configs written to {exp_dir}")
        return 0

    result = Backtester(cfg, workers=args.workers).run()
    store = ResultStore(cfg.output.results_dir, cfg.name)
    t3, t4 = table3(result.table3_rows), table4(result.table4_rows)
    store.write_summary("table3.csv", t3)
    store.write_summary("table4.csv", t4)
    print("\nTable 3 — spikes\n" + format_table3(t3))
    print("\nTable 4 — strategies\n" + format_table4(t4))
    print(f"\nresults in {exp_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
