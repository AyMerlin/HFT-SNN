# HFT-SNN

Master's thesis code: a reproducible comparison, on identical Binance BTCUSDT perpetual
futures data, between

- **Baseline** — a faithful reimplementation of Gao, Luk, Weston, *High-Frequency Trading and
  Financial Time-Series Prediction with Spiking Neural Networks*, Wilmott (2021): vwap bars,
  double-input SNN with unsupervised STDP, the paper's spike definitions and its three
  trading strategies; and
- **Improved model** — the same pipeline with two changes: a bivariate marked Hawkes
  intensity as input (memory of past jumps), and reward-modulated STDP pools that learn the
  momentum/reversion distinction during training.

The full specification is [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md); every
interpretation and deviation is logged in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

For the exact package versions used for the results, install from `requirements-lock.txt`.

## Usage

```bash
.venv/bin/python -m experiments.run_experiment --config experiments/configs/paper_baseline.yaml
```

- `--set key.path=value` overrides any config value, e.g. `--set backtest.seeds=[0]`.
- `--resolve-only` validates the config and writes the resolved run configs without running.

Results go to `results/<experiment>/<run_id>/` (not committed); downloaded data to `data/`.

Download and cache the study period (about 4 GB of parquet), then write the coverage report:

```bash
.venv/bin/python -m experiments.download_data --config experiments/configs/base.yaml
.venv/bin/python -m experiments.data_report --config experiments/configs/base.yaml
```

Run the tests:

```bash
.venv/bin/python -m pytest
```

## Layout

| Path | Contents |
|---|---|
| `snn_hft/config` | config schemas holding all defaults, YAML loading |
| `snn_hft/data` | Binance / local / synthetic sources, parquet store |
| `snn_hft/preprocessing` | causal preprocessing steps (bars, differences, normalisation, Hawkes steps) |
| `snn_hft/models/hawkes` | bivariate marked Hawkes process |
| `snn_hft/snn` | numba SNN engine shared by both models |
| `snn_hft/signals` | signal models: paper SNN, Hawkes + R-STDP SNN, benchmarks |
| `snn_hft/evaluation` | spike metrics and diagnostics |
| `snn_hft/strategy` | the paper's direction rules and execution |
| `snn_hft/backtest` | walk-forward backtester, performance, result store |
| `snn_hft/analysis` | comparison tables, statistics, plots |
| `snn_hft/testing` | causality harness |
| `experiments/` | CLI and YAML configs |

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | skeleton, config system, CLI, design-decisions log | done |
| M1 | data loading and parquet store | done — [coverage](docs/reports/m1_data_coverage.md) |
| M2 | baseline preprocessing, causality harness | done |
| M3 | SNN engine and profiling | done |
| M4 | paper SNN signal model, spike evaluator | done — [spike check](docs/reports/m4_spike_check.md) |
| M5 | strategies, backtester, E1 | done — [E1 report](docs/reports/m5_e1_baseline.md) |
| M6 | Hawkes module | done — [real-data review](docs/reports/m6_hawkes_review.md) |
| M7 | improved preprocessing | done — [check](docs/reports/m7_improved_preprocessing.md) |
| M8 | R-STDP, improved model, E2, A1 | in progress |
| M9 | window experiments | |
| M10 | analysis and comparison report | |
