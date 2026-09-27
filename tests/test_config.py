from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from snn_hft.config import (
    ExperimentConfig,
    RunConfig,
    apply_override,
    expand_runs,
    load_config,
    load_raw,
    to_yaml,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "configs"


def make(models=({"model": "paper_snn"},), **kw) -> ExperimentConfig:
    return ExperimentConfig.model_validate({"name": "t", "models": list(models), **kw})


def write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


# --------------------------------------------------------------------------- defaults


def test_defaults_match_plan_section_12():
    cfg = make(models=[{"model": "paper_snn"}, {"model": "hawkes_rstdp"}])
    paper, improved = cfg.models
    core = paper.core
    assert cfg.bars.vwap_num == 10
    assert paper.input.ticks_per_bar == 10 and paper.input.new_mean == 0.2
    assert paper.zscore.new_std == 0.1 and paper.zscore.stats_source == "train_window"
    assert (core.lif.u_rest, core.lif.threshold, core.lif.leak, core.lif.t_ref) == (0.0, 1.0, 0.05, 2)
    assert core.lif.leak_mode == "subtractive"
    assert (core.synapse.w_init_low, core.synapse.w_init_high, core.synapse.w_max) == (0.2, 0.6, 1.0)
    assert core.synapse.delay_ticks == 1
    assert (core.stdp.A, core.stdp.B, core.stdp.tau_plus, core.stdp.tau_minus) == (0.01, -0.0105, 5.0, 5.0)
    assert core.stdp.pairing == "all_to_all"
    assert core.hidden_size == 64 and not core.warm_start and not core.learn_during_test
    assert improved.rstdp.gamma == 1.0 and improved.rstdp.tau_z_bars == 3.0
    assert improved.rstdp.delivery == "bar_end" and not improved.rstdp.reset_eligibility_on_reward
    assert improved.hawkes.restarts == 5
    assert improved.hawkes.mark_fn == "linear_normalized" and improved.hawkes.time_axis == "bar_index"
    assert improved.use_rstdp_pools and improved.keep_direction_pools
    assert (cfg.strategy.momentum_window, cfg.strategy.alf_n, cfg.strategy.stoch_n) == (3, 1, 3)
    assert (cfg.execution.holding, cfg.execution.entry_delay) == (3, 1)
    assert cfg.execution.fee_rate == 0.0 and cfg.execution.latency_ms == 0.0
    assert cfg.backtest.seeds == (0, 1, 2, 3, 4) and cfg.benchmarks.naive_reps == 100
    assert cfg.evaluation.eval_window == 3 and cfg.evaluation.annualization_days == 365


def test_defaults_match_user_decisions():
    cfg = make()
    assert cfg.data.dataset == "aggTrades"
    assert cfg.benchmarks.big_move
    assert cfg.backtest.split == "validation"
    assert cfg.periods.data.start == date(2025, 9, 27) and cfg.periods.data.end == date(2026, 9, 26)
    assert len(cfg.periods.validation.days()) == 45
    assert len(cfg.periods.test.days()) == 300


# --------------------------------------------------------------------------- validation


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        make(bars={"vwap_num": 10, "vwap_nun": 5})


def test_baseline_rejects_improved_model_options():
    for extra in ({"use_rstdp_pools": False}, {"hawkes": {}}, {"rstdp": {"gamma": 2.0}}):
        with pytest.raises(ValidationError):
            make(models=[{"model": "paper_snn", **extra}])


def test_improved_model_needs_a_hidden_pool():
    with pytest.raises(ValidationError):
        make(models=[{"model": "hawkes_rstdp", "use_rstdp_pools": False, "keep_direction_pools": False}])


def test_model_names_unique_and_valid():
    with pytest.raises(ValidationError):
        make(models=[{"model": "paper_snn"}, {"model": "paper_snn"}])
    with pytest.raises(ValidationError):
        make(models=[{"model": "paper_snn", "name": "a__b"}])


def test_history_must_cover_largest_windows():
    make(models=[{"model": "hawkes_rstdp"}], backtest={"w_snn": [1, 10], "w_h": [1, 10]})
    with pytest.raises(ValidationError, match="history_days"):
        make(models=[{"model": "hawkes_rstdp"}], backtest={"w_snn": [10], "w_h": [11]})
    # W_h does not count for a model without Hawkes input.
    make(models=[{"model": "paper_snn"}], backtest={"w_snn": [20], "w_h": [10]})


def test_period_layout_is_checked():
    with pytest.raises(ValidationError, match="history"):
        make(periods={"validation": {"start": "2025-10-01", "end": "2025-11-30"}})
    with pytest.raises(ValidationError, match="after the validation"):
        make(periods={"test": {"start": "2025-11-30", "end": "2026-09-26"}})
    with pytest.raises(ValidationError, match="beyond"):
        make(periods={"test": {"start": "2025-12-01", "end": "2026-09-27"}})


def test_stdp_and_lif_constraints():
    with pytest.raises(ValidationError):
        make(models=[{"model": "paper_snn", "core": {"stdp": {"A": -0.01}}}])
    with pytest.raises(ValidationError):
        make(models=[{"model": "paper_snn", "core": {"lif": {"threshold": 0.0}}}])
    with pytest.raises(ValidationError):
        make(models=[{"model": "paper_snn", "core": {"synapse": {"w_init_high": 1.5}}}])


# --------------------------------------------------------------------------- loading


def test_inherits_merges_mappings_and_replaces_lists(tmp_path):
    write(tmp_path / "base.yaml", {"backtest": {"seeds": [0, 1, 2], "w_snn": [1]}, "bars": {"vwap_num": 20}})
    child = write(
        tmp_path / "child.yaml",
        {"inherits": "base.yaml", "name": "c", "models": [{"model": "paper_snn"}], "backtest": {"seeds": [7]}},
    )
    cfg = load_config(child)
    assert cfg.backtest.seeds == (7,)
    assert cfg.backtest.w_snn == (1,)
    assert cfg.bars.vwap_num == 20


def test_inherits_cycle_is_an_error(tmp_path):
    write(tmp_path / "a.yaml", {"inherits": "b.yaml"})
    write(tmp_path / "b.yaml", {"inherits": "a.yaml"})
    with pytest.raises(ValueError, match="cyclic"):
        load_raw(tmp_path / "a.yaml")


def test_overrides_parse_yaml_values_and_list_indices():
    raw = {"models": [{"model": "paper_snn"}], "backtest": {"seeds": [0]}}
    apply_override(raw, "models.0.core.lif.threshold=1.25")
    apply_override(raw, "backtest.seeds=[3, 4]")
    apply_override(raw, "name=o")
    cfg = ExperimentConfig.model_validate(raw)
    assert cfg.models[0].core.lif.threshold == 1.25
    assert cfg.backtest.seeds == (3, 4)
    with pytest.raises(ValueError):
        apply_override(raw, "no_equals_sign")
    with pytest.raises(KeyError):
        apply_override(raw, "models.5.name=x")


@pytest.mark.parametrize("name", ["paper_baseline.yaml", "improved.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.backtest.split == "test"
    assert cfg.data.dataset == "aggTrades"


# --------------------------------------------------------------------------- runs


def test_expand_runs_counts_and_ids():
    cfg = make(
        models=[{"model": "paper_snn"}, {"model": "hawkes_rstdp"}],
        backtest={"seeds": [0, 1], "w_snn": [1, 3], "w_h": [1, 5]},
        strategy={"rules": ["momentum", "alexanders_filter"]},
    )
    runs = expand_runs(cfg)
    paper = [r for r in runs if r.signal.model == "paper_snn"]
    improved = [r for r in runs if r.signal.model == "hawkes_rstdp"]
    assert len(paper) == 2 * 2 * 2  # rules × w_snn × seeds
    assert len(improved) == 2 * 2 * 2 * 2  # rules × w_snn × w_h × seeds
    assert all(r.w_h is None for r in paper)
    assert len({r.run_id for r in runs}) == len(runs)
    assert "improved__momentum__Wsnn3__Wh5__seed1" in {r.run_id for r in improved}
    assert "paper__alexanders_filter__Wsnn1__seed0" in {r.run_id for r in paper}


def test_model_id_scope():
    def model_id(**kw):
        cfg = make(**kw)
        return expand_runs(cfg)[0].model_id

    base = model_id()
    assert base == model_id()
    assert base.startswith("paper_snn-")
    # Not part of the id: display name, seed, windows, rule, execution.
    assert base == model_id(models=[{"model": "paper_snn", "name": "renamed"}])
    assert base == model_id(backtest={"seeds": [9], "w_snn": [3]})
    assert base == model_id(strategy={"rules": ["stochastic_oscillator"]}, execution={"holding": 5})
    # Part of the id: anything that changes the signals.
    assert base != model_id(models=[{"model": "paper_snn", "core": {"lif": {"threshold": 1.1}}}])
    assert base != model_id(bars={"vwap_num": 20})
    assert base != model_id(data={"dataset": "trades"})


def test_run_config_yaml_roundtrip():
    cfg = make(models=[{"model": "hawkes_rstdp"}])
    run = expand_runs(cfg)[0]
    again = RunConfig.model_validate(yaml.safe_load(to_yaml(run)))
    assert again == run
    assert again.model_id == run.model_id
