"""Configuration schemas.

Every default of the implementation plan (§11, §12) and of the decisions in
docs/DESIGN_DECISIONS.md lives here, so a YAML file only states what differs.
Unknown keys are rejected, which keeps typos from silently falling back to
defaults and keeps the baseline model from accepting improved-model options.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from snn_hft.utils.repro import stable_hash

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")

# Bump whenever code that determines signals changes (preprocessing, encoder, SNN kernels,
# Hawkes model): it is part of every model id, so stale cached signals are never reused.
SIGNAL_CODE_VERSION = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DateRange(_Strict):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.end < self.start:
            raise ValueError(f"end {self.end} is before start {self.start}")
        return self

    def days(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]

    def __contains__(self, day: date) -> bool:
        return self.start <= day <= self.end


# --------------------------------------------------------------------------- data


class DataConfig(_Strict):
    source: Literal["binance", "local", "synthetic"] = "binance"
    venue: str = "binance_um"
    symbol: str = "BTCUSDT"
    dataset: Literal["aggTrades", "trades"] = "aggTrades"  # decision U1
    raw_dir: Path = Path("data/raw")
    local_dir: Path | None = None  # input directory for source="local"


class PeriodsConfig(_Strict):
    """Study period (decision U4). All development happens on `validation`."""

    data: DateRange = DateRange(start=date(2025, 9, 27), end=date(2026, 9, 26))
    history_days: int = Field(20, ge=0)
    validation: DateRange = DateRange(start=date(2025, 10, 17), end=date(2025, 11, 30))
    test: DateRange = DateRange(start=date(2025, 12, 1), end=date(2026, 9, 26))

    @model_validator(mode="after")
    def _layout(self) -> PeriodsConfig:
        first_evaluable = self.data.start + timedelta(days=self.history_days)
        if self.validation.start < first_evaluable:
            raise ValueError(
                f"validation starts {self.validation.start}, but needs {self.history_days} "
                f"history days after data start {self.data.start}"
            )
        if self.test.start <= self.validation.end:
            raise ValueError("test period must start after the validation period ends")
        if self.test.end > self.data.end:
            raise ValueError("test period extends beyond the data range")
        return self


class BarsConfig(_Strict):
    vwap_num: int = Field(10, gt=0)


# --------------------------------------------------------------------------- SNN engine (shared)


class InputConfig(_Strict):
    ticks_per_bar: int = Field(10, gt=0)  # T, §11 #5
    new_mean: float = Field(0.2, gt=0, le=1)  # mean per-tick input spike probability


class ZScoreConfig(_Strict):
    """Baseline-only normalisation (§4.1)."""

    new_std: float = Field(0.1, gt=0)
    stats_source: Literal["train_window", "same_day"] = "train_window"  # §11 #4


class LIFConfig(_Strict):
    u_rest: float = 0.0
    threshold: float = 1.0
    leak: float = Field(0.05, ge=0)  # subtractive leak per tick
    leak_mode: Literal["subtractive", "exponential"] = "subtractive"
    tau_m: float = Field(20.0, gt=0)  # ticks; only used by leak_mode="exponential"
    t_ref: int = Field(2, ge=0)

    @model_validator(mode="after")
    def _threshold_above_rest(self) -> LIFConfig:
        if self.threshold <= self.u_rest:
            raise ValueError("threshold must exceed u_rest")
        return self


class SynapseConfig(_Strict):
    delay_ticks: int = Field(1, ge=1)  # §11 #17; ≥ 1 so layer update order never matters
    w_init_low: float = Field(0.2, ge=0)
    w_init_high: float = 0.6
    w_max: float = Field(1.0, gt=0)

    @model_validator(mode="after")
    def _init_range(self) -> SynapseConfig:
        if not self.w_init_low <= self.w_init_high <= self.w_max:
            raise ValueError("need 0 <= w_init_low <= w_init_high <= w_max")
        return self


class STDPConfig(_Strict):
    A: float = Field(0.01, gt=0)
    B: float = -0.0105  # §11 #1: the paper prints B > 0, its text says "weakened"
    tau_plus: float = Field(5.0, gt=0)  # ticks
    tau_minus: float = Field(5.0, gt=0)  # ticks
    pairing: Literal["all_to_all", "nearest"] = "all_to_all"


class SNNCoreConfig(_Strict):
    """Engine settings shared by both models (§6.1)."""

    hidden_size: int = Field(64, gt=0)  # a hyperparameter in the paper
    lif: LIFConfig = Field(default_factory=LIFConfig)
    synapse: SynapseConfig = Field(default_factory=SynapseConfig)
    stdp: STDPConfig = Field(default_factory=STDPConfig)
    warm_start: bool = False  # §11 #10
    epochs: int = Field(1, ge=1)
    learn_during_test: bool = False  # §11 #8
    reset_state_per_day: bool = True  # §11 #9


# --------------------------------------------------------------------------- signal models


class HawkesConfig(_Strict):
    time_axis: Literal["bar_index", "wallclock"] = "bar_index"
    mark_fn: Literal["unmarked", "linear_normalized", "saturating"] = "linear_normalized"
    limit: Literal["right", "left"] = "right"  # λ(t+) or λ(t-) as network input
    restarts: int = Field(5, ge=1)
    beta_init_grid: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
    max_iter: int = Field(500, gt=0)
    # Decision U9: a bar is an event only if |d_t| exceeds this quantile of the non-zero |d| on the
    # fit window (frozen in θ_d). None = every non-zero move (spec §4.2).
    event_quantile: float | None = Field(0.9, gt=0, lt=1)


class RSTDPConfig(_Strict):
    gamma: float = 1.0
    tau_z_bars: float = Field(3.0, ge=0)  # 0 means E = ξ (no eligibility memory)
    reset_eligibility_on_reward: bool = False
    delivery: Literal["bar_end", "every_tick"] = "bar_end"


class _SignalBase(_Strict):
    name: str

    @model_validator(mode="after")
    def _valid_name(self):
        if not _NAME_RE.match(self.name) or "__" in self.name:
            raise ValueError(f"invalid model name {self.name!r}")
        return self

    @property
    def uses_hawkes(self) -> bool:
        return False


class PaperSNNConfig(_SignalBase):
    """Baseline (§6.2). Accepts only the §11 ambiguity settings, not architecture changes."""

    model: Literal["paper_snn"] = "paper_snn"
    name: str = "paper"
    input: InputConfig = Field(default_factory=InputConfig)
    zscore: ZScoreConfig = Field(default_factory=ZScoreConfig)
    core: SNNCoreConfig = Field(default_factory=SNNCoreConfig)


class HawkesRSTDPConfig(_SignalBase):
    """Improved model (§6.3)."""

    model: Literal["hawkes_rstdp"] = "hawkes_rstdp"
    name: str = "improved"
    input: InputConfig = Field(default_factory=InputConfig)
    core: SNNCoreConfig = Field(default_factory=SNNCoreConfig)
    hawkes: HawkesConfig = Field(default_factory=HawkesConfig)
    rstdp: RSTDPConfig = Field(default_factory=RSTDPConfig)
    use_rstdp_pools: bool = True  # false = ablation A1 (Problem 1 only)
    keep_direction_pools: bool = True  # false = ablation A2

    @model_validator(mode="after")
    def _has_hidden_pools(self) -> HawkesRSTDPConfig:
        if not (self.use_rstdp_pools or self.keep_direction_pools):
            raise ValueError("at least one of use_rstdp_pools / keep_direction_pools must be true")
        return self

    @property
    def uses_hawkes(self) -> bool:
        return True


SignalConfig = Annotated[PaperSNNConfig | HawkesRSTDPConfig, Field(discriminator="model")]


# --------------------------------------------------------------------------- strategy / evaluation


RuleName = Literal["momentum", "alexanders_filter", "stochastic_oscillator"]


class StrategyConfig(_Strict):
    rules: tuple[RuleName, ...] = ("momentum", "alexanders_filter", "stochastic_oscillator")
    momentum_window: int = Field(3, gt=0)
    alf_n: int = Field(1, gt=0)
    stoch_n: int = Field(3, gt=1)


class ExecutionConfig(_Strict):
    """Paper's common assumptions (§7.2) plus the latency option (decision U2)."""

    entry_delay: int = Field(1, ge=1)
    holding: int = Field(3, ge=1)
    latency_ms: float = Field(0.0, ge=0)  # 0 reproduces the paper's entry rule
    fee_rate: float = Field(0.0, ge=0)  # decision U3: costs out of scope for now


class HealthConfig(_Strict):
    output_rate_min: float = 0.001  # fraction of bars with an output signal
    output_rate_max: float = 0.5


class EvaluationConfig(_Strict):
    eval_window: int = Field(3, gt=0)  # §11 #12
    annualization_days: int = Field(365, gt=0)  # §11 #14
    health: HealthConfig = Field(default_factory=HealthConfig)


class BenchmarksConfig(_Strict):
    naive_reps: int = Field(100, ge=0)  # random-timing strategies, §6.6
    big_move: bool = True  # decision U5


class BacktestConfig(_Strict):
    # Defaults to the validation period so the test period is only used on purpose.
    split: Literal["validation", "test"] = "validation"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    w_snn: tuple[int, ...] = (1,)
    w_h: tuple[int, ...] = (1,)
    latencies_ms: tuple[float, ...] = (0.0,)

    @model_validator(mode="after")
    def _non_empty(self) -> BacktestConfig:
        for field in ("seeds", "w_snn", "w_h", "latencies_ms"):
            values = getattr(self, field)
            if not values:
                raise ValueError(f"backtest.{field} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"backtest.{field} contains duplicates")
        if min(self.w_snn) < 1 or min(self.w_h) < 1:
            raise ValueError("window sizes must be >= 1 day")
        if min(self.latencies_ms) < 0:
            raise ValueError("latencies must be >= 0")
        return self


class OutputConfig(_Strict):
    results_dir: Path = Path("results")
    cache_dir: Path = Path("cache")
    # trades.parquet costs ~1–4 MB per run and day: keep it for the lowest seed of each job by default
    save_trades: Literal["none", "first_seed", "all"] = "first_seed"
    # per-bar pool spike counts and input probabilities in the fold cache (needed for signals.parquet
    # and the diagnostics of §9.2; tuning only needs metrics)
    cache_diagnostics: bool = True


# --------------------------------------------------------------------------- experiment / run


class _Common(_Strict):
    data: DataConfig = Field(default_factory=DataConfig)
    periods: PeriodsConfig = Field(default_factory=PeriodsConfig)
    bars: BarsConfig = Field(default_factory=BarsConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    benchmarks: BenchmarksConfig = Field(default_factory=BenchmarksConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class ExperimentConfig(_Common):
    """One YAML file: one or more signal models evaluated on identical days."""

    name: str
    description: str = ""
    models: tuple[SignalConfig, ...] = Field(min_length=1)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @model_validator(mode="after")
    def _consistent(self) -> ExperimentConfig:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid experiment name {self.name!r}")
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError(f"model names must be unique, got {names}")
        # §8.1: the first evaluated day needs max(W_snn) + max(W_h) prior days.
        needed = max(self.backtest.w_snn) + (
            max(self.backtest.w_h) if any(m.uses_hawkes for m in self.models) else 0
        )
        if self.periods.history_days < needed:
            raise ValueError(
                f"periods.history_days={self.periods.history_days} < max(w_snn)+max(w_h)={needed}"
            )
        return self

    @property
    def eval_period(self) -> DateRange:
        return getattr(self.periods, self.backtest.split)


class RunConfig(_Common):
    """Fully resolved description of one run (one model, rule, window pair and seed)."""

    experiment: str
    run_id: str
    signal: SignalConfig
    rule: RuleName
    split: Literal["validation", "test"]
    w_snn: int
    w_h: int | None
    seed: int
    latencies_ms: tuple[float, ...]

    @property
    def model_id(self) -> str:
        """Stable id of everything that determines the signals, except windows and seed.

        Windows and seed are separate parts of the signal-cache key (§8.3). The
        model's display name is excluded, so renaming a model keeps its cache.
        """
        payload = {
            "code": SIGNAL_CODE_VERSION,
            "data": self.data.model_dump(mode="json", include={"venue", "symbol", "dataset"}),
            "bars": self.bars.model_dump(mode="json"),
            "signal": self.signal.model_dump(mode="json", exclude={"name"}),
        }
        return f"{self.signal.model}-{stable_hash(payload)}"


def make_run_id(model_name: str, rule: str, w_snn: int, w_h: int | None, seed: int) -> str:
    parts = [model_name, rule, f"Wsnn{w_snn}"]
    if w_h is not None:
        parts.append(f"Wh{w_h}")
    parts.append(f"seed{seed}")
    return "__".join(parts)


def expand_runs(cfg: ExperimentConfig) -> list[RunConfig]:
    """All runs of an experiment: models × rules × W_snn × W_h (Hawkes models only) × seeds."""
    common = {field: getattr(cfg, field) for field in _Common.model_fields}
    runs = []
    for model in cfg.models:
        w_h_values: tuple[int | None, ...] = cfg.backtest.w_h if model.uses_hawkes else (None,)
        for rule in cfg.strategy.rules:
            for w_snn in cfg.backtest.w_snn:
                for w_h in w_h_values:
                    for seed in cfg.backtest.seeds:
                        runs.append(
                            RunConfig(
                                **common,
                                experiment=cfg.name,
                                run_id=make_run_id(model.name, rule, w_snn, w_h, seed),
                                signal=model,
                                rule=rule,
                                split=cfg.backtest.split,
                                w_snn=w_snn,
                                w_h=w_h,
                                seed=seed,
                                latencies_ms=cfg.backtest.latencies_ms,
                            )
                        )
    return runs
