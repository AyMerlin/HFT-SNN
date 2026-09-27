# Implementation Plan — SNN High-Frequency Trading: Paper Baseline vs. Hawkes + R-STDP Model

**Audience:** an implementing agent.
**Language:** Python ≥ 3.11, object-oriented.
**Reference paper:** K. Gao, W. Luk, S. Weston, *High-Frequency Trading and Financial Time-Series Prediction with Spiking Neural Networks*, Wilmott Magazine (2021). https://www.doc.ic.ac.uk/~wl/papers/21/wilm21kg.pdf — read it before starting.

---

## Instructions to the implementing agent

**Your task:** build the Python framework described in this document, from an empty repository, so it produces a reproducible comparison between the paper's method and the improved method on BTCUSDT futures data.

**How to work:**
1. Read this entire document and the reference paper before writing code.
2. Before coding, reply with:
   - a short summary of your understanding;
   - any contradictions or gaps you find in this spec;
   - your planned order of work.
3. Implement milestone by milestone, in the order of §14 (M0 → M10). Do not start a milestone until the previous milestone's acceptance criteria and tests pass.
4. Write the tests from §13 alongside the code they cover, not at the end.
5. After each milestone, commit with a message naming the milestone, then give a short status report:
   - what was built;
   - test results;
   - any deviations from this spec;
   - open questions.
6. Record every interpretation or deviation in `docs/DESIGN_DECISIONS.md`: what the spec says, what you did, and why.

**When to decide yourself and when to ask:**
- **Decide yourself:** naming, internal code structure, and minor implementation details not fixed by this spec. Log them in `DESIGN_DECISIONS.md`.
- **Stop and ask:**
  - if a spec requirement seems wrong, contradictory, or infeasible;
  - if you would need to change anything in the baseline model's architecture or preprocessing;
  - if data cannot be obtained, e.g. Binance is unreachable from your environment. Then implement `LocalFileDataSource` and `SyntheticDataSource`, continue with synthetic data, and ask for the files;
  - if simulating one day of data is too slow to make the experiments practical.

**Mandatory check-in points.** Pause and wait for approval before continuing:
- After **M1**: data coverage (days, trades per day, bars per day at `num=10`).
- After **M5**: baseline replication results (Table-3/Table-4-style output for E1). This validates the baseline before anything is built on it.
- After **M8**: first improved-model results (E2, A1).

**Hard rules:**
- Do not implement anything listed as out of scope in §0.
- Do not tune hyperparameters on the test period.
- Never let the causality tests (§13) fail.

**Done means:** all milestones M0–M10 meet their acceptance criteria, all tests pass, and the final comparison report (§9, §14) is produced from saved results.

---

## 0. Goal and scope

Build a modular framework that lets us **compare, on identical data and identical evaluation, the paper's original method against an improved method**, on Binance BTC perpetual futures (BTCUSDT, USDⓈ-M).

### In scope
1. **Baseline:** a faithful reimplementation of the paper's pipeline (preprocessing, double-input SNN, unsupervised STDP, spike definitions, three strategies, naive strategies, metrics).
2. **Improved model:** the baseline with exactly two problems fixed (§1.2):
   - **Problem 1:** memoryless Poisson input → replaced by a bivariate, marked Hawkes intensity.
   - **Problem 2:** momentum/reversion distinction absent from training → added via reward-modulated STDP (MSTDPET) and two new hidden pools.
3. **Training-window experiments:** rolling training windows of 1, 3, 5, and 10 days, varied separately for the Hawkes fit and for SNN training.

### Out of scope (do NOT implement; leave clean extension points only)
- Nessler/Kappel winner-take-all or Bayesian reformulation, lateral inhibition, softmax firing.
- Loihi / neuromorphic deployment or resource analysis.
- The paper's LSTM and ARIMA comparison baselines. Leave a `SignalModel` extension point for them.
- Resonate-and-fire neurons, surrogate-gradient training, deeper networks.
- Any change to the strategies that consumes momentum/reversion information. Strategies must stay identical to the paper.

### Non-negotiable principles
1. **The baseline is not "improved."** Where the paper is ambiguous, use the default in §11 and make it configurable. Log every interpretation in `docs/DESIGN_DECISIONS.md`.
2. **The improved model differs from the baseline only by the changes in §6.3–§6.5.**
   - Everything else is shared code, not copied code: vwap bars, LIF neuron model, STDP constants, Poisson encoder, output layer, signal extraction, strategies, evaluation.
   - Shared code makes the difference between the models structural and auditable.
3. **Causality contract (critical).**
   - Anything used to produce a signal on day `d` may depend only on data strictly before day `d`, plus day-`d` data up to the current bar.
   - Every `fit()` receives only training data. Every `transform()`/`generate()` uses fitted state plus same-day past.
   - There must be an automated test for this (§13).
4. **Fairness.** Both models are evaluated on the same test days, with the same seeds, the same hyperparameter-tuning budget, and **matched mean input firing rate** (§6.4.3).
5. **Reproducibility.** Every run is fully described by one YAML config plus a seed. Configs, git hash, and package versions are saved with results.

---

## 1. Background for the implementer

### 1.1 Paper pipeline (baseline)

The pipeline runs in this order:

1. Raw trades → volume-weighted average price (vwap) bars of `num` consecutive transactions (paper: `num = 10`).
2. Price difference `d_t = V_t − V_{t−1}`, giving two series:
   - price difference `A_t = d_t`
   - negative price difference `B_t = −d_t`
3. Each series is normalized:
   ```
   z = max((x − mean(x)) / stdev(x) · new_stdev + new_mean, 0)
   ```
4. Each normalized value is Poisson-encoded into a spike train of `T` ticks.
5. The spike trains drive the network:
   - Double-input SNN: X1 → H1 (64 LIF), X2 → H2 (64 LIF), H1 ∪ H2 → one output LIF.
   - Weights are learned by unsupervised pairwise STDP.
6. Output spikes are price-spike signals.
7. The signals are evaluated as real/fake and momentum/reversion. They also drive three strategies (momentum, Alexanders filter, stochastic oscillator), which choose direction from price only.
8. Training is walk-forward: train on day `k`, test on day `k+1`.

### 1.2 The two problems being fixed

**Problem 1 — memoryless input encoding.**
- Poisson encoding of a per-bar normalized value means the encoded spike probability depends only on the current bar.
- Past jumps have no influence on it. A jump inside an active cluster is encoded exactly like an isolated jump of the same size.
- **Fix:** model up- and down-moves as a bivariate, mutually-exciting, marked Hawkes point process. Encode its conditional intensity instead of the z-score.
- The four excitation channels are:

  | Channel | Meaning |
  |---|---|
  | `α_uu` | up excites up (momentum-type) |
  | `α_dd` | down excites down (momentum-type) |
  | `α_ud` | down excites up (reversion-type) |
  | `α_du` | up excites down (reversion-type) |

- Parameters are fit by MLE on a rolling window of past days, frozen, and applied to the next day.

**Problem 2 — no momentum/reversion distinction during training.**
- The paper only labels spikes as momentum/reversion after training, for evaluation. Its own Future Work section names this as a limitation.
- **Fix, three parts:**
  - (a) Use the Hawkes fit's causal per-event branching probabilities as a momentum/reversion signal.
  - (b) Feed that signal into training through reward-modulated STDP (MSTDPET) with eligibility traces. The Hebbian timing rule is preserved inside the eligibility trace.
  - (c) Add two new hidden pools, `H_mom` and `H_rev`. Each receives input from both X1 and X2 and gets its own reward signal. Without separate pools, momentum and reversion rewards would collide in the same synapses.
- The single output neuron and the strategies stay unchanged.

---

## 2. Framework architecture

### 2.1 Modules

```
DataLoading ──► Preprocessing ──► SignalCreation ──► Strategy ──► Backtesting ──► Analysis
 (raw→std)      (per model)        (SNN models)       (paper)      (walk-forward)   (compare)
```

### 2.2 Repository layout

```
snn_hft/
  config/                 # pydantic/dataclass config schemas + YAML loader
  data/
    sources.py            # DataSource ABC, BinanceFuturesDataSource, LocalFileDataSource, SyntheticDataSource
    store.py              # DataStore (parquet cache), TradingCalendar (UTC days)
    containers.py         # TradeFrame, BarSeries, DayData
  preprocessing/
    base.py               # PreprocessingStep ABC, Pipeline
    bars.py               # VWAPBarAggregator
    returns.py            # PriceDifferencer, DirectionalSplitter
    normalization.py      # ShiftedZScoreNormalizer, IntensityRateScaler
    events.py             # EventExtractor
    hawkes_steps.py       # HawkesIntensityTransformer, BranchingRewardComputer
  models/hawkes/
    kernels.py            # Kernel ABC, ExponentialKernel
    marks.py              # MarkFunction ABC, Unmarked, LinearNormalized, Saturating
    process.py            # BivariateHawkesProcess (fit, intensity, branching, loglik, simulate)
    params.py             # HawkesParameters (frozen dataclass, JSON-serializable)
  snn/
    encoding.py           # SpikeEncoder ABC, PoissonEncoder
    neurons.py            # NeuronPopulation ABC, LIFPopulation
    synapses.py           # SynapseGroup
    learning.py           # LearningRule ABC, NoLearning, PairwiseSTDP, RewardModulatedSTDP
    network.py            # SpikingNetwork (generic), NetworkBuilder
    simulator.py          # numba-jitted simulation kernels
  signals/
    base.py               # SignalModel ABC, SignalSeries
    paper_snn.py          # PaperSNNSignalModel (baseline, frozen spec)
    hawkes_snn.py         # HawkesRSTDPSignalModel (improved)
    random_signal.py      # RandomSignalModel (for naive strategies)
  evaluation/
    spike_metrics.py      # SpikeEvaluator (real/fake, momentum/reversion)
    diagnostics.py        # Problem-1 / Problem-2 / Hawkes diagnostics
  strategy/
    direction_rules.py    # DirectionRule ABC, MomentumRule, AlexandersFilterRule, StochasticOscillatorRule
    execution.py          # ExecutionConfig, Trade
    strategy.py           # Strategy
  backtest/
    splitter.py           # WalkForwardSplitter
    backtester.py         # Backtester
    performance.py        # PerformanceCalculator
    store.py              # ResultStore
    cache.py              # SignalCache, HawkesParamCache
  analysis/
    loader.py             # ResultLoader
    compare.py            # ComparisonTable, statistical tests
    plots.py              # plotting utilities
experiments/
  configs/                # paper_baseline.yaml, improved.yaml, sweeps ...
  run_experiment.py       # CLI: python -m experiments.run_experiment --config ...
tests/
docs/DESIGN_DECISIONS.md
README.md
```

**Stack:** numpy, pandas, pyarrow, numba, scipy, pyyaml, pydantic (or dataclasses), matplotlib, requests, pytest.

- Implement the SNN simulator yourself with numba. Do **not** use Brian2, BindsNET, or snnTorch. We need exact control over the paper's discrete update rules.
- Implement the Hawkes process yourself with numba plus scipy.optimize. The `tick` library is unmaintained.

### 2.3 Core data containers

| Container | Content |
|---|---|
| `TradeFrame` | One symbol, one UTC day. DataFrame columns: `ts_ns:int64, price:float64, qty:float64, is_buyer_maker:bool, trade_id:int64`, sorted by `(ts_ns, trade_id)`. Metadata: `symbol, venue, day`. |
| `BarSeries` | Columns: `bar_idx:int, ts_end_ns:int64, vwap:float64, volume:float64, n_trades:int`. |
| `DayData` | A day moving through the pipeline: `day`, `bars`, and optional fields `d`, `channels_raw (N×2)`, `channel_prob (N×2, per-tick spike prob ∈ [0,1])`, `events`, `lambda_u/lambda_d`, `branching`, `reward_mom/reward_rev`. |
| `SignalSeries` | `day`, `bar_idx: np.ndarray[int]` (bars where the model emitted a signal), `model_id`, `diagnostics: dict` (e.g. per-bar pool spike counts). |
| `Trade` | `day, signal_bar, entry_bar, exit_bar, direction(±1), entry_price, exit_price, gross_return, fee, net_return`. |
| `FoldSpec` | `train_days: list[date]`, `test_day: date`. |

### 2.4 Key abstract interfaces (sketch)

```python
class DataSource(ABC):
    def fetch_day(self, symbol: str, day: date) -> TradeFrame: ...

class PreprocessingStep(ABC):
    def fit(self, train_days: list[DayData]) -> "PreprocessingStep": ...   # learns state ONLY from training data
    def transform(self, day: DayData) -> DayData: ...                        # causal: fitted state + same-day past

class SignalModel(ABC):
    def fit(self, train_days: list[DayData], seed: int) -> "SignalModel": ...
    def generate(self, day: DayData, seed: int) -> SignalSeries: ...
    @property
    def model_id(self) -> str: ...                                             # stable hash of config

class LearningRule(ABC):
    def on_tick(self, synapses: SynapseGroup, pre_s, post_s, tick: int) -> None: ...
    def on_reward(self, synapses: SynapseGroup, reward: float) -> None: ...  # no-op for plain STDP

class DirectionRule(ABC):
    def direction(self, vwap: np.ndarray, t: int) -> int: ...                # +1 long, −1 short, 0 none

class Strategy:
    def __init__(self, signal_model: SignalModel, direction_rule: DirectionRule, execution: ExecutionConfig): ...
    def trades_for_day(self, day: DayData, signals: SignalSeries) -> list[Trade]: ...
```

**Inheritance, where it is needed:**
- `PaperSNNSignalModel` and `HawkesRSTDPSignalModel` share a common `SNNSignalModel` base. The base owns the pipeline, network, and simulator plumbing and the signal extraction.
- `RewardModulatedSTDP` reuses `PairwiseSTDP`'s trace computation. Its per-tick STDP term `ξ` must come from the same function (§6.5).

---

## 3. Module 1 — Data Loading

### 3.1 Sources
- **`BinanceFuturesDataSource`** (primary).
  - Bulk historical data from Binance's public data archive, daily files for USDⓈ-M futures. Expected path (verify against current Binance docs):
    `https://data.binance.vision/data/futures/um/daily/trades/BTCUSDT/BTCUSDT-trades-YYYY-MM-DD.zip`
    Each file has a `.CHECKSUM` companion; verify it.
  - Default dataset: `trades` (individual transactions, closest to the paper's "transactions"). `aggTrades` is available as an option.
  - Handle files with and without header rows. Expected columns: `id, price, qty, quote_qty, time, is_buyer_maker`.
  - REST fallback for recent days: `GET /fapi/v1/aggTrades` on `fapi.binance.com` (paginate, `limit ≤ 1000`). Verify endpoint names and limits against current docs before relying on them.
- **`LocalFileDataSource`:** reads already-downloaded zip/CSV/parquet files. Use this if the environment cannot reach Binance.
- **`SyntheticDataSource`:** generates trade streams, for example prices driven by a simulated bivariate Hawkes process. Used for tests and for the Hawkes parameter-recovery test.

### 3.2 Standardization and storage
- Convert to `TradeFrame` and store as parquet: `data/raw/{venue}/{symbol}/{YYYY-MM-DD}.parquet`.
- **Trading day:** crypto trades 24/7, so one trading day = one UTC calendar day `[00:00:00, 24:00:00)`. End-of-day position closing happens at the last bar of that UTC day.
- **Contract:** BTCUSDT perpetual (a single contract, so no dominant-contract roll is needed). Leave an extension point for dated quarterly contracts with dominant-by-volume selection, mirroring the paper's "dominant contract."
- `DataStore.get_days(symbol, start, end) → list[TradeFrame]`, with lazy download and caching.

---

## 4. Module 2 — Preprocessing

All steps implement `PreprocessingStep`. A `Pipeline` fits steps in order: step *i* is fit on training days that have been transformed by steps `< i`.

Each step that learns state uses its own trailing window from the training days it receives (§8.2).

### 4.1 Baseline steps (paper-exact)

| Step | Spec |
|---|---|
| `VWAPBarAggregator(num=10)` | Group every `num` consecutive trades of the day. `vwap = Σ p_i q_i / Σ q_i`. Bar timestamp = last trade's timestamp. Drop an incomplete final group. Paper: "window length 10". |
| `PriceDifferencer` | `d_t = V_t − V_{t−1}` for `t ≥ 1`. Bar 0 has no `d`. |
| `DirectionalSplitter` | `channels_raw[:,0] = d_t` ("price difference", X1). `channels_raw[:,1] = −d_t` ("negative price difference", X2). Paper: obtained by multiplying by −1. |
| `ShiftedZScoreNormalizer(new_mean, new_std, stats_source)` | Per channel: `z = max((x − mean(x))/stdev(x) · new_std + new_mean, 0)`, then `channel_prob = clip(z, 0, 1)`, interpreted as per-tick spike probability (§6.2). `mean`/`stdev` come from the training window (default, causal), or from the same day if `stats_source="same_day"` (paper-literal option). The paper's typeset `x_i − * mean(x)` is a typo for `x_i − mean(x)`. |

### 4.2 Improved-model steps (in place of `ShiftedZScoreNormalizer`)

These replace **only** the normalization step. Bars, differencing, and splitting are shared.

| Step | Spec |
|---|---|
| `EventExtractor` | Up-event at bar `t` if `d_t > 0`, mark `m = d_t`. Down-event if `d_t < 0`, mark `m = |d_t|`. `d_t = 0` → no event. No threshold. Time axis: see §5.1. |
| `HawkesIntensityTransformer(window_days)` | Uses the frozen `HawkesParameters θ_d` for this day (§5.5). Computes `λ_u(t⁺), λ_d(t⁺)` at every bar (§5.3). Writes `lambda_u`, `lambda_d`. |
| `IntensityRateScaler(new_mean)` | `channel_prob[:,c] = clip(λ_c(t⁺) · s_c, 0, 1)`, with `s_c = new_mean / mean_train(λ_c)`. Uses the **same `new_mean`** as the baseline, so mean input firing rates match. No `max(·,0)` is needed, because intensities are non-negative. |
| `BranchingRewardComputer` | At every event bar computes the causal branching decomposition and the pool rewards `reward_mom[t]`, `reward_rev[t]` (§5.4, §6.5). Non-event bars get reward 0. |

---

## 5. Hawkes model specification (`models/hawkes`)

### 5.1 Time axis and events
- Default time axis `time_axis="bar_index"`: the event time is the integer bar index (trade-time clock). All other windows in the paper are also measured in bars.
- Option `"wallclock"`: seconds since UTC day start, using the bar's end timestamp. Break ties with a strict ordering (event order within equal timestamps).
- With `bar_index`, each bar carries at most one event, so ties are impossible.
- Each day is an independent realization on `[0, N_d]`, where `N_d` is the number of bars. History resets at the start of each day.

### 5.2 Model

Types `m ∈ {u, d}`. Mark function `g(·)` with per-source-type normalization (§5.6).

```
λ_u(t) = μ_u + Σ_{t_i^u < t} α_uu · g_u(m_i) · exp(−β_uu (t − t_i^u))
             + Σ_{t_j^d < t} α_ud · g_d(m_j) · exp(−β_ud (t − t_j^d))

λ_d(t) = μ_d + Σ_{t_j^d < t} α_dd · g_d(m_j) · exp(−β_dd (t − t_j^d))
             + Σ_{t_i^u < t} α_du · g_u(m_i) · exp(−β_du (t − t_i^u))
```

- Notation: `α_{target,source}`, e.g. `α_ud` = a **d**own-event exciting **u**p-intensity.
- Parameter vector `θ = (μ_u, μ_d, α_uu, β_uu, α_ud, β_ud, α_du, β_du, α_dd, β_dd)`: 10 parameters, all > 0.

### 5.3 Efficient recursion (O(N))

For target `m` and source `k`, define over the merged, time-sorted event list:

```
S_mk(t_n⁻) = Σ_{events i of type k, t_i < t_n} g_k(m_i) · exp(−β_mk (t_n − t_i))

S_mk(t_n⁻) = exp(−β_mk (t_n − t_{n−1})) · [ S_mk(t_{n−1}⁻) + 1{event n−1 is type k} · g_k(m_{n−1}) ]

λ_m(t_n⁻) = μ_m + Σ_k α_mk · S_mk(t_n⁻)                         (left limit: excludes event n)
λ_m(t_n⁺) = λ_m(t_n⁻) + 1{event n is type k} · α_mk · g_k(m_n)  (right limit: includes event n)
```

- The same recursion evaluates `λ` at every bar (event or not) by stepping bar by bar.
- **Network input** at bar `t` uses `λ_c(t⁺)`. This includes the jump observed at bar `t`, which is causal because `d_t` is known at bar `t`. This mirrors the baseline, which also encodes the current `d_t`. Make it configurable (`limit="right"|"left"`).

### 5.4 Causal branching decomposition

For an event `n` of type `m` at time `t_n`, with frozen `θ`:

```
p_bg(n)    = μ_m                     / λ_m(t_n⁻)
p_same(n)  = α_mm · S_mm(t_n⁻)       / λ_m(t_n⁻)     # triggered by same-direction history  → momentum
p_cross(n) = α_mk · S_mk(t_n⁻)       / λ_m(t_n⁻)     # k ≠ m, opposite-direction history   → reversion
p_bg + p_same + p_cross = 1
momentum_score M(n) = p_same(n) − p_cross(n) ∈ [−1, 1]
```

This uses only events before `t_n`, so it is causal by construction.

### 5.5 Fitting (MLE) and freezing

**Log-likelihood** over a window of days `D`:

```
log L(θ) = Σ_{day ∈ D} Σ_{m ∈ {u,d}} [ Σ_{events n of type m} log λ_m(t_n⁻)  −  Λ_m(N_day) ]

Λ_m(T) = μ_m · T + Σ_k Σ_{events i of type k} (α_mk · g_k(m_i) / β_mk) · (1 − exp(−β_mk (T − t_i)))
```

**Optimization:**
- scipy `L-BFGS-B` on log-parameters (positivity), numba-jitted objective, analytic or finite-difference gradient.
- ≥ 5 random restarts; keep the best.
- Initialization: `μ_m ≈ 0.5 · event_rate_m`, `α/β ≈ 0.2`, `β` drawn from `{0.1, 0.5, 1, 2}` per bar.

**Stationarity:**
- Branching matrix `A_mk = α_mk · E[g_k] / β_mk`, which equals `α_mk/β_mk` under normalized marks.
- Require spectral radius `ρ(A) < 1`. Add a penalty, and reject solutions with `ρ ≥ 1`.
- Log convergence status, final log-likelihood, and `ρ(A)`.

**Freezing rule (causality, critical):** the parameters used to transform day `d`, for both training days and test days, are always:

```
θ_d = fit(days [d − W_h, d − 1])
```

where `W_h` is the Hawkes window (1, 3, 5, 10).
- Training-day inputs and test-day inputs are therefore produced the same way.
- No day is ever encoded or rewarded with parameters fit on itself.
- Cache `θ_d` per `(day, W_h, mark_fn, time_axis)` in `HawkesParamCache`. It is reused across folds and strategies.
- Store `A` (the branching matrix) in the JSON for evaluation only. It is **not** used to generate rewards.

### 5.6 Mark functions (`MarkFunction` ABC)

| Name | Formula |
|---|---|
| `Unmarked` | `g(m) = 1` |
| `LinearNormalized` (default) | `g_k(m) = m / mean_k(m)`. `mean_k` is the mean mark of source type `k` over the fit window, frozen into `θ_d`. |
| `Saturating` | `g_k(m) = (1 − exp(−m/c_k)) / E[·]`, with `c_k` = median mark of type `k` in the fit window, normalized to mean 1. |

### 5.7 Simulation (for tests)
- Implement Ogata thinning for the bivariate marked process.
- Marks are drawn from an empirical or configurable distribution.

---

## 6. Module 3 — Signal Creation

### 6.1 Shared SNN engine (`snn/`)

**Time discretization:**
- Bar `b ∈ {0..N−1}` occupies `T` ticks. Global tick `τ = b·T + s`, with `s ∈ {0..T−1}`.
- The whole day is one continuous simulation of `N·T` ticks.
- Membrane potentials and traces reset **at the start of each day**. They are **not** reset between bars.

**LIF neuron, paper-literal discrete form.** Per tick, for each non-input neuron `j`:

```
I_j = Σ_i w_ij · s_i[τ − delay]                  # synaptic_delay_ticks, default 1
if ref_j > 0:  ref_j -= 1;  u_j = u_rest;  s_j[τ] = 0
else:
    u_j += I_j
    if u_j ≥ θ_thr:  s_j[τ] = 1;  u_j = u_rest;  ref_j = t_ref
    else:            s_j[τ] = 0;  u_j = max(u_j − leak, u_rest)     # "decreases by a small amount"
```

- Weights are excitatory, clipped to `[0, w_max]`.
- Option: `leak_mode="exponential"` (`u ← u_rest + (u − u_rest)·exp(−1/τ_m)`). The default is the paper-literal subtractive leak.

**STDP traces and term (shared by both models).** Per synapse group, per tick:

```
x_i⁻ = x_i · exp(−1/τ_plus)        # pre trace, decayed, before adding the current spike
y_j⁻ = y_j · exp(−1/τ_minus)       # post trace, decayed, before adding the current spike

ξ_ij(τ) = A · (x_i⁻ + s_i[τ]) · s_j[τ]  +  B · y_j⁻ · s_i[τ]      with A > 0, B < 0

x_i = x_i⁻ + s_i[τ];   y_j = y_j⁻ + s_j[τ]
```

- This is the trace form of the paper's rule:
  - `Δw = A·exp(−|t_pre − t_post|/τ)` if `t_pre − t_post ≤ 0`
  - `Δw = B·exp(−|t_pre − t_post|/τ)` otherwise
- Simultaneous pre and post spikes (`t_pre = t_post`) fall into the potentiation branch, as in the paper.
- The default is all-to-all pairing (the trace sums all past pairs). Option `pairing="nearest"` resets the trace to 1 instead of adding.
- `τ_plus = τ_minus = τ` by default.
- **Sign fix:** the paper prints `B > 0`, which contradicts its own text ("weakened"). Implement `B < 0` and log this in DESIGN_DECISIONS.
- The function computing `ξ` is written **once** and used by both learning rules.

**Learning rules:**
- `PairwiseSTDP`: `w_ij ← clip(w_ij + ξ_ij(τ), 0, w_max)` every tick.
- `RewardModulatedSTDP` (MSTDPET): see §6.5.
- `NoLearning`: used for all test-day simulation. Weights are frozen during testing.

**Poisson encoder:** for input channel `c` at bar `b`, independently at each tick `s`: `X_c[b·T + s] ~ Bernoulli(channel_prob[b, c])`. It is seeded.

**Signal extraction (shared):**
- Bar `b` emits a signal if the output neuron spikes at any tick in `[b·T, (b+1)·T)`. At most one signal per bar.
- Record per-bar spike counts of every hidden pool in `SignalSeries.diagnostics`.

**Performance:**
- BTCUSDT has on the order of millions of trades per day. With `num=10` that is up to ~10⁵ bars per day, and `N·T` up to ~10⁶ ticks.
- Write the tick loop as numba kernels over dense arrays.
- Profile early (milestone M3). Target: simulating one day in well under a minute.

### 6.2 `PaperSNNSignalModel` (baseline — frozen spec)

- **Pipeline:** `VWAPBarAggregator → PriceDifferencer → DirectionalSplitter → ShiftedZScoreNormalizer`.
- **Topology** (Figure 4 of the paper):
  ```
  X1 ──all-to-all──► H1 (64 LIF)        [PairwiseSTDP]
  X2 ──all-to-all──► H2 (64 LIF)        [PairwiseSTDP]
  H1 ∪ H2 ──all-to-all──► Out (1 LIF)   [PairwiseSTDP]
  ```
  No intra-layer connections, and no H1 ↔ H2 connections.
- **`fit(train_days)`:**
  1. Fresh weight initialization (seeded), unless `warm_start=true`.
  2. Simulate each training day in chronological order with STDP on (`epochs=1`).
  3. Weights carry across training days. Membrane potentials and traces reset per day.
- **`generate(test_day)`:** simulate with `NoLearning` and return a `SignalSeries`.
- **This class must not accept options that change the paper's architecture or preprocessing.** Only the ambiguity defaults of §11 are configurable.

### 6.3 `HawkesRSTDPSignalModel` (improved)

- **Pipeline:** `VWAPBarAggregator → PriceDifferencer → DirectionalSplitter → EventExtractor → HawkesIntensityTransformer → IntensityRateScaler → BranchingRewardComputer`.
- **Topology:**
  ```
  X1 (λ_u-encoded) ──all-to-all──► H1    (64 LIF)   [PairwiseSTDP]            # kept, as in paper
  X2 (λ_d-encoded) ──all-to-all──► H2    (64 LIF)   [PairwiseSTDP]            # kept, as in paper
  X1, X2 ─────────all-to-all──► H_mom (64 LIF)   [RewardModulatedSTDP, reward = reward_mom]
  X1, X2 ─────────all-to-all──► H_rev (64 LIF)   [RewardModulatedSTDP, reward = reward_rev]
  H1 ∪ H2 ∪ H_mom ∪ H_rev ──all-to-all──► Out (1 LIF)   [PairwiseSTDP]
  ```
- The LIF parameters, STDP constants (`A, B, τ`), encoder, delay, and output layer are identical to the baseline.
- **Single output neuron.** The output format is identical to the baseline, so strategies plug in unchanged.
- Flags, for the ablations in §10:
  - `use_rstdp_pools: bool = true` — false gives the "Problem 1 only" model: Hawkes input, plain topology, plain STDP.
  - `keep_direction_pools: bool = true` — false removes H1/H2.
  - `mark_fn`, `time_axis`, `W_h`.
- **`fit`:**
  1. Identical to the baseline, but every training day is transformed with its own `θ_d` (§5.5).
  2. R-STDP rewards are delivered during training.
- **`generate`:** `NoLearning` on all groups. The test day is transformed with `θ_{test_day}`. Rewards are ignored.

### 6.4 Input encoding details (improved)

1. `λ_c(t⁺)` from §5.3, with frozen `θ_d`.
2. `channel_prob[t, c] = clip(λ_c(t⁺) · new_mean / mean_train(λ_c), 0, 1)`. `mean_train` is taken over the training window's bars.
3. **Matched firing rate:** because both models use the same `new_mean`, the average input spike probability is (up to clipping) equal. Measured differences therefore come from temporal structure, not from input intensity. Log the realized mean input rate of both models per day.

### 6.5 Reward-modulated STDP (MSTDPET) — exact specification

Apply this only to synapse groups `X→H_mom` and `X→H_rev`. For each group `g` with reward stream `R_g`:

```
ξ_ij(τ)  = identical function as PairwiseSTDP (§6.1)             # Hebbian timing term, unchanged
E_ij(τ)  = E_ij(τ−1) · exp(−1/τ_z) + ξ_ij(τ)                       # eligibility trace
at reward-delivery ticks:
    w_ij ← clip( w_ij + γ · R_g(b) · E_ij(τ),  0, w_max )
```

**Reward streams** (from `BranchingRewardComputer`, per bar `b`):

```
if bar b has an event:   reward_mom[b] = M(b) = p_same(b) − p_cross(b)
                         reward_rev[b] = −M(b)
else:                    reward_mom[b] = reward_rev[b] = 0
```

- Events explained mostly by the background get rewards near 0.
- `M(b)` uses only history before bar `b` and frozen `θ_b`, so it is causal.

**Delivery and trace behavior:**
- Rewards are delivered at the **last tick of bar `b`**, i.e. `τ = b·T + T − 1`.
- `E_ij` is **not** reset after delivery (option `reset_eligibility_on_reward`).
- `τ_z` is configured in bars (`tau_z_bars`, default 3 = the paper's holding and window length) and converted to ticks (`τ_z = tau_z_bars · T`). It determines how many preceding bars' coincidences a reward credits.

**Mandatory reduction test:** with `γ = 1`, reward `R ≡ 1` delivered every tick, and `τ_z → 0` (so `E = ξ`), `RewardModulatedSTDP` must produce bit-identical weights to `PairwiseSTDP` on the same spike input.

### 6.6 `RandomSignalModel` (naive strategies)

- Given a reference `SignalSeries` with `n` signals on a day, sample `n` distinct bar indices uniformly from the eligible bars.
- Eligible bars have enough history for the direction rule and at least `entry_delay` bars remaining.
- The paper's naive versions use random timestamps with the same count and repeat 100 times. Use seeds `0..R−1`, default `R = 100`, and average metrics over repetitions.
- The spike accuracy of `RandomSignalModel` is also the **empirical chance level** for spike accuracy. Report it.

---

## 7. Module 4 — Strategy (paper-exact)

### 7.1 Direction rules (`DirectionRule`), evaluated at signal bar `t` on the vwap series `P`

| Rule | Formula | Decision |
|---|---|---|
| `MomentumRule(window=3)` | `position_flag = mean(P_{t−window} … P_{t−1}) − P_t` | `> 0` → short, `< 0` → long, `= 0` → none (Strategy Logic 1) |
| `AlexandersFilterRule(n=1)` | `ALF = (P_t / P_{t−n} − 1) · 100` | `> 0` → long, `< 0` → short, `= 0` → none |
| `StochasticOscillatorRule(n=3)` | `L_n = min(P_{t−n+1..t})`, `H_n = max(P_{t−n+1..t})`, `%K = (P_t − L_n)/(H_n − L_n) · 100` | `> 50` → long, `< 50` → short, `= 50` or `H_n = L_n` → none |

The paper's `position_flag` formula has an index typo (`P_{t+window}`). Implement `P_{t−window} … P_{t−1}` and log it.

### 7.2 Execution (`ExecutionConfig`, common assumptions of the paper)

- Initial capital 1. Each order has notional 1. No leverage. Every signal opens its own position, so overlapping positions are allowed.
- Entry price = vwap at bar `t + entry_delay` (`entry_delay = 1`).
- Exit price = vwap at bar `t + entry_delay + holding` (`holding = 3`).
- If the exit bar is beyond the day's last bar, exit at the last bar of the day (end-of-day market close). If the entry bar is beyond the last bar, no trade.
- Per-trade net return: `r = direction · (P_exit / P_entry − 1) − 2 · fee_rate`.
  - `fee_rate = 0` by default. The paper models no costs.
  - Keep `fee_rate` as a parameter for later sensitivity runs. It is not part of the main comparison.

### 7.3 `Strategy`

```python
Strategy(signal_model=PaperSNNSignalModel(cfg.signal), direction_rule=MomentumRule(3), execution=ExecutionConfig())
Strategy(signal_model=HawkesRSTDPSignalModel(cfg.signal), direction_rule=MomentumRule(3), execution=ExecutionConfig())
Strategy.from_config(cfg)      # factory; signal model and rule chosen by name in YAML
```

- The same `Strategy` class and direction rules work with any `SignalModel`. **This swap is the core modularity requirement.**
- A "naive" strategy is the same strategy with `RandomSignalModel(reference=<spike-based SignalSeries>)`.

---

## 8. Module 5 — Backtesting

### 8.1 `WalkForwardSplitter`

- For each test day `k+1` in the configured test period, `FoldSpec(train_days=[k − W_snn + 1 … k], test_day=k+1)`.
- The paper's protocol is `W_snn = 1`.
- **All configurations in one comparison must use the identical list of test days.**
  - The first test day must have `max(W_snn) + max(W_h)` prior days of data available, so every window size is evaluable.
  - This also removes the day-1 cold-start problem.

### 8.2 Windows
- `W_snn` = number of days the SNN is trained on per fold, presented chronologically, one pass each.
- `W_h` = number of days per Hawkes fit (§5.5). It is independent of `W_snn`.
- The baseline normalizer uses `W_snn` for its statistics (`stats_source="train_window"`).

### 8.3 `Backtester.run(strategy, backtest_cfg) → RunResult`

For each fold and each seed in `seeds` (default 5 seeds, because Poisson encoding and initialization are stochastic):

1. Build `DayData` for the train and test days (cached).
2. `signal_model.fit(train_days, seed)`, then `signals = signal_model.generate(test_day, seed)`.
   - **Cache** by `(model_id, test_day, W_snn, W_h, seed)` in `SignalCache`.
   - Signals do not depend on the direction rule, so all three strategies and all naive variants reuse the same signals.
3. Evaluate spike metrics (§8.4) on the **test day**. Also record them on the training day(s) from the training simulation (the paper reports train and test).
4. Compute trades via `strategy.trades_for_day`.
5. Naive versions: `R` random repetitions, matched to each day's spike count.

### 8.4 `SpikeEvaluator` (paper definitions, identical for both models)

```
r_t        = | X_{t+1} / X_t − 1 |                 for t = 1 … n−1   (X = vwap)
r_pivot    = median(r_t)  over the whole day        # evaluation only, allowed to use the full day (paper)
S_strength = ( |r_{t+1}| + … + |r_{t+w}| ) / w       # w = eval_window, default 3
real spike ⇔ S_strength > r_pivot

P_prior_avg = mean(P_{t−w} … P_{t−1})               # paper typo P_{t+window} fixed
P_post_avg  = mean(P_{t+1} … P_{t+w})
mom_rev_flag = (P_prior_avg − P_spike) · (P_post_avg − P_spike)
reversion ⇔ mom_rev_flag > 0,  else momentum
```

- Metrics per day:
  - spike accuracy = real / total
  - momentum spike percentage = momentum / total
  - number of spikes
- Signals too close to the day's edges for a full window are excluded from these metrics. Count and log them.

### 8.5 `PerformanceCalculator` (paper Table 4 metrics)

- Daily P&L = sum of net returns of trades closed that day (notional 1 each, additive, no compounding).
- Accumulated return = Σ daily P&L.
- Annualized volatility = `std(daily P&L) · √365`. Crypto trades 365 days a year; document this deviation from futures conventions.
- Sharpe = `mean(daily P&L) / std(daily P&L) · √365`, with risk-free rate 0.
- Win rate = share of trades with `r > 0`.
- Profit/loss ratio = `mean(r | r > 0) / |mean(r | r < 0)|`.
- Transactions per day (average).
- Report mean ± std across seeds. For naive strategies, report the mean across repetitions.

### 8.6 `ResultStore` layout

```
results/
  <experiment_name>/
    <run_id>/                     # e.g. improved__momentum__Wsnn1__Wh5__seed0
      config.yaml                 # fully resolved config
      meta.json                   # git commit, package versions, seed, timestamps, machine
      signals.parquet             # day, bar_idx, per-pool spike counts (improved)
      trades.parquet
      daily_pnl.parquet
      spike_metrics.csv           # per day: split(train/test), n_spikes, accuracy, momentum_pct, n_excluded
      performance.json            # Table-4 metrics
      input_stats.csv             # per day: realized mean input firing rate per channel
      hawkes/<YYYY-MM-DD>.json    # improved only: θ_d, marks normalization, logL, converged, ρ(A), A
      diagnostics/                # §9.2 outputs
```

---

## 9. Module 6 — Analysis

### 9.1 Generic
- `ResultLoader`: query runs by experiment, model, strategy, window, or seed.
- `ComparisonTable`: reproduces the paper's Table 3 (spike accuracy, momentum %) and Table 4 (strategy performance) for any set of runs. Includes naive rows and the random-signal chance level.
- **Plots:**
  - cumulative P&L per strategy/model
  - daily return distributions
  - per-day spike accuracy (both models + chance level)
  - momentum percentage per day
  - window-sweep curves
- **Statistics:**
  - Paired comparisons on identical test days: Wilcoxon signed-rank on daily spike accuracy and daily P&L, baseline vs improved.
  - Bootstrap confidence interval for the Sharpe difference.

### 9.2 Problem-specific diagnostics (`evaluation/diagnostics.py`)

- **Problem 1 — does the encoding carry more information?**
  - Spearman correlation, per test day, between the encoded input probability at bar `t` (per channel, summed) and the forward realized move `S_strength(t)`.
  - Compare baseline z-score input vs improved `λ` input.
- **Problem 2 — did the pools specialize?**
  - For each test-day output spike at bar `t`, compute `D(t) = spikes(H_mom) − spikes(H_rev)` over bars `[t − tau_z_bars + 1, t]`.
  - Report the AUC of `D(t)` for predicting (i) the paper's `mom_rev_flag` label and (ii) the Hawkes momentum score `M` (at the nearest event).
  - An AUC ≈ 0.5 means no specialization. The baseline has no pools, so report n/a.
  - Also compare the momentum spike percentage of both models.
- **Hawkes window diagnostics:**
  - Parameter stability across consecutive days, as the coefficient of variation of each `α`, `β`, `μ`. Focus on `α_ud` and `α_du`.
  - Next-day out-of-sample log-likelihood per event, as a function of `W_h`.
  - `ρ(A)` over time.

---

## 10. Experiments to run

Hyperparameters are tuned first on the **validation period** (§12). All experiments below then run on the **test period**, with 5 seeds.

| ID | Description | Models | Windows |
|---|---|---|---|
| E1 | Baseline replication | Paper SNN, 3 strategies + naive (R=100) | `W_snn=1` |
| E2 | Improved, paper-analogous | Hawkes + R-STDP, 3 strategies + naive | `W_snn=1, W_h=1` |
| E3 | Hawkes window sweep | Improved | `W_h ∈ {1,3,5,10}`, `W_snn=1` |
| E4 | SNN window sweep | Baseline and improved | `W_snn ∈ {1,3,5,10}` (improved uses the `W_h` chosen **on validation**) |
| E5 (optional) | Full grid | Improved | `W_h × W_snn` |
| A1 | Ablation: Problem 1 only | Improved with `use_rstdp_pools=false` | as E2 |
| A2 | Ablation: no direction pools | Improved with `keep_direction_pools=false` | as E2 |
| A3 | Ablation: mark function | Improved with `Unmarked` vs `LinearNormalized` | as E2 |

**Primary comparison: E1 vs E2**, plus the best window configurations from E3/E4, with A1 separating the effect of Problem 1 from Problem 2.

**Data period:**
- Configurable. Recommended: at least 90 consecutive UTC days of BTCUSDT.
  - Up to 20 days reserved for window history (`max W_snn + max W_h`).
  - About 20 validation days.
  - The remaining days are the test period.
- Never tune on the test period.

---

## 11. Paper ambiguities and resolved defaults

Log each item in `docs/DESIGN_DECISIONS.md`. All items are configurable.

| # | Ambiguity in paper | Default |
|---|---|---|
| 1 | STDP depression constant printed `B > 0` | `B < 0` (text says "weakened") |
| 2 | `P_prior_avg` / `position_flag` index `P_{t+window}` | `P_{t−window} … P_{t−1}` |
| 3 | z-score typo `x_i − * mean(x)` | `x_i − mean(x)` |
| 4 | Source of `mean`/`stdev` for normalization | training window (causal); option `same_day` |
| 5 | Poisson encoding parameters (T, rate mapping) | normalized value = per-tick spike probability, clipped to [0,1], `T=10` |
| 6 | LIF constants, weight init, bounds | §12 defaults; weights in `[0, w_max]` |
| 7 | STDP pairing scheme, `A`, `B`, `τ` | all-to-all traces, §12 defaults |
| 8 | Learning during test | off (weights frozen) |
| 9 | State continuity | continuous within a day; reset per day |
| 10 | Fresh vs continued model per fold | fresh init per fold (`warm_start=false`) |
| 11 | Multiple output spikes within one bar | one signal per bar |
| 12 | Evaluation window for real/fake, mom/rev | 3 bars |
| 13 | Return aggregation | additive per-trade returns on notional 1, overlapping allowed |
| 14 | Annualization | √365 (crypto) |
| 15 | Trading day for 24/7 market | UTC calendar day; close positions at the day's last bar |
| 16 | Stochastic oscillator window | includes current bar `t`; `H_n = L_n` → no trade |
| 17 | Synaptic delay | 1 tick |

---

## 12. Hyperparameters and tuning protocol

**Starting defaults** (to be tuned; shared between models unless marked):

| Parameter | Default |
|---|---|
| `vwap_num` | 10 |
| `T` (ticks per bar) | 10 |
| `new_mean`, `new_std` | 0.2, 0.1 (per-tick probability units) |
| LIF `u_rest`, `θ_thr`, `leak`, `t_ref` | 0, 1.0, 0.05/tick, 2 ticks |
| Weight init / `w_max` | U(0.2, 0.6), 1.0 |
| STDP `A`, `B`, `τ` | 0.01, −0.0105, 5 ticks |
| Hidden pool sizes | 64 each |
| R-STDP `γ`, `tau_z_bars` (improved only) | 1.0, 3 |
| Hawkes restarts, mark fn, time axis (improved only) | 5, `LinearNormalized`, `bar_index` |
| Strategy windows | momentum 3, ALF 1, stochastic 3, holding 3, entry delay 1 |
| Seeds / naive reps | 5 / 100 |

**Tuning protocol (fairness):**
- Tune on the validation period only.
- Objective: mean validation spike accuracy. Do not tune on P&L, to avoid overfitting the strategies.
- **Same search budget for both models.** The shared parameters (`θ_thr`, `leak`, `new_mean`, `A/B`) get the same grid for both. The improved model additionally tunes `γ` and `tau_z_bars` within the same total number of trials.

**Health checks:** warn and flag the run if the output spike rate falls outside [0.1 %, 50 %] of bars, or if any hidden pool is silent or saturated.

---

## 13. Testing requirements (pytest)

**Data**
- Parsing of header and no-header Binance files.
- Checksum verification.
- UTC day boundaries.

**Preprocessing**
- vwap against a hand calculation.
- Differencer and splitter.
- z-score with clipping.
- Event extraction: sign rule, marks, `d=0` produces no event.

**Causality harness (critical)**
- For every `PreprocessingStep` and every `SignalModel`: perturb all data after bar `t` (and all later days). Assert that outputs up to bar `t` are unchanged.
- Signal models are compared with a fixed seed.

**Encoder**
- The empirical spike rate converges to `channel_prob`.

**LIF**
- Integration, threshold, reset, refractory period, subtractive leak floor at `u_rest`.

**STDP**
- Pre-before-post increases `w`.
- Post-before-pre decreases `w`.
- Simultaneous spikes potentiate.
- Magnitude follows `exp(−|Δt|/τ)`, checked against the closed-form pairwise rule for single-pair inputs.

**R-STDP**
- The reduction test in §6.5 gives bit-identical weights to plain STDP.
- A zero reward leaves weights unchanged.
- The sign of the reward flips the update direction.

**Hawkes**
- Recursive intensity and log-likelihood equal the brute-force O(N²) computation.
- Branching probabilities sum to 1.
- Parameter recovery on data simulated by Ogata thinning (within tolerance).
- Stationarity check.
- `θ_d` is never fit on day `d` (freezing test).

**Strategy**
- Hand-computed examples for each direction rule.
- Entry and exit bars.
- End-of-day truncation.

**Backtest**
- Known P&L on a synthetic day.
- Identical test-day lists across configurations.
- Signal cache reuse.

**End-to-end smoke test**
- A few synthetic days through the full pipeline, for both models.

---

## 14. Milestones and acceptance criteria

| M | Deliverable | Accept when |
|---|---|---|
| M0 | Skeleton, config system, CLI, DESIGN_DECISIONS.md | `run_experiment --config` resolves and saves a config |
| M1 | Data loading (Binance + local + synthetic), parquet store | ≥ 90 days of BTCUSDT trades cached and standardized; tests pass |
| M2 | Baseline preprocessing + causality harness | all §13 preprocessing and causality tests pass |
| M3 | SNN engine (encoder, LIF, STDP, simulator) + profiling | unit tests pass; one BTC day simulates in acceptable time |
| M4 | `PaperSNNSignalModel` + `SpikeEvaluator` | runs on real days; health checks pass; Table-3-style output |
| M5 | Strategies, execution, backtester, performance, ResultStore, naive | E1 completes; Table-4-style output; results saved as specified |
| M6 | Hawkes module (fit, intensity, branching, simulate) | recursion, recovery, and freezing tests pass |
| M7 | Improved preprocessing (events, intensity, scaler, rewards) | causality tests pass; matched mean input rate verified |
| M8 | R-STDP + `HawkesRSTDPSignalModel` | reduction test passes; E2 and A1 complete |
| M9 | Window experiments | E3, E4 (and optionally E5), A2, A3 complete on identical test days |
| M10 | Analysis module + comparison report | notebook or script produces all §9 tables, plots, and statistical tests for E1 vs E2 and the sweeps |

**Final deliverable:** a reproducible comparison of the paper's method against the improved method on identical BTCUSDT test days. It consists of Table-3/Table-4-style tables, naive and chance baselines, the Problem-1/Problem-2 diagnostics, the window-sweep results, and every design decision documented.
