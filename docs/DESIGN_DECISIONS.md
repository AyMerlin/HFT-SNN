# Design decisions

Every interpretation of, or deviation from, the implementation plan
([IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)) and the reference paper
(Gao, Luk, Weston, *Wilmott* 2021) is recorded here: what the spec or paper says,
what the code does, and why. Config keys refer to `snn_hft/config/schema.py`;
`models.*` means a key inside one entry of an experiment's `models:` list.

Sections: [P] paper ambiguities (§11 of the plan) · [U] decisions agreed with the
thesis author · [I] implementation decisions, added per milestone · [O] open questions.

---

## [P] Paper ambiguities (plan §11)

| # | Paper | Decision | Config key |
|---|---|---|---|
| P1 | STDP depression constant printed `B > 0` | `B < 0` (default −0.0105); the text says the synapse is "weakened" | `models.*.core.stdp.B` |
| P2 | `P_prior_avg` and `position_flag` sum from `P_{t+window}` to `P_{t−1}` | `P_{t−window} … P_{t−1}` | fixed in code, see note |
| P3 | z-score typeset as `x_i − * mean(x)` | `x_i − mean(x)` | fixed in code, see note |
| P4 | `mean`/`stdev` are those "of the initial data", i.e. the whole day including future bars | training window (causal) by default; `same_day` reproduces the paper literally | `models.*.zscore.stats_source` |
| P5 | Poisson encoding parameters not given | normalised value = per-tick spike probability, clipped to [0, 1]; `T = 10` ticks per bar | `models.*.input.ticks_per_bar`, `models.*.input.new_mean` |
| P6 | LIF constants, weight initialisation and bounds not given | plan §12 defaults; weights in `[0, w_max]` | `models.*.core.lif.*`, `models.*.core.synapse.*` |
| P7 | STDP pairing scheme, `A`, `B`, `τ` not given | all-to-all traces, plan §12 defaults | `models.*.core.stdp.*` |
| P8 | Learning during the test day not stated | off (weights frozen) | `models.*.core.learn_during_test` |
| P9 | State continuity not stated | continuous within a day, reset at the start of each day | `models.*.core.reset_state_per_day` |
| P10 | Fresh vs continued model per fold not stated | fresh initialisation per fold | `models.*.core.warm_start` |
| P11 | Several output spikes within one bar | one signal per bar; spike counts kept as diagnostics | fixed in code, see note |
| P12 | Evaluation window for real/fake and momentum/reversion | 3 bars | `evaluation.eval_window` |
| P13 | Return aggregation | additive per-trade returns on notional 1, overlapping positions allowed | fixed in code, see note |
| P14 | Annualisation | √365, since crypto trades every day (futures convention would be √252) | `evaluation.annualization_days` |
| P15 | Trading day for a 24/7 market | UTC calendar day; positions close at the day's last bar | fixed in code, see note |
| P16 | Stochastic oscillator window | includes the current bar `t`; `H_n = L_n` → no trade | `strategy.stoch_n` |
| P17 | Synaptic delay not given | 1 tick (at least 1, see I20) | `models.*.core.synapse.delay_ticks` |

**Note on items fixed in code.** The plan says every §11 item is configurable. P2 and P3
are typesetting errors whose literal reading is not a usable formula (P2 would average
future prices before the signal), so they have no switch. P11, P13 and P15 are definitions
of the evaluation itself; they stay fixed so that both models are always evaluated
identically, and can be made configurable later if needed.

**Other paper details followed literally.**
- Spike strength `S_strength = (|r_{t+1}| + … + |r_{t+w}|)/w` with `r_t = |X_{t+1}/X_t − 1|`
  skips the move from the spike bar to the next bar and needs bars up to `t+w+1`.
  Signals without a full window are excluded from spike metrics and counted (§8.4).
- The paper states that 50 % spike accuracy is the chance level. On real BTCUSDT data about
  62 % of all bars qualify as "real" spikes (mean of three right-skewed returns vs their median,
  plus volatility clustering). Accuracy is therefore always reported next to the empirical
  chance level of the random-timing benchmark and the big-move benchmark (U5).

---

## [U] Decisions agreed with the thesis author (2026-09-27)

**U1 — Main dataset: Binance `aggTrades` instead of `trades`.**
Spec: `trades` by default, `aggTrades` optional (§3.1).
Decision: `aggTrades` is the main dataset (`data.dataset`); `trades` remains available.
Why: the author asked for the data closest to real execution. A `trades` row is one fill
against one resting order, so one market order produces many rows in the same millisecond.
Measured on 2026-09-16: `trades` gives 412k bars/day with a median bar duration of 1 ms,
`aggTrades` 161k bars/day with 10 ms. An `aggTrades` row is one taker order's execution at one
price, the closest thing Binance publishes to a single transaction.

**U2 — Execution latency option.**
Spec: entry at the vwap of bar `t + entry_delay` (§7.2).
Decision: `execution.latency_ms`; the entry bar is the first bar at or after `t + entry_delay`
whose first trade is at least `latency_ms` after the signal bar's last trade; exit is
`holding` bars after entry. `latency_ms = 0` reproduces the paper's rule exactly. Several latencies can
be evaluated on the same signals (`backtest.latencies_ms`); the replication reports 0 ms.
Why: on BTCUSDT the next bar usually starts in the same millisecond the signal bar ends
(67 % of bars with `aggTrades`), often as part of the same market order, so the paper's
rule assumes an execution no trader can achieve.

**U3 — Trading costs out of scope for now.**
Spec: `fee_rate = 0` by default, kept for later sensitivity runs (§7.2).
Decision: `fee_rate` stays at 0 in all runs; the parameter remains as the extension point.
Why: the author will add costs later. A typical 3-bar move (~0.001 % of price) is about
100× smaller than a round trip at Binance's base taker fee, so with fees every strategy loses
for both models; the thesis must state that results are fee-free.

**U4 — Study period: 12 months instead of ≥ 90 days.**
Data 2025-09-27 → 2026-09-26: 20 history days (→ 2025-10-16), 45 validation days
(2025-10-17 → 2025-11-30), 300 test days (2025-12-01 → 2026-09-26). Keys: `periods.*`.
Why: for P&L the effective sample size is days, not trades. With 50 test days the 95 %
interval of an annualised Sharpe near 20 is about ±6.6, with 300 days about ±2.7. The last
three months were also the quietest of the year (daily `aggTrades` files 11–19 MB vs 24–52 MB
in Oct 2025 – Feb 2026). All development and debugging happens on the validation period;
`backtest.split` defaults to `validation` so the test period is only used on purpose.

**U5 — Big-move benchmark.**
Not in the spec. A benchmark signal that fires after large price changes, matched in count
to the SNN signals and causal (threshold from training data). `benchmarks.big_move`.
Why: volatility persists, so any signal placed after large moves beats random timing on
spike accuracy. This benchmark shows how much of the SNN's accuracy that alone explains.

**U6 — Hawkes process on the paper's bar clock, reviewed after M6.**
Spec: `time_axis = "bar_index"`, events without threshold (§4.2, §5.1).
Decision: implemented as specified. After M6 the author reviews fitted parameters on real
days: share of events attributed to the background rate, spread of the momentum score, and
whether the intensity carries information beyond `d_t`.
Why: 96 % of bars have a non-zero price change, so on the bar clock there is an event at
almost every time step. The model can still learn direction and size effects, but the fit
may attribute most events to the background rate, which would shrink the R-STDP rewards
towards 0. Alternatives (e.g. a causal event threshold) need the author's approval.

**U7 — Each model tunes `new_mean` independently (2026-09-28).**
Spec conflict: plan §0 principle 4 and §6.4.3 require the same `new_mean` for both models so
their mean input firing rates match; plan §12 lists `new_mean` among the parameters each model
tunes on the same grid.
Decision: each model tunes `new_mean` independently on the same grid, like the other shared
parameters (`models.*.input.new_mean`). The improved model's `IntensityRateScaler` uses the
improved model's own value.
Consequence: mean input rates are no longer matched by construction. The realised mean input
firing rate of both models is logged per day (`input_stats.csv`), so any difference is visible.
Proposed for the M8 check-in: a control run of the improved model at the baseline's tuned
`new_mean`, which separates the input-rate effect from the effect of the Hawkes memory.


**U8 — Bar size: 100 `aggTrades` per vwap bar (2026-09-28).**
Paper and spec: `num = 10` transactions per bar.
Decision: `bars.vwap_num = 100` for all main experiments (`base.yaml`); the literal `num = 10`
results are kept as E1-num10 (`experiments/configs/paper_baseline_num10.yaml`).
Why: E1 at `num = 10` showed random timing as profitable as the SNN. At 10 BTCUSDT `aggTrades`
a bar lasts ~0.1 s, and consecutive vwap moves continue (sign autocorrelation 0.71 at lag 1,
0.52 at lag 3), so the paper's direction rules win ~75 % of trades at *any* time; in the paper's
crude-oil data random timing won 51.2 %. The bar size was chosen by a rule that does not use the
model: the smallest `num` at which the random-timing momentum strategy has no edge (win rate
≤ 52 %) on validation days (2025-11-01 → 11-10):

| num | bars/day | median bar | sign autocorr lag 1 / 3 | random-timing win rate |
|---|---|---|---|---|
| 10 | 184k | 0.1 s | 0.71 / 0.52 | 75.3 % |
| 30 | 61k | 0.9 s | 0.59 / 0.24 | 60.2 % |
| 100 | 18k | 4.6 s | 0.28 / 0.04 | 52.5 % |
| 300 | 6k | 16 s | 0.17 / 0.01 | 50.3 % |

The paper does not report its bar count; with the tuned signal rate (~16 %), 100-trade bars give
~2,900 signals per day, close to the paper's 2,586. Consequences: holding 3 bars ≈ 15 s, next-bar
entry is achievable within seconds, and latencies up to 1 s are evaluated.


**U9 — Hawkes events: causal threshold on the bar clock (2026-09-28).**
Spec: every bar with `d_t ≠ 0` is an event, no threshold (§4.2); bar clock (§5.1).
Decision: bar clock, but a bar is an event only if |d_t| exceeds the `q`-quantile of the non-zero
|d| on the fit window [d − W_h, d − 1]. The threshold is part of θ_d (frozen, cached, never
computed on day d). `q = hawkes.event_quantile` is tuned on validation in {0.8, 0.9, 0.95} within
the improved model's trial budget; `null` restores the spec behaviour.
Why: with every move an event (99.99 % of bars at 100 trades per bar), the fitted process puts
95–97 % of events into the background, so the momentum score M behind the R-STDP rewards is
nearly constant and never negative (Problem 2 would get no training signal). With the threshold
(validation days, W_h = 1): |M| > 0.2 for 78–93 % of events and the intensity's information
about the next moves beyond |d_t| rises from 0.19 to 0.23–0.24 (partial Spearman). The
`wallclock` axis also made the rewards informative but reduced that information to 0.06.
Evidence: [docs/reports/m6_hawkes_review.md](reports/m6_hawkes_review.md).


**U10 — Third model "hawkes_typed" with a momentum/reversion strategy (2026-09-28).**
Scope change requested by the author after M8. E2 showed that the improved model's R-STDP pools do
not specialise (AUC 0.501) and that, by design, their information cannot reach the strategies (one
output neuron; direction from price only). A further suspected cause: with excitatory weights only,
a pool responds monotonically to (λ_u, λ_d) and cannot fire for "one intensity high, the other low"
(momentum) without also firing for "both high" (reversion).
Decision: (A) premise checks first (`experiments/typed_premise_checks.py`, validation data for
every design-relevant check; test-period analyses only as critique, never for design); gate:
check in with the author, who chooses the teacher and the tuning objective. (B) Only then a third
model `hawkes_typed` with two typed outputs (momentum, reversion) and a new "typed" strategy that
follows momentum signals and fades reversion signals; ablations T1–T3; experiment E6.
Constraints: E1, E2, the baseline and the improved model stay unchanged (same model ids, cached
results valid); the paper strategies stay unchanged; tuning on validation only, same 80-trial
budget and shared grid; M9 sweeps stay pending until after E6.


**U11 — Typed line stopped after Phase A; M9 and M10 next (2026-09-28).**
The Phase A gate failed (best Hawkes feature AUC 0.504 vs 0.53) and the trade label is not
predictable from the price-derived inputs (held-out AUC 0.512; typing would add ≈ +0.01 bp per trade
against +2.4 bp for a perfect type). Decision: Phase B is not built; Phase A is reported as a negative
premise result. A short order-flow check (validation only) is added as an appendix item: does
order-flow information make the trade outcome predictable at all? The tuning grid is not widened
(the E2 conclusions are unlikely to change and the sweeps keep the tuned settings for
comparability). M9 runs E3, E4, A2 and A3; the optional E5 grid is skipped. Then M10.

---

## [I] Implementation decisions

### M0 — skeleton and configuration

**I1 — Environment.** Python 3.13 in a project virtual environment (`.venv`); dependencies
declared in `pyproject.toml`, exact versions in `requirements-lock.txt`. Every run's
`meta.json` records the git commit (and whether the tree was dirty), package versions,
Python version and machine.

**I2 — One schema holds all defaults.** Plan §11/§12 defaults live in
`snn_hft/config/schema.py`; YAML files state only differences. Unknown keys are rejected.
Files can inherit (`inherits: base.yaml`); mappings merge, lists replace. Command-line
overrides use dotted paths (`--set models.0.core.lif.threshold=1.2`).

**I3 — Baseline config cannot express improved-model options.** `PaperSNNConfig` and
`HawkesRSTDPConfig` are separate schemas selected by `model:`; the baseline rejects
`hawkes`, `rstdp` and the pool flags (§6.2).

**I4 — Hidden pool size is configurable for both models** (`core.hidden_size`, default 64).
The paper calls it a hyperparameter chosen heuristically; the topology itself is fixed.

**I5 — Exponential-leak time constant.** The plan offers `leak_mode = "exponential"` without
a `τ_m`; default `tau_m = 20` ticks (only used with that option).

**I6 — Experiments and runs.** One experiment YAML lists one or more models evaluated on
identical days. It expands into runs over models × strategy rules × `W_snn` × `W_h` (Hawkes
models only) × seeds, with ids like `improved__momentum__Wsnn1__Wh5__seed0` (plan §8.6);
baseline ids omit `Wh`. Latencies are not a run dimension: they only change execution, so
they are evaluated inside a run on the same signals.

**I7 — Model id.** `RunConfig.model_id` hashes everything that determines the signals
(venue, symbol, dataset, bar size, full signal-model config except its display name).
Windows and seed are separate parts of the signal-cache key (§8.3), so strategy, execution
and benchmark settings never invalidate cached signals.

### M1 — data loading

**I8 — Cache path includes the dataset.** Spec: `data/raw/{venue}/{symbol}/{YYYY-MM-DD}.parquet`.
Code: `data/raw/{venue}/{symbol}/{dataset}/{YYYY-MM-DD}.parquet`, so `aggTrades` and `trades`
can be cached side by side (U1). Files are written atomically (temporary file, then rename)
with zstd compression. The parquet metadata stores provenance and coverage: source URL,
sha256 of the verified zip, trade count, trades dropped outside the UTC day, first/last
timestamp and the longest gap between trades.

**I9 — No REST fallback.** Spec §3.1 lists `GET /fapi/v1/aggTrades` as a fallback for recent
days. Not implemented: the archive covers the whole study period (the last day, 2026-09-26,
was already published on 2026-09-27), and a REST download of one day needs ~1,600 paginated
requests. `DataSource` is the extension point if it is needed later.

**I10 — Standard columns from `aggTrades`.** `trade_id` = `agg_trade_id`, `qty` = `quantity`,
`ts_ns` = `transact_time`. Files with and without a header row are both accepted. The
timestamp unit (ms, µs or ns) is detected from its magnitude, because some Binance archives
(spot, from 2025) use microseconds; the futures files in this study use milliseconds.

**I11 — Trades outside the UTC day are dropped and counted** (`n_outside_day` in the
metadata and coverage report). Ties in time are ordered by trade id.

**I12 — Memory.** `DataStore.get_days` returns a list as specified, but a year of trades does
not fit in memory, so `iter_days` streams days; later milestones cache bars per day instead
of trades.

**I13 — Containers.** `BarSeries` gets an extra column `ts_start_ns` (first trade of the bar),
needed for the latency option (U2). `DayData` is immutable; preprocessing steps return a
copy with more fields filled, so the causality harness can compare outputs safely.

**I14 — Synthetic source.** Deterministic per `(seed, day)`, millisecond timestamps (so ties
occur as on Binance), prices from a `PriceModel`. M1 ships a random-walk model with a
one-tick spread; a Hawkes-driven price model plugs into the same interface at M6.

### M2 — baseline preprocessing and causality harness

**I15 — Days enter a pipeline as trades.** `DayData.trades` holds the day's `TradeFrame`;
`VWAPBarAggregator` replaces it by `bars` (trades are dropped unless `keep_trades=True`),
so a pipeline never holds more than one day's raw trades per day in memory.

**I16 — Normalisation statistics.** `mean` and `stdev` are pooled over all bars with a price
difference in all training days (not averaged per day). `stdev` is the population standard
deviation (ddof = 0); with ≥ 20k bars per day the choice is immaterial.

**I17 — Bar 0 carries no input.** The first bar of each day has no price difference, so its
spike probability is 0 on both channels. The paper does not say how it handles bar 0.

**I18 — Causality harness** (`snn_hft/testing/causality.py`). For each tested bar `t` it
perturbs the evaluated day after bar `t` and every later day entirely, in three ways (new
values, truncated day, remainder replaced by a different number of trades), and requires
every output up to bar `t` to be bit-identical. Tests show that it detects both the
paper-literal `stats_source = "same_day"` option (which uses the day's future bars, see P4) and
a step fitted on later days. The same harness will cover the Hawkes steps (M7) and the signal
models (M4, M8). It was also run on real days (2025-10-18, 80k bars; 2025-11-21, 592k bars).

Real-data check (fit 2025-10-17, transform 2025-10-18): realised mean input probability 0.201
(X1) and 0.200 (X2) for `new_mean = 0.2`; 0.2 % of bars clipped at 0, 0.02 % at 1; 35 % of bars
lie within ±0.02 of `new_mean`, i.e. most bars produce nearly the same input on both channels.

### M3 — SNN engine

**I19 — Learning rules are declarative; one numba kernel executes them.** Spec §2.4 sketches
`LearningRule.on_tick` / `on_reward` Python callbacks. A Python call per tick would cost
seconds per million ticks, so rules carry parameters and a rule code, and
`snn/simulator.py` executes them. The Hebbian term is computed by the single function
`stdp_xi`, and the per-tick plasticity by `learning_tick`, which both `simulate_day` and the
test driver `replay_learning` call. R-STDP therefore uses exactly the STDP code path inside
its eligibility trace, and the §6.5 reduction test passes bit-identically.

**I20 — Synaptic delay ≥ 1 tick** (`core.synapse.delay_ticks`, default 1). With at least one
tick of delay the order in which layers are updated within a tick cannot matter.

**I21 — STDP uses emission times.** `ξ` uses `s_i[τ]` of the pre neuron's emission, not its
arrival. With the default 1-tick delay, an input spike that makes a hidden neuron fire one
tick later is a pair with `t_pre − t_post = −1`, potentiated by `A·e^{−1/τ}`.

**I22 — Spike traces are per neuron.** `x_i` and `y_j` depend only on one neuron's spikes, so
they are stored per neuron rather than per synapse. This requires all learning groups of a
network to share `τ+`, `τ−` and the pairing scheme (and all R-STDP groups `γ`, `τ_z`, delivery),
which both models do; the network refuses mixed constants.

**I23 — One neuron per input population** (the paper's X1 and X2).

**I24 — Encoder randomness is prefix-stable.** Uniforms are drawn row by row (tick, channel)
in chunks, so the spikes of the first k bars depend only on the seed and those k bars, not on
the day's length. A test checks this and the chunk-size invariance.

**I25 — Recorded output.** Every simulation returns spike counts per bar and population,
including the input populations (the realised input rate for `input_stats.csv`) and every
hidden pool (diagnostics and health checks).

**I26 — Rewards.** Each R-STDP group reads a named reward stream (`mom`, `rev`); learning
without the needed rewards is an error; test days (`learn = False`) ignore rewards.

Profiling (M3, validation days, default hyperparameters, Apple M2, one core):
paper network 1.2M ticks/s training, 2.4M ticks/s test; improved topology 0.6M / 1.2M
ticks/s. The busiest day of the study (9.5M ticks) therefore takes about 8 s + 4 s (paper) or
16 s + 8 s (improved) for training plus test; a typical day (1.5M ticks) about 2 s / 4 s.
With the untuned §12 defaults the output neuron fires in 91–96 % of bars (outside the
health-check band), which tuning addresses in M5; with random rewards and `γ = 1` the R-STDP
pools went silent within one day, which is examined with the real rewards in M8.

### M4 — paper SNN signal model and spike evaluator

**I27 — Shared SNN signal plumbing.** `SNNSignalModel` owns preprocessing, encoding,
simulation and signal extraction; `PaperSNNSignalModel` only supplies the paper pipeline and
topology (the improved model will do the same). `fit` simulates the training days in
chronological order with learning on and records their signals for the training-day spike
metrics; `generate` simulates the test day with learning off (`core.learn_during_test`).

**I28 — Seeds.** Every random draw comes from a named stream `rng_for(seed, purpose, day, epoch)`:
initial weights depend on the seed only (the same in every fold), encoder spikes on seed,
purpose (train/test), day and epoch. A training day therefore gets the same spikes in every fold
that contains it, and both models use the same seeds.

**I29 — Evaluation bars are built outside the models** with the shared `VWAPBarAggregator`,
so the spike evaluator and the strategies never depend on a model's internals.

**I30 — Spike evaluation details.** A bar is evaluable if both windows fit into the day
(`w ≤ t ≤ n − 2 − w`); the same set is used for real/fake and momentum/reversion, so both have
one denominator. Window means are computed per window, and price differences within 1e-12 of
the price are treated as zero (flat prices → momentum). The chance level is reported exactly as
the share of real bars among evaluable bars, i.e. the expected accuracy of signals at random
times, together with the base momentum share.

**I31 — Health checks.** Output rate outside [0.1 %, 50 %] of bars; a hidden pool is *silent*
if it does not spike all day, *saturated* if its mean rate is ≥ 90 % of the maximum
`T / (t_ref + 1)` spikes per neuron per bar.

**I32 — One LIF parameter set for all layers.** Plan §6.1/§12 give one set of LIF constants.
The output neuron sums 128 hidden neurons while each hidden neuron has a single input, so the
threshold that keeps the output in the health band (≈ 8–32 on validation days) also makes the
hidden neurons integrate over ~10 bars. This follows the spec; per-layer constants would be a
change to discuss, not an interpretation.

M4 check (15 validation folds × 3 seeds, provisional threshold 16):
[docs/reports/m4_spike_check.md](reports/m4_spike_check.md). Test spike accuracy 53.8 % vs a
chance level of 53.5 %; momentum share 80.3 % vs 80.6 % for random timing; no health warnings.

### M5 — strategies, backtester, benchmarks, tuning

**I33 — Zero tolerance in the direction rules.** `position_flag = 0`, `ALF = 0` and `%K = 50`
mean "no transaction" (Strategy Logic 1–3). Differences within 1e-12 of the price count as
exactly zero, so floating-point noise on flat prices never opens a position.

**I34 — Entries on the last bar.** A signal whose entry bar is the day's last bar opens and
closes on that bar (return 0) and counts as a trade; a signal with no bar left to enter is not
traded. With latency, the exit is `holding` bars after the delayed entry, capped at the last bar.

**I35 — Naive benchmark.** For each day, rule and repetition r (0..R−1, R = 100), the model's
number of signals is drawn uniformly without replacement from the bars the rule can trade
(`t ≥ min_history`, `t ≤ n − 1 − entry_delay`); the random stream depends on r, the day and the
model's seed. The same samples serve every latency. Table-4 metrics are computed per
repetition and then averaged (mean ± std over repetitions); averaging the daily P&L first would
understate the volatility of a single naive backtest. The naive spike accuracy is the empirical
chance level; the exact expected chance level (`base_accuracy`) is reported alongside.

**I36 — Big-move benchmark (U5) details.** Target rate = the model's signal rate on the test day
(the same count matching the naive benchmark uses); the threshold is the matching quantile of
`|d_t|` on the fold's training days; a signal fires when `|d_t|` is strictly above it. It is
evaluated exactly like the model (spikes and all strategies). *Revised 2026-09-28:* the first
version matched the model's training-day rate, but the trained model fires about twice as often
on test days (8.2 % vs 4.4 % of bars at 100 trades per bar), so the benchmark fired half as often
as the model, which favoured its accuracy.

**I37 — Performance details.** Daily P&L includes every test day (0 on days without trades);
standard deviations use ddof = 1; win rate and profit/loss ratio pool all trades of the test
period; transactions per day = trades / test days.

**I38 — Two-phase backtester.** Phase 1 computes and caches one fold result per
(model id, test day, W_snn, W_h, seed) in parallel worker processes; phase 2 evaluates each
signal job (all rules, latencies, naive repetitions and the big-move benchmark) from the cache.
Result-layout additions to §8.6: `health.csv` (health warnings per day), and in
`daily_pnl.parquet` a `source` column (`model`, `big_move`, `naive_mean`) and `latency_ms`.
`signals.parquet` is identical for the three rules and is hard-linked. `trades.parquet` costs
about 4 MB per run and day with three latencies, so by default it is written only for the lowest
seed of each job (`output.save_trades = "first_seed"`; sweeps use `"none"`).

**I39 — Tuning protocol (§12).** `experiments/tune.py`. Shared grid, identical for both models:
LIF threshold {4, 8, 16, 32}, leak {0.02, 0.05, 0.1} per tick, `new_mean` {0.1, 0.2, 0.3},
STDP scale {0.5, 1, 2} (multiplies `A` and `B`). The improved model adds `γ` {0.1, 0.3, 1} and
`tau_z_bars` {1, 3, 10}. Budget: 40 distinct grid points drawn uniformly (grid seed 0) for each
model, evaluated with `W_snn = 1` (and `W_h = 1`) on all 45 validation days with seeds 0 and 1.
Objective: mean test-day spike accuracy. A trial is admissible if at most 5 % of its
validation days fail a health check. Only the signal phase runs, so P&L never enters the choice.
The chosen config is written to `experiments/configs/tuned/`.

First baseline tuning (num = 10): threshold 4, leak 0.1, `new_mean` 0.1, STDP scale 2 (validation
accuracy 54.61 % vs chance 52.45 %) — [m5_tuning_paper_num10.md](reports/m5_tuning_paper_num10.md);
E1 at num = 10: [m5_e1_baseline_num10.md](reports/m5_e1_baseline_num10.md).

**I41 — Tuning revision (after E1-num10).** (a) The paper names two encoding hyperparameters,
`new_mean` and `new_stdev`; the first grid omitted `new_stdev`, which decides how much of the move
size reaches the network (with heavy-tailed `d_t`, `new_std = 0.1` leaves the summed input of
~80 % of bars at exactly `2·new_mean`). It is now tuned for the baseline, {0.05, 0.1, 0.2, 0.4};
it has no counterpart in the improved model, which tunes `γ` and `tau_z_bars` instead.
(b) The first tuning chose values at the edges of the grid, so the shared grid was widened:
threshold {2, 4, 8, 16, 32}, leak {0.02, 0.05, 0.1, 0.2}, `new_mean` {0.05, 0.1, 0.2, 0.3},
STDP scale {0.5, 1, 2, 4}. (c) Budget: 80 trials for each model (runs are ~10× cheaper at
num = 100). The improved model has not been tuned yet, so both models get the same budget.

**I40 — Signal code version in model ids.** `SIGNAL_CODE_VERSION` (schema.py) is part of every
model id; it is bumped whenever preprocessing, encoding, the SNN kernels or the Hawkes model
change, so cached signals from older code are never reused.

### M6 — Hawkes module

**I42 — Module layout.** `models/hawkes/` holds, as in the spec, `kernels.py`, `marks.py`,
`params.py` and `process.py`, plus two additions: `events.py` (events and event times from bar
price differences, wrapped by the M7 `EventExtractor` step) and `provider.py`
(`HawkesParamProvider`, which enforces the freezing rule). `HawkesParamCache` lives in
`backtest/cache.py` as specified.

**I43 — Likelihood and fitting.** The log-likelihood and its analytic gradient use the O(N)
recursion of §5.3 plus a companion recursion for ∂S/∂β (checked against finite differences and
against the brute-force O(N²) sum). L-BFGS-B works on log-parameters with bounds
μ, α ∈ [1e-8, 1e3] and β ∈ [1e-4, 1e3]; the objective is −ℓ / n_events. Stationarity: a penalty
100·(ρ − 0.99)² with its closed-form gradient when ρ(A) > 0.99, and every restart with ρ(A) ≥ 1 is
rejected. If all restarts are rejected, the provider retries once with 4× the restarts and a
penalty of 1e4, then fails loudly. Restart initialisation: β drawn from `beta_init_grid`,
α = 0.2·β·U(0.8, 1.2), μ = 0.5·(event rate of the type)·U(0.8, 1.2).

**I44 — Freezing and caching.** `θ_d` is fitted on exactly the `W_h` calendar days before `d`
(tests check that day `d` is never requested and that changing day `d` leaves `θ_d` unchanged).
Fits use their own random stream, independent of the SNN seed, so every fold, seed and strategy
shares one `θ_d`. Cache key: data identity, bar size, `SIGNAL_CODE_VERSION`, mark function, time
axis, `W_h`, a hash of the fit settings (restarts, β grid, max_iter), and the day.

**I45 — Bars and limits.** On the bar clock the event window of a day is [0, N_bars] and bar b is
at time b; on the wall clock, at the bar's last trade in seconds since the UTC day start
([0, 86 400]). `λ(b⁻)` excludes and `λ(b⁺)` includes the event of bar b. Events with equal times
are strictly ordered; the branching split of an event uses the history before it, including
earlier events at the same time.

M6 review on real validation days (U6): [docs/reports/m6_hawkes_review.md](reports/m6_hawkes_review.md).

### M7 — improved preprocessing

**I46 — Where the threshold lives.** `EventExtractor` writes every move to `day.events`; the
Hawkes step keeps those above θ_d's threshold (U9), because the threshold is part of θ_d. The
provider learns it from the candidate moves of the fit window.

**I47 — History source.** θ_d needs the bars of days before `d`, which are not among the
training days `fit()` receives (planned item of M0). The provider gets a history function
(`candidate_events_source`, built on the shared bar cache) and only ever calls it for the `W_h`
days before `d`. The causality tests run the full improved pipeline with history drawn from the
perturbed world, and a provider that fits on day `d` itself is flagged.

**I48 — Encoding details.** `IntensityRateScaler` divides by the mean intensity over all bars of
all training days, per channel (channel 0 = λ_u → X1, channel 1 = λ_d → X2). Bar 0 gets
λ = μ. The first events of a day have no history, so their momentum score (and reward) is 0.

**I49 — Branching output.** `day.branching` is a table (bar_idx, p_bg, p_same, p_cross,
momentum); θ_d is kept in `day.extras["hawkes_params"]` so runs can store it (§8.6 `hawkes/`).

**I50 — Improved tuning grid.** `hawkes.event_quantile` {0.8, 0.9, 0.95} joins `γ` and
`tau_z_bars` as the improved model's own parameters, within the same 80-trial budget.

M7 check on real validation days: [docs/reports/m7_improved_preprocessing.md](reports/m7_improved_preprocessing.md).
With equal `new_mean` both encodings give similar realised input rates (0.069 vs 0.071 at
`new_mean = 0.05`); both exceed the target on test days by 30–40 % because they are scaled with
training-day statistics. The Hawkes input is about three times as strongly related to the next
moves as the z-score input (Spearman 0.25 vs 0.07–0.09).

### M8 — improved signal model, E2, A1

**I51 — Hawkes precompute phase.** Before the signal phase, the backtester collects every
(day, Hawkes setting, W_h) that any fold needs (training and test days), fits the missing θ_d in
parallel and caches them. Signal workers then only read θ_d, so no two workers fit the same day.

**I52 — Stored Hawkes parameters.** Each improved run stores θ_d of every day it used (training
and test days) as `hawkes/<YYYY-MM-DD>.json` (§8.6): the ten parameters, event threshold, ρ(A),
branching matrix, log-likelihood per event, convergence, event count and fit days. The files are
shared by the three rule directories of a job via hard links.

Improved-model tuning (validation): [m8_tuning_improved.md](reports/m8_tuning_improved.md); only 4 of
80 trials pass the health rule because the sparse Hawkes input often leaves the output nearly
silent; the chosen trial sits at two grid corners (`new_mean` 0.3, threshold 32).
E2, A1 and input-rate controls on the test period: [m8_e2_results.md](reports/m8_e2_results.md).

**I53 — Model structure.** `HawkesRSTDPSignalModel` subclasses the shared `SNNSignalModel` and only
supplies the Hawkes pipeline, the extended topology and the reward streams (`mom`, `rev`). It keeps
one parameter provider for its lifetime, so θ_d is reused across folds.

### U10 Phase A — premise checks for the typed model

**I54 — Definitions and additions.** Trade label at bar t: dir_t = MomentumRule(3) direction,
follow_ret_t = dir_t · (vwap[t+4] / vwap[t+1] − 1); momentum = following wins (> 0), reversion =
fading wins (< 0); bars with dir_t = 0, no exit bar inside the day or a zero return are excluded.
The gate compares each Hawkes feature in its expected orientation (p_cross is expected to be
higher for reversion, so 1 − AUC counts). Added beyond the work package: A1b, a held-out logistic
regression of the trade label on causal features the network could see (signed moves and sizes at
t, t−1, t−2, the intensity asymmetry at t, t−1, t−2, the log total intensity), because a teacher can
only train what the inputs make predictable. Pool maps use quantile bins of the encoded input
probabilities (50/75/90/97 %). A4 uses seed-0 test signals and is reported for the critique only.
Report: [docs/reports/typed_premise_checks.md](reports/typed_premise_checks.md).

**I55 — VWAP smoothing check.** The vwap bars match the paper (non-overlapping blocks of `num`
transactions, volume-weighted) and remove the bid-ask zig-zag (raw aggTrade changes: lag-1
autocorrelation −0.52). Averaging itself creates lag-1 autocorrelation ≈ 0.23–0.25 in vwap
differences (Working effect), which makes the paper's momentum label call about 54 % of bars of a
pure random walk "momentum" — the paper's reported momentum share. Strategies and the typed-model
trade label enter at t+1 and exit at t+4 and are not affected (random-walk win rate 50 %). No code
change: the paper's smoothing is kept for the replication; the effect is reported.
[docs/reports/vwap_smoothing_check.md](reports/vwap_smoothing_check.md).

**I56 — Price-difference check.** Differencing removes the intraday trend as the paper intends:
the drift left in d_t is < 2 % of its standard deviation on every validation day and explains
≤ 0.03 % of its variance. The paper's "magnitude bias" between hours is mostly removed by the
trade-count bars (per-bar |d| varies ≈ 1.2× across hours while bar counts vary ≈ 3×), not by the
differencing. d_t stays in USD as in the paper; for multi-day training windows (E4) price-level
changes between days rescale it relative to the z-score statistics. No code change.
[docs/reports/differencing_check.md](reports/differencing_check.md).

**I57 — Order-flow appendix check (U11).** Order-flow imbalance per bar = (taker-buy − taker-sell
volume) / bar volume from the aggTrades side flag, oriented by the trade direction. On held-out
validation days it predicts the trade outcome with AUC 0.538 (price features: 0.512; current bar's
imbalance alone: 0.541), and typing with it earns +0.203 bp per trade vs +0.087 bp for always
following (oracle +2.524 bp). This passes the 0.53 premise threshold that price features failed;
it is reported as an appendix and future work, not built into a model (scope decision U11).
[docs/reports/orderflow_check.md](reports/orderflow_check.md).

### Planned (to be recorded in detail when implemented)

- M10: in the Problem-2 AUC, "nearest event" means the event at bar `t`, else the next one,
  matching the reward timing.

---

## [O] Open questions

None at the moment.

**O2 → resolved as U9 (2026-09-28).**

**O1 → resolved as U7 (2026-09-28).**
