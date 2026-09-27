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
| P17 | Synaptic delay not given | 1 tick | `models.*.core.synapse.delay_ticks` |

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

### Planned (to be recorded in detail when implemented)

- M2/M7: the Hawkes step needs the `W_h` days before each transformed day, which can lie
  outside the training days passed to `fit()`. It receives a history source that can only
  return days strictly before the day being transformed; the causality harness covers it.
- M4: health checks for silent or saturated pools; the paper's topology drives each
  64-neuron pool from a single input spike train, so redundant neurons and weights drifting
  to their bounds are expected and not "fixed".
- M8: reward delivery schedule is configurable (`rstdp.delivery`), so the mandatory
  reduction test can deliver a reward every tick.
- M10: in the Problem-2 AUC, "nearest event" means the event at bar `t`, else the next one,
  matching the reward timing.

---

## [O] Open questions

**O1 — Tuning `new_mean` vs matched input firing rate.** Plan §6.4.3 requires both models to
use the same `new_mean` so mean input rates match; plan §12 lists `new_mean` among the
parameters each model tunes on the same grid, which can yield different values.
Proposed: tune `new_mean` for the baseline and reuse that value for the improved model, which
spends the same trial budget on `γ` and `tau_z_bars` instead. This is conservative for the
improved model. Needs the author's decision before tuning (M5).
