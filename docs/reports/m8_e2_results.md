# M8 — E2 improved model, A1 ablation and input-rate controls (check-in)

**Setup.** Same data, bars (100 `aggTrades`), test days (2025-12-01 → 2026-09-26, 300 days), seeds
(0–4), benchmarks and latencies as E1. Improved model tuned on the validation period with the same
grid and budget as the baseline ([m8_tuning_improved.md](m8_tuning_improved.md), trial t076:
threshold 32, leak 0.1, `new_mean` 0.3, STDP scale 0.5, γ 0.1, τ_z 1 bar, event quantile 0.95),
`W_snn = 1`, `W_h = 1`. Full tables: [e2_tables.md](e2_tables.md); baseline: [m5_e1_baseline.md](m5_e1_baseline.md).

```bash
.venv/bin/python -m experiments.run_experiment --config experiments/configs/improved.yaml
```

## Spikes

| Model | Spike accuracy | Signals per day | Share of bars | Big-move at the same rate | Gap to big-move | Momentum spike % |
|---|---|---|---|---|---|---|
| paper SNN (E1) | 57.39 % | 1,418 | 8.2 % | 61.28 % | −3.9 pp | 59.1 % |
| **improved** | **58.57 %** | 2,614 | 15.6 % | 60.00 % | **−1.4 pp** | 55.9 % |
| A1: Hawkes input, no R-STDP pools | 58.42 % | 1,827 | 10.2 % | 61.19 % | −2.8 pp | 55.9 % |
| control: improved at `new_mean` 0.05 | 75.60 % | 50 | 0.4 % | 77.85 % | −2.3 pp | 53.8 % |
| control: paper SNN at `new_mean` 0.3 | 52.45 % | 9,050 | 56.5 % | 54.22 % | −1.8 pp | 57.7 % |

Chance level (random timing) 51.75 % on these days; random-timing momentum share 56.3 %.

- **The improved model is more accurate than the baseline**: +1.22 pp per day (higher on 186 of
  297 days with signals from both, Wilcoxon p ≈ 1·10⁻⁷), while firing about twice as often.
  Because accuracy falls as a model fires more (see the tuning reports), the rate-matched gap to
  the big-move benchmark is the fairer yardstick: it shrinks from −3.9 pp to −1.4 pp.
- **Almost all of the gain comes from the Hawkes input (Problem 1).** A1 alone is +1.06 pp over
  the baseline (p ≈ 1·10⁻⁶); the R-STDP pools add +0.16 pp over A1 (p ≈ 2·10⁻⁷).
- **The R-STDP pools do not specialise (Problem 2).** At the improved model's output spikes,
  D = spikes(H_mom) − spikes(H_rev) does not separate the paper's momentum from reversion spikes:
  AUC 0.501 over 3.9 million evaluable signals (0.5 = no specialisation). Both pools fire about
  7.5 spikes per signal bar.
- **Input rates are not interchangeable between encodings.** At the baseline's `new_mean` the
  improved model hardly fires (0.4 % of bars, unhealthy on 81 % of seed-days); the baseline at the
  improved model's `new_mean` saturates (56 % of bars, unhealthy on 99.7 %). This supports tuning
  `new_mean` per model (U7).
- Health: the improved model has warnings on 52 of 1,500 test seed-days (3.5 %), A1 on 26 (1.7 %),
  the baseline on 0.

## Strategies (fee-free)

- As for the baseline, **no strategy of the improved model beats random timing** in daily P&L
  (paired Wilcoxon over 300 days, p = 0.04–0.7; with next-bar execution it is slightly behind,
  with 10–1000 ms latency slightly ahead, neither consistently).
- Per trade the improved model earns about the same as the baseline with next-bar execution
  (0.066–0.077 bp) and somewhat more with latency (e.g. momentum at 100 ms: 0.019 vs 0.012 bp).
- The big-move benchmark earns more than the improved model in every configuration.

## Summary for the thesis

| Question | Answer on BTCUSDT, 300 test days |
|---|---|
| Does the Hawkes input (Problem 1) help? | Yes: +1.06 pp spike accuracy at a higher signal rate; the rate-matched gap to a volatility filter halves. |
| Do the R-STDP pools learn momentum vs reversion (Problem 2)? | No: AUC 0.50; their accuracy contribution is +0.16 pp. |
| Do either model's spikes improve trading over random timing? | No (fee-free, any latency). |
| Does either model beat a simple big-move filter? | No, on accuracy or P&L. |
