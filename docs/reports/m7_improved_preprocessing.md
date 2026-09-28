# M7 — improved preprocessing on real validation days

Setup: 100 trades per bar, bar clock, events above the fit-window 0.9 quantile of |d| (U9),
`LinearNormalized` marks, W_h = 1, W_snn = 1; first 10 validation folds
(2025-10-17 → 2025-10-26). Both pipelines are fitted on the
training day and applied to the test day with the same `new_mean`. Runtime 4 s.

## Input rates (§6.4.3)

With equal `new_mean`, both encodings deliver the configured mean input probability per channel on
the test days, so input rates match when the models are configured alike (U7 lets each model tune
its own `new_mean`; realised rates are logged per day in every run's `input_stats.csv`).

| new_mean | encoding | mean input prob X1 | mean input prob X2 | share clipped at 1 | share at 0 | Spearman(input, next moves) |
|---|---|---|---|---|---|---|
| 0.05 | improved (Hawkes) | 0.0686 | 0.0687 | 0.0006 | 0.0000 | 0.253 |
| 0.05 | paper (z-score) | 0.0712 | 0.0707 | 0.0001 | 0.3543 | 0.094 |
| 0.1 | improved (Hawkes) | 0.1352 | 0.1351 | 0.0049 | 0.0000 | 0.253 |
| 0.1 | paper (z-score) | 0.1077 | 0.1071 | 0.0002 | 0.1811 | 0.092 |
| 0.2 | improved (Hawkes) | 0.2611 | 0.2593 | 0.0257 | 0.0000 | 0.252 |
| 0.2 | paper (z-score) | 0.2011 | 0.2004 | 0.0002 | 0.0113 | 0.066 |

*Next moves* = spike strength S_strength (mean absolute return over the next 3 bars).

## Reward streams of the R-STDP pools (new_mean = 0.05)

`reward_mom = M = p_same − p_cross` at event bars, `reward_rev = −reward_mom`, 0 elsewhere.

| event_share | threshold | rho | p_bg | p_same | p_cross | reward_mom_mean | reward_mom_std | share_positive | share_negative | bars_with_reward |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.107 | 20.435 | 0.635 | 0.399 | 0.370 | 0.231 | 0.139 | 0.327 | 0.628 | 0.370 | 0.107 |

## Causality on real data

The full improved pipeline passed the causality harness on 2025-10-19 (history days, test day and the
following day perturbed after bars 0, 100, 6253, 12504; three perturbation modes each),
in 2 s.
