# M6 — Hawkes fits on real validation days (U6 review)

Setup: BTCUSDT `aggTrades`, 100 trades per bar, bar clock, `LinearNormalized` marks,
5 restarts; θ_d = fit(days d − W_h … d − 1), applied to day d. Validation days
2025-10-17 → 2025-11-30 (45 days × 4 windows, 14 s on 8 workers).
Means over days; per-day values in `results/m6_hawkes_review.csv`.

## Events and fit

| W_h | bars/day | events/day | event share of bars | up share | ρ(A) | fit seconds |
|---|---|---|---|---|---|---|
| 1 | 19836.133 | 19835.133 | 1.000 | 0.500 | 0.077 | 0.000 |
| 3 | 19836.133 | 19835.133 | 1.000 | 0.500 | 0.061 | 0.000 |
| 5 | 19836.133 | 19835.133 | 1.000 | 0.500 | 0.056 | 0.000 |
| 10 | 19836.133 | 19835.133 | 1.000 | 0.500 | 0.046 | 0.000 |

All fits converged: True.

## Parameters (mean over days) and branching matrix A = α/β

| W_h | mu_u | mu_d | alpha_uu | beta_uu | alpha_ud | beta_ud | alpha_du | beta_du | alpha_dd | beta_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.470 | 0.476 | 0.002 | 0.134 | 0.000 | 462.479 | 0.000 | 185.635 | 0.002 | 0.142 |
| 3 | 0.471 | 0.484 | 0.002 | 0.033 | 0.000 | 308.628 | 0.000 | 307.415 | 0.002 | 33.146 |
| 5 | 0.472 | 0.484 | 0.002 | 0.035 | 0.000 | 303.450 | 0.000 | 234.920 | 0.002 | 0.173 |
| 10 | 0.476 | 0.488 | 0.002 | 0.038 | 0.000 | 348.171 | 0.000 | 330.180 | 0.001 | 5.352 |

| W_h | A_uu (up→up) | A_ud (down→up) | A_du (up→down) | A_dd (down→down) |
|---|---|---|---|---|
| 1 | 0.059 | 0.004 | 0.003 | 0.046 |
| 3 | 0.056 | 0.000 | 0.000 | 0.034 |
| 5 | 0.054 | 0.000 | 0.000 | 0.035 |
| 10 | 0.044 | 0.000 | 0.000 | 0.028 |

Parameter stability across consecutive days (coefficient of variation, std / mean):

| W_h | mu_u | mu_d | alpha_uu | beta_uu | alpha_ud | beta_ud | alpha_du | beta_du | alpha_dd | beta_dd | ρ(A) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.06 | 0.05 | 1.14 | 4.03 | 3.10 | 1.03 | 1.81 | 1.94 | 1.42 | 4.59 | 0.62 |
| 3 | 0.04 | 0.04 | 0.89 | 0.72 | 3.25 | 1.38 | 4.66 | 1.34 | 1.15 | 4.93 | 0.57 |
| 5 | 0.03 | 0.03 | 0.77 | 0.61 | 2.43 | 1.26 | 1.93 | 1.35 | 0.91 | 4.62 | 0.51 |
| 10 | 0.03 | 0.02 | 0.77 | 0.58 | 2.72 | 1.29 | 2.63 | 1.25 | 0.93 | 5.54 | 0.57 |

## Branching split and momentum score (drives the R-STDP rewards)

| W_h | background p_bg | same direction p_same | opposite p_cross | M mean | M std | M 10 % | M 90 % | share abs(M) > 0.2 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.946 | 0.052 | 0.002 | 0.050 | 0.030 | 0.018 | 0.088 | 0.005 |
| 3 | 0.956 | 0.044 | 0.000 | 0.044 | 0.021 | 0.021 | 0.071 | 0.001 |
| 5 | 0.957 | 0.043 | 0.000 | 0.043 | 0.017 | 0.023 | 0.065 | 0.000 |
| 10 | 0.965 | 0.035 | 0.000 | 0.035 | 0.014 | 0.019 | 0.053 | 0.000 |

## Next-day fit and information content

| W_h | in-sample loglik/event | next-day loglik/event | CV of λ_u+λ_d over bars | Spearman(λ⁺, next moves) | Spearman(abs d_t, next moves) | partial Spearman(λ⁺ given abs d_t) |
|---|---|---|---|---|---|---|
| 1 | -1.693 | -1.693 | 0.011 | 0.157 | 0.063 | 0.149 |
| 3 | -1.693 | -1.693 | 0.008 | 0.168 | 0.063 | 0.160 |
| 5 | -1.693 | -1.693 | 0.008 | 0.181 | 0.063 | 0.172 |
| 10 | -1.693 | -1.693 | 0.006 | 0.181 | 0.063 | 0.171 |

*Next moves* = the paper's spike strength S_strength (mean absolute return over the next 3 bars).
The partial correlation measures what the intensity adds beyond the current price change,
i.e. the memory Problem 1 is about.

## Alternative event definitions (first 10 validation days, W_h = 1)

Not in the spec except `wallclock`. The threshold variants count a bar as an event only if
abs(d_t) exceeds the given quantile of abs(d) on the fit day (causal; frozen with θ_d).

| time axis | events | event share of bars | ρ(A) | p_bg | p_same | p_cross | M std | share abs(M) > 0.2 | CV of λ | Spearman(λ⁺, next moves) | Spearman(abs d_t, next moves) | partial (λ⁺ given abs d_t) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bar_index | every move | 1.000 | 0.089 | 0.935 | 0.065 | 0.000 | 0.034 | 0.006 | 0.017 | 0.212 | 0.094 | 0.194 |
| wallclock | every move | 1.000 | 0.590 | 0.413 | 0.329 | 0.258 | 0.582 | 0.810 | 1.269 | 0.093 | 0.094 | 0.056 |
| bar_index | abs d above q80 | 0.200 | 0.577 | 0.452 | 0.338 | 0.210 | 0.455 | 0.932 | 0.274 | 0.252 | 0.094 | 0.239 |
| bar_index | abs d above q90 | 0.107 | 0.633 | 0.443 | 0.395 | 0.162 | 0.361 | 0.781 | 0.421 | 0.245 | 0.094 | 0.233 |
| wallclock | abs d above q90 | 0.107 | 0.778 | 0.276 | 0.346 | 0.378 | 0.528 | 0.776 | 2.348 | 0.248 | 0.094 | 0.233 |
