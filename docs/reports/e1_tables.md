# E1 — paper SNN baseline, 100 trades per bar, test period (2025-12-01 → 2026-09-26)

## Table 3 — spikes

| Model | Split | Signals | Seeds | Spike accuracy | ± seeds | Chance level | Momentum % | Momentum base % | Spikes per day |
|---|---|---|---|---|---|---|---|---|---|
| paper | test | model | 5 | 57.39 % | 0.26 pp | 51.75 % | 59.11 % | 56.26 % | 1,418 |
| paper | test | random timing | 5 | 51.76 % | 0.01 pp | 51.75 % | 56.25 % | 56.26 % | 1,418 |
| paper | test | big-move | 5 | 61.28 % | 0.71 pp | 51.75 % | 63.02 % | 56.26 % | 1,606 |
| paper | train | model | 5 | 60.63 % | 0.20 pp | 51.74 % | 58.65 % | 56.25 % | 850 |

## Table 4 — strategies, paper execution (next bar)

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 296.14 % ± 42.64 | 38.68 % ± 2.54 | 9.38 ± 1.74 | 51.49 % ± 0.06 | 0.99 ± 0.00 | 1,418 |
| alexanders filter | random timing | 320.94 % ± 32.21 | 34.17 % ± 0.92 | 11.45 ± 0.89 | 51.31 % ± 0.01 | 1.01 ± 0.00 | 1,418 |
| alexanders filter | big-move | 452.07 % ± 67.66 | 58.04 % ± 2.59 | 9.45 ± 1.13 | 51.88 % ± 0.06 | 0.99 ± 0.00 | 1,606 |
| momentum | model | 279.27 % ± 42.63 | 39.53 % ± 2.94 | 8.64 ± 1.46 | 51.46 % ± 0.07 | 0.99 ± 0.00 | 1,418 |
| momentum | random timing | 276.03 % ± 27.26 | 34.00 % ± 1.18 | 9.88 ± 0.68 | 51.26 % ± 0.01 | 1.00 ± 0.00 | 1,418 |
| momentum | big-move | 454.44 % ± 68.25 | 57.59 % ± 2.62 | 9.57 ± 1.13 | 51.87 % ± 0.06 | 0.99 ± 0.00 | 1,606 |
| stochastic oscillator | model | 289.69 % ± 39.79 | 39.46 % ± 2.57 | 8.99 ± 1.55 | 51.50 % ± 0.05 | 0.99 ± 0.00 | 1,418 |
| stochastic oscillator | random timing | 298.98 % ± 28.71 | 34.27 % ± 1.18 | 10.62 ± 0.72 | 51.29 % ± 0.01 | 1.00 ± 0.00 | 1,418 |
| stochastic oscillator | big-move | 450.88 % ± 67.43 | 58.00 % ± 2.65 | 9.43 ± 1.11 | 51.88 % ± 0.06 | 0.99 ± 0.00 | 1,606 |

## Table 4 — strategies, 10 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 99.79 % ± 27.35 | 31.12 % ± 1.40 | 3.89 ± 0.96 | 50.72 % ± 0.09 | 0.99 ± 0.00 | 1,418 |
| alexanders filter | random timing | 86.77 % ± 7.53 | 25.05 % ± 1.22 | 4.21 ± 0.17 | 50.55 % ± 0.01 | 0.99 ± 0.00 | 1,418 |
| alexanders filter | big-move | 248.63 % ± 24.89 | 42.66 % ± 2.01 | 7.08 ± 0.40 | 51.10 % ± 0.02 | 0.99 ± 0.00 | 1,606 |
| momentum | model | 100.24 % ± 20.06 | 31.47 % ± 2.00 | 3.86 ± 0.64 | 50.76 % ± 0.06 | 0.99 ± 0.00 | 1,418 |
| momentum | random timing | 85.78 % ± 8.95 | 26.44 % ± 1.42 | 3.94 ± 0.21 | 50.61 % ± 0.02 | 0.99 ± 0.00 | 1,418 |
| momentum | big-move | 243.50 % ± 24.96 | 41.99 % ± 2.13 | 7.04 ± 0.39 | 51.10 % ± 0.02 | 0.99 ± 0.00 | 1,606 |
| stochastic oscillator | model | 102.83 % ± 19.01 | 31.43 % ± 1.54 | 3.97 ± 0.62 | 50.75 % ± 0.06 | 0.99 ± 0.00 | 1,418 |
| stochastic oscillator | random timing | 87.53 % ± 7.65 | 25.81 % ± 1.48 | 4.12 ± 0.13 | 50.59 % ± 0.01 | 0.99 ± 0.00 | 1,418 |
| stochastic oscillator | big-move | 241.23 % ± 24.82 | 42.16 % ± 2.08 | 6.95 ± 0.40 | 51.09 % ± 0.01 | 0.99 ± 0.00 | 1,606 |

## Table 4 — strategies, 100 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 51.80 % ± 27.52 | 36.05 % ± 1.46 | 1.75 ± 0.95 | 50.56 % ± 0.08 | 0.99 ± 0.00 | 1,418 |
| alexanders filter | random timing | 52.51 % ± 4.10 | 24.70 % ± 1.22 | 2.58 ± 0.08 | 50.44 % ± 0.01 | 0.99 ± 0.00 | 1,418 |
| alexanders filter | big-move | 198.92 % ± 20.45 | 44.11 % ± 1.62 | 5.48 ± 0.38 | 50.89 % ± 0.02 | 0.99 ± 0.00 | 1,606 |
| momentum | model | 52.24 % ± 19.12 | 35.59 % ± 1.97 | 1.78 ± 0.62 | 50.59 % ± 0.06 | 0.99 ± 0.00 | 1,418 |
| momentum | random timing | 54.35 % ± 5.21 | 25.96 % ± 1.33 | 2.54 ± 0.12 | 50.50 % ± 0.02 | 0.99 ± 0.00 | 1,418 |
| momentum | big-move | 191.79 % ± 20.52 | 42.54 % ± 1.69 | 5.47 ± 0.39 | 50.89 % ± 0.02 | 0.99 ± 0.00 | 1,606 |
| stochastic oscillator | model | 53.80 % ± 19.33 | 36.92 % ± 1.83 | 1.76 ± 0.60 | 50.59 % ± 0.06 | 0.99 ± 0.00 | 1,418 |
| stochastic oscillator | random timing | 54.31 % ± 4.53 | 25.47 % ± 1.41 | 2.59 ± 0.08 | 50.48 % ± 0.01 | 0.99 ± 0.00 | 1,418 |
| stochastic oscillator | big-move | 193.72 % ± 19.79 | 44.15 % ± 1.72 | 5.33 ± 0.36 | 50.88 % ± 0.02 | 0.99 ± 0.00 | 1,606 |

## Table 4 — strategies, 1000 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 36.99 % ± 21.33 | 31.83 % ± 1.38 | 1.41 ± 0.80 | 50.46 % ± 0.07 | 0.99 ± 0.00 | 1,418 |
| alexanders filter | random timing | 18.99 % ± 1.33 | 23.65 % ± 1.15 | 0.97 ± 0.08 | 50.33 % ± 0.01 | 0.99 ± 0.00 | 1,418 |
| alexanders filter | big-move | 123.18 % ± 17.93 | 36.16 % ± 2.42 | 4.13 ± 0.35 | 50.65 % ± 0.03 | 0.99 ± 0.00 | 1,606 |
| momentum | model | 43.96 % ± 14.37 | 33.02 % ± 1.68 | 1.61 ± 0.48 | 50.48 % ± 0.06 | 0.99 ± 0.00 | 1,418 |
| momentum | random timing | 25.43 % ± 2.33 | 24.98 % ± 1.37 | 1.24 ± 0.07 | 50.40 % ± 0.01 | 0.99 ± 0.00 | 1,418 |
| momentum | big-move | 122.81 % ± 17.29 | 36.92 % ± 2.31 | 4.03 ± 0.34 | 50.65 % ± 0.03 | 0.99 ± 0.00 | 1,606 |
| stochastic oscillator | model | 40.53 % ± 13.50 | 31.89 % ± 1.87 | 1.54 ± 0.47 | 50.48 % ± 0.06 | 0.99 ± 0.00 | 1,418 |
| stochastic oscillator | random timing | 20.28 % ± 2.09 | 24.16 % ± 1.34 | 1.02 ± 0.11 | 50.37 % ± 0.01 | 0.99 ± 0.00 | 1,418 |
| stochastic oscillator | big-move | 120.24 % ± 17.73 | 35.88 % ± 2.32 | 4.06 ± 0.36 | 50.65 % ± 0.03 | 0.99 ± 0.00 | 1,606 |

Mean over seeds; ± is the standard deviation across seeds (random timing: across its seeds' repetition means). Returns are fee-free (decision U3).
