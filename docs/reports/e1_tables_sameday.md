# E1 sensitivity — paper-literal same-day normalisation, 100 trades per bar

## Table 3 — spikes

| Model | Split | Signals | Seeds | Spike accuracy | ± seeds | Chance level | Momentum % | Momentum base % | Spikes per day |
|---|---|---|---|---|---|---|---|---|---|
| paper | test | model | 5 | 57.61 % | 0.32 pp | 51.75 % | 59.02 % | 56.26 % | 1,370 |
| paper | test | random timing | 5 | 51.76 % | 0.03 pp | 51.75 % | 56.27 % | 56.26 % | 1,370 |
| paper | test | big-move | 5 | 61.31 % | 0.78 pp | 51.75 % | 62.94 % | 56.26 % | 1,496 |
| paper | train | model | 5 | 60.63 % | 0.20 pp | 51.74 % | 58.65 % | 56.25 % | 850 |

## Table 4 — strategies, paper execution (next bar)

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 257.66 % ± 33.46 | 36.74 % ± 2.14 | 8.56 ± 1.27 | 51.36 % ± 0.06 | 0.99 ± 0.00 | 1,369 |
| alexanders filter | random timing | 301.57 % ± 28.40 | 31.80 % ± 0.68 | 11.56 ± 0.86 | 51.27 % ± 0.01 | 1.01 ± 0.00 | 1,370 |
| alexanders filter | big-move | 398.27 % ± 60.74 | 53.15 % ± 1.70 | 9.09 ± 1.17 | 51.80 % ± 0.05 | 0.99 ± 0.00 | 1,496 |
| momentum | model | 239.69 % ± 32.14 | 37.06 % ± 1.69 | 7.88 ± 1.11 | 51.35 % ± 0.07 | 0.99 ± 0.00 | 1,369 |
| momentum | random timing | 255.24 % ± 22.28 | 31.92 % ± 0.82 | 9.74 ± 0.62 | 51.21 % ± 0.01 | 1.00 ± 0.00 | 1,370 |
| momentum | big-move | 401.25 % ± 61.30 | 52.56 % ± 1.80 | 9.26 ± 1.19 | 51.79 % ± 0.05 | 0.99 ± 0.00 | 1,496 |
| stochastic oscillator | model | 250.25 % ± 28.80 | 37.69 % ± 1.26 | 8.09 ± 1.07 | 51.38 % ± 0.04 | 0.99 ± 0.00 | 1,369 |
| stochastic oscillator | random timing | 278.82 % ± 23.93 | 31.91 % ± 0.85 | 10.65 ± 0.67 | 51.24 % ± 0.01 | 1.01 ± 0.00 | 1,370 |
| stochastic oscillator | big-move | 396.97 % ± 60.71 | 53.06 % ± 1.78 | 9.08 ± 1.16 | 51.80 % ± 0.05 | 0.99 ± 0.00 | 1,496 |

## Table 4 — strategies, 10 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 69.43 % ± 21.90 | 29.56 % ± 2.17 | 2.84 ± 0.78 | 50.61 % ± 0.07 | 0.99 ± 0.00 | 1,369 |
| alexanders filter | random timing | 73.60 % ± 6.13 | 24.07 % ± 1.10 | 3.72 ± 0.15 | 50.50 % ± 0.01 | 0.99 ± 0.00 | 1,370 |
| alexanders filter | big-move | 223.80 % ± 23.28 | 40.66 % ± 1.99 | 6.68 ± 0.41 | 51.05 % ± 0.02 | 0.99 ± 0.00 | 1,496 |
| momentum | model | 71.03 % ± 13.50 | 30.28 % ± 1.26 | 2.85 ± 0.51 | 50.65 % ± 0.03 | 0.99 ± 0.00 | 1,369 |
| momentum | random timing | 70.75 % ± 5.82 | 25.62 % ± 1.20 | 3.35 ± 0.13 | 50.56 % ± 0.01 | 0.99 ± 0.00 | 1,370 |
| momentum | big-move | 218.52 % ± 23.13 | 40.73 % ± 2.00 | 6.51 ± 0.42 | 51.05 % ± 0.02 | 0.99 ± 0.00 | 1,496 |
| stochastic oscillator | model | 72.32 % ± 15.72 | 30.16 % ± 1.37 | 2.91 ± 0.56 | 50.65 % ± 0.05 | 0.99 ± 0.00 | 1,369 |
| stochastic oscillator | random timing | 73.25 % ± 5.43 | 24.81 % ± 1.23 | 3.59 ± 0.10 | 50.54 % ± 0.01 | 0.99 ± 0.00 | 1,370 |
| stochastic oscillator | big-move | 215.87 % ± 23.28 | 40.38 % ± 2.09 | 6.49 ± 0.41 | 51.04 % ± 0.02 | 0.99 ± 0.00 | 1,496 |

## Table 4 — strategies, 100 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 24.46 % ± 19.26 | 31.63 % ± 2.73 | 0.92 ± 0.71 | 50.45 % ± 0.06 | 0.99 ± 0.00 | 1,369 |
| alexanders filter | random timing | 41.36 % ± 3.30 | 23.72 % ± 1.11 | 2.12 ± 0.08 | 50.40 % ± 0.02 | 0.99 ± 0.00 | 1,369 |
| alexanders filter | big-move | 180.93 % ± 19.03 | 42.04 % ± 1.76 | 5.23 ± 0.37 | 50.87 % ± 0.02 | 1.00 ± 0.00 | 1,496 |
| momentum | model | 25.24 % ± 12.37 | 31.82 % ± 2.43 | 0.95 ± 0.44 | 50.48 % ± 0.04 | 0.99 ± 0.00 | 1,369 |
| momentum | random timing | 41.03 % ± 3.76 | 25.20 % ± 1.22 | 1.97 ± 0.11 | 50.45 % ± 0.02 | 0.99 ± 0.00 | 1,369 |
| momentum | big-move | 173.86 % ± 19.01 | 40.93 % ± 1.75 | 5.16 ± 0.38 | 50.86 % ± 0.02 | 0.99 ± 0.00 | 1,496 |
| stochastic oscillator | model | 26.52 % ± 12.30 | 32.70 % ± 1.95 | 0.98 ± 0.43 | 50.48 % ± 0.04 | 0.99 ± 0.00 | 1,369 |
| stochastic oscillator | random timing | 41.62 % ± 3.20 | 24.46 % ± 1.22 | 2.07 ± 0.06 | 50.43 % ± 0.01 | 0.99 ± 0.00 | 1,369 |
| stochastic oscillator | big-move | 176.43 % ± 19.19 | 41.61 % ± 1.81 | 5.15 ± 0.37 | 50.86 % ± 0.02 | 0.99 ± 0.00 | 1,496 |

## Table 4 — strategies, 1000 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 11.57 % ± 17.10 | 29.13 % ± 1.80 | 0.46 ± 0.69 | 50.39 % ± 0.06 | 0.99 ± 0.00 | 1,369 |
| alexanders filter | random timing | 11.92 % ± 0.72 | 23.01 % ± 1.07 | 0.62 ± 0.06 | 50.30 % ± 0.00 | 0.99 ± 0.00 | 1,369 |
| alexanders filter | big-move | 113.54 % ± 11.88 | 34.47 % ± 1.99 | 4.00 ± 0.22 | 50.64 % ± 0.02 | 0.99 ± 0.00 | 1,496 |
| momentum | model | 21.20 % ± 10.02 | 30.66 % ± 1.24 | 0.84 ± 0.38 | 50.42 % ± 0.03 | 0.99 ± 0.00 | 1,369 |
| momentum | random timing | 17.20 % ± 0.90 | 24.41 % ± 1.18 | 0.85 ± 0.05 | 50.37 % ± 0.01 | 0.99 ± 0.00 | 1,369 |
| momentum | big-move | 110.98 % ± 11.92 | 35.01 % ± 1.93 | 3.85 ± 0.23 | 50.63 % ± 0.02 | 0.99 ± 0.00 | 1,496 |
| stochastic oscillator | model | 16.92 % ± 11.96 | 29.85 % ± 1.82 | 0.68 ± 0.46 | 50.41 % ± 0.04 | 0.99 ± 0.00 | 1,369 |
| stochastic oscillator | random timing | 12.96 % ± 0.78 | 23.68 % ± 1.13 | 0.66 ± 0.06 | 50.34 % ± 0.01 | 0.99 ± 0.00 | 1,369 |
| stochastic oscillator | big-move | 109.59 % ± 11.97 | 34.12 % ± 1.87 | 3.90 ± 0.24 | 50.63 % ± 0.02 | 0.99 ± 0.00 | 1,496 |

Mean over seeds; ± is the standard deviation across seeds (random timing: across its seeds' repetition means). Returns are fee-free (decision U3).
