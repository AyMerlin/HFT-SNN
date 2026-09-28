# E1 sensitivity — paper-literal same-day normalisation, 10 trades per bar

## Table 3 — spikes

| Model | Split | Signals | Seeds | Spike accuracy | ± seeds | Chance level | Momentum % | Momentum base % | Spikes per day |
|---|---|---|---|---|---|---|---|---|---|
| paper | test | model | 5 | 56.53 % | 0.01 pp | 53.37 % | 80.51 % | 77.98 % | 25,793 |
| paper | test | random timing | 5 | 53.37 % | 0.00 pp | 53.37 % | 77.98 % | 77.98 % | 25,793 |
| paper | test | big-move | 5 | 65.18 % | 0.00 pp | 53.37 % | 84.53 % | 77.98 % | 28,086 |
| paper | train | model | 5 | 56.50 % | 0.02 pp | 53.35 % | 80.52 % | 78.00 % | 25,570 |

## Table 4 — strategies, paper execution (next bar)

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 18,278.25 % ± 12.08 | 852.63 % ± 4.56 | 26.08 ± 0.13 | 74.05 % ± 0.01 | 0.84 ± 0.00 | 25,793 |
| alexanders filter | random timing | 18,726.62 % ± 10.43 | 904.02 % ± 4.45 | 25.20 ± 0.11 | 73.83 % ± 0.00 | 0.94 ± 0.00 | 25,792 |
| alexanders filter | big-move | 21,957.32 % ± 9.58 | 1,189.81 % ± 5.18 | 22.45 ± 0.09 | 73.86 % ± 0.00 | 0.85 ± 0.00 | 28,086 |
| momentum | model | 17,008.56 % ± 11.82 | 789.06 % ± 4.11 | 26.23 ± 0.12 | 73.03 % ± 0.01 | 0.83 ± 0.00 | 25,793 |
| momentum | random timing | 16,801.20 % ± 10.10 | 806.00 % ± 4.38 | 25.36 ± 0.13 | 72.17 % ± 0.00 | 0.92 ± 0.00 | 25,793 |
| momentum | big-move | 21,311.15 % ± 9.74 | 1,147.49 % ± 4.92 | 22.60 ± 0.09 | 73.50 % ± 0.00 | 0.85 ± 0.00 | 28,085 |
| stochastic oscillator | model | 17,467.91 % ± 9.77 | 809.56 % ± 4.54 | 26.25 ± 0.14 | 73.42 % ± 0.01 | 0.83 ± 0.00 | 25,793 |
| stochastic oscillator | random timing | 17,636.99 % ± 9.51 | 843.53 % ± 4.45 | 25.44 ± 0.12 | 72.94 % ± 0.00 | 0.93 ± 0.00 | 25,793 |
| stochastic oscillator | big-move | 21,647.10 % ± 9.43 | 1,168.80 % ± 4.98 | 22.53 ± 0.09 | 73.70 % ± 0.00 | 0.85 ± 0.00 | 28,085 |

## Table 4 — strategies, 10 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 13,269.87 % ± 12.54 | 633.75 % ± 3.11 | 25.48 ± 0.12 | 65.20 % ± 0.01 | 0.96 ± 0.00 | 25,792 |
| alexanders filter | random timing | 13,168.78 % ± 7.51 | 654.15 % ± 3.59 | 24.49 ± 0.12 | 64.95 % ± 0.00 | 1.00 ± 0.00 | 25,792 |
| alexanders filter | big-move | 14,442.09 % ± 8.59 | 784.97 % ± 3.85 | 22.38 ± 0.10 | 65.52 % ± 0.00 | 0.91 ± 0.00 | 28,086 |
| momentum | model | 12,631.48 % ± 11.78 | 602.71 % ± 3.48 | 25.50 ± 0.14 | 64.68 % ± 0.00 | 0.95 ± 0.00 | 25,793 |
| momentum | random timing | 12,149.58 % ± 8.64 | 603.96 % ± 3.67 | 24.48 ± 0.14 | 64.06 % ± 0.00 | 0.99 ± 0.00 | 25,793 |
| momentum | big-move | 14,209.43 % ± 8.80 | 767.45 % ± 3.85 | 22.53 ± 0.10 | 65.38 % ± 0.00 | 0.91 ± 0.00 | 28,085 |
| stochastic oscillator | model | 12,876.48 % ± 15.20 | 613.11 % ± 3.45 | 25.55 ± 0.14 | 64.89 % ± 0.01 | 0.95 ± 0.00 | 25,793 |
| stochastic oscillator | random timing | 12,610.24 % ± 7.51 | 623.16 % ± 3.54 | 24.62 ± 0.13 | 64.50 % ± 0.00 | 0.99 ± 0.00 | 25,793 |
| stochastic oscillator | big-move | 14,343.04 % ± 8.44 | 778.19 % ± 3.80 | 22.43 ± 0.10 | 65.46 % ± 0.00 | 0.91 ± 0.00 | 28,085 |

## Table 4 — strategies, 100 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 6,592.13 % ± 15.55 | 246.97 % ± 2.42 | 32.48 ± 0.29 | 59.69 % ± 0.01 | 0.91 ± 0.00 | 25,792 |
| alexanders filter | random timing | 6,459.65 % ± 4.81 | 252.79 % ± 1.12 | 31.09 ± 0.12 | 59.46 % ± 0.00 | 0.93 ± 0.00 | 25,792 |
| alexanders filter | big-move | 8,372.69 % ± 3.93 | 347.72 % ± 1.32 | 29.30 ± 0.10 | 60.45 % ± 0.00 | 0.91 ± 0.00 | 28,086 |
| momentum | model | 6,288.48 % ± 16.05 | 235.28 % ± 2.14 | 32.52 ± 0.27 | 59.42 % ± 0.01 | 0.91 ± 0.00 | 25,793 |
| momentum | random timing | 5,892.42 % ± 4.28 | 229.17 % ± 1.15 | 31.29 ± 0.14 | 58.95 % ± 0.00 | 0.92 ± 0.00 | 25,793 |
| momentum | big-move | 8,346.18 % ± 4.36 | 345.48 % ± 1.32 | 29.39 ± 0.10 | 60.43 % ± 0.00 | 0.91 ± 0.00 | 28,085 |
| stochastic oscillator | model | 6,412.49 % ± 12.46 | 238.97 % ± 2.06 | 32.65 ± 0.26 | 59.54 % ± 0.01 | 0.91 ± 0.00 | 25,793 |
| stochastic oscillator | random timing | 6,159.92 % ± 3.78 | 238.02 % ± 1.20 | 31.49 ± 0.14 | 59.22 % ± 0.00 | 0.93 ± 0.00 | 25,793 |
| stochastic oscillator | big-move | 8,370.44 % ± 4.08 | 347.71 % ± 1.31 | 29.29 ± 0.10 | 60.45 % ± 0.00 | 0.91 ± 0.00 | 28,085 |

Mean over seeds; ± is the standard deviation across seeds (random timing: across its seeds' repetition means). Returns are fee-free (decision U3).
