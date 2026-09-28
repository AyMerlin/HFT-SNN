# E1 — paper SNN baseline, literal 10 trades per bar, test period (2025-12-01 → 2026-09-26)

## Table 3 — spikes

| Model | Split | Signals | Seeds | Spike accuracy | ± seeds | Chance level | Momentum % | Momentum base % | Spikes per day |
|---|---|---|---|---|---|---|---|---|---|
| paper | test | model | 5 | 56.46 % | 0.01 pp | 53.37 % | 80.42 % | 77.98 % | 26,464 |
| paper | test | random timing | 5 | 53.37 % | 0.00 pp | 53.37 % | 77.98 % | 77.98 % | 26,464 |
| paper | test | big-move | 5 | 65.09 % | 0.01 pp | 53.37 % | 84.41 % | 77.98 % | 29,073 |
| paper | train | model | 5 | 56.50 % | 0.02 pp | 53.35 % | 80.52 % | 78.00 % | 25,570 |

## Table 4 — strategies, paper execution (next bar)

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 18,721.30 % ± 14.26 | 925.00 % ± 5.08 | 24.62 ± 0.12 | 73.99 % ± 0.01 | 0.83 ± 0.00 | 26,463 |
| alexanders filter | random timing | 19,188.64 % ± 10.47 | 985.65 % ± 4.65 | 23.69 ± 0.10 | 73.80 % ± 0.00 | 0.94 ± 0.00 | 26,463 |
| alexanders filter | big-move | 22,800.37 % ± 9.22 | 1,353.53 % ± 5.06 | 20.50 ± 0.07 | 74.02 % ± 0.00 | 0.85 ± 0.00 | 29,073 |
| momentum | model | 17,417.66 % ± 10.51 | 859.05 % ± 4.53 | 24.67 ± 0.12 | 72.99 % ± 0.01 | 0.82 ± 0.00 | 26,463 |
| momentum | random timing | 17,212.79 % ± 9.95 | 879.75 % ± 4.57 | 23.81 ± 0.11 | 72.16 % ± 0.00 | 0.91 ± 0.00 | 26,464 |
| momentum | big-move | 22,134.80 % ± 9.03 | 1,306.53 % ± 4.92 | 20.61 ± 0.07 | 73.66 % ± 0.00 | 0.84 ± 0.00 | 29,073 |
| stochastic oscillator | model | 17,891.10 % ± 10.38 | 880.18 % ± 4.40 | 24.73 ± 0.11 | 73.38 % ± 0.01 | 0.82 ± 0.00 | 26,464 |
| stochastic oscillator | random timing | 18,070.89 % ± 9.52 | 920.55 % ± 4.56 | 23.88 ± 0.11 | 72.92 % ± 0.00 | 0.92 ± 0.00 | 26,464 |
| stochastic oscillator | big-move | 22,486.41 % ± 9.22 | 1,331.09 % ± 5.00 | 20.55 ± 0.07 | 73.87 % ± 0.00 | 0.85 ± 0.00 | 29,073 |

## Table 4 — strategies, 10 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 13,626.01 % ± 16.73 | 698.94 % ± 2.90 | 23.72 ± 0.09 | 65.16 % ± 0.01 | 0.95 ± 0.00 | 26,463 |
| alexanders filter | random timing | 13,532.78 % ± 7.74 | 719.46 % ± 3.72 | 22.89 ± 0.11 | 64.93 % ± 0.00 | 1.00 ± 0.00 | 26,463 |
| alexanders filter | big-move | 15,042.53 % ± 8.08 | 913.07 % ± 3.98 | 20.04 ± 0.08 | 65.59 % ± 0.00 | 0.91 ± 0.00 | 29,073 |
| momentum | model | 12,977.31 % ± 13.62 | 666.90 % ± 3.27 | 23.68 ± 0.11 | 64.65 % ± 0.01 | 0.95 ± 0.00 | 26,463 |
| momentum | random timing | 12,490.08 % ± 8.80 | 665.61 % ± 3.80 | 22.83 ± 0.12 | 64.05 % ± 0.00 | 0.99 ± 0.00 | 26,464 |
| momentum | big-move | 14,800.75 % ± 7.91 | 894.15 % ± 3.91 | 20.14 ± 0.08 | 65.45 % ± 0.00 | 0.91 ± 0.00 | 29,072 |
| stochastic oscillator | model | 13,223.06 % ± 20.46 | 677.78 % ± 3.03 | 23.74 ± 0.09 | 64.86 % ± 0.01 | 0.95 ± 0.00 | 26,463 |
| stochastic oscillator | random timing | 12,961.12 % ± 7.74 | 686.37 % ± 3.69 | 22.98 ± 0.11 | 64.48 % ± 0.00 | 0.99 ± 0.00 | 26,463 |
| stochastic oscillator | big-move | 14,941.91 % ± 8.07 | 905.89 % ± 3.93 | 20.07 ± 0.08 | 65.54 % ± 0.00 | 0.91 ± 0.00 | 29,072 |

## Table 4 — strategies, 100 ms latency

| Strategy | Signals | Accumulated return | Annualized volatility | Sharpe | Win rate | Profit/loss | Trades per day |
|---|---|---|---|---|---|---|---|
| alexanders filter | model | 6,764.80 % ± 10.14 | 270.08 % ± 1.74 | 30.47 ± 0.16 | 59.63 % ± 0.01 | 0.91 ± 0.00 | 26,463 |
| alexanders filter | random timing | 6,622.10 % ± 4.18 | 274.00 % ± 1.00 | 29.41 ± 0.09 | 59.43 % ± 0.00 | 0.93 ± 0.00 | 26,463 |
| alexanders filter | big-move | 8,666.69 % ± 3.75 | 410.06 % ± 1.07 | 25.71 ± 0.06 | 60.46 % ± 0.00 | 0.91 ± 0.00 | 29,073 |
| momentum | model | 6,449.35 % ± 13.11 | 258.78 % ± 1.95 | 30.32 ± 0.19 | 59.37 % ± 0.01 | 0.91 ± 0.00 | 26,463 |
| momentum | random timing | 6,046.20 % ± 3.69 | 249.84 % ± 1.11 | 29.45 ± 0.12 | 58.92 % ± 0.00 | 0.92 ± 0.00 | 26,463 |
| momentum | big-move | 8,635.01 % ± 3.19 | 407.45 % ± 0.98 | 25.78 ± 0.06 | 60.43 % ± 0.00 | 0.91 ± 0.00 | 29,072 |
| stochastic oscillator | model | 6,577.72 % ± 11.06 | 262.81 % ± 1.53 | 30.45 ± 0.14 | 59.48 % ± 0.01 | 0.91 ± 0.00 | 26,463 |
| stochastic oscillator | random timing | 6,318.24 % ± 3.18 | 259.04 % ± 1.09 | 29.68 ± 0.11 | 59.19 % ± 0.00 | 0.93 ± 0.00 | 26,463 |
| stochastic oscillator | big-move | 8,663.86 % ± 3.72 | 409.94 % ± 1.07 | 25.71 ± 0.06 | 60.46 % ± 0.00 | 0.91 ± 0.00 | 29,072 |

Mean over seeds; ± is the standard deviation across seeds (random timing: across its seeds' repetition means). Returns are fee-free (decision U3).
