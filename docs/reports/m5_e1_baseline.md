# M5 — E1 baseline replication, 100 trades per bar (main result)

**Setup.** Paper double-input SNN with pairwise STDP, 100 `aggTrades` per vwap bar (decision U8),
tuned on the validation period ([m5_tuning_paper.md](m5_tuning_paper.md): threshold 4, leak 0.2,
`new_mean` 0.05, `new_std` 0.2, STDP scale 1), walk-forward `W_snn = 1` on the 300 test days
(2025-12-01 → 2026-09-26), seeds 0–4, random-timing benchmark with 100 repetitions per day,
big-move benchmark matched to the model's daily signal count, latencies 0 / 10 / 100 / 1000 ms,
fee-free (U3). Full tables: [e1_tables.md](e1_tables.md). The literal 10-trade version is in
[m5_e1_baseline_num10.md](m5_e1_baseline_num10.md).

```bash
.venv/bin/python -m experiments.run_experiment --config experiments/configs/paper_baseline.yaml
```

## Comparison with the paper

| | Paper (crude oil / gold, May 2017) | E1, 100 trades per bar | E1, 10 trades per bar |
|---|---|---|---|
| Chance level (random timing) | assumed 50 % | 51.76 % | 53.37 % |
| Spike accuracy, test | 65.94 % / 58.70 % | **57.39 %** | 56.46 % |
| Spike accuracy, training | 66.07 % / 58.92 % | 60.63 % | 56.50 % |
| Momentum spike share | 54.0 % / 54.5 % | 59.1 % (random: 56.2 %) | 80.4 % (random: 78.0 %) |
| Momentum strategy win rate, spike / naive | 52.42 % / 51.20 % | 51.46 % / 51.26 % | 72.99 % / 72.16 % |
| Profit/loss ratio, spike / naive | 1.022 / 1.013 | 0.991 / 1.002 | 0.82 / 0.91 |
| Sharpe, spike / naive (momentum) | 21.28 / 17.91 | 8.64 / 9.88 | 24.67 / 23.81 |
| Trades per day | 2,586 | 1,418 | 26,464 |

At 100 trades per bar the market regime matches the paper's closely: chance level near 50 %,
random-timing win rate 51.3 % (paper 51.2 %), a momentum share in the mid-50s, profit/loss ratios
near 1 and trade counts of the same order.

## Spikes (Table 3)

| Signals | Spike accuracy | Momentum spike % | Spikes per day |
|---|---|---|---|
| paper SNN, test days | **57.39 %** | 59.11 % | 1,418 |
| paper SNN, training days | 60.63 % | 58.65 % | 850 |
| random timing (= chance level) | 51.76 % | 56.25 % | 1,418 |
| big-move benchmark | **61.28 %** | 63.02 % | 1,606 |

- The SNN beats the chance level by **5.6 pp** on average, on all 300 test days (Wilcoxon on
  daily values, p ≈ 6·10⁻⁵¹). This reproduces the paper's core spike claim in direction, at a
  smaller size than its crude-oil result and close to its gold result.
- The big-move benchmark is **3.9 pp more accurate than the SNN**, on 285 of 300 days.
- The SNN is more accurate on training days (60.6 %) than on test days, because it fires about
  half as often while its weights are still learning (4.4 % vs 8.2 % of bars).

## Strategies (Table 4, fee-free)

Momentum strategy (the other two rules behave the same way; see [e1_tables.md](e1_tables.md)):

| Execution | Signals | Accumulated return | Sharpe | Win rate |
|---|---|---|---|---|
| next bar (paper) | paper SNN | 279 % | 8.64 | 51.46 % |
| next bar (paper) | random timing | 276 % | 9.88 | 51.26 % |
| next bar (paper) | big-move | 454 % | 9.57 | 51.87 % |
| 1 s latency | paper SNN | 44 % | 1.61 | 50.48 % |
| 1 s latency | random timing | 25 % | 1.24 | 50.40 % |
| 1 s latency | big-move | 123 % | 4.03 | 50.65 % |

- **The paper's strategy claim does not reproduce.** Daily P&L of the SNN strategies is not
  distinguishable from random timing for any of the three rules and any latency (paired
  Wilcoxon over 300 days, p = 0.4–1.0; the SNN is ahead on 144–159 of 300 days). In the paper,
  spike-based momentum returned 105 % against 77 % for its naive version.
- The big-move benchmark earns more than the SNN in every configuration (p ≤ 3·10⁻⁴).
- Per trade the SNN earns about 0.07 bp before costs (paper ≈ 0.18 bp), far below a taker fee.

## Paper-literal normalisation (sensitivity)

The paper normalises with the whole day's mean and standard deviation, which uses future bars.
Re-running E1 with that setting ([e1_tables_sameday.md](e1_tables_sameday.md),
[e1_tables_num10_sameday.md](e1_tables_num10_sameday.md)) changes spike accuracy by only
+0.2 pp (57.61 %) at 100 trades per bar and +0.1 pp at 10; the look-ahead does not explain the
gap to the paper's 66 %.

## Verdict

On BTCUSDT, in a regime comparable to the paper's, the reimplementation reproduces the paper's
spike-level finding (accuracy clearly above chance, momentum spikes over-represented) but not its
trading finding (spike timing beats random timing). A simple threshold on the current price change
is both more accurate and more profitable than the SNN. The improved model (M8) is therefore
compared against three references: the paper SNN, random timing and the big-move benchmark.
