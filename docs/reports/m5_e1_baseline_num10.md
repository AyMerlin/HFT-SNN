# M5 — E1 baseline replication, literal bar size (num = 10)

*Superseded as the main result by [m5_e1_baseline.md](m5_e1_baseline.md) (100 trades per bar,
decision U8). Big-move numbers updated after matching the benchmark to the model's test-day
signal count (I36).*

**Setup.** Paper double-input SNN with pairwise STDP, tuned on the validation period
([m5_tuning_paper_num10.md](m5_tuning_paper_num10.md): threshold 4, leak 0.1, `new_mean` 0.1, A/B = 0.02/−0.021),
walk-forward `W_snn = 1` on the 300 test days (2025-12-01 → 2026-09-26), BTCUSDT `aggTrades`,
`vwap_num = 10`, seeds 0–4, naive benchmark with 100 repetitions per day, big-move benchmark,
latencies 0 / 10 / 100 ms, fee-free (U3). Runtime 21 min on 8 cores. Full tables:
[e1_tables_num10.md](e1_tables_num10.md); per-run files in `results/e1_paper_baseline_num10/`.

```bash
.venv/bin/python -m experiments.run_experiment --config experiments/configs/paper_baseline_num10.yaml
```

## Spikes (Table 3)

| Signals | Spike accuracy | Momentum spike % | Spikes per day |
|---|---|---|---|
| paper SNN, test days | **56.46 %** | 80.42 % | 26,464 |
| paper SNN, training days | 56.50 % | 80.52 % | 25,570 |
| random timing (= chance level) | 53.37 % | 77.98 % | 26,464 |
| big-move benchmark | **65.09 %** | 84.41 % | 29,073 |
| *paper, crude oil (test)* | *65.94 %* | *54.00 %* | |
| *paper, gold (test)* | *58.70 %* | *54.52 %* | |

- The SNN beats the chance level by **3.09 pp** on average and on 299 of 300 days
  (Wilcoxon signed-rank on daily values, p ≈ 6·10⁻⁵¹). Training and test accuracy are equal,
  as in the paper.
- The big-move benchmark — a signal whenever the current price change is large, fired about as
  often as the SNN — is **8.6 pp more accurate than the SNN on every single test day**. Most of what the
  paper's accuracy metric rewards is volatility persistence, which a threshold on `|d_t|` captures
  better than the SNN.
- The chance level is 53.4 %, not the 50 % the paper assumes, and ~78 % of all bars are
  "momentum" bars by the paper's definition (vs ~54 % in the paper's commodity data), because
  consecutive `aggTrades` vwap moves mostly continue.
- Seed variation is negligible (±0.01 pp); health checks passed on all 1,500 test folds; the
  realised input rate was 0.105 per tick per channel (`new_mean` 0.1).

## Strategies (Table 4, paper execution = next bar, fee-free)

| Strategy | Signals | Accumulated return | Sharpe | Win rate | Trades per day |
|---|---|---|---|---|---|
| momentum | paper SNN | 17,418 % | 24.67 | 72.99 % | 26,463 |
| momentum | random timing | 17,213 % | 23.81 | 72.16 % | 26,464 |
| momentum | big-move | 22,135 % | 20.61 | 73.66 % | 29,073 |
| Alexander's filter | paper SNN | 18,721 % | 24.62 | 73.99 % | 26,463 |
| Alexander's filter | random timing | 19,189 % | 23.69 | 73.80 % | 26,463 |
| stochastic oscillator | paper SNN | 17,891 % | 24.73 | 73.38 % | 26,464 |
| stochastic oscillator | random timing | 18,071 % | 23.88 | 72.92 % | 26,464 |
| *paper, crude oil, momentum* | *spike / naive* | *105 % / 77 % (1 month)* | *21.3 / 17.9* | *52.4 % / 51.2 %* | *2,586* |

- Accumulated returns are sums of ~8 million fee-free trades of notional 1 (about 0.2 bp per
  trade on average), so their size says nothing about tradability: a single Binance taker fee is
  about 5 bp. **Random timing earns almost the same**, so the profits come from the direction
  rules plus the persistence of vwap moves, not from the spikes.
- SNN vs random timing, daily P&L (paired over 300 days): with the paper's next-bar execution the
  SNN is ahead for the momentum strategy (+0.007 per day, 222/300 days), behind for Alexander's
  filter (−0.016, 138/300) and level for the stochastic oscillator. With 10 ms or 100 ms latency
  the SNN is ahead for all three strategies (positive on 202–280 of 300 days, p < 10⁻⁸). The
  SNN's Sharpe is about 0.9 higher than random timing's in every configuration, mostly through
  lower volatility.
- The big-move benchmark earns more than the SNN for every strategy and latency (daily P&L,
  p < 10⁻⁴), at a lower Sharpe.
- Latency matters: from next bar to 100 ms latency the win rate falls from ~73 % to ~59 % and the
  accumulated return by ~60 %.

## Replication verdict

The pipeline reproduces the paper's qualitative claims on BTCUSDT — accuracy above the naive
level, spike-based strategies with win rates above 50 % and profit/loss ratios near 1, Sharpe
ratios above 20 without costs — but the controls added here (exact chance level, random timing
with the same count, big-move benchmark, latency) show the SNN's contribution is small: +3 pp of
spike accuracy over chance, and a large deficit to a trivial volatility filter.
