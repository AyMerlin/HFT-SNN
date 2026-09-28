# VWAP smoothing check

Five validation days (2025-11-03 → 11-07) of BTCUSDT `aggTrades`, and a synthetic random walk
with the same number of trades (Gaussian steps, trades at bid or ask, no real momentum), both
aggregated exactly like the pipeline (non-overlapping blocks of `num` trades, volume-weighted).
"ac1" = lag-1 autocorrelation of first differences.

| data | trades per bar | ac1 raw trade diffs | ac1 bar close diffs | ac1 vwap diffs | momentum share, paper label | random-timing win rate, momentum strategy |
|---|---|---|---|---|---|---|
| BTCUSDT aggTrades | 10 | -0.522 | 0.158 | 0.338 | 0.813 | 0.742 |
| random walk + bid-ask bounce | 10 | -0.010 | -0.001 | 0.173 | 0.530 | 0.499 |
| BTCUSDT aggTrades | 100 | -0.522 | 0.234 | 0.290 | 0.573 | 0.523 |
| random walk + bid-ask bounce | 100 | -0.010 | -0.002 | 0.231 | 0.538 | 0.500 |

Working (1960) value for block averages of a random walk: 0.246 (10 points), 0.250 (100 points).

## Reading

- **The implementation matches the paper** (non-overlapping blocks of `num` transactions,
  volume-weighted; the paper's Figure 2 shows 10 raw points per vwap point) **and removes the
  bid-ask zig-zag**: raw trade-to-trade changes have lag-1 autocorrelation −0.52 on BTCUSDT, vwap
  differences are positive.
- **Averaging creates momentum by itself.** On the random walk, the vwap differences have lag-1
  autocorrelation ≈ 0.23 (Working effect), while bar closing prices have none.
- **The paper's momentum label is biased by it.** mom_rev_flag compares the average price before and
  after the spike with the spike's vwap; on a pure random walk it labels about 53–54 % of bars
  "momentum" — the paper's reported momentum spike share (≈ 54 %) is what averaging alone produces.
- **The strategies are not biased by it.** They enter at bar t+1 and exit at t+4, skipping the
  lag-1 dependence: on the random walk the momentum strategy wins 50.0 % at random times. The
  typed-model trade label (entry t+1, exit t+4) is unaffected for the same reason.
- On BTCUSDT at 100 trades per bar some real persistence remains beyond the averaging (vwap 0.29 vs
  0.23; closing prices 0.23 vs 0.0), which gives random timing its 52 % win rate.
