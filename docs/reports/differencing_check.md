# Price-difference check (paper Step 2)

Validation days 2025-10-17 → 2025-11-30 (45 days), 100 aggTrades per bar. The paper differences
the vwap to (1) remove the intraday trend and (2) remove a "magnitude bias" between trading hours.

## 1. Trend

| measure (per day) | mean | min | max |
|---|---|---|---|
| daily move (%) | -0.3759 | -5.3603 | 3.5627 |
| drift per bar = mean d (USD) | 0.0031 | -0.1664 | 0.2771 |
| std of d (USD) | 13.5437 | 10.6969 | 15.5526 |
| abs(mean d) / std d | 0.0068 | 0.0013 | 0.0181 |
| share of d variance due to drift | 0.0001 | 0.0000 | 0.0003 |
| R² of d on time | 0.0001 | 0.0000 | 0.0007 |
| R² of the level on a straight trend | -0.4754 | -3.6053 | 0.8045 |
| lag-1 autocorr of the level | 0.9997 | 0.9983 | 1.0000 |
| lag-1 autocorr of d | 0.3045 | 0.2068 | 0.4079 |

- The drift that a day's trend leaves in d_t is below 2 % of the standard deviation of d_t on every day
  and explains at most 0.03 % of its variance; d_t has no time trend (R² ≤ 0.0007). The level is
  non-stationary (lag-1 autocorrelation ≈ 1), d_t is not. **Differencing removes the trend as intended.**
- Intraday price paths are not straight-line trends: a line from the day's first to its last price
  explains less than the day's mean on average (negative R²). The "trend" the paper removes is a
  random-walk level, which differencing handles regardless of its shape.
- The remaining lag-1 autocorrelation of d_t (≈ 0.30) is mostly the averaging effect of the vwap
  ([vwap_smoothing_check.md](vwap_smoothing_check.md)), not trend.

## 2. Scale across hours

| UTC hour | bars per day | mean abs(d) / day mean | input prob X1 + X2 | SNN signal rate |
|---|---|---|---|---|
| 0 | 738 | 1.010 | 0.226 | 0.097 |
| 1 | 755 | 1.019 | 0.228 | 0.101 |
| 2 | 710 | 0.991 | 0.224 | 0.095 |
| 3 | 658 | 0.962 | 0.219 | 0.090 |
| 4 | 739 | 0.974 | 0.220 | 0.091 |
| 5 | 631 | 0.923 | 0.213 | 0.082 |
| 6 | 633 | 0.930 | 0.213 | 0.081 |
| 7 | 705 | 0.998 | 0.224 | 0.096 |
| 8 | 673 | 0.950 | 0.217 | 0.085 |
| 9 | 686 | 0.947 | 0.219 | 0.094 |
| 10 | 586 | 0.930 | 0.215 | 0.084 |
| 11 | 648 | 0.942 | 0.217 | 0.088 |
| 12 | 832 | 0.967 | 0.222 | 0.095 |
| 13 | 1,004 | 0.994 | 0.225 | 0.098 |
| 14 | 1,566 | 1.039 | 0.234 | 0.109 |
| 15 | 1,476 | 1.041 | 0.235 | 0.113 |
| 16 | 1,323 | 1.005 | 0.229 | 0.103 |
| 17 | 1,145 | 1.006 | 0.228 | 0.104 |
| 18 | 1,020 | 1.037 | 0.235 | 0.115 |
| 19 | 796 | 1.014 | 0.230 | 0.103 |
| 20 | 790 | 1.024 | 0.232 | 0.107 |
| 21 | 592 | 1.042 | 0.235 | 0.116 |
| 22 | 545 | 1.123 | 0.252 | 0.138 |
| 23 | 582 | 1.021 | 0.233 | 0.103 |

- Busy hours produce up to 2.9× more bars than quiet hours, but each bar's move changes little:
  mean |d| per hour varies by only 1.22× (max / min). With bars of a fixed number of trades, the
  bar clock itself equalises most intraday volatility ("business time").
- Differencing removes level effects: within a day the price level changes by a few percent, so d_t
  in USD changes scale by the same few percent. It does not remove volatility differences between
  hours; those are small here because of the trade-count bars.
- The SNN's signal rate follows the hourly |d| closely (correlation 0.97) and amplifies it
  (1.70× max / min), which is the intended behaviour (it fires on larger moves), not a bias
  from price levels.
- d_t is in USD, not in returns. Over one training day this does not matter; for multi-day training
  windows (E4, W_snn up to 10 days) price-level changes between days (typically a few percent, more in
  volatile weeks) rescale d_t by the same amount relative to the z-score statistics.
