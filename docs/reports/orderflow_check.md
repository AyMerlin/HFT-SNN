# Appendix — does order flow make the trade outcome predictable? (U11)

Validation days 2025-10-17 → 2025-11-30, 100 aggTrades per bar, 891,546 bars with a decided trade label.
Order-flow imbalance (OFI) of a bar = (taker-buy − taker-sell volume) / bar volume; count imbalance
= the same for trade counts. All features are oriented by the trade direction dir_t (positive = in the
direction the momentum rule would trade). Fit on the first 22 days, evaluated on the rest.

| features | AUC fit half | AUC held-out half |
|---|---|---|
| price features | 0.515 | 0.512 |
| order flow only | 0.548 | 0.538 |
| price + order flow | 0.549 | 0.539 |

Single order-flow features:

| feature | AUC (all days) |
|---|---|
| ofi_0 | 0.541 |
| ofi_1 | 0.507 |
| ofi_2 | 0.501 |
| ofi_3 | 0.502 |
| ofi_4 | 0.503 |
| ofi_sum5 | 0.518 |
| ofi_sum20 | 0.510 |
| count_imb_0 | 0.540 |

Value of typing with price + order flow (fee-free, paper execution):

| momentum strategy, held-out validation bars | bp per trade | trades faded |
|---|---|---|
| always follow | +0.087 |  |
| fade when classifier says reversion (p < 0.5) | +0.203 | 29% |
| fade the 10 % most reversion-like bars | +0.144 | 10 % |
| oracle (not causal) | +2.524 |  |
