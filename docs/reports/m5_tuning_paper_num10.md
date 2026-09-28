# M5 tuning — paper SNN, literal bar size (num = 10)

Protocol: [DESIGN_DECISIONS I39](../DESIGN_DECISIONS.md). 40 distinct points of the shared grid
(threshold × leak × `new_mean` × STDP scale), walk-forward `W_snn = 1` on all 45 validation days
(2025-10-17 → 2025-11-30), seeds 0 and 1 (90 folds per trial). Objective: mean test-day spike
accuracy; admissible if at most 5 % of the days fail a health check. Chance level (random
timing) on these days: 52.45 %.

```bash
.venv/bin/python -m experiments.tune --config experiments/configs/paper_baseline_num10.yaml \
  --trials 40 --seeds 0 1 --out experiments/configs/tuned/paper_snn_num10.yaml
```

**Chosen: t009** — threshold 4, leak 0.1 per tick, `new_mean` 0.1, STDP scale 2
(A = 0.02, B = −0.021). Validation accuracy 54.61 % (+2.17 pp over chance),
signal rate 16.6 %.

Notes:
- The two most accurate trials (t037, t038) almost never fire (signal rate ≈ 0–1 %) and fail the
  health check on 98 % and 69 % of the days; the admissibility rule excludes them.
- `new_mean = 0.1` (the lowest input rate in the grid) is in every one of the top admissible trials.
- Every admissible trial beats the chance level; the untuned defaults of M4 reached about +0.35 pp.

| Trial | Threshold | Leak | new_mean | STDP scale | Accuracy | Edge over chance | Signal rate | Unhealthy days | Admissible |
|---|---|---|---|---|---|---|---|---|---|
| t009 | 4.0 | 0.1 | 0.1 | 2.0 | 54.61 % | +2.17 pp | 16.6 % | 0 % | yes |
| t008 | 4.0 | 0.1 | 0.1 | 1.0 | 54.59 % | +2.15 pp | 16.8 % | 0 % | yes |
| t033 | 32.0 | 0.05 | 0.1 | 0.5 | 53.70 % | +1.25 pp | 2.8 % | 0 % | yes |
| t003 | 4.0 | 0.05 | 0.1 | 0.5 | 53.54 % | +1.09 pp | 27.7 % | 0 % | yes |
| t005 | 4.0 | 0.05 | 0.1 | 2.0 | 53.54 % | +1.09 pp | 27.4 % | 0 % | yes |
| t004 | 4.0 | 0.05 | 0.1 | 1.0 | 53.54 % | +1.09 pp | 27.6 % | 0 % | yes |
| t015 | 8.0 | 0.05 | 0.1 | 0.5 | 53.51 % | +1.06 pp | 13.9 % | 0 % | yes |
| t016 | 8.0 | 0.05 | 0.1 | 1.0 | 53.50 % | +1.05 pp | 13.8 % | 0 % | yes |
| t027 | 32.0 | 0.02 | 0.1 | 2.0 | 53.41 % | +0.96 pp | 5.1 % | 0 % | yes |
| t026 | 32.0 | 0.02 | 0.1 | 1.0 | 53.36 % | +0.91 pp | 5.1 % | 0 % | yes |
| t013 | 8.0 | 0.02 | 0.1 | 0.5 | 53.35 % | +0.90 pp | 19.3 % | 0 % | yes |
| t021 | 16.0 | 0.02 | 0.1 | 1.0 | 53.30 % | +0.85 pp | 10.0 % | 0 % | yes |
| t025 | 32.0 | 0.02 | 0.1 | 0.5 | 53.24 % | +0.80 pp | 5.1 % | 0 % | yes |
| t034 | 32.0 | 0.05 | 0.1 | 2.0 | 53.17 % | +0.72 pp | 3.3 % | 0 % | yes |
| t020 | 8.0 | 0.1 | 0.2 | 2.0 | 52.90 % | +0.45 pp | 23.3 % | 0 % | yes |
| t010 | 4.0 | 0.1 | 0.2 | 1.0 | 52.88 % | +0.43 pp | 45.1 % | 0 % | yes |
| t011 | 4.0 | 0.1 | 0.2 | 2.0 | 52.87 % | +0.42 pp | 44.5 % | 0 % | yes |
| t022 | 16.0 | 0.05 | 0.2 | 0.5 | 52.75 % | +0.30 pp | 17.5 % | 0 % | yes |
| t017 | 8.0 | 0.05 | 0.2 | 0.5 | 52.75 % | +0.30 pp | 32.6 % | 0 % | yes |
| t023 | 16.0 | 0.05 | 0.2 | 1.0 | 52.74 % | +0.29 pp | 17.4 % | 0 % | yes |
| t035 | 32.0 | 0.05 | 0.2 | 0.5 | 52.70 % | +0.25 pp | 9.1 % | 0 % | yes |
| t028 | 32.0 | 0.02 | 0.2 | 0.5 | 52.69 % | +0.25 pp | 10.7 % | 0 % | yes |
| t029 | 32.0 | 0.02 | 0.2 | 2.0 | 52.61 % | +0.17 pp | 10.5 % | 0 % | yes |
| t019 | 8.0 | 0.05 | 0.3 | 2.0 | 52.59 % | +0.15 pp | 47.8 % | 0 % | yes |
| t018 | 8.0 | 0.05 | 0.3 | 0.5 | 52.59 % | +0.15 pp | 48.8 % | 0 % | yes |
| t030 | 32.0 | 0.02 | 0.3 | 0.5 | 52.59 % | +0.14 pp | 16.2 % | 0 % | yes |
| t032 | 32.0 | 0.02 | 0.3 | 2.0 | 52.55 % | +0.10 pp | 15.8 % | 0 % | yes |
| t024 | 16.0 | 0.05 | 0.3 | 2.0 | 52.54 % | +0.09 pp | 26.8 % | 0 % | yes |
| t039 | 32.0 | 0.1 | 0.3 | 0.5 | 52.54 % | +0.09 pp | 11.9 % | 0 % | yes |
| t031 | 32.0 | 0.02 | 0.3 | 1.0 | 52.51 % | +0.06 pp | 16.0 % | 0 % | yes |
| t037 | 32.0 | 0.1 | 0.1 | 2.0 | 82.00 % | +29.55 pp | 0.0 % | 98 % | no |
| t038 | 32.0 | 0.1 | 0.2 | 0.5 | 71.32 % | +18.88 pp | 1.1 % | 69 % | no |
| t006 | 4.0 | 0.05 | 0.2 | 1.0 | 52.71 % | +0.26 pp | 56.6 % | 100 % | no |
| t000 | 4.0 | 0.02 | 0.2 | 1.0 | 52.65 % | +0.20 pp | 60.7 % | 100 % | no |
| t001 | 4.0 | 0.02 | 0.2 | 2.0 | 52.65 % | +0.20 pp | 60.6 % | 100 % | no |
| t012 | 4.0 | 0.1 | 0.3 | 1.0 | 52.62 % | +0.17 pp | 69.4 % | 100 % | no |
| t007 | 4.0 | 0.05 | 0.3 | 1.0 | 52.58 % | +0.13 pp | 76.5 % | 100 % | no |
| t002 | 4.0 | 0.02 | 0.3 | 2.0 | 52.57 % | +0.12 pp | 77.8 % | 100 % | no |
| t014 | 8.0 | 0.02 | 0.3 | 1.0 | 52.54 % | +0.10 pp | 52.4 % | 100 % | no |
| t036 | 32.0 | 0.1 | 0.1 | 0.5 | nan % | +nan pp | 0.0 % | 100 % | no |
