# M4 spike check — paper SNN on validation days

Purpose: show that `PaperSNNSignalModel` runs on real data, passes the health checks and
produces Table-3-style output. **Not a result**: hyperparameters are not tuned yet (M5);
the threshold below was picked only to get the output rate into the health band.

Setup: BTCUSDT `aggTrades`, `vwap_num = 10`, `T = 10`, walk-forward `W_snn = 1` on the first
15 validation days (2025-10-17 → 2025-10-31), seeds 0–2, §12 defaults except LIF threshold 16
(the default threshold 1 makes the output fire in 96 % of bars).

```bash
.venv/bin/python -m experiments.spike_report --config experiments/configs/paper_baseline.yaml \
  --set backtest.split=validation --set 'backtest.seeds=[0,1,2]' \
  --set models.0.core.lif.threshold=16 --set name=m4_spike_check --max-folds 15
```

| Split | Days | Spike accuracy | Chance level | Momentum spike % | Momentum base % | Spikes per day | Signal rate | Health warnings |
|---|---|---|---|---|---|---|---|---|
| train | 15 | 53.70 % | 53.39 % | 80.39 % | 80.50 % | 30,089 | 17.2 % | 0 |
| test | 15 | 53.82 % | 53.48 % | 80.34 % | 80.57 % | 29,425 | 17.4 % | 0 |

*Chance level* is the share of all evaluable bars that are "real" spikes, i.e. the expected
accuracy of signals at random times; *momentum base* is the same for the momentum label.

Observations:
- Test accuracy exceeds the chance level on every day, by 0.35 percentage points on average
  (standard deviation 0.40 pp across days, largest 1.4 pp). Seeds change daily accuracy by
  about 0.26 pp.
- About 80 % of all bars are "momentum" bars under the paper's definition, for the model and for
  random timing alike, because consecutive `aggTrades` vwap moves mostly continue.
- The chance level is about 53 %, not the 50 % the paper assumes.
- The signal rate is 17.0–17.9 % on every day regardless of activity, which suggests the output
  neuron fires at a rate set mostly by the network's integration dynamics rather than by the input.
