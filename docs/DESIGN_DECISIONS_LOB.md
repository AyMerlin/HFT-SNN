# Design decisions — FI-2010 project: DeepLOB replication, then LIF and BRF spiking networks

Interpretations of and deviations from the LOB project plan, in the same format as
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md): what the plan says, what was done, and why.
Sections: [S] scope, [B] branch and repository, [A] decisions agreed with the thesis
author, [R] reference material, [P] replication specification, [F…] implementation
decisions per milestone, [O] open questions.

---

## [S] Scope (reduced 2026-09-28)

The project runs on **FI-2010 only**, in two phases:

1. **Replicate DeepLOB** (Zhang, Zohren & Roberts, IEEE TSP 2019) on FI-2010, following the
   paper's own evaluation setup, and compare with the published numbers and LOBCAST's
   reproduction. The replicated code is then frozen.
2. **Compare spiking networks against it on the same data**: one recurrent spiking network
   in which only the neuron type differs (LIF vs. BRF), plus a non-spiking recurrent twin,
   on the identical FI-2010 splits and horizons.

Kept from the original plan: RQ1 (replication), RQ2 (simple baselines: majority class and
logistic regression on the FI-2010 features), RQ3 (spiking vs. non-spiking twin vs.
DeepLOB), RQ4 (LIF vs. BRF), RQ5 (accuracy vs. operation count / energy proxy); plan §6–§8.2
as far as they apply to FI-2010; ablations A1 (input encoding), A4 (1 vs. 2 spiking
layers), A5 (window length T).

Removed: all Bybit/BTC work (retrieval, standardization, BTC baselines `DeepLOB-paper` /
`DeepLOB-tuned` on BTC, BTC preprocessing), the cross-venue test (RQ6), the trading
evaluation (RQ7), and ablations A2 (label threshold) and A3 (price representation), since
FI-2010 ships fixed labels and pre-normalized features. The removed code is in the branch
history up to commit `b55d3bc`.

### Milestones

| M | Deliverable | Accept when |
|---|---|---|
| F1 | FI-2010 download and conversion; data source; window dataset; classification metrics; trainer | Shapes, splits and label counts match the DeepLOB paper's setup (and LOBCAST's); tests pass |
| F2 | DeepLOB replication on FI-2010, plus majority and logistic baselines | Within ~2 points of the paper's published values per reported horizon (metric: see O2); code frozen, commit recorded → **check-in** |
| F3 | `SpikingCell`, `LIFCell`, `BRFCell`; BRF validation | Plan §7.3 gates: unit tests against the reference code and one paper benchmark (ECG) reproduced |
| F4 | `SpikingLOBNet` (LIF, BRF) and `RecurrentTwinModel` on FI-2010; equal tuning budget; 5 seeds | Results for all models on the identical test set → **check-in** |
| F5 | Efficiency analysis (operation counts, energy proxy, latency); ablations A1, A4, A5; report | Report produced from saved results; branch pushed; `main` unchanged |

---

## [B] Branch and repository

**B1 — Branch.** All work is on `feature/lob-snn-benchmark`, created from `main` on
2026-09-28. Branch point (`git merge-base main feature/lob-snn-benchmark`):
`38f473c46d618d3a8eb65881d748079b4b549c77`. No tags on `main`; nothing from this project is
merged into `main`.

**B2 — Changes to shared files** (all additive; existing behavior unchanged):

| File | Change | Why |
|---|---|---|
| `.gitignore` | `data/` → `/data/` | The unanchored rule also matched the package `snn_hft/data/` (see B3). Approved by the author. |
| `pyproject.toml` | packages `snn_hft*` → `snn_hft*`, `retrieval*` | Installs the retrieval package. |

No existing module under `snn_hft/` has been modified.

**B3 — `snn_hft/data/` is missing from git.** Because of the old `data/` rule, the package
`snn_hft/data/` (containers, sources, store) was never committed on any branch. On a fresh
clone 11 of 13 existing test modules fail at import (`No module named 'snn_hft.data'`). The
author will push the files; until then the existing suite and the E1 bit-identical
regression check (plan §1.2) cannot run.

---

## [A] Decisions agreed with the thesis author (2026-09-28)

**A1 — New branch** `feature/lob-snn-benchmark` as in the plan.

**A2 — `.gitignore` fix** as in B2; the author pushes the missing `snn_hft/data/`.

**A3 — Network access** opened for the data and reference hosts (Fairdata, arXiv, PyTorch
wheel index, among others).

**A4 — Scope reduced to FI-2010** as in [S]; all BTC-only code, tests and docs deleted.

**A5 — Neurons:** LIF and BRF (balanced resonate-and-fire), as in the plan.

**A6 — Identical replication** of the paper's architecture (§IV) and of both FI-2010
setups (§V-B), per the author's instruction; specification in [P].

---

## [R] Reference material (pinned)

| Reference | Commit | License | Use |
|---|---|---|---|
| DeepLOB (zcakhaa/DeepLOB-…) | `ff14d7c` | none stated | architecture cross-check; its `data/data.zip` is FI-2010 NoAuction DecPre, fold 7 only |
| LOBCAST `main` (mini-LOBCAST) | `fa37c97` | none stated | FI-2010 loading; no DeepLOB on this branch |
| LOBCAST `v0-LOBCAST` | `c2a81d1` | none stated | DeepLOB hyperparameters (Adam lr 0.01, ε 1, batch 32, ≤ 100 epochs) and the paper's published FI-2010 numbers; cross-check only, no code copied |
| BRF (AdaptiveAILab/brf-neurons) | `1a42b8c` | MIT | ground truth for the BRF update; ECG (QTDB) data ships in the repo |
| SeqSNN (microsoft/SeqSNN) | `15cbacb` | MIT | depends on spikingjelly, snntorch, utilsd → re-implement the conv encoder only (plan §7.5 fallback) |

---

## [P] Replication specification (DeepLOB paper, arXiv 1808.03668v6)

The author asked for an identical replication of the architecture of §IV and of both
FI-2010 setups of §V-B. The paper is the specification; the authors' notebooks and LOBCAST
are used only where the paper is silent. "Paper" below cites the text or Figures 3–4.

**P1 — Architecture (Fig. 3, 4, §IV-B).** Input 100 × 40 (100 most recent LOB states;
per level {ask price, ask volume, bid price, bid volume}, levels 1–10).
Convolution block, 16 filters per layer, Leaky-ReLU (slope 0.01) after every convolution,
zero padding in time so the time length stays 100:
`1×2 stride 1×2 → 4×1 → 4×1 → 1×2 stride 1×2 → 4×1 → 4×1 → 1×10 → 4×1 → 4×1`
(feature maps 100 × 20 → 100 × 10 → 100 × 1).
Inception@32: branches `1×1@32 → 3×1@32`, `1×1@32 → 5×1@32`, `max-pool 3×1 (stride 1,
zero padding) → 1×1@32`, concatenated (96 channels), Leaky-ReLU after every convolution.
LSTM with 64 units over the 100 time steps; its last output feeds a 3-unit softmax layer.
No batch normalization, no dropout (neither appears in the paper). About 60,000 parameters
(paper §IV-B c); this configuration has ~60.7k (Keras counting).

The authors' public code differs from the paper and is **not** used: Keras notebooks
(32 filters, Inception@64, dropout 0.2 before the LSTM, lr 1e-4; 142k parameters); PyTorch
notebook and LOBCAST v0 (32/64 filters, batch normalization, tanh in the second block, no
time padding; 143,907 parameters).

**P2 — Details the paper leaves open, taken from Keras (the paper's framework, §V-A):**
- padding for even kernels (4×1) as Keras/TensorFlow `same`: one zero row before, two after;
- initialization: Glorot-uniform kernels, zero biases, orthogonal LSTM recurrent kernel,
  LSTM forget-gate bias 1 (Keras `unit_forget_bias`), one bias vector per LSTM gate;
- LSTM gate activation: sigmoid (the authors' GPU code uses CuDNN LSTM, which has no
  hard-sigmoid);
- Adam with ε = 1 in the Keras/TensorFlow form: the update is
  `lr·√(1−β₂ᵗ)/(1−β₁ᵗ) · m / (√v + ε)`, i.e. ε is "ε̂" of Kingma & Ba. PyTorch's `Adam`
  puts ε on the bias-corrected √v̂, which with ε = 1 is a materially different optimizer,
  so a Keras-form Adam is implemented;
- training samples are reshuffled every epoch (Keras `fit` default).

**P3 — Training (§V-A).** Categorical cross-entropy; Adam, learning rate 0.01, ε = 1
(β₁ = 0.9, β₂ = 0.999, Keras defaults); mini-batches of 32; training stops when validation
accuracy has not improved for 20 epochs (~100 epochs on FI-2010). Open in the paper: which
weights are evaluated. Primary: the weights with the best validation accuracy; the
last-epoch weights are evaluated too and reported as a sensitivity. Safety cap: 300 epochs.

**P4 — Data (§III-C).** FI-2010 z-score normalized, labels used as provided (Eq. 3 of the
paper, threshold fixed by the dataset). The paper does not say Auction or NoAuction; the
NoAuction version is used, as in the authors' notebook and LOBCAST. Label rows 1–5 of the
dataset are horizons k = 10, 20, 30, 50, 100 events; label values 1 = up, 2 = stationary,
3 = down (verified: mean future change of the best ask is positive for 1, negative for 3).
Each input window is the 100 rows ending at the labelled row (authors' code).

**P5 — Setup 1 (§V-B).** Nine anchored folds: fold i trains on days 1…i and tests on day
i+1 (the dataset's `Train_…_CF_i` / `Test_…_CF_i` files). Horizons k = 10, 50, 100
(Table I). Reported: accuracy, precision, recall, F1, each the mean over the 9 folds.

**P6 — Setup 2 (§V-B).** Train on days 1–7 (`Train_…_CF_7`), test on days 8–10
(`Test_…_CF_7`, `CF_8`, `CF_9` concatenated, as in the authors' code). Horizons k = 10, 20,
50 (Table II).

**P7 — Validation split (not in the paper).** The last 20 % of each training file, as in
the authors' notebook and LOBCAST (`TRAIN_SPLIT_VAL = 0.8`). Because FI-2010 training files
are ordered stock by stock (see F1), this validation set is mostly the last stock.

**P8 — Metrics.** Precision, recall and F1 are support-weighted averages over the three
classes: in all six DeepLOB rows of Tables I–II recall equals accuracy, which only weighted
averaging produces. Macro averages are reported as well (LOBCAST reports macro-F1).
Target values (DeepLOB rows):

| Setup | k | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 10 | 78.91 | 78.47 | 78.91 | 77.66 |
| 1 | 50 | 75.01 | 75.10 | 75.01 | 74.96 |
| 1 | 100 | 76.66 | 76.77 | 76.66 | 76.58 |
| 2 | 10 | 84.47 | 84.00 | 84.47 | 83.40 |
| 2 | 20 | 74.85 | 74.06 | 74.85 | 72.82 |
| 2 | 50 | 80.51 | 80.38 | 80.51 | 80.35 |

The paper reports single runs; we report mean ± std over 5 seeds.

---

## [F0] Scope reduction (cleanup)

**F0-1 — Kept after removing the BTC work:** `retrieval/http.py` (resumable download with
retries and a sha256 sidecar; generic, reused for FI-2010), the dependency-direction test
(`tests/lob/test_architecture.py`; the retrieval layer may import no framework module
until the FI-2010 format module exists), the `scripts/data/` package, and the `.gitignore`
fix.

**F0-2 — Environment.** Python 3.11 virtual environment; PyTorch 2.14.0 CPU build and
scikit-learn 1.9 (extra `lob` in `pyproject.toml`); no GPU in this container (cost: F1-8).

---

## [F1] FI-2010 data

**F1-1 — Source.** The official archive `BenchmarkDatasets.zip` (1.86 GB, CC BY 4.0) from
Fairdata dataset `73eb48d7-4dbc-4a10-a52a-da745b47a649`: Etsin issues the download URL
(`POST /api/download/authorize`), and the file is verified against the sha256 published in
Metax (`cea93692…f664`). `python -m scripts.data.fi2010_prepare` downloads, verifies and
converts in ~4 minutes. The DeepLOB repository's `data.zip` (DecPre, fold 7 only) is not
used.

**F1-2 — What the files contain (verified on the archive).**
- 9 folds × (training, test) files per variant; `Train_CF_i` holds days 1…i, `Test_CF_i`
  day i+1. 394,337 samples over 10 days.
- Each fold is z-scored with the statistics of its own training days (the LOB rows of
  `Train_CF_7` have mean 0 and std 1), so the same day has different values in different
  folds; verified to be an exact affine re-normalization (residual ≤ 1.3e-6, print
  precision).
- Samples are ordered **stock by stock**: `Train_CF_i` = stock 1 days 1…i, then stock 2
  days 1…i, …; test files hold one day for the 5 stocks. Stock boundaries are found from
  simultaneous jumps of all 20 price rows (4th-largest jump ≥ 19× the 5th on every day)
  and verified: the labels of every training file equal the stock-by-stock concatenation of
  the day files exactly.
- Label values 1 = up, 2 = stationary, 3 = down (mean future change of the best ask is
  +0.0032 / +0.0008 / −0.0009 z-units for 1 / 2 / 3 at k = 100 on day 1).

Samples per day and stock:

| day | stock 1 | stock 2 | stock 3 | stock 4 | stock 5 | total |
|---|---|---|---|---|---|---|
| 1 | 3,454 | 6,318 | 4,922 | 10,719 | 14,099 | 39,512 |
| 2 | 5,079 | 6,122 | 5,965 | 6,712 | 14,519 | 38,397 |
| 3 | 3,903 | 5,411 | 4,027 | 5,186 | 10,008 | 28,535 |
| 4 | 2,806 | 6,992 | 5,342 | 10,819 | 11,064 | 37,023 |
| 5 | 3,030 | 6,728 | 5,946 | 6,378 | 12,703 | 34,785 |
| 6 | 2,263 | 6,243 | 5,607 | 7,007 | 18,032 | 39,152 |
| 7 | 2,801 | 7,060 | 6,740 | 7,854 | 12,891 | 37,346 |
| 8 | 2,647 | 8,662 | 8,591 | 13,229 | 22,349 | 55,478 |
| 9 | 1,873 | 9,271 | 10,036 | 12,880 | 18,112 | 52,172 |
| 10 | 1,888 | 5,128 | 5,722 | 5,821 | 13,378 | 31,937 |

**F1-3 — Converted format** (`snn_hft/data/fi2010/format.py`, the only framework module the
retrieval layer imports). Per file three arrays with one row per sample: LOB (N, 40) and
hand-crafted features (N, 104) as float32, labels (N, 5) as int8 with the dataset's values;
`manifest.json` lists each file's (stock, day) segments, label counts per horizon, the
archive checksum, the code commit and the verification results. float32 keeps 7 significant
digits of the ~8 printed in the text files. Only NoAuction/Zscore is converted by default
(`--auction`, `--normalization` select other variants).

**F1-4 — Splits** (`snn_hft/data/fi2010/source.py`). Setup 2 gives 203,800 / 50,950 /
139,587 training / validation / test samples, exactly as in the authors' notebook. Setup 1
fold i uses `train_cf<i>` (80 / 20) and `test_cf<i>`. Consequences of following the
authors' conventions, kept for fidelity:
- the validation part is the tail of the stock-by-stock file, i.e. almost only stock 5
  (Setup 2: stock 5, days 4–7);
- the Setup 2 test days 9 and 10 come from folds 8 and 9, so they are z-scored with
  statistics that include days 8 (and 9), which are themselves test days. Only feature
  statistics are affected, not labels.

Setup 2 class counts (up / stationary / down):

| k | train | validation | test |
|---|---|---|---|
| 10 | 40,957 / 121,924 / 40,919 | 9,305 / 32,283 / 9,362 | 21,167 / 98,638 / 19,782 |
| 20 | 52,535 / 99,044 / 52,221 | 12,074 / 26,818 / 12,058 | 27,470 / 86,618 / 25,499 |
| 50 | 71,482 / 61,827 / 70,491 | 16,123 / 18,847 / 15,980 | 38,467 / 66,007 / 35,113 |

**F1-5 — Input windows** (`snn_hft/training/dataset.py`). Sample i is the 100 rows ending at
a labelled row; every row from the 100th on is a sample, and windows may span a stock or
day change, exactly as in the authors' code (Setup 2 training: 203,701 windows, of which
~1.6 % span a change). An option restricts windows to one (stock, day) segment; it is not
used for the replication. Windows are gathered per batch, not stored.

**F1-6 — Training loop** (`snn_hft/training/trainer.py`, `optim.py`).
- Keras-form Adam (P2), cross-entropy on logits (Keras applies it to softmax outputs
  clipped at 1e-7; numerically the same away from saturation).
- Samples reshuffled every epoch from a seeded generator; the last batch may be smaller.
- Early stopping as Keras `EarlyStopping`: an epoch improves only if validation accuracy is
  strictly higher than the best; stop after 20 epochs without improvement; best and last
  weights kept.
- The complete state is checkpointed after every epoch and `fit` resumes from it; a test
  shows that an interrupted and resumed run is identical to an uninterrupted one.

**F1-7 — Metrics** (`snn_hft/evaluation/classification.py`): accuracy, weighted and macro
precision/recall/F1, Matthews' correlation and the confusion matrix, checked against
scikit-learn; Setup 1 averages each metric over the 9 folds without weights.

**F1-8 — Measured cost of the paper-sized DeepLOB on this container** (4 CPUs, no GPU,
PyTorch 2.14 CPU): 24.8 ms per training step of 32 with 4 threads, i.e. 2.6 min per Setup 2
epoch plus ~0.6 min validation. Four single-thread runs in parallel reach ~23 % more total
throughput (~81 ms per step each). With ~120 epochs per run (the paper's ~100 plus the
patience of 20), one Setup 2 run is ~4 machine-hours; Setup 2 (3 horizons × 5 seeds) is
~3 days; Setup 1 (9 folds × 3 horizons, smaller training sets) is ~4 days per seed.

---

## [O] Open questions

1. **Compute** (plan: stop if insufficient). See F1-8: the full grid (Setup 2 × 5 seeds,
   Setup 1 × 5 seeds) is ~3 weeks of this container's CPU, and the container is reclaimed
   after inactivity (checkpoints live on its disk).
2. **F2 acceptance metric:** compare like with like — support-weighted P/R/F1 (see P8)
   against the paper, macro-F1 against LOBCAST.
3. **Tuning budget** for LIF, BRF and the twin: once per model family on one reference
   horizon, or per horizon.
