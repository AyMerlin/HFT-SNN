# Design decisions — FI-2010 project: DeepLOB replication, then LIF and BRF spiking networks

Interpretations of and deviations from the LOB project plan, in the same format as
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md): what the plan says, what was done, and why.
Sections: [S] scope, [B] branch and repository, [A] decisions agreed with the thesis
author, [R] reference material, [F…] implementation decisions per milestone, [O] open
questions.

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

## [F0] Scope reduction (cleanup)

**F0-1 — Kept after removing the BTC work:** `retrieval/http.py` (resumable download with
retries and a sha256 sidecar; generic, reused for FI-2010), the dependency-direction test
(`tests/lob/test_architecture.py`; the retrieval layer may import no framework module
until the FI-2010 format module exists), the `scripts/data/` package, and the `.gitignore`
fix.

**F0-2 — Environment.** Python 3.11 virtual environment; PyTorch 2.14.0 CPU build (no GPU
in this container). Measured DeepLOB training cost on the 4 CPUs: ~1.8 ms per sample at
batch 32, i.e. ~6 min per epoch on FI-2010's ~204k training windows.

---

## [O] Open questions

1. **Compute** (plan: stop if insufficient). No GPU here; one DeepLOB run with early
   stopping is a few CPU hours, so the replication (3 horizons × 5 seeds) is tens of CPU
   hours. The spiking models (BPTT over 100 steps) and tuning add more.
2. **F2 acceptance metric.** The DeepLOB paper's FI-2010 table has recall = accuracy in every
   row, the signature of support-weighted averaging. Proposal: weighted-F1 vs. the paper,
   macro-F1 vs. LOBCAST.
3. **Tuning budget** for LIF, BRF and the twin: once per model family on one reference
   horizon, or per horizon.
4. To settle in F1 from the paper: which of the paper's two FI-2010 setups is primary
   (proposal: Setup 2, first 7 days train / last 3 test), which normalization (the public
   notebook uses DecPre, LOBCAST Z-score), the mapping of the published horizons to FI-2010's
   label rows, and the training details (paper vs. public notebook).
