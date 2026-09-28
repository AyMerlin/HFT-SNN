# Design decisions — LOB price-trend project (LIF vs. BRF vs. DeepLOB)

Interpretations of and deviations from the LOB project plan, in the same format as
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md): what the plan says, what was done, and why.
Sections: [B] branch and repository, [A] decisions agreed with the thesis author,
[R] reference material, [L0]… implementation decisions per milestone, [O] open questions.

---

## [B] Branch and repository

**B1 — Branch.** All work is on `feature/lob-snn-benchmark`, created from `main` on
2026-09-28. Branch point (`git merge-base main feature/lob-snn-benchmark`):
`38f473c46d618d3a8eb65881d748079b4b549c77`. No tags on `main`; nothing from this project is
merged into `main`.

**B2 — Changes to shared files** (all additive; existing behavior unchanged):

| File | Change | Why |
|---|---|---|
| `.gitignore` | `data/` → `/data/` | The unanchored rule also matched the package `snn_hft/data/` (see B3) and would have hidden the new `snn_hft/data/lob/`. Approved by the author. |
| `pyproject.toml` | packages `snn_hft*` → `snn_hft*`, `retrieval*`; extra `lob = ["orjson"]` | Installs the retrieval package; faster JSON parsing (optional, identical results). |

No existing module under `snn_hft/` has been modified.

**B3 — `snn_hft/data/` is missing from git.** Because of the old `data/` rule, the package
`snn_hft/data/` (containers, sources, store) was never committed on any branch. On a fresh
clone 11 of 13 existing test modules fail at import (`No module named 'snn_hft.data'`). The
author will push the files; until then the existing suite and the E1 bit-identical
regression check (plan §1.2) cannot run. The new LOB code does not need them:
`snn_hft/data/lob/` works as a namespace portion until `snn_hft/data/__init__.py` arrives.

---

## [A] Decisions agreed with the thesis author (2026-09-28)

**A1 — New branch** `feature/lob-snn-benchmark` as in the plan (not the session's default
branch).

**A2 — `.gitignore` fix** as in B2, and the author pushes the missing `snn_hft/data/`.

**A3 — Network access** opened for Bybit, Tardis, Fairdata, arXiv, Binance data and the
PyTorch wheel index. Still unreachable from this container: `www.bybit.com` (403) and
`datasets.tardis.dev` (Cloudflare "Attention Required" challenge page; see L0-12).

Still open from the pre-implementation review: where training runs (no GPU here), the L2
metric (weighted vs. macro F1), and the tuning budget per horizon — see [O].

---

## [R] Reference material (pinned)

| Reference | Commit | License | Use |
|---|---|---|---|
| DeepLOB (zcakhaa/DeepLOB-…) | `ff14d7c` | none stated | architecture cross-check; its `data/data.zip` is FI-2010 NoAuction DecPre, fold 7 only |
| LOBCAST `main` (mini-LOBCAST) | `fa37c97` | none stated | no DeepLOB on this branch |
| LOBCAST `v0-LOBCAST` | `c2a81d1` | none stated | DeepLOB hyperparameters (Adam lr 0.01, ε 1, batch 32, ≤ 100 epochs) and the paper's published FI-2010 numbers; cross-check only, no code copied |
| BRF (AdaptiveAILab/brf-neurons) | `1a42b8c` | MIT | ground truth for the BRF update; ECG (QTDB) data ships in the repo |
| SeqSNN (microsoft/SeqSNN) | `15cbacb` | MIT | depends on spikingjelly, snntorch, utilsd → re-implement the conv encoder only (plan §7.5 fallback) |

---

## [L0] Retrieval layer and Bybit data

**L0-1 — Bybit source.** Files come from the public directory listing
`https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/`, one zip per UTC day. The depth
changed from 500 to 200 levels on 2025-08-21 (`…_ob500.data.zip` → `…_ob200.data.zip`).
Format details are in [DATA_RETRIEVAL.md](DATA_RETRIEVAL.md).

**L0-2 — Study window.** 2026-06-29 … 2026-09-26: the 90 most recent days published when
the work started (all `ob200`, no missing days in the listing).

**L0-3 — Timestamps.** `ts_ns` = Bybit `ts` (time the message was generated, ms → ns),
which is monotone in the files. The matching-engine time `cts` is not kept (not in the
plan's schema; can be added as an optional column later). `recv_ts_ns` is null for Bybit.

**L0-4 — Sequence check.** Generic rule in `OrderBookBuilder`: a message may state the
update id the previous message must have had (`prev_seq`). Bybit deltas: `u` must equal the
previous `u + 1`; snapshots start a new sequence. Tardis files carry no ids, so no check
(plan §4.2 cannot apply there; re-snapshots mark reconnections).

**L0-5 — Snapshot reconciliation.** Every snapshot that arrives while the book is valid is
compared level by level with the reconstructed book (on the snapshot's depth). Bybit files
end with a snapshot stamped ~0.13 s after the next midnight, so every day gets at least one
full 200-level check. A mismatch downgrades an `ok` day to `flagged`. Sample day
2026-09-26: 861,975 messages, no sequence gap, exact match on all 200 levels per side.

**L0-6 — Rows.** One row per applied message whose timestamp lies inside the UTC day and
leaves a valid book (plan: "after every applied message"). Messages stamped outside the day
are applied for continuity but emit nothing. Consequence: Bybit places each day's boundary
snapshot 0.13–0.73 s after midnight (00:00:00.129 in the September files, 00:00:00.728 in
July), and the deltas stamped between midnight and that snapshot (1–7 messages) sit at the
end of the previous day's file. They are not emitted in either day: at most ~0.7 s and 7
rows of ~860,000 per day. Not worth a cross-file spill-over: the first 100 events of a day
cannot start a model window anyway (windows never cross day boundaries).

**L0-7 — Gap time and day status.** Gap time = time inside the day with an invalid book
(from the last good message to the restoring snapshot) + message-free stretches longer than
60 s while the book is valid (including from the last message to midnight when a file ends
early). Status per plan: ≤ 2 % of the day `ok`, ≤ 5 % `flagged`, else `excluded`; a day with
no valid row is `excluded`; a snapshot mismatch makes an `ok` day `flagged`. The 60 s
threshold is a parameter (`--silence-gap-s`); Bybit publishes every ~100 ms, and the
longest silence in the sample days was 0.6 s.

**L0-8 — Removed rows.** Rows with fewer than N levels on a side, a crossed or locked touch
(`ask_px_1 ≤ bid_px_1`) or a non-positive value are removed and counted per reason; they do
not count as gap time. The schema therefore has no missing levels: every served row has
all N levels.

**L0-9 — Excluded days get no file** (manifest entry and quality row only), since the
framework never serves them. Re-standardizing a day replaces its file, quality row and gaps.

**L0-10 — Raw files.** Zips are stored as downloaded under
`data/raw/lob/bybit/<symbol>/<original file name>` (the plan's `<day>.zip` loses the depth
suffix) with a `.sha256` sidecar; a file counts as downloaded once the sidecar exists, which
makes downloads resumable. Zips are streamed, never unzipped to disk. **Deviation:** the 90
zips take ~16 GB of the ~27 GB free, so `--delete-raw` removes each zip after its day is
written. The sha256 stays in the manifest, and re-downloading a day takes seconds.

**L0-11 — Extra modules.** Besides the plan's file list: `retrieval/http.py` (retries,
atomic writes, sidecars), `retrieval/standardize.py` (day loop, gap accounting, row
cleaning) and `scripts/data/_common.py`. Test files have unique base names because pytest
imports them by base name.

**L0-12 — Tardis at L0.** `TardisL2Parser` exists now for the parser-equivalence test
(plan §11); its downloader and standardize script follow at L7. `datasets.tardis.dev`
answers this container with a Cloudflare challenge page, so the Binance days may have to be
downloaded elsewhere and copied in, or recorded with the optional Binance recorder.

**L0-13 — Parallel standardization.** `--workers N` processes days independently (each
Bybit file opens with a snapshot); workers write their day files and the main process
owns the manifest and CSVs. Frames are identical to the sequential run; the reports differ
only by the opening-snapshot check and the ~0.13 s before it.

**L0-14 — Schema details.** Parquet with zstd; file-level metadata under key
`snn_hft.lob` (venue, symbol, day, source, raw file and sha256, code commit, schema
version, depth). `recv_ts_ns` is a nullable int64. Manifest adds `schema_version`,
`code_git_dirty`, `updated_utc` and per-day `raw_file`, `raw_sha256`, `notes`.

---

## [O] Open questions

1. **Compute** (plan: stop if insufficient). This container has no GPU, 4 CPUs, 15 GB RAM
   and is reclaimed after inactivity. L2 onward needs a GPU machine or a reduced scope.
2. **L2 acceptance metric.** The DeepLOB paper's FI-2010 table has recall = accuracy in every
   row, the signature of support-weighted averaging. Proposal: weighted-F1 vs. the paper,
   macro-F1 vs. LOBCAST.
3. **Tuning budget** once per model family on one reference horizon, or per horizon.
4. To settle at L1/L2 (proposals from the pre-implementation review): paper vs. public
   notebook hyperparameters (paper); FI-2010 Setup 2 primary, Setup 1 optional; horizon
   mapping of the published numbers (to verify against the paper); FI-2010 label definition
   differs from `TrendLabeler`; `RollingDayZScoreNormalizer` reads causal per-day statistics
   (the step interface sees one day); `FoldSpec` gains only `val_days`, LOB experiments use
   a separate chronological split; horizons reported in wall-clock time per venue; the BRF
   cell owns its surrogate (double Gaussian); compact result summaries committed under
   `docs/reports/lob/` because `results/` is not versioned.
