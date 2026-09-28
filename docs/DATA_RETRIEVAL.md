# Data retrieval (LOB project)

The retrieval layer (`retrieval/`, scripts in `scripts/data/`) downloads venue order-book
files, reconstructs the book, checks quality and writes standardized day files. The
framework (`snn_hft/`) only reads those files; it never downloads or reconstructs anything
(plan §3.1). Decisions are logged in [DESIGN_DECISIONS_LOB.md](DESIGN_DECISIONS_LOB.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,lob]"
```

## Bybit BTCUSDT perpetual (primary dataset)

```bash
# what is available
.venv/bin/python -m scripts.data.bybit_download --list

# 1. download raw day files (resumable; completed files are skipped)
.venv/bin/python -m scripts.data.bybit_download --symbol BTCUSDT --start 2026-06-29 --end 2026-09-26

# 2. reconstruct, quality-check and write standardized days
.venv/bin/python -m scripts.data.bybit_standardize --symbol BTCUSDT --start 2026-06-29 --end 2026-09-26 \
    --depth 20 --workers 4 --delete-raw

# 3. quality report (Table 0)
.venv/bin/python -m scripts.data.lob_quality_report data/standardized/lob/bybit/BTCUSDT \
    --out docs/reports/lob/l0_bybit_quality.md
```

Outputs (not committed; `data/` is ignored):

```
data/raw/lob/bybit/BTCUSDT/<day>_BTCUSDT_ob200.data.zip (+ .sha256)   # removed by --delete-raw
data/standardized/lob/bybit/BTCUSDT/<day>.parquet
data/standardized/lob/bybit/BTCUSDT/manifest.json
data/standardized/lob/bybit/BTCUSDT/quality_report.csv
data/standardized/lob/bybit/BTCUSDT/gaps.csv
```

Useful options: `--skip-existing` (only days missing from the manifest), `--workers 1`
(days in order with one book carried across midnight), `--silence-gap-s` (default 60).

### Bybit file format (verified 2026-09-28 on 2026-07-10, 2026-09-25 and 2026-09-26)

| Item | Finding |
|---|---|
| Download | Public directory listing `https://quote-saver.bycsi.com/orderbook/linear/<SYMBOL>/`; plain HTTPS GET per file, no key, no 7-day form limit. Listing starts 2023-01-18 with no missing days. |
| File | `<YYYY-MM-DD>_<SYMBOL>_ob<depth>.data.zip` holding one JSON-lines file `<…>.data` (~640 MB unzipped, 85–200 MB zipped for recent days). |
| Depth | `ob500` until 2025-08-20, `ob200` from 2025-08-21. The book never exceeds the stated depth: levels leaving the window arrive as deletes. |
| Messages | `{"topic":"orderbook.200.BTCUSDT","type":"snapshot"|"delta","ts":<ms>,"cts":<ms>,"data":{"s","b":[[price,size],…],"a":[…],"u","seq"}}`; prices and sizes are strings; size `"0"` deletes a level. |
| Update frequency | Batched every ~100 ms (≈ 862,000 messages per day). |
| Sequence | `u` increases by exactly 1 per message across the whole file; `seq` increases but is not contiguous. |
| Day boundaries | UTC. First message: a snapshot shortly after midnight (00:00:00.129 in September 2026, 00:00:00.728 in July 2026). Last message: the same boundary snapshot of the next day, preceded by the 1–7 deltas stamped between midnight and it. The closing snapshot repeats the `u` of the last delta. |
| Snapshots | Only the opening and closing snapshots in the sample days; a mid-day snapshot (after a Bybit-side problem) resets the book. |
| Timestamps | `ts` monotone; `cts` (matching engine) slightly earlier. |

Reconstruction check on 2026-09-26: replaying all deltas from the opening snapshot
reproduces the closing snapshot exactly on all 200 levels of both sides, with no crossed or
locked book at any message.

### Cost per day (this container, 4 CPUs)

Download ~3–15 s. Standardization ~60–90 s per day per worker (JSON parsing ≈ 40 %,
book updates ≈ 40 %), ~1.5 GB peak memory per worker. Standardized file ≈ 35–60 MB.

## Standardized format

Defined once in `snn_hft/data/lob/schema.py` (plan §4.5): one parquet file per venue,
symbol and UTC day with columns `ts_ns`, `recv_ts_ns`, `ask_px_1..N`, `ask_qty_1..N`,
`bid_px_1..N`, `bid_qty_1..N` (int64 timestamps, float64 prices, float32 quantities), sorted
by `ts_ns` then message order. Every row has all N levels, positive values, strictly
ordered levels and an uncrossed touch. `manifest.json` records source, venue, symbol,
depth, date range, the retrieval code commit, script arguments, times and the status of
every day (`ok` / `flagged` / `excluded`).

## Quality control

Per day (`quality_report.csv`): messages, snapshots, messages outside the day, sequence
gaps, deltas dropped while the book was invalid, invalid levels, snapshot checks and
mismatches, rows emitted / removed (shallow, crossed or locked, non-positive) / written,
invalid-book time, silence time, longest interval between messages. Gap intervals are
listed in `gaps.csv`. Status: gap time ≤ 2 % of the day `ok`, ≤ 5 % `flagged`, otherwise
`excluded`; see DESIGN_DECISIONS_LOB.md L0-7.

## Other sources

- **FI-2010** (L1): `scripts/data/fi2010_prepare.py`, to follow.
- **Binance futures via Tardis free days** (L7): parser in `retrieval/tardis.py`;
  downloader and script to follow. `datasets.tardis.dev` currently serves this container a
  Cloudflare challenge page instead of the file.
