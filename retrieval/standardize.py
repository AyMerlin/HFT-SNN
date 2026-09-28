"""One UTC day of venue messages -> a clean top-N LOBFrame plus its quality report."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from retrieval.book import BookMessage, OrderBookBuilder
from retrieval.quality import NS_PER_S, DayQualityReport, GapRecord
from snn_hft.data.lob.schema import day_bounds_ns, frame_from_arrays

DEFAULT_DEPTH = 20
DEFAULT_SILENCE_GAP_S = 60.0


@dataclass
class DayResult:
    frame: pd.DataFrame
    report: DayQualityReport
    gaps: list[GapRecord]


class _RowBuffer:
    """Growable (rows, depth) arrays for the emitted top-N book."""

    def __init__(self, depth: int, capacity: int = 1 << 20):
        self.depth = depth
        self.n = 0
        self.has_recv = False
        self._alloc(capacity)

    def _alloc(self, capacity: int) -> None:
        d = self.depth
        old = getattr(self, "ts", None)
        new = {
            "ts": np.empty(capacity, np.int64),
            "recv": np.empty(capacity, np.int64),
            "ap": np.full((capacity, d), np.nan),
            "aq": np.full((capacity, d), np.nan, np.float32),
            "bp": np.full((capacity, d), np.nan),
            "bq": np.full((capacity, d), np.nan, np.float32),
        }
        if old is not None:
            for k, arr in new.items():
                arr[: self.n] = getattr(self, k)[: self.n]
        for k, arr in new.items():
            setattr(self, k, arr)

    def append(self, ts: int, recv: int | None, ap: list, aq: list, bp: list, bq: list) -> None:
        if self.n == len(self.ts):
            self._alloc(2 * len(self.ts))
        i = self.n
        self.ts[i] = ts
        if recv is not None:
            self.recv[i] = recv
            self.has_recv = True
        else:
            self.recv[i] = 0
        self.ap[i, : len(ap)] = ap
        self.aq[i, : len(aq)] = aq
        self.bp[i, : len(bp)] = bp
        self.bq[i, : len(bq)] = bq
        self.n += 1

    def arrays(self) -> tuple[np.ndarray, ...]:
        n = self.n
        return self.ts[:n], self.recv[:n], self.ap[:n], self.aq[:n], self.bp[:n], self.bq[:n]


class DayStandardizer:
    """Runs one day's messages through an `OrderBookBuilder` and cleans the output.

    - Every applied message whose timestamp lies inside the UTC day and leaves a valid book
      emits one row with the top-`depth` levels. Messages outside the day are applied (for
      continuity) but emit nothing.
    - Gap time = time inside the day with an invalid book, plus message-free stretches
      longer than `silence_gap_s` while the book was valid (including the stretch from the
      last message to midnight when the file ends early).
    - Rows with fewer than `depth` levels on a side, a crossed or locked touch, or a
      non-positive value are removed and counted. Rows are sorted by `ts_ns` (stable, so ties
      keep message order).
    """

    def __init__(self, depth: int = DEFAULT_DEPTH, silence_gap_s: float = DEFAULT_SILENCE_GAP_S):
        self.depth = depth
        self.silence_gap_ns = int(silence_gap_s * NS_PER_S)

    def run(
        self, messages: Iterable[BookMessage], day: date, builder: OrderBookBuilder, raw_file: str | None = None
    ) -> DayResult:
        lo, hi = day_bounds_ns(day)
        builder.reset_stats()
        rep = DayQualityReport(day=day, raw_file=raw_file)
        buf = _RowBuffer(self.depth)
        prev_ts, prev_valid = lo, builder.valid
        invalid_ns = silence_ns = max_int = 0

        def account(a: int, b: int, valid: bool) -> None:
            nonlocal invalid_ns, silence_ns, max_int
            a, b = max(a, lo), min(b, hi)
            if b <= a:
                return
            dur = b - a
            if not valid:
                invalid_ns += dur
            elif dur > self.silence_gap_ns:
                silence_ns += dur
            max_int = max(max_int, dur)
            if dur > NS_PER_S:
                rep.n_intervals_over_1s += 1

        depth = self.depth
        for msg in messages:
            ts = msg.ts_ns
            if ts < prev_ts:
                rep.n_ts_backwards += 1
            account(prev_ts, ts, prev_valid)
            valid = builder.apply(msg)
            if ts < lo or ts >= hi:
                rep.n_outside_day += 1
            else:
                if rep.first_ts_ns is None:
                    rep.first_ts_ns = ts
                rep.last_ts_ns = ts
                if valid:
                    buf.append(ts, msg.recv_ts_ns, *builder.top(depth))
            prev_ts, prev_valid = max(prev_ts, ts), valid
        account(prev_ts, hi, prev_valid)

        st = builder.stats
        rep.n_messages = st.n_messages
        rep.n_snapshots = st.n_snapshots
        rep.n_seq_gaps = st.n_seq_gaps
        rep.n_deltas_dropped = st.n_deltas_dropped
        rep.n_invalid_levels = st.n_invalid_levels
        rep.n_snapshot_checks = st.n_snapshot_checks
        rep.n_snapshot_mismatches = st.n_snapshot_mismatches
        rep.invalid_book_s = invalid_ns / NS_PER_S
        rep.silence_s = silence_ns / NS_PER_S
        rep.max_interval_s = max_int / NS_PER_S
        gaps = [GapRecord(day, g.start_ns, g.end_ns, g.reason) for g in st.gaps]

        frame = self._clean(buf, rep)
        rep.decide_status()
        return DayResult(frame, rep, gaps)

    def _clean(self, buf: _RowBuffer, rep: DayQualityReport) -> pd.DataFrame:
        ts, recv, ap, aq, bp, bq = buf.arrays()
        rep.n_rows_emitted = len(ts)
        px = np.concatenate([ap, bp], axis=1)
        qty = np.concatenate([aq, bq], axis=1)
        shallow = np.isnan(px).any(axis=1) | np.isnan(qty).any(axis=1)
        crossed = ~shallow & (ap[:, 0] <= bp[:, 0])
        nonpos = ~shallow & ((px <= 0).any(axis=1) | (qty <= 0).any(axis=1))
        rep.n_rows_shallow = int(shallow.sum())
        rep.n_rows_crossed_or_locked = int(crossed.sum())
        rep.n_rows_nonpositive = int(nonpos.sum())
        keep = ~(shallow | crossed | nonpos)
        order = np.flatnonzero(keep)
        if len(order) and np.any(np.diff(ts[order]) < 0):
            order = order[np.argsort(ts[order], kind="stable")]
        rep.n_rows_written = len(order)
        return frame_from_arrays(
            ts[order], recv[order] if buf.has_recv else None, ap[order], aq[order], bp[order], bq[order]
        )
