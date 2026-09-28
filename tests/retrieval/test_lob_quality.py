"""Day standardization and quality control (plan §4.2, §11)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from retrieval.book import BookMessage, OrderBookBuilder
from retrieval.standardize import DayStandardizer
from snn_hft.data.lob.schema import NS_PER_DAY, day_bounds_ns, day_status, validate_frame

DAY = date(2026, 7, 1)
T0, T_END = day_bounds_ns(DAY)
S = 1_000_000_000
BIDS = [(99.0, 1.0), (98.0, 2.0), (97.0, 3.0)]
ASKS = [(101.0, 1.0), (102.0, 2.0), (103.0, 3.0)]


def snap(ts, seq, bids=BIDS, asks=ASKS):
    return BookMessage(ts, True, bids, asks, seq=seq)


def delta(ts, seq, bids=(), asks=(), prev=None):
    return BookMessage(ts, False, list(bids), list(asks), seq=seq, prev_seq=seq - 1 if prev is None else prev)


def steady(start_ns, end_ns, step_ns, seq0):
    """Deltas every `step_ns` in [start_ns, end_ns), alternating the best bid size."""
    return [delta(t, seq0 + i, bids=[(99.0, 1.0 + (i % 2))]) for i, t in enumerate(range(start_ns, end_ns, step_ns))]


def run(messages, depth=3, builder=None, silence_s=60.0):
    return DayStandardizer(depth, silence_s).run(messages, DAY, builder or OrderBookBuilder())


def test_clean_day_emits_one_row_per_message_and_is_ok():
    msgs = [snap(T0, 0)] + steady(T0 + S, T_END, 30 * S, 1)
    res = run(msgs)
    assert res.report.status == "ok"
    assert res.report.n_rows_written == len(msgs)
    assert res.report.gap_s == 0
    validate_frame(res.frame, 3, DAY)
    assert list(res.frame["bid_qty_1"][:3]) == [1.0, 1.0, 2.0]


def test_crossed_and_locked_rows_are_removed_and_counted():
    msgs = [
        snap(T0, 0),
        delta(T0 + 1 * S, 1, bids=[(101.0, 1.0)]),  # locked: bid == ask
        delta(T0 + 2 * S, 2, bids=[(102.0, 1.0)]),  # crossed
        delta(T0 + 3 * S, 3, bids=[(101.0, 0.0), (102.0, 0.0)]),  # back to normal
    ]
    rep = run(msgs).report
    assert (rep.n_rows_emitted, rep.n_rows_crossed_or_locked, rep.n_rows_written) == (4, 2, 2)


def test_shallow_rows_are_removed_and_counted():
    msgs = [snap(T0, 0), delta(T0 + S, 1, asks=[(103.0, 0.0)]), delta(T0 + 2 * S, 2, asks=[(104.0, 1.0)])]
    res = run(msgs)
    assert (res.report.n_rows_shallow, res.report.n_rows_written) == (1, 2)
    assert list(res.frame["ask_px_3"]) == [103.0, 104.0]


@pytest.mark.parametrize("gap_share, status", [(0.01, "ok"), (0.03, "flagged"), (0.10, "excluded")])
def test_day_status_from_sequence_gap_time(gap_share, status):
    gap_ns = int(gap_share * NS_PER_DAY)
    first = steady(T0 + S, T0 + 3600 * S, 10 * S, 1)
    gap_start = first[-1].ts_ns
    lost = delta(gap_start + 5 * S, 999, prev=998)  # breaks the sequence
    resync_ts = gap_start + gap_ns
    rest = steady(resync_ts + S, T_END, 10 * S, 1001)
    rep = run([snap(T0, 0), *first, lost, snap(resync_ts, 1000), *rest]).report
    assert rep.n_seq_gaps == 1
    assert rep.invalid_book_s == pytest.approx(gap_ns / S - 5, abs=1e-6)
    assert rep.status == status


def test_status_thresholds_match_plan():
    assert [day_status(x) for x in (0.0, 0.02, 0.0201, 0.05, 0.0501)] == ["ok", "ok", "flagged", "flagged", "excluded"]


def test_long_silence_counts_as_gap_short_one_does_not():
    quiet = [snap(T0, 0)] + steady(T0 + S, T0 + 3600 * S, 50 * S, 1)  # 50 s steps: below threshold
    assert run(quiet + steady(T0 + 3600 * S, T_END, 50 * S, 10_000)).report.silence_s == 0
    rep = run(quiet).report  # file ends after one hour: the rest of the day is silence
    assert rep.silence_s == pytest.approx((T_END - quiet[-1].ts_ns) / S)
    assert rep.status == "excluded"


def test_snapshot_mismatch_downgrades_ok_day_to_flagged():
    msgs = [snap(T0, 0), delta(T0 + S, 1, bids=[(99.0, 5.0)])] + steady(T0 + 2 * S, T_END - S, 30 * S, 2)
    msgs.append(snap(T_END - 1, 10**6, bids=[(99.0, 7.0), (98.0, 2.0), (97.0, 3.0)]))
    rep = run(msgs).report
    assert rep.n_snapshot_mismatches == 1 and rep.status == "flagged"


def test_messages_outside_the_day_are_applied_but_not_emitted():
    msgs = [snap(T0 - S, 0), delta(T0 + S, 1, bids=[(99.0, 4.0)]), delta(T_END + 1, 2, bids=[(99.0, 5.0)])]
    b = OrderBookBuilder()
    res = run(msgs, builder=b)
    assert res.report.n_outside_day == 2 and res.report.n_rows_written == 1
    assert b.top(1)[3] == [5.0]


def test_day_continues_from_previous_book_without_opening_snapshot():
    b = OrderBookBuilder()
    run([snap(T0, 0)] + steady(T0 + S, T_END, 3600 * S, 1), builder=b)
    next_day = date(2026, 7, 2)
    n0 = day_bounds_ns(next_day)[0]
    res = DayStandardizer(3).run([delta(n0 + S, 100, prev=b.last_seq)], next_day, b)
    assert res.report.n_rows_written == 1 and res.report.invalid_book_s == 0


def test_missing_previous_day_drops_until_first_snapshot():
    res = run([delta(T0 + S, 5), delta(T0 + 2 * S, 6), snap(T0 + 3 * S, 7), delta(T0 + 4 * S, 8)])
    assert res.report.n_rows_written == 2
    assert res.report.invalid_book_s == pytest.approx(3.0)
    assert [g.reason for g in res.gaps] == ["no_snapshot"]


def test_rows_are_sorted_by_timestamp_keeping_message_order():
    msgs = [snap(T0, 0), delta(T0 + 2 * S, 1, bids=[(99.0, 2.0)]), delta(T0 + S, 2, bids=[(99.0, 3.0)])]
    res = run(msgs)
    assert res.report.n_ts_backwards == 1
    assert np.all(np.diff(res.frame["ts_ns"].to_numpy()) >= 0)
    assert list(res.frame["bid_qty_1"]) == [1.0, 3.0, 2.0]
