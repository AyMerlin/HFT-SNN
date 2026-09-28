"""OrderBookBuilder (plan §4.2, §11): snapshots, deltas, top-N, sequence gaps, snapshot checks."""

from __future__ import annotations

from retrieval.book import BookMessage, BookSide, OrderBookBuilder


def snap(ts, bids, asks, seq=None):
    return BookMessage(ts_ns=ts, is_snapshot=True, bids=bids, asks=asks, seq=seq)


def delta(ts, bids=(), asks=(), seq=None, prev=None):
    return BookMessage(ts_ns=ts, is_snapshot=False, bids=list(bids), asks=list(asks), seq=seq, prev_seq=prev)


BIDS = [(99.0, 1.0), (98.0, 2.0), (97.0, 3.0)]
ASKS = [(101.0, 1.5), (102.0, 2.5), (103.0, 3.5)]


def test_book_side_orders_best_first_and_deletes():
    bids, asks = BookSide(is_bid=True), BookSide(is_bid=False)
    for p in (97.0, 99.0, 98.0):
        bids.set(p, p / 10)
        asks.set(p + 5, 1.0)
    assert bids.top(2) == ([99.0, 98.0], [9.9, 9.8])
    assert asks.top(5)[0] == [102.0, 103.0, 104.0]
    bids.set(99.0, 0)
    bids.set(50.0, 0)  # delete of an unknown level is a no-op
    assert bids.best() == 98.0 and len(bids) == 2


def test_snapshot_resets_the_book():
    b = OrderBookBuilder()
    assert b.apply(snap(1, BIDS, ASKS, seq=10))
    b.apply(delta(2, bids=[(99.5, 4.0)], seq=11, prev=10))
    b.apply(snap(3, [(90.0, 1.0)], [(110.0, 1.0)], seq=50))
    assert b.top(5) == ([110.0], [1.0], [90.0], [1.0])


def test_delta_insert_update_delete_against_hand_built_book():
    b = OrderBookBuilder()
    b.apply(snap(1, BIDS, ASKS, seq=1))
    b.apply(delta(2, bids=[(99.5, 4.0), (98.0, 0)], asks=[(101.0, 0.5)], seq=2, prev=1))
    b.apply(delta(3, asks=[(100.5, 7.0), (103.0, 0)], seq=3, prev=2))
    ap, aq, bp, bq = b.top(3)
    assert (ap, aq) == ([100.5, 101.0, 102.0], [7.0, 0.5, 2.5])
    assert (bp, bq) == ([99.5, 99.0, 97.0], [4.0, 1.0, 3.0])


def test_top_n_returns_fewer_levels_when_side_is_shallow():
    b = OrderBookBuilder()
    b.apply(snap(1, BIDS[:1], ASKS, seq=1))
    ap, _, bp, _ = b.top(3)
    assert len(ap) == 3 and bp == [99.0]


def test_sequence_gap_invalidates_until_next_snapshot():
    b = OrderBookBuilder()
    b.apply(snap(100, BIDS, ASKS, seq=1))
    assert b.apply(delta(200, bids=[(99.0, 5.0)], seq=2, prev=1))
    assert not b.apply(delta(300, bids=[(99.0, 6.0)], seq=4, prev=3))  # update 3 missing
    assert not b.apply(delta(400, bids=[(99.0, 7.0)], seq=5, prev=4))  # dropped
    assert b.stats.n_seq_gaps == 1 and b.stats.n_deltas_dropped == 2
    assert b.apply(snap(500, BIDS, ASKS, seq=9))
    assert b.apply(delta(600, bids=[(99.0, 8.0)], seq=10, prev=9))
    (gap,) = b.stats.gaps
    assert (gap.start_ns, gap.end_ns, gap.reason) == (200, 500, "sequence")
    assert b.top(1)[3] == [8.0]


def test_deltas_before_the_first_snapshot_are_dropped():
    b = OrderBookBuilder()
    assert not b.apply(delta(10, bids=[(99.0, 1.0)], seq=5, prev=4))
    assert b.apply(snap(20, BIDS, ASKS, seq=6))
    (gap,) = b.stats.gaps
    assert (gap.start_ns, gap.end_ns, gap.reason) == (10, 20, "no_snapshot")


def test_messages_without_sequence_ids_are_not_checked():
    b = OrderBookBuilder()
    b.apply(snap(1, BIDS, ASKS))
    assert b.apply(delta(2, bids=[(99.0, 3.0)]))
    assert b.stats.n_seq_gaps == 0


def test_snapshot_check_counts_mismatches():
    b = OrderBookBuilder()
    b.apply(snap(1, BIDS, ASKS, seq=1))
    b.apply(delta(2, bids=[(99.0, 5.0)], seq=2, prev=1))
    b.apply(snap(3, [(99.0, 5.0)] + BIDS[1:], ASKS, seq=3))  # agrees with the book
    b.apply(snap(4, BIDS, ASKS, seq=4))  # 99.0 has size 1.0 again: disagrees
    assert (b.stats.n_snapshot_checks, b.stats.n_snapshot_mismatches) == (2, 1)


def test_invalid_levels_are_counted_and_skipped():
    b = OrderBookBuilder()
    b.apply(snap(1, BIDS, ASKS, seq=1))
    b.apply(delta(2, bids=[(99.0, -1.0), (float("nan"), 1.0)], seq=2, prev=1))
    assert b.stats.n_invalid_levels == 2
    assert b.top(1)[3] == [1.0]


def test_book_continues_across_calls_and_reset_forgets_it():
    b = OrderBookBuilder()
    b.apply(snap(1, BIDS, ASKS, seq=1))
    b.reset_stats()
    assert b.apply(delta(2, asks=[(101.0, 9.0)], seq=2, prev=1))  # "next day", no snapshot
    b.reset()
    assert not b.apply(delta(3, asks=[(101.0, 8.0)], seq=3, prev=2))
