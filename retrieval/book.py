"""Order-book reconstruction shared by all venues (plan §4.2).

Parsers turn venue messages into `BookMessage`s; `OrderBookBuilder` applies them:

- snapshot: reset the book to the snapshot (checking it against the current book first);
- delta: for each (price, size), size 0 deletes the level, otherwise sets it;
- sequence check: a message may name the update id the previous message must have had
  (`prev_seq`). On a mismatch the book becomes invalid until the next snapshot and the gap
  is recorded.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field

Level = tuple[float, float]  # (price, size)


@dataclass(frozen=True)
class BookMessage:
    """One venue message in normalized form."""

    ts_ns: int  # exchange timestamp
    is_snapshot: bool
    bids: Sequence[Level]
    asks: Sequence[Level]
    recv_ts_ns: int | None = None  # local receive timestamp, if the source has one
    seq: int | None = None  # update id after this message
    prev_seq: int | None = None  # update id the previous message must have had; None = no check


@dataclass
class Gap:
    """An interval with an invalid book: from the last good message to the restoring snapshot."""

    start_ns: int
    end_ns: int | None  # None while still open
    reason: str  # "sequence" or "no_snapshot"

    @property
    def duration_ns(self) -> int | None:
        return None if self.end_ns is None else self.end_ns - self.start_ns


class BookSide:
    """One side of the book: price -> size, prices kept sorted best-first."""

    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self._keys: list[float] = []  # ascending sort keys: -price for bids, price for asks
        self._sizes: dict[float, float] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._sizes.clear()

    def set(self, price: float, size: float) -> None:
        """Size 0 deletes the level (a delete of an unknown level is a no-op)."""
        key = -price if self.is_bid else price
        if size == 0:
            if self._sizes.pop(price, None) is not None:
                i = bisect_left(self._keys, key)
                del self._keys[i]
            return
        if price not in self._sizes:
            i = bisect_left(self._keys, key)
            self._keys.insert(i, key)
        self._sizes[price] = size

    def best(self) -> float | None:
        if not self._keys:
            return None
        k = self._keys[0]
        return -k if self.is_bid else k

    def top(self, n: int) -> tuple[list[float], list[float]]:
        """Prices and sizes of the best `n` levels (fewer if the side is shallower)."""
        keys = self._keys[:n]
        prices = [-k for k in keys] if self.is_bid else list(keys)
        return prices, [self._sizes[p] for p in prices]

    def levels(self) -> list[Level]:
        prices, sizes = self.top(len(self._keys))
        return list(zip(prices, sizes))


@dataclass
class BuilderStats:
    n_messages: int = 0
    n_snapshots: int = 0
    n_deltas_dropped: int = 0  # deltas received while the book was invalid
    n_seq_gaps: int = 0
    n_invalid_levels: int = 0  # negative or non-finite sizes/prices in messages (skipped)
    n_snapshot_checks: int = 0  # snapshots received while the book was valid
    n_snapshot_mismatches: int = 0  # ... whose levels differed from the reconstructed book
    gaps: list[Gap] = field(default_factory=list)


class OrderBookBuilder:
    """Applies `BookMessage`s and exposes the top-N book (plan §4.2).

    The builder is venue-agnostic and keeps its state across calls, so consecutive days can
    be processed with one builder (a day file that does not start with a snapshot continues
    from the previous day's book).
    """

    def __init__(self) -> None:
        self.bids = BookSide(is_bid=True)
        self.asks = BookSide(is_bid=False)
        self.valid = False
        self.last_seq: int | None = None
        self.last_ts_ns: int | None = None
        self.stats = BuilderStats()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Forget the book (e.g. when the previous day is missing)."""
        self.bids.clear()
        self.asks.clear()
        self.valid = False
        self.last_seq = None

    def reset_stats(self) -> BuilderStats:
        """Return the stats collected so far and start new ones (the open gap carries over)."""
        old = self.stats
        self.stats = BuilderStats()
        if old.gaps and old.gaps[-1].end_ns is None:
            self.stats.gaps.append(old.gaps[-1])
        return old

    @property
    def open_gap(self) -> Gap | None:
        g = self.stats.gaps[-1] if self.stats.gaps else None
        return g if g is not None and g.end_ns is None else None

    # ------------------------------------------------------------------ messages

    def apply(self, msg: BookMessage) -> bool:
        """Apply one message; return whether the book is valid afterwards."""
        st = self.stats
        st.n_messages += 1
        if msg.is_snapshot:
            st.n_snapshots += 1
            if self.valid:
                st.n_snapshot_checks += 1
                if not self._matches(msg):
                    st.n_snapshot_mismatches += 1
            self.bids.clear()
            self.asks.clear()
            self._apply_levels(msg)
            gap = self.open_gap
            if gap is not None:
                gap.end_ns = msg.ts_ns
            self.valid = True
        elif not self.valid:
            st.n_deltas_dropped += 1
            if self.open_gap is None:
                start = self.last_ts_ns if self.last_ts_ns is not None else msg.ts_ns
                st.gaps.append(Gap(start, None, "no_snapshot"))
        elif msg.prev_seq is not None and self.last_seq is not None and msg.prev_seq != self.last_seq:
            st.n_seq_gaps += 1
            st.n_deltas_dropped += 1
            st.gaps.append(Gap(self.last_ts_ns if self.last_ts_ns is not None else msg.ts_ns, None, "sequence"))
            self.valid = False
        else:
            self._apply_levels(msg)
        if self.valid:
            self.last_seq = msg.seq
        self.last_ts_ns = msg.ts_ns
        return self.valid

    def _apply_levels(self, msg: BookMessage) -> None:
        for side, levels in ((self.bids, msg.bids), (self.asks, msg.asks)):
            for price, size in levels:
                if not (price > 0 and size >= 0) or price == float("inf") or size == float("inf"):
                    self.stats.n_invalid_levels += 1  # NaN fails both comparisons
                    continue
                side.set(price, size)

    def _matches(self, snap: BookMessage) -> bool:
        """Does the reconstructed book agree with a snapshot on the snapshot's depth?"""
        for side, levels in ((self.bids, snap.bids), (self.asks, snap.asks)):
            ref = sorted(((p, s) for p, s in levels if s > 0), key=lambda ps: -ps[0] if side.is_bid else ps[0])
            prices, sizes = side.top(len(ref))
            if len(prices) != len(ref):
                return False
            for (p, s), p2, s2 in zip(ref, prices, sizes):
                if p != p2 or abs(s - s2) > 1e-9 * max(1.0, abs(s)):
                    return False
        return True

    # ------------------------------------------------------------------ output

    def top(self, n: int) -> tuple[list[float], list[float], list[float], list[float]]:
        """(ask prices, ask sizes, bid prices, bid sizes) of the best `n` levels, best first."""
        ap, aq = self.asks.top(n)
        bp, bq = self.bids.top(n)
        return ap, aq, bp, bq
