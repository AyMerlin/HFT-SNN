import hashlib
import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from snn_hft.data.containers import TRADE_COLUMNS, TradeFrame, day_bounds_ns
from snn_hft.data.sources import (
    BinanceFuturesDataSource,
    ChecksumError,
    DataNotAvailableError,
    LocalFileDataSource,
    SyntheticDataSource,
    parse_binance_csv,
    standardize,
    to_ns,
    verify_checksum,
)
from snn_hft.data.store import DataStore, TradingCalendar

DAY = date(2026, 9, 16)
DAY_MS = 1_789_516_800_000  # 2026-09-16 00:00:00 UTC in ms (first trade of the real file is +2 ms)

TRADES_HEADER = "id,price,qty,quote_qty,time,is_buyer_maker\n"
TRADES_ROWS = (
    f"101,75599.9,0.001,75.5999,{DAY_MS + 2},true\n"
    f"102,75600.0,0.006,453.6,{DAY_MS + 2},false\n"
    f"103,75600.1,0.010,756.001,{DAY_MS + 5},True\n"
)
AGG_HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
AGG_ROWS = (
    f"7,75599.9,0.5,101,104,{DAY_MS + 2},true\n"
    f"8,75600.0,1.25,105,105,{DAY_MS + 9},false\n"
)


def zip_bytes(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def checksum_for(blob: bytes, name: str) -> bytes:
    return f"{hashlib.sha256(blob).hexdigest()}  {name}\n".encode()


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "dataset,header,rows",
    [("trades", TRADES_HEADER, TRADES_ROWS), ("aggTrades", AGG_HEADER, AGG_ROWS)],
)
def test_parse_with_and_without_header(dataset, header, rows):
    with_header = parse_binance_csv((header + rows).encode(), dataset)
    without = parse_binance_csv(rows.encode(), dataset)
    pd.testing.assert_frame_equal(with_header, without)
    assert list(with_header.columns) == list(TRADE_COLUMNS)
    assert with_header.dtypes.astype(str).to_dict() == TRADE_COLUMNS
    assert with_header["ts_ns"].iloc[0] == (DAY_MS + 2) * 1_000_000


def test_parse_maps_aggtrades_columns():
    df = parse_binance_csv((AGG_HEADER + AGG_ROWS).encode(), "aggTrades")
    assert df["trade_id"].tolist() == [7, 8]
    assert df["qty"].tolist() == [0.5, 1.25]
    assert df["is_buyer_maker"].tolist() == [True, False]


def test_timestamp_unit_detection():
    ms = np.array([DAY_MS], dtype=np.int64)
    assert to_ns(ms)[0] == to_ns(ms * 1_000)[0] == to_ns(ms * 1_000_000)[0] == DAY_MS * 1_000_000


def test_checksum_verification():
    blob = b"payload"
    verify_checksum(blob, checksum_for(blob, "f.zip").decode(), "f.zip")
    with pytest.raises(ChecksumError):
        verify_checksum(blob + b"x", checksum_for(blob, "f.zip").decode(), "f.zip")
    with pytest.raises(ChecksumError):
        verify_checksum(blob, "", "f.zip")


# --------------------------------------------------------------------------- UTC day


def test_day_bounds_are_utc():
    start, end = day_bounds_ns(DAY)
    assert start == DAY_MS * 1_000_000
    assert end - start == 86_400 * 10**9


def test_standardize_keeps_only_the_utc_day_and_sorts():
    start_ms, end_ms = DAY_MS, DAY_MS + 86_400_000
    df = pd.DataFrame(
        {
            "ts_ns": np.array([end_ms - 1, start_ms - 1, start_ms, end_ms, start_ms], dtype=np.int64) * 1_000_000,
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
            "qty": [1.0] * 5,
            "is_buyer_maker": [True] * 5,
            "trade_id": np.array([5, 1, 3, 6, 2], dtype=np.int64),
        }
    ).astype(TRADE_COLUMNS)
    frame, stats = standardize(df, "BTCUSDT", "v", DAY, "aggTrades")
    assert stats.n_raw == 5 and stats.n_outside_day == 2  # previous-day and next-day midnight trades
    assert frame.df["trade_id"].tolist() == [2, 3, 5]  # ties at 00:00:00.000 broken by trade id
    assert frame.df["ts_ns"].iloc[-1] == (end_ms - 1) * 1_000_000  # 23:59:59.999 is kept


def test_tradeframe_rejects_bad_input():
    good = pd.DataFrame(
        {"ts_ns": [DAY_MS * 1_000_000, DAY_MS * 1_000_000], "price": [1.0, 1.0], "qty": [1.0, 1.0],
         "is_buyer_maker": [True, False], "trade_id": [1, 2]}
    ).astype(TRADE_COLUMNS)
    TradeFrame(good, "S", "v", DAY)
    with pytest.raises(ValueError, match="sorted"):
        TradeFrame(good.iloc[::-1].reset_index(drop=True), "S", "v", DAY)
    with pytest.raises(ValueError, match="outside"):
        TradeFrame(good, "S", "v", date(2026, 9, 15))
    with pytest.raises(ValueError, match="dtype"):
        TradeFrame(good.astype({"trade_id": "float64"}), "S", "v", DAY)


# --------------------------------------------------------------------------- sources


class FakeArchive:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self.files:
            raise DataNotAvailableError(url)
        return self.files[url]


def binance_files(dataset: str, text: str, corrupt: bool = False) -> tuple[BinanceFuturesDataSource, FakeArchive]:
    probe = BinanceFuturesDataSource(dataset=dataset)
    url = probe.url("BTCUSDT", DAY)
    name = url.rsplit("/", 1)[-1]
    blob = zip_bytes(name.replace(".zip", ".csv"), text)
    checksum = checksum_for(blob + (b"x" if corrupt else b""), name)
    archive = FakeArchive({url: blob, url + ".CHECKSUM": checksum})
    return BinanceFuturesDataSource(dataset=dataset, fetcher=archive), archive


def test_binance_source_url_and_parse():
    source, archive = binance_files("aggTrades", AGG_HEADER + AGG_ROWS)
    result = source.fetch("BTCUSDT", DAY)
    assert archive.calls[0] == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-09-16.zip"
    )
    assert len(result.frame) == 2 and result.sha256 is not None
    assert result.frame.venue == "binance_um" and result.frame.dataset == "aggTrades"


def test_binance_source_rejects_bad_checksum():
    source, _ = binance_files("trades", TRADES_ROWS, corrupt=True)
    with pytest.raises(ChecksumError):
        source.fetch("BTCUSDT", DAY)


def test_binance_source_missing_day():
    source = BinanceFuturesDataSource(fetcher=FakeArchive({}))
    with pytest.raises(DataNotAvailableError):
        source.fetch("BTCUSDT", DAY)


def test_local_source_reads_zip_csv_and_parquet(tmp_path):
    stem = "BTCUSDT-trades-2026-09-16"
    blob = zip_bytes(f"{stem}.csv", TRADES_HEADER + TRADES_ROWS)
    (tmp_path / f"{stem}.zip").write_bytes(blob)
    (tmp_path / f"{stem}.zip.CHECKSUM").write_bytes(checksum_for(blob, f"{stem}.zip"))
    from_zip = LocalFileDataSource(tmp_path, dataset="trades").fetch("BTCUSDT", DAY)
    assert from_zip.sha256 is not None and len(from_zip.frame) == 3

    other = tmp_path / "csv" / "BTCUSDT"
    other.mkdir(parents=True)
    (other / f"{stem}.csv").write_text(TRADES_ROWS)
    from_csv = LocalFileDataSource(tmp_path / "csv", dataset="trades").fetch_day("BTCUSDT", DAY)
    pd.testing.assert_frame_equal(from_csv.df, from_zip.frame.df)

    pq_dir = tmp_path / "pq"
    pq_dir.mkdir()
    from_zip.frame.df.to_parquet(pq_dir / f"{stem}.parquet")
    from_pq = LocalFileDataSource(pq_dir, dataset="trades").fetch_day("BTCUSDT", DAY)
    pd.testing.assert_frame_equal(from_pq.df, from_zip.frame.df)

    with pytest.raises(DataNotAvailableError):
        LocalFileDataSource(tmp_path, dataset="trades").fetch("BTCUSDT", date(2026, 9, 17))


def test_synthetic_source_is_deterministic_per_seed_and_day():
    src = SyntheticDataSource(trades_per_day=5_000, seed=3)
    a = src.fetch_day("SYN", DAY)
    b = SyntheticDataSource(trades_per_day=5_000, seed=3).fetch_day("SYN", DAY)
    pd.testing.assert_frame_equal(a.df, b.df)
    assert not a.df.equals(src.fetch_day("SYN", date(2026, 9, 17)).df)
    assert not a.df.equals(SyntheticDataSource(trades_per_day=5_000, seed=4).fetch_day("SYN", DAY).df)
    assert len(a) == 5_000 and (a.df["price"] > 0).all()


# --------------------------------------------------------------------------- store


class CountingSource(SyntheticDataSource):
    def __init__(self, fail_on: set[date] = frozenset(), **kw):
        super().__init__(**kw)
        self.fail_on = fail_on
        self.fetched: list[date] = []

    def fetch(self, symbol, day):
        self.fetched.append(day)
        if day in self.fail_on:
            raise DataNotAvailableError(str(day))
        return super().fetch(symbol, day)


def test_store_caches_and_roundtrips(tmp_path):
    source = CountingSource(trades_per_day=2_000)
    store = DataStore(source, raw_dir=tmp_path)
    first = store.get_day("SYN", DAY)
    second = store.get_day("SYN", DAY)
    assert source.fetched == [DAY]
    pd.testing.assert_frame_equal(first.df, second.df)
    pd.testing.assert_frame_equal(first.df, source.fetch_day("SYN", DAY).df)
    assert store.path("SYN", DAY) == tmp_path / "synthetic" / "SYN" / "aggTrades" / "2026-09-16.parquet"
    rec = store.record("SYN", DAY)
    assert rec.n_trades == 2_000 and rec.day == "2026-09-16" and rec.max_gap_s > 0


def test_store_ensure_days_reports_failures_and_coverage(tmp_path):
    days = TradingCalendar.days(date(2026, 9, 14), date(2026, 9, 17))
    source = CountingSource(fail_on={date(2026, 9, 15)}, trades_per_day=1_000)
    store = DataStore(source, raw_dir=tmp_path)
    failures = store.ensure_days("SYN", days, workers=2)
    assert set(failures) == {date(2026, 9, 15)}
    cov = store.coverage("SYN", days[0], days[-1], vwap_num=10)
    assert cov["cached"].tolist() == [True, False, True, True]
    assert cov.loc[0, "n_bars"] == 100
    # Already cached days are not fetched again.
    source.fetched.clear()
    store.ensure_days("SYN", days, workers=2)
    assert source.fetched == [date(2026, 9, 15)]


def test_trading_calendar():
    assert TradingCalendar.days(date(2026, 2, 27), date(2026, 3, 1)) == [
        date(2026, 2, 27), date(2026, 2, 28), date(2026, 3, 1)
    ]
    assert TradingCalendar.previous(date(2026, 3, 1), 2) == [date(2026, 2, 27), date(2026, 2, 28)]
