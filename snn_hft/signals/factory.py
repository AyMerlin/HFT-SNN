"""Signal model construction from a resolved run config."""

from __future__ import annotations

from snn_hft.config.schema import HawkesRSTDPConfig, PaperSNNConfig, RunConfig
from snn_hft.signals.base import SignalModel
from snn_hft.signals.paper_snn import PaperSNNSignalModel


def hawkes_sources(run: RunConfig):
    """Bars of any day (for the Hawkes history) and the θ_d cache, from the run's data settings."""
    from snn_hft.backtest.cache import BarCache, HawkesParamCache
    from snn_hft.data.store import make_store

    store = make_store(run.data)
    bar_cache = BarCache(store, run.data.symbol, run.bars.vwap_num, run.output.cache_dir)
    params = HawkesParamCache(run.output.cache_dir, store.source.venue, run.data.symbol, run.data.dataset, run.bars.vwap_num)
    return bar_cache.get, params


def make_signal_model(run: RunConfig) -> SignalModel:
    if isinstance(run.signal, PaperSNNConfig):
        return PaperSNNSignalModel.from_run(run)
    if isinstance(run.signal, HawkesRSTDPConfig):
        from snn_hft.signals.hawkes_snn import HawkesRSTDPSignalModel

        if run.w_h is None:
            raise ValueError("a Hawkes model needs w_h")
        bars_for_day, params = hawkes_sources(run)
        return HawkesRSTDPSignalModel(run.signal, run.bars, run.model_id, run.w_h, bars_for_day, params)
    raise NotImplementedError(f"signal model {run.signal.model!r}")
