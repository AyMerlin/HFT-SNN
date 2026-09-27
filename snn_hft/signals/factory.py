"""Signal model construction from a resolved run config."""

from __future__ import annotations

from snn_hft.config.schema import PaperSNNConfig, RunConfig
from snn_hft.signals.base import SignalModel
from snn_hft.signals.paper_snn import PaperSNNSignalModel


def make_signal_model(run: RunConfig) -> SignalModel:
    if isinstance(run.signal, PaperSNNConfig):
        return PaperSNNSignalModel.from_run(run)
    raise NotImplementedError(f"signal model {run.signal.model!r} arrives in a later milestone")
