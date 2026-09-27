"""Baseline: the paper's double-input SNN with unsupervised pairwise STDP (§6.2).

Frozen specification: the config schema (`PaperSNNConfig`) only admits the §11
ambiguity settings and hyperparameters, never architecture or preprocessing changes.
"""

from __future__ import annotations

from snn_hft.config.schema import BarsConfig, PaperSNNConfig, RunConfig
from snn_hft.preprocessing.base import Pipeline
from snn_hft.preprocessing.pipelines import paper_pipeline
from snn_hft.signals.base import SNNSignalModel
from snn_hft.snn.network import NetworkBuilder, SpikingNetwork


class PaperSNNSignalModel(SNNSignalModel):
    def __init__(self, cfg: PaperSNNConfig, bars: BarsConfig, model_id: str):
        if not isinstance(cfg, PaperSNNConfig):
            raise TypeError("PaperSNNSignalModel needs a PaperSNNConfig")
        super().__init__(cfg.core, cfg.input.ticks_per_bar, model_id)
        self.cfg = cfg
        self.bars = bars

    @classmethod
    def from_run(cls, run: RunConfig) -> PaperSNNSignalModel:
        return cls(run.signal, run.bars, run.model_id)

    def build_pipeline(self) -> Pipeline:
        return paper_pipeline(self.bars, self.cfg)

    def build_network(self) -> SpikingNetwork:
        return NetworkBuilder(self.cfg.core).paper()
