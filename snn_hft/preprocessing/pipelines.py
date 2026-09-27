"""Pipelines of the signal models, built from their configs."""

from __future__ import annotations

from snn_hft.config.schema import BarsConfig, PaperSNNConfig
from snn_hft.preprocessing.bars import VWAPBarAggregator
from snn_hft.preprocessing.base import Pipeline
from snn_hft.preprocessing.normalization import ShiftedZScoreNormalizer
from snn_hft.preprocessing.returns import DirectionalSplitter, PriceDifferencer


def paper_pipeline(bars: BarsConfig, signal: PaperSNNConfig) -> Pipeline:
    """VWAPBarAggregator → PriceDifferencer → DirectionalSplitter → ShiftedZScoreNormalizer (§6.2)."""
    return Pipeline(
        [
            VWAPBarAggregator(bars.vwap_num),
            PriceDifferencer(),
            DirectionalSplitter(),
            ShiftedZScoreNormalizer(
                new_mean=signal.input.new_mean,
                new_std=signal.zscore.new_std,
                stats_source=signal.zscore.stats_source,
            ),
        ]
    )
