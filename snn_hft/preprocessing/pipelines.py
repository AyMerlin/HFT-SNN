"""Pipelines of the signal models, built from their configs."""

from __future__ import annotations

from snn_hft.config.schema import BarsConfig, HawkesRSTDPConfig, PaperSNNConfig
from snn_hft.models.hawkes.provider import HawkesParamProvider
from snn_hft.preprocessing.bars import VWAPBarAggregator
from snn_hft.preprocessing.base import Pipeline
from snn_hft.preprocessing.events import EventExtractor
from snn_hft.preprocessing.hawkes_steps import BranchingRewardComputer, HawkesIntensityTransformer
from snn_hft.preprocessing.normalization import IntensityRateScaler, ShiftedZScoreNormalizer
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


def hawkes_pipeline(bars: BarsConfig, signal: HawkesRSTDPConfig, provider: HawkesParamProvider) -> Pipeline:
    """Shared bar, difference and split steps, then EventExtractor → HawkesIntensityTransformer →
    IntensityRateScaler → BranchingRewardComputer (§6.3). Only the normalisation is replaced."""
    return Pipeline(
        [
            VWAPBarAggregator(bars.vwap_num),
            PriceDifferencer(),
            DirectionalSplitter(),
            EventExtractor(signal.hawkes.time_axis),
            HawkesIntensityTransformer(provider, limit=signal.hawkes.limit),
            IntensityRateScaler(new_mean=signal.input.new_mean),
            BranchingRewardComputer(),
        ]
    )
