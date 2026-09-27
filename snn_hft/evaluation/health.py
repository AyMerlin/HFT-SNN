"""Run health checks (§12): output spike rate and silent or saturated hidden pools."""

from __future__ import annotations

from dataclasses import dataclass, field

from snn_hft.config.schema import HealthConfig
from snn_hft.signals.base import SignalSeries


@dataclass
class HealthReport:
    output_rate: float
    pool_rates: dict[str, float]  # spikes per neuron per bar
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def check_health(
    series: SignalSeries,
    pool_sizes: dict[str, int],
    ticks_per_bar: int,
    t_ref: int,
    cfg: HealthConfig,
    saturation: float = 0.9,
) -> HealthReport:
    """Flag an output rate outside [output_rate_min, output_rate_max] and hidden pools that are
    silent (no spike all day) or saturated (≥ `saturation` × the maximum rate T / (t_ref + 1))."""
    report = HealthReport(output_rate=series.rate, pool_rates={})
    if not cfg.output_rate_min <= series.rate <= cfg.output_rate_max:
        report.warnings.append(
            f"output signal in {series.rate:.2%} of bars, outside [{cfg.output_rate_min:.1%}, {cfg.output_rate_max:.0%}]"
        )
    max_rate = ticks_per_bar / (t_ref + 1)
    for pool, size in pool_sizes.items():
        counts = series.diagnostics[f"spikes_{pool}"]
        rate = counts.sum() / (size * max(series.n_bars, 1))
        report.pool_rates[pool] = rate
        if counts.sum() == 0:
            report.warnings.append(f"pool {pool} is silent")
        elif rate >= saturation * max_rate:
            report.warnings.append(f"pool {pool} is saturated ({rate:.2f} spikes/neuron/bar, max {max_rate:.2f})")
    return report
