"""Neuron populations (§6.1)."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from snn_hft.config.schema import LIFConfig


@dataclass(frozen=True)
class NeuronPopulation(ABC):
    name: str
    size: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"population {self.name!r} must have at least one neuron")


@dataclass(frozen=True)
class InputPopulation(NeuronPopulation):
    """Neurons whose spikes come from an encoder channel."""

    channel: int = 0


@dataclass(frozen=True)
class LIFPopulation(NeuronPopulation):
    """Leaky integrate-and-fire neurons with the paper-literal discrete update."""

    params: LIFConfig = field(default_factory=LIFConfig)
