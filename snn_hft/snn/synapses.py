"""Synapse groups (§6.1): all-to-all excitatory connections between two populations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snn_hft.snn.learning import LearningRule, NoLearning
from snn_hft.snn.neurons import NeuronPopulation


@dataclass
class SynapseGroup:
    name: str
    pre: NeuronPopulation
    post: NeuronPopulation
    rule: LearningRule = field(default_factory=NoLearning)
    w_max: float = 1.0
    weights: np.ndarray | None = None  # (pre.size, post.size)

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.zeros((self.pre.size, self.post.size))
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.weights.shape != (self.pre.size, self.post.size):
            raise ValueError(f"{self.name}: weights shape {self.weights.shape} != ({self.pre.size}, {self.post.size})")
        if (self.weights < 0).any() or (self.weights > self.w_max).any():
            raise ValueError(f"{self.name}: weights must lie in [0, w_max]")

    def init_uniform(self, low: float, high: float, rng: np.random.Generator) -> None:
        self.weights = rng.uniform(low, high, size=(self.pre.size, self.post.size))

    @property
    def n_synapses(self) -> int:
        return self.pre.size * self.post.size
