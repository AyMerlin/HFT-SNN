"""Learning rules (§6.1, §6.5).

The rules are declarative: they carry parameters and a rule code, and the numba
kernel in `simulator.py` executes them. Both STDP rules compute the Hebbian term
with the single function `simulator.stdp_xi`, so R-STDP keeps the timing rule of
plain STDP inside its eligibility trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from snn_hft.config.schema import RSTDPConfig, STDPConfig
from snn_hft.snn import simulator


class LearningRule:
    code: int = simulator.RULE_NONE


@dataclass(frozen=True)
class NoLearning(LearningRule):
    code: int = simulator.RULE_NONE


@dataclass(frozen=True)
class PairwiseSTDP(LearningRule):
    """w ← clip(w + ξ, 0, w_max) every tick."""

    A: float = 0.01
    B: float = -0.0105
    tau_plus: float = 5.0
    tau_minus: float = 5.0
    pairing: Literal["all_to_all", "nearest"] = "all_to_all"
    code: int = simulator.RULE_STDP

    @classmethod
    def from_config(cls, cfg: STDPConfig) -> PairwiseSTDP:
        return cls(A=cfg.A, B=cfg.B, tau_plus=cfg.tau_plus, tau_minus=cfg.tau_minus, pairing=cfg.pairing)

    @property
    def trace_params(self) -> tuple[float, float, float, float, bool]:
        """(A, B, decay of x per tick, decay of y per tick, nearest pairing)."""
        return (
            self.A,
            self.B,
            math.exp(-1.0 / self.tau_plus),
            math.exp(-1.0 / self.tau_minus),
            self.pairing == "nearest",
        )


@dataclass(frozen=True)
class RewardModulatedSTDP(PairwiseSTDP):
    """MSTDPET (§6.5): E ← E·exp(−1/τ_z) + ξ each tick; w ← clip(w + γ·R·E) when a reward arrives.

    `tau_z` is in ticks; 0 means E = ξ. `reward_stream` names the reward array this group uses.
    """

    gamma: float = 1.0
    tau_z: float = 30.0
    reward_stream: str = ""
    delivery: Literal["bar_end", "every_tick"] = "bar_end"
    reset_eligibility_on_reward: bool = False
    code: int = simulator.RULE_RSTDP

    @classmethod
    def from_configs(
        cls, stdp: STDPConfig, rstdp: RSTDPConfig, ticks_per_bar: int, reward_stream: str
    ) -> RewardModulatedSTDP:
        return cls(
            A=stdp.A,
            B=stdp.B,
            tau_plus=stdp.tau_plus,
            tau_minus=stdp.tau_minus,
            pairing=stdp.pairing,
            gamma=rstdp.gamma,
            tau_z=rstdp.tau_z_bars * ticks_per_bar,
            reward_stream=reward_stream,
            delivery=rstdp.delivery,
            reset_eligibility_on_reward=rstdp.reset_eligibility_on_reward,
        )

    @property
    def eligibility_decay(self) -> float:
        return 0.0 if self.tau_z == 0 else math.exp(-1.0 / self.tau_z)

    @property
    def reward_params(self) -> tuple[float, float, bool, bool]:
        """(γ, decay of E per tick, reset E after delivery, deliver every tick)."""
        return self.gamma, self.eligibility_decay, self.reset_eligibility_on_reward, self.delivery == "every_tick"
