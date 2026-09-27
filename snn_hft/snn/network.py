"""Generic spiking network, compiled to flat arrays for the numba kernel, and the builders
of the paper topology (§6.2) and the improved topology (§6.3)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from snn_hft.config.schema import RSTDPConfig, SNNCoreConfig
from snn_hft.snn import simulator
from snn_hft.snn.learning import PairwiseSTDP, RewardModulatedSTDP
from snn_hft.snn.neurons import InputPopulation, LIFPopulation, NeuronPopulation
from snn_hft.snn.synapses import SynapseGroup


@dataclass
class DayRun:
    """Result of simulating one day: spike counts per bar and population."""

    counts: np.ndarray  # (n_bars, n_pops) int32
    pop_names: tuple[str, ...]
    trace_u: np.ndarray  # membrane potential of the recorded neuron per tick (may be empty)

    def pop_counts(self, name: str) -> np.ndarray:
        return self.counts[:, self.pop_names.index(name)]


class SpikingNetwork:
    """Populations and all-to-all synapse groups, with state that persists between calls."""

    def __init__(self, populations: Sequence[NeuronPopulation], groups: Sequence[SynapseGroup], delay_ticks: int = 1):
        if delay_ticks < 1:
            raise ValueError("delay_ticks must be >= 1")
        names = [p.name for p in populations]
        if len(set(names)) != len(names):
            raise ValueError(f"population names must be unique: {names}")
        for g in groups:
            if g.pre not in populations or g.post not in populations:
                raise ValueError(f"group {g.name} connects populations outside the network")
            if isinstance(g.post, InputPopulation):
                raise ValueError(f"group {g.name} targets an input population")
        self.populations = list(populations)
        self.groups = list(groups)
        self.delay_ticks = delay_ticks
        self._compile()
        self.reset_state()

    # ------------------------------------------------------------------ compilation

    def _compile(self) -> None:
        offsets, n = {}, 0
        for pop in self.populations:
            offsets[pop.name] = n
            n += pop.size
        self.n_neurons = n
        self.offsets = offsets

        inputs = [p for p in self.populations if isinstance(p, InputPopulation)]
        for p in inputs:
            if p.size != 1:
                raise ValueError(f"input population {p.name} must have one neuron (one per encoder channel)")
        self.input_neuron = np.array([offsets[p.name] for p in inputs], dtype=np.int64)
        self.input_channel = np.array([p.channel for p in inputs], dtype=np.int64)

        self.is_lif = np.zeros(n, dtype=np.bool_)
        self.u_rest = np.zeros(n)
        self.thr = np.full(n, np.inf)
        self.leak = np.zeros(n)
        self.use_exp = np.zeros(n, dtype=np.bool_)
        self.exp_decay = np.ones(n)
        self.t_ref = np.zeros(n, dtype=np.int64)
        self.pop_of = np.zeros(n, dtype=np.int64)
        for idx, pop in enumerate(self.populations):
            sl = slice(offsets[pop.name], offsets[pop.name] + pop.size)
            self.pop_of[sl] = idx
            if isinstance(pop, LIFPopulation):
                p = pop.params
                self.is_lif[sl] = True
                self.u_rest[sl] = p.u_rest
                self.thr[sl] = p.threshold
                self.leak[sl] = p.leak
                self.use_exp[sl] = p.leak_mode == "exponential"
                self.exp_decay[sl] = math.exp(-1.0 / p.tau_m)
                self.t_ref[sl] = p.t_ref

        pre, post, rule, stream, w_max, self._slices = [], [], [], [], [], []
        stdp_rules = [g.rule for g in self.groups if isinstance(g.rule, PairwiseSTDP)]
        rstdp_rules = [g.rule for g in self.groups if isinstance(g.rule, RewardModulatedSTDP)]
        if len({r.trace_params for r in stdp_rules}) > 1:
            raise ValueError("all STDP groups of a network must share A, B, τ and pairing")
        if len({r.reward_params for r in rstdp_rules}) > 1:
            raise ValueError("all R-STDP groups of a network must share γ, τ_z and delivery")
        self.reward_streams = tuple(dict.fromkeys(r.reward_stream for r in rstdp_rules))
        start = 0
        for g in self.groups:
            i, j = np.meshgrid(np.arange(g.pre.size), np.arange(g.post.size), indexing="ij")
            pre.append(offsets[g.pre.name] + i.ravel())
            post.append(offsets[g.post.name] + j.ravel())
            rule.append(np.full(g.n_synapses, g.rule.code))
            sid = self.reward_streams.index(g.rule.reward_stream) if isinstance(g.rule, RewardModulatedSTDP) else -1
            stream.append(np.full(g.n_synapses, sid))
            w_max.append(np.full(g.n_synapses, g.w_max))
            self._slices.append(slice(start, start + g.n_synapses))
            start += g.n_synapses
        self.pre = np.concatenate(pre).astype(np.int64)
        self.post = np.concatenate(post).astype(np.int64)
        self.rule = np.concatenate(rule).astype(np.int64)
        self.stream = np.concatenate(stream).astype(np.int64)
        self.w_max = np.concatenate(w_max).astype(np.float64)
        self.stdp_params = stdp_rules[0].trace_params if stdp_rules else (0.0, 0.0, 0.0, 0.0, False)
        self.rstdp_params = rstdp_rules[0].reward_params if rstdp_rules else (0.0, 0.0, False, False)

    # ------------------------------------------------------------------ state and weights

    def reset_state(self) -> None:
        """Membrane potentials, refractory counters, traces and in-flight spikes back to rest."""
        self.u = self.u_rest.copy()
        self.ref = np.zeros(self.n_neurons, dtype=np.int64)
        self.x = np.zeros(self.n_neurons)
        self.y = np.zeros(self.n_neurons)
        self.E = np.zeros(len(self.pre))
        self.hist = np.zeros((self.delay_ticks + 1, self.n_neurons), dtype=np.uint8)

    def init_weights(self, low: float, high: float, rng: np.random.Generator) -> None:
        for g in self.groups:
            g.init_uniform(low, high, rng)

    def flat_weights(self) -> np.ndarray:
        return np.concatenate([g.weights.ravel() for g in self.groups])

    def set_flat_weights(self, w: np.ndarray) -> None:
        for g, sl in zip(self.groups, self._slices):
            g.weights = w[sl].reshape(g.pre.size, g.post.size).copy()

    def group(self, name: str) -> SynapseGroup:
        return next(g for g in self.groups if g.name == name)

    @property
    def pop_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.populations)

    # ------------------------------------------------------------------ simulation

    def run_day(
        self,
        channel_spikes: np.ndarray,
        ticks_per_bar: int,
        learn: bool,
        rewards: Mapping[str, np.ndarray] | None = None,
        record: tuple[str, int] | None = None,
    ) -> DayRun:
        """Simulate one day of encoder spikes (n_ticks, n_channels), continuing from the current state.

        With `learn=False` no weight changes (NoLearning, used on test days). `rewards` maps each
        reward stream to per-bar rewards; it is required when learning with R-STDP groups.
        """
        n_ticks = channel_spikes.shape[0]
        if n_ticks % ticks_per_bar:
            raise ValueError("number of ticks must be a multiple of ticks_per_bar")
        n_bars = n_ticks // ticks_per_bar
        input_spikes = np.ascontiguousarray(channel_spikes[:, self.input_channel], dtype=np.uint8)
        reward_arr = np.zeros((n_bars, len(self.reward_streams)))
        if learn and self.reward_streams:
            if rewards is None or set(self.reward_streams) - set(rewards):
                raise ValueError(f"rewards needed for streams {self.reward_streams}")
            for k, name in enumerate(self.reward_streams):
                reward_arr[:, k] = rewards[name]
        record_idx = -1
        if record is not None:
            record_idx = self.offsets[record[0]] + record[1]
        w = self.flat_weights()
        A, B, dplus, dminus, nearest = self.stdp_params
        gamma, dz, reset_elig, every_tick = self.rstdp_params
        counts, trace_u = simulator.simulate_day(
            input_spikes, self.input_neuron, ticks_per_bar,
            self.is_lif, self.u_rest, self.thr, self.leak, self.use_exp, self.exp_decay, self.t_ref,
            self.pre, self.post, w, self.w_max, self.rule, self.stream,
            A, B, dplus, dminus, nearest, self.delay_ticks,
            learn, reward_arr, gamma, dz, reset_elig, every_tick,
            self.pop_of, len(self.populations),
            self.u, self.ref, self.x, self.y, self.E, self.hist,
            record_idx,
        )
        if learn:
            self.set_flat_weights(w)
        return DayRun(counts=counts, pop_names=self.pop_names, trace_u=trace_u)


class NetworkBuilder:
    """Builds the paper topology and the improved topology from the shared engine config."""

    def __init__(self, core: SNNCoreConfig):
        self.core = core
        self.stdp = PairwiseSTDP.from_config(core.stdp)

    def _lif(self, name: str, size: int) -> LIFPopulation:
        return LIFPopulation(name, size, params=self.core.lif)

    def _group(self, pre: NeuronPopulation, post: NeuronPopulation, rule) -> SynapseGroup:
        return SynapseGroup(f"{pre.name}->{post.name}", pre, post, rule=rule, w_max=self.core.synapse.w_max)

    def paper(self) -> SpikingNetwork:
        """X1 → H1, X2 → H2, H1 ∪ H2 → Out (Figure 4), all pairwise STDP."""
        return self.hawkes_rstdp(None, ticks_per_bar=1, use_rstdp_pools=False, keep_direction_pools=True)

    def hawkes_rstdp(
        self,
        rstdp: RSTDPConfig | None,
        ticks_per_bar: int,
        use_rstdp_pools: bool = True,
        keep_direction_pools: bool = True,
    ) -> SpikingNetwork:
        """Paper topology plus H_mom / H_rev fed by both inputs and trained with R-STDP (§6.3)."""
        if not (use_rstdp_pools or keep_direction_pools):
            raise ValueError("the network needs at least one hidden pool")
        size = self.core.hidden_size
        x1, x2 = InputPopulation("X1", 1, channel=0), InputPopulation("X2", 1, channel=1)
        pops: list[NeuronPopulation] = [x1, x2]
        groups: list[SynapseGroup] = []
        hidden: list[LIFPopulation] = []
        if keep_direction_pools:
            h1, h2 = self._lif("H1", size), self._lif("H2", size)
            pops += [h1, h2]
            hidden += [h1, h2]
            groups += [self._group(x1, h1, self.stdp), self._group(x2, h2, self.stdp)]
        if use_rstdp_pools:
            if rstdp is None:
                raise ValueError("R-STDP pools need an RSTDPConfig")
            h_mom, h_rev = self._lif("H_mom", size), self._lif("H_rev", size)
            pops += [h_mom, h_rev]
            hidden += [h_mom, h_rev]
            for pool, stream in ((h_mom, "mom"), (h_rev, "rev")):
                rule = RewardModulatedSTDP.from_configs(self.core.stdp, rstdp, ticks_per_bar, stream)
                groups += [self._group(x1, pool, rule), self._group(x2, pool, rule)]
        out = self._lif("Out", 1)
        pops.append(out)
        groups += [self._group(h, out, self.stdp) for h in hidden]
        return SpikingNetwork(pops, groups, delay_ticks=self.core.synapse.delay_ticks)
