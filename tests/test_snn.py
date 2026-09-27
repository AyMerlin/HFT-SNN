import math

import numpy as np
import pytest

from snn_hft.config.schema import LIFConfig, RSTDPConfig, SNNCoreConfig
from snn_hft.snn import simulator
from snn_hft.snn.encoding import PoissonEncoder
from snn_hft.snn.learning import NoLearning, PairwiseSTDP, RewardModulatedSTDP
from snn_hft.snn.network import NetworkBuilder, SpikingNetwork
from snn_hft.snn.neurons import InputPopulation, LIFPopulation
from snn_hft.snn.synapses import SynapseGroup

TAU = 5.0
A, B = 0.01, -0.0105


# --------------------------------------------------------------------------- encoder


def test_poisson_rate_converges_to_channel_prob():
    prob = np.tile([[0.05, 0.9], [0.2, 0.5]], (10_000, 1))
    spikes = PoissonEncoder().encode(prob, 10, np.random.default_rng(0))
    assert spikes.shape == (200_000, 2) and spikes.dtype == np.uint8
    rates = spikes.reshape(-1, 20, 2).mean(axis=0)  # period of the tiled pattern: 2 bars × 10 ticks
    np.testing.assert_allclose(rates[:10].mean(axis=0), [0.05, 0.9], atol=0.005)
    np.testing.assert_allclose(rates[10:].mean(axis=0), [0.2, 0.5], atol=0.005)


def test_poisson_prefix_does_not_depend_on_day_length_or_chunking():
    prob = np.random.default_rng(1).uniform(0, 1, (1_000, 2))
    full = PoissonEncoder(chunk_bars=7).encode(prob, 10, np.random.default_rng(5))
    prefix = PoissonEncoder(chunk_bars=7).encode(prob[:333], 10, np.random.default_rng(5))
    other_chunks = PoissonEncoder(chunk_bars=100_000).encode(prob, 10, np.random.default_rng(5))
    np.testing.assert_array_equal(prefix, full[:3_330])
    np.testing.assert_array_equal(other_chunks, full)


def test_poisson_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        PoissonEncoder().encode(np.array([[1.2, 0.0]]), 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        PoissonEncoder().encode(np.array([0.5]), 10, np.random.default_rng(0))


# --------------------------------------------------------------------------- LIF


def one_neuron_net(w: float, **lif) -> SpikingNetwork:
    x = InputPopulation("X1", 1, channel=0)
    h = LIFPopulation("H", 1, params=LIFConfig(**lif))
    return SpikingNetwork([x, h], [SynapseGroup("X1->H", x, h, NoLearning(), weights=[[w]])])


def drive(net: SpikingNetwork, input_ticks, n_ticks: int, T: int = 1):
    spikes = np.zeros((n_ticks, 1), dtype=np.uint8)
    spikes[list(input_ticks), 0] = 1
    return net.run_day(spikes, T, learn=False, record=("H", 0))


def test_lif_integration_threshold_reset_refractory_and_leak():
    run = drive(one_neuron_net(0.4), range(6), n_ticks=8, T=4)
    # τ0: no input yet (1-tick delay) · τ1–2 integrate 0.4 and leak 0.05 · τ3 crosses 1.0 and resets
    # τ4–5 refractory (t_ref = 2): input discarded · τ6 integrates the τ5 input · τ7 leaks
    np.testing.assert_allclose(run.trace_u, [0.0, 0.35, 0.70, 0.0, 0.0, 0.0, 0.35, 0.30], atol=1e-12)
    assert run.pop_counts("H").tolist() == [1, 0]  # bars of T = 4 ticks
    assert run.pop_counts("X1").tolist() == [4, 2]


def test_lif_fires_at_threshold_exactly():
    run = drive(one_neuron_net(0.5, leak=0.0), [0, 1], n_ticks=3)
    np.testing.assert_allclose(run.trace_u, [0.0, 0.5, 0.0])
    assert run.pop_counts("H").tolist() == [0, 0, 1]


def test_subtractive_leak_is_floored_at_rest():
    run = drive(one_neuron_net(0.08, u_rest=0.1), [0], n_ticks=4)
    np.testing.assert_allclose(run.trace_u, [0.1, 0.13, 0.1, 0.1], atol=1e-12)


def test_exponential_leak_option():
    run = drive(one_neuron_net(0.5, leak_mode="exponential", tau_m=10.0), [0], n_ticks=3)
    decay = math.exp(-0.1)
    np.testing.assert_allclose(run.trace_u, [0.0, 0.5 * decay, 0.5 * decay**2])


# --------------------------------------------------------------------------- STDP and R-STDP (replayed spike trains)


def replay(spikes, rule_code, w0, rewards=None, T=1, gamma=1.0, dz=0.0, every_tick=True, reset=False,
           pre=(0,), post=(1,), nearest=False, w_max=1.0):
    spikes = np.asarray(spikes, dtype=np.uint8)
    n_syn = len(pre)
    w = np.full(n_syn, float(w0)) if np.isscalar(w0) else np.array(w0, dtype=float)
    n_bars = -(-spikes.shape[0] // T)
    rewards = np.ones((n_bars, 1)) if rewards is None else np.asarray(rewards, dtype=float).reshape(n_bars, 1)
    n = spikes.shape[1]
    d = math.exp(-1.0 / TAU)
    simulator.replay_learning(
        spikes, T, np.array(pre), np.array(post), w, np.full(n_syn, w_max), np.full(n_syn, rule_code),
        np.zeros(n_syn, dtype=np.int64), A, B, d, d, nearest, rewards, gamma, dz, reset, every_tick,
        np.zeros(n), np.zeros(n), np.zeros(n_syn),
    )
    return w


def pair(t_pre: int, t_post: int, n_ticks: int = 20) -> np.ndarray:
    spikes = np.zeros((n_ticks, 2), dtype=np.uint8)
    spikes[t_pre, 0] = 1
    spikes[t_post, 1] = 1
    return spikes


def test_stdp_single_pairs_match_closed_form_rule():
    for dt in range(-6, 7):  # dt = t_pre − t_post
        t_post = 8
        dw = replay(pair(t_post + dt, t_post), simulator.RULE_STDP, 0.5)[0] - 0.5
        closed = (A if dt <= 0 else B) * math.exp(-abs(dt) / TAU)
        assert dw == pytest.approx(closed, rel=1e-12), dt


def test_stdp_directions():
    assert replay(pair(3, 6), simulator.RULE_STDP, 0.5)[0] > 0.5  # pre before post: strengthened
    assert replay(pair(6, 3), simulator.RULE_STDP, 0.5)[0] < 0.5  # post before pre: weakened
    assert replay(pair(4, 4), simulator.RULE_STDP, 0.5)[0] == pytest.approx(0.5 + A)  # simultaneous: potentiation


def test_stdp_weights_stay_in_bounds():
    assert replay(pair(3, 4), simulator.RULE_STDP, 0.999)[0] == 1.0
    assert replay(pair(4, 3), simulator.RULE_STDP, 0.001)[0] == 0.0


def test_stdp_all_to_all_vs_nearest_pairing():
    spikes = np.zeros((10, 2), dtype=np.uint8)
    spikes[[1, 2], 0] = 1
    spikes[5, 1] = 1
    d = math.exp(-1 / TAU)
    all_pairs = replay(spikes, simulator.RULE_STDP, 0.5)[0] - 0.5
    nearest = replay(spikes, simulator.RULE_STDP, 0.5, nearest=True)[0] - 0.5
    assert all_pairs == pytest.approx(A * (d**3 + d**4))
    assert nearest == pytest.approx(A * d**3)


def random_trains(seed=0, n_ticks=3_000, n=6, p=0.15):
    return (np.random.default_rng(seed).random((n_ticks, n)) < p).astype(np.uint8)


ALL_PAIRS = dict(pre=(0, 0, 0, 1, 1, 1), post=(2, 3, 4, 3, 4, 5))


def test_rstdp_reduces_bit_identically_to_stdp():
    """§6.5 mandatory test: γ = 1, R ≡ 1 every tick, τ_z → 0 (E = ξ) ⇒ identical weights."""
    spikes = random_trains()
    w0 = np.random.default_rng(1).uniform(0.2, 0.6, 6)
    plain = replay(spikes, simulator.RULE_STDP, w0, **ALL_PAIRS)
    reward = replay(spikes, simulator.RULE_RSTDP, w0, gamma=1.0, dz=0.0, every_tick=True, **ALL_PAIRS)
    assert not np.array_equal(plain, w0)
    np.testing.assert_array_equal(reward, plain)


def test_rstdp_zero_reward_leaves_weights_unchanged():
    w0 = np.full(6, 0.4)
    w = replay(random_trains(), simulator.RULE_RSTDP, w0, rewards=np.zeros(3_000), dz=0.9, **ALL_PAIRS)
    np.testing.assert_array_equal(w, w0)


def test_rstdp_reward_sign_flips_the_update():
    spikes, w0 = random_trains(2), np.full(6, 0.5)
    kw = dict(dz=math.exp(-1 / 30), T=10, every_tick=False, **ALL_PAIRS)
    up = replay(spikes, simulator.RULE_RSTDP, w0, rewards=np.full(300, 0.3), **kw) - w0
    down = replay(spikes, simulator.RULE_RSTDP, w0, rewards=np.full(300, -0.3), **kw) - w0
    assert np.abs(up).max() > 0
    np.testing.assert_allclose(up, -down, rtol=1e-9, atol=1e-15)


def test_rstdp_eligibility_decays_until_the_reward_arrives():
    dz = math.exp(-1 / 30)
    w = replay(pair(2, 3, n_ticks=10), simulator.RULE_RSTDP, 0.5, rewards=[0.0, 2.0], T=5, dz=dz, every_tick=False)
    xi = A * math.exp(-1 / TAU)  # coincidence at tick 3, delivered at the end of bar 1 (tick 9)
    assert w[0] - 0.5 == pytest.approx(2.0 * xi * dz**6)


def test_rstdp_bar_end_delivery_only_at_last_tick_of_bar():
    # Coincidence in bar 0; bar 0's reward is 0, bar 1's reward is 1: only the decayed trace is credited.
    w_bar_end = replay(pair(1, 2, n_ticks=8), simulator.RULE_RSTDP, 0.5, rewards=[0.0, 1.0], T=4, dz=0.5, every_tick=False)
    xi = A * math.exp(-1 / TAU)
    assert w_bar_end[0] - 0.5 == pytest.approx(xi * 0.5**5)


# --------------------------------------------------------------------------- networks


def core(**kw) -> SNNCoreConfig:
    return SNNCoreConfig(**kw)


def test_paper_topology():
    net = NetworkBuilder(core()).paper()
    assert net.pop_names == ("X1", "X2", "H1", "H2", "Out")
    assert [g.name for g in net.groups] == ["X1->H1", "X2->H2", "H1->Out", "H2->Out"]
    assert len(net.pre) == 64 + 64 + 64 + 64
    assert set(net.rule) == {simulator.RULE_STDP}
    assert net.reward_streams == ()


def test_improved_topology_and_ablations():
    rstdp = RSTDPConfig()
    net = NetworkBuilder(core()).hawkes_rstdp(rstdp, ticks_per_bar=10)
    assert net.pop_names == ("X1", "X2", "H1", "H2", "H_mom", "H_rev", "Out")
    assert net.reward_streams == ("mom", "rev")
    assert (net.rule == simulator.RULE_RSTDP).sum() == 2 * 2 * 64  # X1, X2 → H_mom and H_rev
    out = net.offsets["Out"]
    assert (net.post == out).sum() == 4 * 64  # H1 ∪ H2 ∪ H_mom ∪ H_rev → Out
    rule = net.group("X1->H_mom").rule
    assert isinstance(rule, RewardModulatedSTDP) and rule.tau_z == 30.0 and rule.reward_stream == "mom"

    only_p1 = NetworkBuilder(core()).hawkes_rstdp(rstdp, 10, use_rstdp_pools=False)
    paper = NetworkBuilder(core()).paper()
    assert only_p1.pop_names == paper.pop_names
    assert [g.name for g in only_p1.groups] == [g.name for g in paper.groups]
    no_dir = NetworkBuilder(core()).hawkes_rstdp(rstdp, 10, keep_direction_pools=False)
    assert no_dir.pop_names == ("X1", "X2", "H_mom", "H_rev", "Out")


def test_network_rejects_mixed_stdp_constants():
    x = InputPopulation("X1", 1)
    h = LIFPopulation("H", 2)
    groups = [
        SynapseGroup("a", x, h, PairwiseSTDP(A=0.01)),
        SynapseGroup("b", x, h, PairwiseSTDP(A=0.02)),
    ]
    with pytest.raises(ValueError, match="share"):
        SpikingNetwork([x, h], groups)


def trained_net(seed=0, **kw):
    net = NetworkBuilder(core(**kw)).paper()
    net.init_weights(0.2, 0.6, np.random.default_rng(seed))
    return net


def input_trains(n_bars=2_000, T=10, seed=3):
    return PoissonEncoder().encode(np.full((n_bars, 2), 0.2), T, np.random.default_rng(seed))


def test_learning_flag_controls_weight_changes():
    net, spikes = trained_net(), input_trains()
    w0 = net.flat_weights()
    net.run_day(spikes, 10, learn=False)
    np.testing.assert_array_equal(net.flat_weights(), w0)
    net.reset_state()
    net.run_day(spikes, 10, learn=True)
    assert not np.array_equal(net.flat_weights(), w0)
    assert ((net.flat_weights() >= 0) & (net.flat_weights() <= 1)).all()


def test_state_carries_over_between_calls_until_reset():
    spikes = input_trains()
    whole = trained_net().run_day(spikes, 10, learn=True).counts
    net = trained_net()
    first = net.run_day(spikes[:7_000], 10, learn=True).counts
    second = net.run_day(spikes[7_000:], 10, learn=True).counts
    np.testing.assert_array_equal(np.vstack([first, second]), whole)


def test_simulation_is_causal_in_its_inputs():
    spikes = input_trains()
    base = trained_net().run_day(spikes, 10, learn=True).counts
    changed = spikes.copy()
    changed[10_000:] = 1 - changed[10_000:]  # alter every tick after bar 999
    out = trained_net().run_day(changed, 10, learn=True).counts
    np.testing.assert_array_equal(out[:1_000], base[:1_000])
    assert not np.array_equal(out[1_000:], base[1_000:])


def test_rstdp_network_needs_rewards_when_learning():
    net = NetworkBuilder(core()).hawkes_rstdp(RSTDPConfig(), ticks_per_bar=10)
    net.init_weights(0.2, 0.6, np.random.default_rng(0))
    spikes = input_trains(n_bars=100)
    with pytest.raises(ValueError, match="rewards"):
        net.run_day(spikes, 10, learn=True)
    net.run_day(spikes, 10, learn=False)  # test days ignore rewards
    zero = {"mom": np.zeros(100), "rev": np.zeros(100)}
    w0 = net.flat_weights()
    net.run_day(spikes, 10, learn=True, rewards=zero)
    w1 = net.flat_weights()
    rstdp = net.rule == simulator.RULE_RSTDP
    np.testing.assert_array_equal(w1[rstdp], w0[rstdp])  # zero reward: R-STDP synapses unchanged
    assert not np.array_equal(w1[~rstdp], w0[~rstdp])  # plain STDP synapses still learn
