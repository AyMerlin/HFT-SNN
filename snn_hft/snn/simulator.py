"""numba kernels for the discrete-time SNN (§6.1, §6.5).

The network is compiled to flat arrays: neurons are indexed 0..N-1, synapses 0..S-1
with `pre[k] -> post[k]`. One call simulates a whole day tick by tick. The update
rules are written once here and shared by both signal models:

- `lif_tick`      — the paper-literal LIF update (§6.1);
- `stdp_xi`       — the Hebbian timing term ξ, used by PairwiseSTDP *and* R-STDP;
- `learning_tick` — applies ξ directly (rule 1) or through an eligibility trace and
                    a reward (rule 2, MSTDPET), then updates the spike traces.

Rule codes: 0 = no learning, 1 = pairwise STDP, 2 = reward-modulated STDP.
"""

from __future__ import annotations

import numpy as np
from numba import njit

RULE_NONE = 0
RULE_STDP = 1
RULE_RSTDP = 2


@njit(cache=True, inline="always")
def stdp_xi(A, B, x_pre_decayed, s_pre, y_post_decayed, s_post):
    """ξ_ij(τ) = A·(x_i⁻ + s_i[τ])·s_j[τ] + B·y_j⁻·s_i[τ]   (A > 0, B < 0).

    x_i⁻ / y_j⁻ are the pre / post traces decayed to tick τ, before adding tick τ's
    spikes. Simultaneous pre and post spikes potentiate (the paper's t_pre − t_post ≤ 0).
    """
    return A * (x_pre_decayed + s_pre) * s_post + B * y_post_decayed * s_pre


@njit(cache=True)
def lif_tick(current, s, u, ref, is_lif, u_rest, thr, leak, use_exp, exp_decay, t_ref):
    """One tick of every LIF neuron; writes spikes into `s` (input neurons are left alone)."""
    for j in range(u.shape[0]):
        if not is_lif[j]:
            continue
        if ref[j] > 0:  # refractory: input is discarded
            ref[j] -= 1
            u[j] = u_rest[j]
            s[j] = 0
            continue
        u[j] += current[j]
        if u[j] >= thr[j]:
            s[j] = 1
            u[j] = u_rest[j]
            ref[j] = t_ref[j]
        else:
            s[j] = 0
            if use_exp[j]:
                u[j] = u_rest[j] + (u[j] - u_rest[j]) * exp_decay[j]
            else:  # "decreases by a small amount", floored at rest
                u[j] = max(u[j] - leak[j], u_rest[j])


@njit(cache=True)
def learning_tick(
    s, x, y, xd, yd, pre, post, w, w_max, rule, stream, E,
    A, B, dplus, dminus, nearest, gamma, dz, reset_elig, deliver, reward_row,
):
    """Plasticity for one tick, after the neurons have spiked (§6.1, §6.5)."""
    for j in range(x.shape[0]):
        xd[j] = x[j] * dplus
        yd[j] = y[j] * dminus
    for k in range(pre.shape[0]):
        r = rule[k]
        if r == RULE_NONE:
            continue
        sp = s[pre[k]]
        so = s[post[k]]
        xi = 0.0
        if sp != 0 or so != 0:
            xi = stdp_xi(A, B, xd[pre[k]], float(sp), yd[post[k]], float(so))
        if r == RULE_STDP:
            if xi != 0.0:
                w[k] = min(max(w[k] + xi, 0.0), w_max[k])
        else:  # RULE_RSTDP: E(τ) = E(τ−1)·exp(−1/τ_z) + ξ(τ); w += γ·R·E at delivery ticks
            E[k] = E[k] * dz + xi
            if deliver:
                reward = reward_row[stream[k]]
                if reward != 0.0:
                    w[k] = min(max(w[k] + gamma * reward * E[k], 0.0), w_max[k])
                if reset_elig:
                    E[k] = 0.0
    for j in range(x.shape[0]):
        if nearest:
            x[j] = 1.0 if s[j] else xd[j]
            y[j] = 1.0 if s[j] else yd[j]
        else:  # all-to-all: traces sum over all past spikes
            x[j] = xd[j] + s[j]
            y[j] = yd[j] + s[j]


@njit(cache=True)
def simulate_day(
    input_spikes, input_neuron, ticks_per_bar,
    is_lif, u_rest, thr, leak, use_exp, exp_decay, t_ref,
    pre, post, w, w_max, rule, stream,
    A, B, dplus, dminus, nearest, delay,
    learn, rewards, gamma, dz, reset_elig, every_tick,
    pop_of, n_pops,
    u, ref, x, y, E, hist,
    record_neuron,
):
    """Simulate one day. State arrays (u, ref, x, y, E, hist) and weights `w` are updated in place.

    Returns per-bar spike counts per population, shape (n_bars, n_pops), and the membrane
    potential of `record_neuron` per tick (empty if record_neuron < 0).
    """
    n_ticks = input_spikes.shape[0]
    n = u.shape[0]
    n_bars = (n_ticks + ticks_per_bar - 1) // ticks_per_bar
    counts = np.zeros((n_bars, n_pops), dtype=np.int32)
    trace_u = np.empty(n_ticks if record_neuron >= 0 else 0, dtype=np.float64)
    current = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.uint8)
    xd = np.zeros(n, dtype=np.float64)
    yd = np.zeros(n, dtype=np.float64)
    slots = delay + 1
    for tau in range(n_ticks):
        b = tau // ticks_per_bar
        # Spikes emitted at τ − delay arrive now (delay ≥ 1, so update order does not matter).
        prev = hist[(tau + slots - delay) % slots]
        current[:] = 0.0
        for k in range(pre.shape[0]):
            if prev[pre[k]]:
                current[post[k]] += w[k]
        for i in range(input_neuron.shape[0]):
            s[input_neuron[i]] = input_spikes[tau, i]
        lif_tick(current, s, u, ref, is_lif, u_rest, thr, leak, use_exp, exp_decay, t_ref)
        if record_neuron >= 0:
            trace_u[tau] = u[record_neuron]
        if learn:
            deliver = every_tick or (tau % ticks_per_bar == ticks_per_bar - 1)
            learning_tick(
                s, x, y, xd, yd, pre, post, w, w_max, rule, stream, E,
                A, B, dplus, dminus, nearest, gamma, dz, reset_elig, deliver, rewards[b],
            )
        hist[tau % slots, :] = s
        for j in range(n):
            if s[j]:
                counts[b, pop_of[j]] += 1
    return counts, trace_u


@njit(cache=True)
def replay_learning(
    spikes, ticks_per_bar, pre, post, w, w_max, rule, stream,
    A, B, dplus, dminus, nearest, rewards, gamma, dz, reset_elig, every_tick, x, y, E,
):
    """Apply the learning rules to given spike trains (n_ticks, N), without neuron dynamics.

    Uses the same `learning_tick` as `simulate_day`; for unit tests of STDP and R-STDP.
    """
    n = spikes.shape[1]
    s = np.zeros(n, dtype=np.uint8)
    xd = np.zeros(n, dtype=np.float64)
    yd = np.zeros(n, dtype=np.float64)
    for tau in range(spikes.shape[0]):
        s[:] = spikes[tau]
        deliver = every_tick or (tau % ticks_per_bar == ticks_per_bar - 1)
        learning_tick(
            s, x, y, xd, yd, pre, post, w, w_max, rule, stream, E,
            A, B, dplus, dminus, nearest, gamma, dz, reset_elig, deliver, rewards[tau // ticks_per_bar],
        )
