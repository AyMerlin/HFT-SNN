"""Time preprocessing, encoding and simulation of the SNN engine on real days (M3).

    python -m experiments.profile_snn --config experiments/configs/base.yaml \
        --pair 2025-10-19 2025-10-20 --pair 2025-11-20 2025-11-21

Each pair is (training day, test day). Both the paper network and the improved
network topology are timed; the improved network gets random rewards, since the
Hawkes rewards arrive in M7. Uses default hyperparameters (§12).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from snn_hft.config.loader import load_raw
from snn_hft.config.schema import BarsConfig, DataConfig, PaperSNNConfig, RSTDPConfig
from snn_hft.data.store import make_store
from snn_hft.preprocessing.pipelines import paper_pipeline
from snn_hft.snn.encoding import PoissonEncoder
from snn_hft.snn.network import NetworkBuilder, SpikingNetwork


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0


def warm_up(builder: NetworkBuilder) -> None:
    """Trigger numba compilation so it is not counted in the timings."""
    spikes = PoissonEncoder().encode(np.full((50, 2), 0.2), 10, np.random.default_rng(0))
    for net in (builder.paper(), builder.hawkes_rstdp(RSTDPConfig(), 10)):
        net.init_weights(0.2, 0.6, np.random.default_rng(0))
        rewards = {s: np.zeros(50) for s in net.reward_streams}
        net.run_day(spikes, 10, learn=True, rewards=rewards)
        net.run_day(spikes, 10, learn=False)


def describe(name: str, run, n_bars: int) -> str:
    out_rate = (run.pop_counts("Out") > 0).mean()
    hidden = [p for p in run.pop_names if p.startswith("H")]
    per_pool = ", ".join(f"{p} {run.pop_counts(p).sum() / n_bars / 64:.2f}" for p in hidden)
    return f"    {name}: output signal in {out_rate:6.1%} of bars; spikes per neuron per bar: {per_pool}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pair", nargs=2, action="append", type=date.fromisoformat, required=True, metavar=("TRAIN", "TEST"))
    args = parser.parse_args(argv)

    raw = load_raw(args.config)
    store = make_store(DataConfig.model_validate(raw.get("data", {})))
    symbol = DataConfig.model_validate(raw.get("data", {})).symbol
    signal = PaperSNNConfig()
    T = signal.input.ticks_per_bar
    builder = NetworkBuilder(signal.core)
    _, t_jit = timed(warm_up, builder)
    print(f"numba warm-up (compile or load cache): {t_jit:.1f}s")

    for train_day, test_day in args.pair:
        (train_raw, test_raw), t_load = timed(lambda: [store.day_data(symbol, d) for d in (train_day, test_day)])
        pipe = paper_pipeline(BarsConfig(), signal)
        (train,), t_fit = timed(pipe.fit_transform, [train_raw])
        test, t_tr = timed(pipe.transform, test_raw)
        rng = np.random.default_rng(0)
        train_spikes, t_enc = timed(PoissonEncoder().encode, train.channel_prob, T, rng)
        test_spikes = PoissonEncoder().encode(test.channel_prob, T, rng)
        print(
            f"\ntrain {train_day} ({train.n_bars:,} bars, {train.n_bars * T / 1e6:.2f}M ticks) → "
            f"test {test_day} ({test.n_bars:,} bars, {test.n_bars * T / 1e6:.2f}M ticks)\n"
            f"  load {t_load:.2f}s · preprocess {t_fit + t_tr:.2f}s · encode train day {t_enc:.2f}s"
        )
        nets: dict[str, SpikingNetwork] = {
            "paper": builder.paper(),
            "improved topology": builder.hawkes_rstdp(RSTDPConfig(), T),
        }
        for name, net in nets.items():
            net.init_weights(0.2, 0.6, np.random.default_rng(1))
            rewards = {s: np.random.default_rng(2).uniform(-1, 1, train.n_bars) for s in net.reward_streams}
            train_run, t_train = timed(net.run_day, train_spikes, T, True, rewards)
            net.reset_state()
            test_run, t_test = timed(net.run_day, test_spikes, T, False)
            print(
                f"  {name:<18} train {t_train:5.2f}s ({train.n_bars * T / t_train / 1e6:.1f}M ticks/s) · "
                f"test {t_test:5.2f}s ({test.n_bars * T / t_test / 1e6:.1f}M ticks/s)"
            )
            print(describe("train", train_run, train.n_bars))
            print(describe("test ", test_run, test.n_bars))
    return 0


if __name__ == "__main__":
    sys.exit(main())
