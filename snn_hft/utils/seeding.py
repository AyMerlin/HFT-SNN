"""Named random streams, so every draw is reproducible from (seed, purpose, day, ...)."""

from __future__ import annotations

from datetime import date

import numpy as np

PURPOSE = {"init": 1, "train": 2, "test": 3, "naive": 4, "hawkes": 5}


def rng_for(seed: int, purpose: str, *keys: int | date) -> np.random.Generator:
    """Independent generator for one purpose, e.g. ``rng_for(seed, "train", day, epoch)``."""
    parts = [int(seed), PURPOSE[purpose]]
    parts += [k.toordinal() if isinstance(k, date) else int(k) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(parts))
