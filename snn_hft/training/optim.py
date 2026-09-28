"""Adam in the Keras / TensorFlow form (DeepLOB §V-A trains with Keras, ε = 1).

Keras and TensorFlow apply ε outside the bias correction:

    lr_t = lr · √(1 − β₂ᵗ) / (1 − β₁ᵗ)
    m ← β₁ m + (1 − β₁) g,   v ← β₂ v + (1 − β₂) g²
    θ ← θ − lr_t · m / (√v + ε)

which is Kingma & Ba's algorithm with "ε̂" in place of ε. `torch.optim.Adam` adds ε to
√v̂ instead; the two agree for tiny ε but not for the paper's ε = 1, where PyTorch's form
takes much larger early steps.
"""

from __future__ import annotations

import math

import torch


class KerasAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-7):
        if lr <= 0 or eps < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError(f"invalid Adam settings lr={lr} betas={betas} eps={eps}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                lr_t = group["lr"] * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-lr_t)
        return loss
