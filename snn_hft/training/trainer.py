"""Supervised training with early stopping (DeepLOB §V-A) and resumable checkpoints.

Early stopping follows Keras `EarlyStopping(monitor="val_accuracy", patience=20)`: an epoch
improves only if its score is strictly higher than the best so far, and training stops once
`patience` epochs in a row did not improve. The weights of the best epoch and of the last
epoch are both kept (the paper does not say which it evaluated; see DESIGN_DECISIONS_LOB P3).

With `checkpoint_path`, the complete state (weights, optimizer, early-stopping counters,
history and random-number states) is saved after every epoch, and `fit` resumes from it, so
an interrupted run continues exactly as if it had not been interrupted.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from snn_hft.evaluation.classification import classification_metrics
from snn_hft.training.dataset import WindowDataset
from snn_hft.training.optim import KerasAdam


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    lr: float = 0.01
    eps: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    optimizer: Literal["keras_adam", "adam"] = "keras_adam"
    monitor: Literal["val_accuracy", "val_f1_macro", "val_f1_weighted"] = "val_accuracy"
    patience: int = 20
    max_epochs: int = 300
    grad_clip: float | None = None  # max gradient norm; None = no clipping (paper)
    eval_batch_size: int = 1024


@dataclass
class EpochRecord:
    epoch: int  # 0-based
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_f1_macro: float
    val_f1_weighted: float
    seconds: float


@dataclass
class FitResult:
    history: list[EpochRecord]
    best_epoch: int
    best_score: float
    stopped_early: bool
    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    seconds: float = 0.0
    resumed_from_epoch: int | None = None

    def history_rows(self) -> list[dict]:
        return [asdict(r) for r in self.history]


@dataclass
class _Progress:
    epoch: int = 0  # next epoch to run
    best_score: float = -np.inf
    best_epoch: int = -1
    wait: int = 0
    history: list[EpochRecord] = field(default_factory=list)
    seconds: float = 0.0


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class Trainer:
    def __init__(self, config: TrainConfig = TrainConfig(), log: Callable[[str], None] = print):
        self.config = config
        self.log = log

    def make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        c = self.config
        if c.optimizer == "keras_adam":
            return KerasAdam(model.parameters(), lr=c.lr, betas=c.betas, eps=c.eps)
        return torch.optim.Adam(model.parameters(), lr=c.lr, betas=c.betas, eps=c.eps)

    # ------------------------------------------------------------------ evaluation

    @torch.no_grad()
    def predict_proba(self, model: nn.Module, data: WindowDataset) -> np.ndarray:
        model.eval()
        out = [torch.softmax(model(x), dim=1) for x, _ in data.batches(self.config.eval_batch_size)]
        return torch.cat(out).numpy() if out else np.empty((0, 3), np.float32)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, data: WindowDataset) -> tuple[float, dict, np.ndarray]:
        """(mean cross-entropy, metrics, class probabilities) on a dataset."""
        model.eval()
        losses, probs = [], []
        for x, y in data.batches(self.config.eval_batch_size):
            logits = model(x)
            losses.append(nn.functional.cross_entropy(logits, y, reduction="sum").item())
            probs.append(torch.softmax(logits, dim=1))
        p = torch.cat(probs).numpy()
        return sum(losses) / len(data), classification_metrics(data.y, p.argmax(axis=1)), p

    # ------------------------------------------------------------------ training

    def fit(
        self,
        model: nn.Module,
        train: WindowDataset,
        val: WindowDataset,
        seed: int,
        checkpoint_path: str | Path | None = None,
    ) -> FitResult:
        c = self.config
        opt = self.make_optimizer(model)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        prog = _Progress()
        best_state = _state_copy(model)
        resumed = None
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ck = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            best_state = ck["best_state"]
            prog = ck["progress"]
            rng.bit_generator.state = ck["numpy_rng"]
            torch.set_rng_state(ck["torch_rng"])
            resumed = prog.epoch
            self.log(f"resumed at epoch {prog.epoch} (best {prog.best_score:.4f} at epoch {prog.best_epoch})")

        loss_fn = nn.CrossEntropyLoss()
        stopped = prog.wait >= c.patience
        while not stopped and prog.epoch < c.max_epochs:
            t0 = time.time()
            model.train()
            tot_loss = tot_correct = n = 0
            for x, y in train.batches(c.batch_size, shuffle=True, generator=rng):
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                if c.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
                opt.step()
                tot_loss += loss.item() * len(y)
                tot_correct += (logits.argmax(1) == y).sum().item()
                n += len(y)
            val_loss, vm, _ = self.evaluate(model, val)
            rec = EpochRecord(prog.epoch, tot_loss / n, tot_correct / n, val_loss, vm["accuracy"],
                              vm["f1_macro"], vm["f1_weighted"], time.time() - t0)
            prog.history.append(rec)
            prog.seconds += rec.seconds
            score = {"val_accuracy": rec.val_accuracy, "val_f1_macro": rec.val_f1_macro,
                     "val_f1_weighted": rec.val_f1_weighted}[c.monitor]
            if score > prog.best_score:
                prog.best_score, prog.best_epoch, prog.wait = score, prog.epoch, 0
                best_state = _state_copy(model)
            else:
                prog.wait += 1
            stopped = prog.wait >= c.patience
            self.log(f"epoch {prog.epoch:3d}  loss {rec.train_loss:.4f}  acc {rec.train_accuracy:.4f}  "
                     f"val_loss {rec.val_loss:.4f}  val_acc {rec.val_accuracy:.4f}  "
                     f"val_f1m {rec.val_f1_macro:.4f}  {'*' if prog.wait == 0 else ' '}  {rec.seconds:.0f}s")
            prog.epoch += 1
            if checkpoint_path is not None:
                self._save(checkpoint_path, model, opt, best_state, prog, rng)
        return FitResult(prog.history, prog.best_epoch, prog.best_score, stopped, best_state,
                         _state_copy(model), prog.seconds, resumed)

    @staticmethod
    def _save(path, model, opt, best_state, prog, rng) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "best_state": best_state,
                    "progress": copy.deepcopy(prog), "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, path)
