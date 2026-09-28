"""Windows, metrics, Keras-form Adam and the trainer (plan §11 "Evaluation", DeepLOB §V-A)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from torch import nn

import snn_hft.data.fi2010.format as fmt
from snn_hft.evaluation.classification import classification_metrics, mean_over_folds
from snn_hft.training.dataset import WindowDataset
from snn_hft.training.optim import KerasAdam
from snn_hft.training.trainer import TrainConfig, Trainer

# --------------------------------------------------------------------------- windows


def authors_windows(X, Y, T):
    """`data_classification` of the authors' notebook."""
    N = len(X)
    return np.stack([X[i - T : i] for i in range(T, N + 1)]), Y[T - 1 : N]


def test_windows_match_the_authors_indexing():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(30, 4)).astype(np.float32), rng.integers(0, 3, 30)
    ds = WindowDataset(X, Y, T=5)
    x, y = ds.batch(np.arange(len(ds)))
    ax, ay = authors_windows(X, Y, 5)
    assert x.shape == (26, 1, 5, 4) and len(ds) == 26
    np.testing.assert_array_equal(x[:, 0].numpy(), ax)
    np.testing.assert_array_equal(y.numpy(), ay)
    np.testing.assert_array_equal(ds.y, ay)


def test_windows_can_stay_inside_segments():
    X, Y = np.zeros((20, 2), np.float32), np.arange(20) % 3
    segs = [fmt.Segment(1, 1, 0, 8), fmt.Segment(2, 1, 8, 10), fmt.Segment(3, 1, 10, 20)]
    ds = WindowDataset(X, Y, T=4, segments=segs, cross_segments=False)
    assert list(ds.end) == [3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19]  # segment 2 is shorter than T


def test_batches_cover_every_sample_once_with_seeded_shuffle():
    ds = WindowDataset(np.zeros((107, 1), np.float32), np.arange(107) % 3, T=8)
    run = lambda seed: torch.cat([y for _, y in ds.batches(32, shuffle=True, generator=np.random.default_rng(seed))])
    a, b, c = run(1), run(1), run(2)
    assert len(a) == 100 and sorted(a.tolist()) == sorted(ds.y.tolist())
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert [len(y) for _, y in ds.batches(32)] == [32, 32, 32, 4]  # last batch smaller, as in Keras


# --------------------------------------------------------------------------- metrics


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_scikit_learn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 3, 500))
    if seed == 2:
        p[p == 2] = 1  # a class that is never predicted
    m = classification_metrics(y, p)
    kw = dict(labels=[0, 1, 2], zero_division=0)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
    for avg in ("weighted", "macro"):
        assert m[f"precision_{avg}"] == pytest.approx(precision_score(y, p, average=avg, **kw))
        assert m[f"recall_{avg}"] == pytest.approx(recall_score(y, p, average=avg, **kw))
        assert m[f"f1_{avg}"] == pytest.approx(f1_score(y, p, average=avg, **kw))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, p))
    assert m["confusion"] == confusion_matrix(y, p, labels=[0, 1, 2]).tolist()
    assert m["recall_weighted"] == pytest.approx(m["accuracy"])  # why the paper's recall equals accuracy


def test_mean_over_folds_is_unweighted():
    assert mean_over_folds([{"accuracy": 0.5}, {"accuracy": 1.0}], keys=["accuracy"]) == {"accuracy": 0.75}


# --------------------------------------------------------------------------- optimizer


def test_keras_adam_follows_the_keras_update_and_differs_from_torch_adam():
    g_seq = [0.5, -1.0, 2.0]
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = KerasAdam([p], lr=0.01, eps=1.0)
    m = v = 0.0
    x = 1.0
    for t, g in enumerate(g_seq, start=1):
        p.grad = torch.tensor([g])
        opt.step()
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x -= 0.01 * math.sqrt(1 - 0.999**t) / (1 - 0.9**t) * m / (math.sqrt(v) + 1.0)
        assert p.item() == pytest.approx(x, rel=1e-6)
    q = torch.nn.Parameter(torch.tensor([1.0]))
    topt = torch.optim.Adam([q], lr=0.01, eps=1.0)
    for g in g_seq:
        q.grad = torch.tensor([g])
        topt.step()
    assert abs(q.item() - p.item()) > 1e-3  # eps placement matters at eps = 1


# --------------------------------------------------------------------------- trainer


class Last(nn.Module):
    """Linear read-out of the last row of the window."""

    def __init__(self, f=4):
        super().__init__()
        self.lin = nn.Linear(f, 3)

    def forward(self, x):
        return self.lin(x[:, 0, -1, :])


def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    Y = np.digitize(X[:, 0], [-0.4, 0.4])  # learnable from the last row
    return WindowDataset(X[: n // 2], Y[: n // 2], T=3), WindowDataset(X[n // 2 :], Y[n // 2 :], T=3)


def quiet(msg):
    pass


def test_training_learns_and_keeps_the_best_weights():
    train, val = toy()
    torch.manual_seed(0)
    model = Last()
    res = Trainer(TrainConfig(max_epochs=15, patience=20, lr=0.05, eps=1e-7), log=quiet).fit(model, train, val, seed=0)
    h = res.history
    assert h[-1].train_loss < h[0].train_loss and max(r.val_accuracy for r in h) > 0.8
    assert res.best_score == max(r.val_accuracy for r in h) and h[res.best_epoch].val_accuracy == res.best_score
    model.load_state_dict(res.best_state)
    _, m, proba = Trainer(log=quiet).evaluate(model, val)
    assert m["accuracy"] == pytest.approx(res.best_score) and np.allclose(proba.sum(axis=1), 1)


def test_early_stopping_after_patience_epochs_without_improvement():
    train, val = toy()
    res = Trainer(TrainConfig(max_epochs=50, patience=4, lr=1e-12), log=quiet).fit(Last(), train, val, seed=0)
    assert res.stopped_early and res.best_epoch == 0 and len(res.history) == 5


def test_resumed_training_equals_uninterrupted_training(tmp_path):
    train, val = toy()
    cfg = TrainConfig(max_epochs=6, patience=20, lr=0.05, eps=1e-7)
    torch.manual_seed(1)
    ref_model = Last()
    init = {k: v.clone() for k, v in ref_model.state_dict().items()}
    ref = Trainer(cfg, log=quiet).fit(ref_model, train, val, seed=3)

    ck = tmp_path / "ck.pt"
    m1 = Last()
    m1.load_state_dict(init)
    Trainer(TrainConfig(max_epochs=2, patience=20, lr=0.05, eps=1e-7), log=quiet).fit(m1, train, val, seed=3, checkpoint_path=ck)
    m2 = Last()  # fresh object, state comes from the checkpoint
    res = Trainer(cfg, log=quiet).fit(m2, train, val, seed=3, checkpoint_path=ck)
    assert res.resumed_from_epoch == 2
    assert [r.val_accuracy for r in res.history] == [r.val_accuracy for r in ref.history]
    for k in init:
        assert torch.equal(res.last_state[k], ref.last_state[k])
