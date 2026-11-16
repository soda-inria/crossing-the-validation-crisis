"""Torch tensors as inputs of RedundancyMonitor (skipped without torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cv_redundancy import RedundancyMonitor  # noqa: E402


def test_tensors_give_the_numpy_result():
    rng = np.random.default_rng(0)
    n = 100
    effect = rng.normal(size=n)
    tests = [rng.permutation(n)[:20] for _ in range(6)]
    losses = [(effect[idx] + rng.normal(size=20)) ** 2 for idx in tests]
    reference = RedundancyMonitor.from_splits(n, tests, losses)
    monitor = RedundancyMonitor(n)
    for idx, loss in zip(tests, losses):
        loss = torch.tensor(loss, requires_grad=True) * 1.0  # a loss still attached to the graph
        monitor.add_split(torch.as_tensor(idx), loss)
    assert monitor.redundancy == reference.redundancy and monitor.n_pairs == reference.n_pairs
    # float32 and bfloat16 losses, (m, 1) shapes, boolean masks
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.as_tensor(tests[0])] = True
    other = RedundancyMonitor(n)
    other.add_split(mask, torch.tensor(losses[0][np.argsort(tests[0])], dtype=torch.float32)[:, None])
    other.add_split(torch.as_tensor(tests[1]), torch.tensor(losses[1], dtype=torch.bfloat16))
    assert other.n_splits == 2 and other.test_fraction == pytest.approx(0.2)
