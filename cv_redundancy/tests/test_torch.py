"""Torch tensors as inputs of RedundancyMonitor (skipped without torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cv_redundancy import RedundancyMonitor  # noqa: E402


def _run(scale=1.0):
    rng = np.random.default_rng(0)
    n = 100
    effect = scale * rng.normal(size=n)
    tests = [rng.permutation(n)[:20] for _ in range(6)]
    return n, tests, [(effect[idx] + rng.normal(size=20)) ** 2 for idx in tests]


class _OnDevice:
    """A stand-in for a tensor on an accelerator, as a CUDA tensor: ``numpy()`` fails until ``cpu()`` copies it."""

    def __init__(self, tensor):
        self._tensor = tensor

    def detach(self):
        return _OnDevice(self._tensor.detach())

    def cpu(self):
        return self._tensor

    def float(self):
        return _OnDevice(self._tensor.float())

    def numpy(self):
        raise TypeError("can't convert a device tensor to numpy: use Tensor.cpu() to copy it to the host first")


def test_tensors_give_the_numpy_result():
    n, tests, losses = _run()
    reference = RedundancyMonitor.from_splits(n, tests, losses)
    monitor = RedundancyMonitor(n)
    for idx, loss in zip(tests, losses):
        loss = torch.tensor(loss, requires_grad=True) * 1.0  # a loss still attached to the graph
        monitor.add_split(torch.as_tensor(idx), loss)
    assert monitor.summary() == reference.summary()
    # a stand-in for an accelerator tensor: copied to the host first
    on_device = RedundancyMonitor.from_splits(n, [_OnDevice(torch.as_tensor(idx)) for idx in tests],
                                              [_OnDevice(torch.tensor(loss, requires_grad=True) * 1.0)
                                               for loss in losses])
    assert on_device.summary() == reference.summary()


def test_low_precision_tensors_and_masks():
    n, tests, losses = _run(scale=300.0)
    assert max(loss.max() for loss in losses) > 65504  # above the float16 range: bfloat16 must not go through it
    # float32 (m, 1) losses on boolean masks, in increasing index order
    masks, as_float32 = [], []
    for idx, loss in zip(tests, losses):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.as_tensor(idx)] = True
        masks.append(mask)
        as_float32.append(torch.tensor(loss[np.argsort(idx)], dtype=torch.float32)[:, None])
    monitor = RedundancyMonitor.from_splits(n, masks, as_float32)
    reference = RedundancyMonitor.from_splits(n, [np.sort(idx) for idx in tests],
                                              [loss.numpy().astype(float).ravel() for loss in as_float32])
    assert monitor.summary() == reference.summary() and monitor.test_fraction == pytest.approx(0.2)
    # bfloat16, which NumPy lacks: the same values through float32
    as_bfloat16 = [torch.tensor(loss, dtype=torch.bfloat16) for loss in losses]
    monitor = RedundancyMonitor.from_splits(n, [torch.as_tensor(idx) for idx in tests], as_bfloat16)
    reference = RedundancyMonitor.from_splits(n, tests, [loss.float().numpy().astype(float) for loss in as_bfloat16])
    assert monitor.summary() == reference.summary() and monitor.is_defined
