"""Stop a Monte Carlo cross-validation with the redundancy, in a PyTorch training loop.

A simulated 3-class problem (600 observations, 20 features). Each split of a ShuffleSplit (test fraction 0.2, budget
40 splits) trains a small MLP on the training part, then scores the test part with the per-sample cross-entropy
(reduction="none"), the per-sample loss of the log-loss; the monitor stops the run once the splits left in the budget
would add less than 10% of gain on the reference curve. For accuracy, feed it the 0-1 errors (predicted class !=
label) instead. Runs in seconds on a CPU, from the repository root:

    python cv_redundancy/examples/torch_example.py
"""
import os
import sys

import torch
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# the repository root, so that the example runs from a clone without installing anything
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cv_redundancy import RedundancyMonitor  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)  # a model this small trains fastest on one CPU thread
n, d, n_classes = 600, 20, 3
X = torch.randn(n, d)
y = torch.distributions.Categorical(logits=X @ torch.randn(d, n_classes)).sample()


def train(X_train, y_train, epochs=15):
    model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n_classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()
    return model


budget = 40
monitor = RedundancyMonitor(n_samples=n, max_splits=budget)  # tol 0.1, min_splits 3 by default
splitter = ShuffleSplit(n_splits=budget, test_size=0.2, random_state=0)
for k, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    train_idx, test_idx = torch.as_tensor(train_idx), torch.as_tensor(test_idx)
    model = train(X[train_idx], y[train_idx])
    model.eval()
    with torch.no_grad():
        losses = F.cross_entropy(model(X[test_idx]), y[test_idx], reduction="none")
    monitor.add_split(test_idx, losses)  # tensors are accepted as they are
    status = (f"redundancy {monitor.redundancy:.3f}, the splits left would add {100 * monitor.remaining_gain():.1f}%"
              " of gain" if monitor.is_defined else "redundancy not defined yet")
    print(f"split {k:2d}: test log-loss {losses.mean():.3f}, {status}")
    if monitor.should_stop():
        break

print(f"\nstopped after {monitor.n_splits} of {budget} splits")
print(monitor)
