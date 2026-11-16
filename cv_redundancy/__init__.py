"""cv_redundancy: when are more cross-validation splits worth their cost?

Monte Carlo cross-validation (repeated random train/test splits, as scikit-learn's ShuffleSplit) averages the test
scores of K splits, and each split costs one more fit. How much precision a new split buys depends on how
differently the split predictors err. The *redundancy* reads this from the first few splits of the run, on the data
at hand only: on the observations that two splits both hold out, it correlates the per-sample losses of the two
split predictors, averages over the pairs of splits, and multiplies by the test fraction t. When the predictors err
alike (a redundancy near its maximum t), more splits mostly repeat what the first ones measured; when they err
independently (a redundancy near 0), every split still adds about as much as the first.

Section 5 of the paper (Eve, Varoquaux and Moreau, "Crossing the Validation Crisis: Cross-Validation Reduces
Benchmarking Variance Surprisingly Well", NeurIPS 2026) uses the redundancy to decide when to stop splitting, and
Appendix D.2 defines it and relates it to the sample gain G_K of K splits (how many times larger the test set of a
single split would need to be to match the variance of K-split cross-validation): G_K ~ K / (1 + (K - 1) omega), up
to a train/test coupling that the study set cannot estimate. It is a triage signal, not an estimator of the gain.

Contents:
    RedundancyMonitor               the statistic, updated split by split, and the stopping rule (any training loop)
    projected_gain, stopping_split  the reference curve, and where the rule stops for a fixed redundancy
    cross_validate_until_redundant  scikit-learn cross-validation that stops by itself
    redundancy_from_cv_results      the monitor of a finished sklearn.model_selection.cross_validate run
    squared_error, zero_one, log_loss, brier, LOSSES   per-sample losses

The core needs NumPy only; scikit-learn and torch are optional. See README.md in this folder.
"""
from .core import DEFAULT_MIN_SPLITS, DEFAULT_TOL, RedundancyMonitor, projected_gain, stopping_split
from .losses import LOSSES, brier, log_loss, squared_error, zero_one
from .sklearn_cv import cross_validate_until_redundant, redundancy_from_cv_results

__all__ = [
    "DEFAULT_MIN_SPLITS", "DEFAULT_TOL", "RedundancyMonitor", "projected_gain", "stopping_split",
    "LOSSES", "brier", "log_loss", "squared_error", "zero_one",
    "cross_validate_until_redundant", "redundancy_from_cv_results",
]
