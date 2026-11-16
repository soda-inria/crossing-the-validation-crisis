"""Study-only redundancy of one Monte Carlo cross-validation run, and the projection-based stopping rule.

The statistic is that of Appendix D.2 of the paper. One CV run has splits a = 1..k; split a holds out the test set
T_a (indices into the n observations the test sets are drawn from, the study set n_tr + n_te of the paper) and gives
the out-of-fold loss e_{a,i} = loss(g_a(x_i), y_i) of each i in T_a, with the per-sample loss of the metric whose
precision matters (squared error for the MSE, 0-1 error for accuracy, negative log-likelihood for the log-loss, Brier
score for the Brier score). For each pair a < b, on the observations both splits hold out, I_ab = T_a & T_b:

    cov_ab = sample covariance of e_a and e_b over I_ab (ddof = 1, as np.cov)
    var_ab = (var(e_a over I_ab) + var(e_b over I_ab)) / 2 (ddof = 1)

pairs with |I_ab| < 2, or a non-finite covariance or variance, are skipped. Then

    loss correlation    rho_hat   = mean_pairs(cov_ab) / mean_pairs(var_ab)   (0 if the mean variance is <= 0)
    redundancy          omega_hat = t * rho_hat,   t = n_te / n the test fraction

so that omega_hat lies in [-t, t]; both are undefined until a pair qualifies. The reference curve of the sample gain,
G_K = K / (1 + (K - 1) * omega) (the gain-redundancy relation of Appendix D.2 with the train/test coupling kappa set
to 0), tends to the ceiling 1 / omega as K grows.

The paper's numbers come from ``_oof_intersection_icc`` in simulated/objective.py and ``_pairwise_oof_icc`` in
pcam/pcam_deep_training_adapted.py and yelp/nlpfinetuning_aligned.py; tests/test_reference.py checks that
``RedundancyMonitor`` gives the same values. Each new split only computes its pairs with the splits before it.

Stopping rule (projection-based): after k >= min_splits splits, once the redundancy is defined, project the gain with
the reference curve, G(K) = K / (1 + (K - 1) * max(omega_hat, 0)), and stop when the splits left in the budget K_max
would add less than ``tol`` of gain: G(K_max) / G(k) - 1 < tol. The coupling kappa is intrinsic to cross-validation
and not estimable from the study set; it can take either sign but is usually >= 0, most at small training sizes. The
gain left is then smaller than projected, so the rule continues longer than strictly needed rather than stopping too
early. This holds for a given redundancy: a reading above the population value, through noise, makes the rule stop
earlier, most visibly with large budgets. The default tol = 0.1: the test-side variance of the CV estimate is about
that of a single split divided by G (single-split variance scaling as 1 / n_te), so the splits left would shrink that
part of the standard error by less than 5% (1 - 1 / sqrt(1.1)), and the total standard error, which also carries the
variation of the study set that no number of splits averages out, by even less.
"""
import math
import os
import sys
import warnings

import numpy as np

DEFAULT_TOL = 0.1
DEFAULT_MIN_SPLITS = 3

# Two Monte Carlo test sets of sizes m_a and m_b out of n are disjoint with the hypergeometric probability
# p_ab = C(n - m_a, m_b) / C(n, m_b). These events are pairwise independent, so the number of disjoint pairs is close
# to a Poisson count of mean sum p_ab. A count far in its upper tail points to K-fold-like test sets (disjoint within
# a repeat). In simulated 200-split Monte Carlo runs (n = 10 to 1,000) this level never fired; it flags one repeat of
# 5-fold after 3 to 5 splits from n = 50 on.
_LOG_DISJOINT_LEVEL = math.log(1e-6)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _warn(message):
    """UserWarning attributed to the first caller outside the modules of this package."""
    level, frame = 1, sys._getframe(1)
    while frame is not None and os.path.dirname(os.path.abspath(frame.f_code.co_filename)) == _PACKAGE_DIR:
        level, frame = level + 1, frame.f_back
    warnings.warn(message, UserWarning, stacklevel=level + 1)


def _as_numpy(x):
    """NumPy array from a list, an array, or a tensor (any object with ``.detach().cpu().numpy()``, as in torch)."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
        try:
            x = x.numpy()
        except TypeError:  # dtypes NumPy lacks, such as bfloat16
            x = x.float().numpy()
    return np.asarray(x)


def _p_disjoint(n, m_a, m_b):
    """Probability that two Monte Carlo test sets of sizes m_a and m_b out of n share no observation."""
    if m_a + m_b > n:
        return 0.0
    return math.exp(math.lgamma(n - m_a + 1) + math.lgamma(n - m_b + 1) - math.lgamma(n + 1)
                    - math.lgamma(n - m_a - m_b + 1))


def _log_poisson_tail(count, mean):
    """Upper bound on log P(X >= count), X ~ Poisson(mean): the first term of the tail over 1 - mean / (count + 1)."""
    if count <= mean:
        return 0.0
    if mean <= 0:
        return -math.inf
    return -mean + count * math.log(mean) - math.lgamma(count + 1) - math.log1p(-mean / (count + 1))


def _check_rule(max_splits, tol, min_splits):
    if max_splits is not None and (int(max_splits) != max_splits or max_splits < 1):
        raise ValueError(f"max_splits must be None or an integer >= 1, got {max_splits!r}")
    if not tol > 0:
        raise ValueError(f"tol must be > 0, got {tol!r}")
    if int(min_splits) != min_splits or min_splits < 2:
        raise ValueError(f"min_splits must be an integer >= 2, got {min_splits!r}")


def projected_gain(redundancy, n_splits):
    """Sample gain that the reference curve gives for ``n_splits`` splits at a redundancy ``redundancy``.

    G(K) = K / (1 + (K - 1) * max(omega, 0)): the gain-redundancy relation of Appendix D.2 with the coupling kappa set
    to 0. G(1) = 1; G(K) = K when the split predictors err independently (omega = 0); G(K) tends to the ceiling
    1 / omega as K grows (``n_splits=np.inf`` returns it). Negative readings count as 0 (with them, the curve would
    exceed K). Scalars or arrays (broadcast together); a NaN redundancy gives NaN.
    """
    omega = np.asarray(redundancy, dtype=float)
    k = np.asarray(n_splits, dtype=float)
    if np.any(k < 1):
        raise ValueError(f"n_splits must be >= 1, got {n_splits!r}")
    w = np.maximum(omega, 0.0)  # NaN stays NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        finite = k / (1.0 + (k - 1.0) * w)
        ceiling = np.where(w > 0, 1.0 / w, np.inf)
    gain = np.where(np.isinf(k), ceiling, finite)
    gain = np.where(np.isnan(w), np.nan, gain)
    return float(gain) if gain.ndim == 0 else gain


def _remaining_gain(redundancy, n_splits, max_splits):
    """G(K_max) / G(k) - 1 on the reference curve (0 once k >= K_max)."""
    if n_splits >= max_splits:
        return 0.0
    return projected_gain(redundancy, max_splits) / projected_gain(redundancy, n_splits) - 1.0


def stopping_split(redundancy, max_splits=None, tol=DEFAULT_TOL, min_splits=DEFAULT_MIN_SPLITS):
    """Split count at which the rule stops if the redundancy reading stays at ``redundancy``.

    The smallest k >= min_splits with G(K_max) / G(k) - 1 < tol, or K_max when the budget comes first.
    ``max_splits=None`` is an unlimited budget: the rule compares with the ceiling 1 / omega and never stops when
    omega <= 0 (returns ``math.inf``). Used for the table of the README; a real run re-reads the redundancy after
    every split.
    """
    _check_rule(max_splits, tol, min_splits)
    omega = float(redundancy)
    if math.isnan(omega):
        raise ValueError("the redundancy is NaN")
    if max_splits is not None:
        max_splits = int(max_splits)
        for k in range(min(min_splits, max_splits), max_splits + 1):
            if k >= min_splits and _remaining_gain(omega, k, max_splits) < tol:
                return k
        return max_splits
    if omega <= 0:
        return math.inf
    # G(inf) / G(k) - 1 = (1 - omega) / (k * omega): below tol once k > (1 - omega) / (tol * omega)
    k = max(int(min_splits), int((1.0 - omega) / (tol * omega)) - 1)
    while _remaining_gain(omega, k, math.inf) >= tol:
        k += 1
    return k


class RedundancyMonitor:
    """Redundancy of one Monte Carlo cross-validation run, updated split by split, and when to stop splitting.

    Parameters
    ----------
    n_samples : int
        Number of observations the test sets are drawn from: the study set, n_tr + n_te in the paper (with a
        validation subset carved out of the training part, it counts in n_tr). Test indices are positions in
        ``range(n_samples)``.
    max_splits : int or None
        Budget K_max of splits. None: no budget; the rule compares with the ceiling 1 / omega (K_max -> infinity).
    tol : float
        Stop once the splits left in the budget would add less than this share of gain, G(K_max) / G(k) - 1 < tol.
        The default 0.1 stops once they would shrink the test-side standard error of the CV estimate by less than 5%.
    min_splits : int
        No stop before this many splits (at least 2; 3 by default). With the default tol the rule cannot stop that
        early anyway (at t = 0.2, not before 13 of 20 splits or 33 of 200): it matters only when the rule can stop
        within the first few splits, with a larger tol or test fraction, or a small budget. Use 5 then on small
        data, where a reading after 3 splits is noisy.

    Use ``add_split(test_indices, losses)`` after each split, then ``should_stop()``. The splits must be Monte Carlo
    splits (test sets drawn at random and independently, as ShuffleSplit or StratifiedShuffleSplit): in K-fold, the
    splits of one repeat have disjoint test sets and the factor t does not apply.

    Example, with the squared error of a regression::

        monitor = RedundancyMonitor(n_samples=len(y), max_splits=50)
        for train, test in ShuffleSplit(n_splits=50, test_size=0.2).split(X):
            model = fit(X[train], y[train])
            monitor.add_split(test, (model.predict(X[test]) - y[test]) ** 2)
            if monitor.should_stop():
                break
    """

    def __init__(self, n_samples, *, max_splits=None, tol=DEFAULT_TOL, min_splits=DEFAULT_MIN_SPLITS):
        if int(n_samples) != n_samples or n_samples < 2:
            raise ValueError(f"n_samples must be an integer >= 2, got {n_samples!r}")
        _check_rule(max_splits, tol, min_splits)
        self.n_samples = int(n_samples)
        self.max_splits = None if max_splits is None else int(max_splits)
        self.tol = float(tol)
        self.min_splits = int(min_splits)
        self._indices = []      # test indices of each split, sorted
        self._losses = []       # their losses, in the same order
        self._test_sizes = []
        self._pairs = []        # (a, b) of the pairs that qualify, a < b
        self._covs = []
        self._vars = []
        self._sizes = []        # |I_ab|
        self._n_disjoint = 0    # pairs of splits with disjoint test sets
        self._expected_disjoint = 0.0  # and their expected number under Monte Carlo CV
        self._warned = set()

    # ------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_splits(cls, n_samples, test_indices, losses, **kwargs):
        """Monitor of a finished run: ``test_indices`` and ``losses`` are sequences with one entry per split."""
        test_indices, losses = list(test_indices), list(losses)
        if len(test_indices) != len(losses):
            raise ValueError(f"{len(test_indices)} test index sets for {len(losses)} loss vectors")
        monitor = cls(n_samples, **kwargs)
        for idx, loss in zip(test_indices, losses):
            monitor.add_split(idx, loss)
        return monitor

    def _warn_once(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            _warn(message)

    def _validate(self, test_indices, losses):
        idx = _as_numpy(test_indices)
        loss = _as_numpy(losses)
        if idx.dtype == bool:
            if idx.shape != (self.n_samples,):
                raise ValueError(f"a boolean test mask must have n_samples = {self.n_samples} entries, "
                                 f"got shape {idx.shape}")
            idx = np.flatnonzero(idx)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"test_indices must be a non-empty 1-D sequence, got shape {idx.shape}")
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"test_indices must be integers or a boolean mask, got dtype {idx.dtype}")
        if loss.ndim == 0:
            if idx.size != 1:
                raise ValueError(f"losses must hold one value per test index ({idx.size} indices), got a scalar: "
                                 "pass the per-sample losses, not their mean (e.g. not "
                                 "sklearn.metrics.mean_squared_error)")
            loss = loss.reshape(1)
        if loss.ndim != 1 and loss.size == idx.size and max(loss.shape) == loss.size:
            loss = loss.reshape(-1)  # e.g. (m, 1) from a regression head
        if loss.shape != idx.shape:
            raise ValueError(f"losses must hold one value per test index: {idx.size} indices, losses of shape "
                             f"{loss.shape}")
        try:
            loss = loss.astype(float)
        except (TypeError, ValueError):
            raise TypeError(f"losses must be numbers, got dtype {loss.dtype}") from None
        if idx.min() < 0 or idx.max() >= self.n_samples:
            raise ValueError(f"test indices must lie in [0, {self.n_samples}), got [{idx.min()}, {idx.max()}]")
        order = np.argsort(idx, kind="stable")
        idx, loss = idx[order], loss[order]
        if np.any(idx[1:] == idx[:-1]):
            raise ValueError("test indices must be unique within a split")
        if not np.all(np.isfinite(loss)):
            self._warn_once("nonfinite", "non-finite losses: the pairs of splits that both hold out such an "
                                         "observation are skipped")
        return idx, loss

    def add_split(self, test_indices, losses):
        """Add one split: its test indices and the per-sample losses of its predictor on them (same order).

        ``test_indices``: integer positions in ``range(n_samples)``, unique, or a boolean mask of length n_samples
        (``losses`` then in increasing index order). Lists, NumPy arrays and torch tensors are accepted. Returns the
        monitor.
        """
        idx, loss = self._validate(test_indices, losses)
        b = len(self._indices)
        for a in range(b):
            shared, pos_a, pos_b = np.intersect1d(self._indices[a], idx, assume_unique=True, return_indices=True)
            self._expected_disjoint += _p_disjoint(self.n_samples, self._test_sizes[a], idx.size)
            if shared.size == 0:
                self._n_disjoint += 1
            if shared.size < 2:
                continue
            pair_cov = np.cov(self._losses[a][pos_a], loss[pos_b])
            cov = float(pair_cov[0, 1])
            var = float(0.5 * (pair_cov[0, 0] + pair_cov[1, 1]))
            if not (np.isfinite(cov) and np.isfinite(var)):
                continue
            self._pairs.append((a, b))
            self._covs.append(cov)
            self._vars.append(var)
            self._sizes.append(int(shared.size))
        self._indices.append(idx)
        self._losses.append(loss)
        self._test_sizes.append(int(idx.size))
        if self._n_disjoint and _log_poisson_tail(self._n_disjoint, self._expected_disjoint) < _LOG_DISJOINT_LEVEL:
            self._warn_once("disjoint", f"{self._n_disjoint} pairs of splits have disjoint test sets, where "
                                        f"independent random test sets would give {self._expected_disjoint:.2g} on "
                                        "average: the splits look like K-fold. The redundancy assumes Monte Carlo "
                                        "cross-validation (test sets drawn independently, e.g. ShuffleSplit); with "
                                        "K-fold the test fraction t does not apply")
        return self

    # ------------------------------------------------------------------------------------------------------------
    @property
    def n_splits(self):
        """Number of splits added so far (k)."""
        return len(self._indices)

    @property
    def n_pairs(self):
        """Number of pairs of splits that qualify (|I_ab| >= 2, finite covariance and variance)."""
        return len(self._covs)

    @property
    def is_defined(self):
        """Whether the redundancy is defined: at least one pair of test sets shares 2 observations."""
        return bool(self._covs)

    @property
    def test_fraction(self):
        """t = n_te / n, with n_te the mean test-set size of the splits so far (NaN before the first split)."""
        if not self._test_sizes:
            return math.nan
        return float(np.mean(self._test_sizes)) / self.n_samples

    @property
    def mean_intersection_size(self):
        """Mean |I_ab| over the pairs that qualify (about t * n_te in Monte Carlo CV)."""
        return float(np.mean(self._sizes)) if self._sizes else math.nan

    @property
    def mean_pair_covariance(self):
        """mean_pairs(cov_ab), the numerator of the loss correlation."""
        return float(np.mean(self._covs)) if self._covs else math.nan

    @property
    def mean_pair_variance(self):
        """mean_pairs(var_ab), the denominator of the loss correlation."""
        return float(np.mean(self._vars)) if self._vars else math.nan

    @property
    def loss_correlation(self):
        """rho_hat = mean_pairs(cov_ab) / mean_pairs(var_ab); 0 if the mean variance is <= 0, NaN while undefined."""
        if not self._covs:
            return math.nan
        var_mean = float(np.mean(self._vars))
        return float(np.mean(self._covs)) / var_mean if var_mean > 0 else 0.0

    @property
    def redundancy(self):
        """omega_hat = t * rho_hat, in [-t, t]; NaN while undefined."""
        return self.test_fraction * self.loss_correlation

    @property
    def ceiling(self):
        """1 / omega_hat: the limit of the reference curve as K grows (inf if omega_hat <= 0, NaN while undefined)."""
        return projected_gain(self.redundancy, math.inf)

    def _leave_one_out(self):
        """omega_hat without each split in turn (array of k values), or None if one of them is undefined."""
        k = self.n_splits
        if k < 3 or not self._covs:
            return None
        pairs = np.asarray(self._pairs)
        covs, vars_ = np.asarray(self._covs), np.asarray(self._vars)
        n_with = np.bincount(pairs[:, 0], minlength=k) + np.bincount(pairs[:, 1], minlength=k)
        if np.any(n_with == len(covs)):
            return None
        cov_with = np.bincount(pairs[:, 0], covs, k) + np.bincount(pairs[:, 1], covs, k)
        var_with = np.bincount(pairs[:, 0], vars_, k) + np.bincount(pairs[:, 1], vars_, k)
        # ratio of means = ratio of sums over the same pairs
        cov_without, var_without = covs.sum() - cov_with, vars_.sum() - var_with
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(var_without > 0, cov_without / var_without, 0.0)
        sizes = np.asarray(self._test_sizes, dtype=float)
        t = (sizes.sum() - sizes) / (k - 1) / self.n_samples
        return t * rho

    @property
    def redundancy_se(self):
        """Leave-one-split-out jackknife standard error of omega_hat (information only, not used by the rule).

        It measures how much the reading depends on the splits drawn, for this study set; it does not include the
        variation from one study set to another. NaN before 3 splits or while a leave-one-out reading is undefined.
        """
        loo = self._leave_one_out()
        if loo is None:
            return math.nan
        k = loo.size
        return float(np.sqrt((k - 1) / k * np.sum((loo - loo.mean()) ** 2)))

    def projected_gain(self, n_splits=None):
        """Gain the reference curve gives for ``n_splits`` splits at the current redundancy (default: k so far)."""
        return projected_gain(self.redundancy, self.n_splits if n_splits is None else n_splits)

    def remaining_gain(self):
        """Share of gain the splits left in the budget would add: G(K_max) / G(k) - 1 (NaN while undefined)."""
        if not self.is_defined:
            return math.nan
        budget = math.inf if self.max_splits is None else self.max_splits
        return _remaining_gain(self.redundancy, self.n_splits, budget)

    def _rule(self):
        """The decision of should_stop, without its warnings."""
        k = self.n_splits
        if self.max_splits is not None and k >= self.max_splits:
            return True
        if k < self.min_splits or not self.is_defined:
            return False
        return self.remaining_gain() < self.tol

    def should_stop(self):
        """True once the budget is spent, or once k >= min_splits, the redundancy is defined and the remaining
        gain is below tol.

        Warns once when the run ends on a reading that carries no information: still undefined, or computed from
        losses that are constant on every shared test set (the loss correlation is then set to 0).
        """
        stop = self._rule()
        if stop and not self.is_defined:
            self._warn_once("undefined", "the budget is spent and no two test sets shared 2 observations: the "
                                         "redundancy is undefined and the rule never applied (larger test sets or "
                                         "more data would help)")
        elif stop and not self.mean_pair_variance > 0:
            self._warn_once("constant", "the losses are constant on the observations shared by every pair of test "
                                        "sets: the loss correlation is set to 0 and the redundancy carries no "
                                        "information (does the loss suit the predictions? e.g. a 0-1 loss on the "
                                        "predictions of a regressor is always 1)")
        return stop

    def summary(self):
        """Every quantity of the monitor, as a dict."""
        budget = math.inf if self.max_splits is None else self.max_splits
        return {
            "n_samples": self.n_samples,
            "n_splits": self.n_splits,
            "max_splits": self.max_splits,
            "tol": self.tol,
            "min_splits": self.min_splits,
            "test_fraction": self.test_fraction,
            "n_pairs": self.n_pairs,
            "mean_intersection_size": self.mean_intersection_size,
            "mean_pair_covariance": self.mean_pair_covariance,
            "mean_pair_variance": self.mean_pair_variance,
            "loss_correlation": self.loss_correlation,
            "redundancy": self.redundancy,
            "redundancy_se": self.redundancy_se,
            "ceiling": self.ceiling,
            "projected_gain": self.projected_gain() if self.n_splits else math.nan,
            "projected_gain_at_max_splits": self.projected_gain(budget),
            "remaining_gain": self.remaining_gain(),
            "should_stop": self._rule(),
        }

    def __repr__(self):
        budget = ", no budget" if self.max_splits is None else f" of {self.max_splits}"
        head = f"RedundancyMonitor(n_samples={self.n_samples}, splits={self.n_splits}{budget}"
        decision = "stop" if self._rule() else "continue"
        if not self.is_defined:
            return f"{head}, redundancy undefined (no pair of test sets shares 2 observations): {decision})"
        se = self.redundancy_se
        se = "" if math.isnan(se) else f" (jackknife SE {se:.3f})"
        at = "the ceiling" if self.max_splits is None else f"{self.max_splits} splits"
        g_max = self.projected_gain(math.inf if self.max_splits is None else self.max_splits)
        remaining = self.remaining_gain()
        compare = "<" if remaining < self.tol else ">="
        return (f"{head}, redundancy={self.redundancy:.3f}{se}, loss correlation={self.loss_correlation:.3f}, "
                f"t={self.test_fraction:.3f}, pairs={self.n_pairs}, "
                f"mean intersection={self.mean_intersection_size:.1f}; projected gain {self.projected_gain():.2f} now, "
                f"{g_max:.2f} at {at} (+{100 * remaining:.2f}% {compare} tol {100 * self.tol:g}%): {decision})")
