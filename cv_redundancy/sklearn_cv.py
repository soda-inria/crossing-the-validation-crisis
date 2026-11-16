"""Monte Carlo cross-validation with scikit-learn that stops once more splits would add little gain.

``cross_validate_until_redundant`` runs the splits one after the other, fits a clone of the estimator on each
training part, scores the per-sample losses of the test part, feeds them to a ``RedundancyMonitor`` and stops when
the monitor says so. ``redundancy_from_cv_results`` computes the same monitor after the fact from the output of
``sklearn.model_selection.cross_validate(..., return_estimator=True, return_indices=True)``.

scikit-learn is imported inside the functions, so that ``import cv_redundancy`` needs NumPy only.
"""
import time

import numpy as np

from .core import DEFAULT_MIN_SPLITS, DEFAULT_TOL, RedundancyMonitor, _warn
from .losses import resolve_loss, squared_error, zero_one

DEFAULT_BUDGET = 100


def _n_samples(X):
    return X.shape[0] if hasattr(X, "shape") else len(X)


def _per_sample(value, n):
    """Whether a fit parameter holds one value per observation (and is then restricted to the training part)."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return len(shape) > 0 and shape[0] == n
    return isinstance(value, (list, tuple)) and len(value) == n


def _split_losses(estimator, X, y, test, loss):
    """Per-sample losses of a fitted estimator on the test part ``test``."""
    from sklearn.utils import _safe_indexing
    func, method, takes_classes = resolve_loss(loss)
    y_pred = getattr(estimator, method)(_safe_indexing(X, test))
    y_test = np.asarray(_safe_indexing(y, test))
    if takes_classes:
        return func(y_test, y_pred, classes=getattr(estimator, "classes_", None))
    return func(y_test, y_pred)


def cross_validate_until_redundant(estimator, X, y, *, cv=None, loss="squared_error", tol=DEFAULT_TOL,
                                   min_splits=DEFAULT_MIN_SPLITS, max_splits=None, groups=None, params=None,
                                   return_estimator=False, return_indices=False):
    """Run Monte Carlo cross-validation split by split and stop once more splits would add less than ``tol`` of gain.

    Parameters
    ----------
    estimator : estimator object
        Cloned and fitted on the training part of each split.
    X : array-like of shape (n_samples, n_features)
        The data: an array, a list or a pandas object.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The targets.
    cv : splitter, default=None
        A Monte Carlo splitter, ShuffleSplit or StratifiedShuffleSplit. None: ``ShuffleSplit(n_splits=max_splits,
        test_size=0.2, random_state=0)``, stratified for a classifier, with a budget of 100 splits when
        ``max_splits`` is None too. Another splitter, or an iterable of (train, test) pairs, runs with a warning:
        the redundancy assumes test sets drawn at random and independently.
    loss : str, callable or (callable, str), default="squared_error"
        The per-sample loss of the metric whose precision matters: "squared_error" (MSE), "zero_one" (accuracy),
        "log_loss", "brier", a metric name such as "accuracy" or "neg_log_loss", a callable
        ``loss(y_true, y_pred)`` on the output of ``predict`` that returns one value per sample, or a pair
        (callable, "predict_proba").
    tol : float, default=0.1
        Stop once the splits left in the budget would add less than this share of gain, G(K_max) / G(k) - 1 < tol.
    min_splits : int, default=3
        No stop before this many splits.
    max_splits : int, default=None
        The budget K_max, at most the number of splits of ``cv`` (its default).
    groups : array-like of shape (n_samples,), default=None
        Passed to ``cv.split``.
    params : dict, default=None
        Passed to ``estimator.fit``; the entries with one value per observation (``sample_weight``, ...) are
        restricted to the training part, as in ``sklearn.model_selection.cross_validate``.
    return_estimator : bool, default=False
        Also return the fitted estimators.
    return_indices : bool, default=False
        Also return the train and test indices of each split.

    Returns
    -------
    result : dict
        As ``sklearn.model_selection.cross_validate``, one entry per split run (``pandas.DataFrame(result)`` gives
        one row per split):

        - "test_loss": mean per-sample loss on each test part (the metric, as a loss: MSE, error rate, ...);
        - "fit_time", "score_time";
        - "redundancy": the redundancy after each split (NaN while undefined);
        - "n_splits": the number of splits run; "stopped_early": whether the rule stopped before the budget;
        - "monitor": the ``RedundancyMonitor``, with every statistic of the last split;
        - "estimator": the fitted estimators, if ``return_estimator``;
        - "indices": {"train": [...], "test": [...]}, if ``return_indices``.
    """
    from sklearn.base import clone, is_classifier, is_regressor
    from sklearn.exceptions import NotFittedError
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
    from sklearn.utils import _safe_indexing
    from sklearn.utils.multiclass import type_of_target

    # fail, or warn, before any fit
    func, method, _ = resolve_loss(loss)
    try:
        getattr(estimator, method)
    except NotFittedError:
        pass
    except AttributeError:
        raise TypeError(f"{type(estimator).__name__} has no {method}, which loss={loss!r} needs") from None
    if is_classifier(estimator) and func is squared_error:
        _warn("loss='squared_error' on the predicted labels of a classifier: pass the per-sample loss of the metric "
              "of interest, e.g. loss='zero_one' (accuracy), 'log_loss' or 'brier'")
    if is_regressor(estimator) and func is zero_one:
        _warn(f"loss={loss!r} compares predicted and true labels, but the estimator is a regressor: its losses are "
              "all 1; pass the per-sample loss of its metric, e.g. loss='squared_error'")

    if cv is None:
        stratify = is_classifier(estimator) and type_of_target(y) in ("binary", "multiclass")
        cv = (StratifiedShuffleSplit if stratify else ShuffleSplit)(
            n_splits=DEFAULT_BUDGET if max_splits is None else max_splits, test_size=0.2, random_state=0)
    elif isinstance(cv, (int, np.integer)):
        raise TypeError("cv must be a splitter: an integer means K-fold in scikit-learn, whose test sets do not "
                        "overlap; use ShuffleSplit(n_splits=..., test_size=...)")
    if hasattr(cv, "split"):
        n_available = cv.get_n_splits(X, y, groups)
        splits = cv.split(X, y, groups)
    else:
        splits = list(cv)
        n_available = len(splits)
    splitter_warning = None
    if isinstance(cv, GroupShuffleSplit):  # a subclass of ShuffleSplit
        splitter_warning = ("GroupShuffleSplit draws the test sets by group: the redundancy and its reference curve "
                            "assume independent observations (Appendix D.2), and the paper has no result on grouped "
                            "splits")
    elif not isinstance(cv, (ShuffleSplit, StratifiedShuffleSplit)):
        splitter_warning = (f"cv is {type(cv).__name__}, not a ShuffleSplit-like splitter: the redundancy assumes "
                            "Monte Carlo cross-validation, whose test sets are drawn independently (with K-fold, the "
                            "splits of one repeat have disjoint test sets and the test fraction t does not apply)")
    if splitter_warning:
        _warn(splitter_warning)
    if max_splits is None:
        max_splits = n_available
    elif max_splits > n_available:
        raise ValueError(f"max_splits = {max_splits} exceeds the {n_available} splits of cv: pass "
                         f"cv=ShuffleSplit(n_splits={max_splits}, test_size=...), or leave cv=None")

    n = _n_samples(X)
    monitor = RedundancyMonitor(n, max_splits=max_splits, tol=tol, min_splits=min_splits)
    if splitter_warning:
        monitor._warned.add("disjoint")  # the splitter is already flagged
    out = {"test_loss": [], "fit_time": [], "score_time": [], "redundancy": []}
    indices = {"train": [], "test": []}
    estimators = []
    for train, test in splits:
        fit_params = {key: _safe_indexing(value, train) if _per_sample(value, n) else value
                      for key, value in (params or {}).items()}
        start = time.perf_counter()
        fitted = clone(estimator).fit(_safe_indexing(X, train), _safe_indexing(y, train), **fit_params)
        out["fit_time"].append(time.perf_counter() - start)
        start = time.perf_counter()
        losses = _split_losses(fitted, X, y, test, loss)
        out["score_time"].append(time.perf_counter() - start)
        monitor.add_split(test, losses)
        out["test_loss"].append(float(np.mean(losses)))
        out["redundancy"].append(monitor.redundancy)
        indices["train"].append(np.asarray(train))
        indices["test"].append(np.asarray(test))
        if return_estimator:
            estimators.append(fitted)
        if monitor.should_stop():
            break

    result = {key: np.asarray(values) for key, values in out.items()}
    result.update(n_splits=monitor.n_splits, stopped_early=monitor.n_splits < max_splits, monitor=monitor)
    if return_estimator:
        result["estimator"] = estimators
    if return_indices:
        result["indices"] = indices
    return result


def redundancy_from_cv_results(cv_results, X, y, *, loss="squared_error", tol=DEFAULT_TOL,
                               min_splits=DEFAULT_MIN_SPLITS, max_splits=None):
    """RedundancyMonitor of a finished ``cross_validate`` run, from its fitted estimators and test indices.

    ``cv_results``: the output of ``sklearn.model_selection.cross_validate(estimator, X, y, cv=...,
    return_estimator=True, return_indices=True)`` (scikit-learn >= 1.3), on the same X and y, with a ShuffleSplit-like
    cv. ``max_splits`` defaults to the number of splits run; pass a larger budget to ask whether more splits would
    pay (``monitor.remaining_gain()``, ``monitor.should_stop()``).
    """
    if "estimator" not in cv_results or "indices" not in cv_results:
        raise ValueError("cv_results needs the fitted estimators and the test indices: call cross_validate with "
                         "return_estimator=True and return_indices=True")
    tests = cv_results["indices"]["test"]
    if max_splits is None:
        max_splits = len(tests)
    monitor = RedundancyMonitor(_n_samples(X), max_splits=max_splits, tol=tol, min_splits=min_splits)
    for fitted, test in zip(cv_results["estimator"], tests):
        monitor.add_split(test, _split_losses(fitted, X, y, test, loss))
    return monitor
