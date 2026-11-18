"""scikit-learn loop end to end, post hoc monitor, per-sample losses against scikit-learn's metrics."""
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.ensemble import StackingClassifier, StackingRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, brier_score_loss, log_loss as sk_log_loss, mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import (  # noqa: E402
    GroupShuffleSplit, KFold, ShuffleSplit, StratifiedShuffleSplit, cross_validate,
)
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from cv_redundancy import (  # noqa: E402
    absolute_error, brier, cross_validate_until_redundant, log_loss, projected_gain, redundancy_from_cv_results,
    squared_error, zero_one,
)
from cv_redundancy.losses import resolve_loss  # noqa: E402


def _regression(n=100, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return X, X @ rng.normal(size=d) + 0.5 * rng.normal(size=n)


def _classification(n=150, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logits = X @ rng.normal(size=(4, n_classes))
    y = (logits + rng.gumbel(size=logits.shape)).argmax(axis=1)
    return X, y


class _Unfittable(Ridge):
    """A regressor whose fit fails: what the loop does before any fit."""

    def fit(self, X, y, sample_weight=None):
        raise RuntimeError("fit was called")


class _ProbaOnceFitted(LogisticRegression):
    """A classifier whose predict_proba exists only once it is fitted (NotFittedError before)."""

    @property
    def predict_proba(self):
        check_is_fitted(self)
        return super().predict_proba


def _brier_expected(y, proba, classes):
    """sum_c (p_c - 1[y = c])^2, with the one-hot labels of the columns' classes."""
    return np.sum((proba - (np.asarray(y)[:, None] == np.asarray(classes)[None, :])) ** 2, axis=1)


def user_log_loss(y_true, proba, *, classes=None):
    """A user's per-sample log-loss with a keyword-only ``classes``."""
    return log_loss(y_true, proba, classes=classes)


def user_brier(y_true, proba, classes=None):
    """A user's per-sample Brier score with a positional-or-keyword ``classes``."""
    return brier(y_true, proba, classes=classes)


def test_stable_learner_stops_before_unstable_ones():
    X, y = _regression()
    cv = ShuffleSplit(n_splits=200, test_size=0.2, random_state=0)
    runs = {name: cross_validate_until_redundant(est, X, y, cv=cv, return_indices=True)
            for name, est in [("ridge", Ridge(alpha=1.0)), ("tree", DecisionTreeRegressor(random_state=0)),
                              ("1-NN", KNeighborsRegressor(n_neighbors=1))]}
    ridge, tree, nn1 = runs["ridge"], runs["tree"], runs["1-NN"]
    assert ridge["stopped_early"] and ridge["n_splits"] < 50 and ridge["monitor"].redundancy > 0.18  # near t = 0.2
    assert tree["n_splits"] > ridge["n_splits"] + 20 and tree["monitor"].redundancy < 0.1
    # 1-NN sits in between: its prediction for an observation changes only when its nearest neighbour leaves the
    # training part, so its split predictors often err alike
    assert nn1["n_splits"] > ridge["n_splits"] and nn1["monitor"].redundancy < ridge["monitor"].redundancy
    for res in runs.values():
        k, omega = res["n_splits"], res["redundancy"]
        assert len(res["test_loss"]) == len(omega) == len(res["indices"]["test"]) == k
        assert res["monitor"].redundancy == omega[-1] and res["monitor"].max_splits == 200
        # the rule held at every split before the stop and failed at the stop
        remaining = [projected_gain(omega[j - 1], 200) / projected_gain(omega[j - 1], j) - 1 for j in range(3, k + 1)]
        assert all(r >= 0.1 for r in remaining[:-1]) and (remaining[-1] < 0.1 or k == 200)


def test_post_hoc_equals_incremental():
    X, y = _regression(seed=1)
    cv = ShuffleSplit(n_splits=8, test_size=0.25, random_state=3)
    cvr = cross_validate(Ridge(), X, y, cv=cv, scoring="neg_mean_squared_error", return_estimator=True,
                         return_indices=True)
    post_hoc = redundancy_from_cv_results(cvr, X, y, loss="squared_error")
    loop = cross_validate_until_redundant(Ridge(), X, y, cv=cv, tol=1e-9, return_estimator=True)
    assert loop["n_splits"] == 8 and not loop["stopped_early"] and post_hoc.should_stop()
    assert loop["test_loss"] == pytest.approx(-cvr["test_score"], rel=1e-12)
    for key, value in post_hoc.summary().items():
        if key not in ("tol", "should_stop", "remaining_gain"):
            assert loop["monitor"].summary()[key] == pytest.approx(value, rel=1e-12, nan_ok=True), key
    larger_budget = redundancy_from_cv_results(cvr, X, y, max_splits=200)
    assert larger_budget.remaining_gain() > 0 and larger_budget.redundancy == post_hoc.redundancy
    with pytest.raises(ValueError, match="return_indices"):
        redundancy_from_cv_results({"test_score": cvr["test_score"]}, X, y)
    with pytest.raises(TypeError):
        redundancy_from_cv_results(cvr, X, y, "squared_error")  # loss is keyword-only
    with pytest.raises(ValueError, match="at least one split"):
        redundancy_from_cv_results({"estimator": [], "indices": {"test": []}}, X, y)
    with pytest.raises(ValueError, match="one of each per split"):
        redundancy_from_cv_results({"estimator": cvr["estimator"][:2], "indices": cvr["indices"]}, X, y)
    with pytest.raises(ValueError, match="one of each per split"):
        redundancy_from_cv_results({"estimator": cvr["estimator"], "indices": {"test": cvr["indices"]["test"][:2]}},
                                   X, y)
    # the loop's loss checks
    with pytest.raises(TypeError, match="predict_proba"):
        redundancy_from_cv_results(cvr, X, y, loss="log_loss")
    with pytest.warns(UserWarning, match="but the estimator is a regressor"):
        redundancy_from_cv_results(cvr, X, y, loss="zero_one")
    Xc, yc = _classification()
    cvr = cross_validate(LogisticRegression(max_iter=1000), Xc, yc, cv=cv, return_estimator=True, return_indices=True)
    with pytest.warns(UserWarning, match="on the predicted labels of a classifier") as record:
        redundancy_from_cv_results(cvr, Xc, yc)  # loss="squared_error" by default
    assert len(record) == 1 and record[0].filename == __file__  # warned once, from the caller's line


@pytest.mark.parametrize("loss", ["zero_one", "accuracy", "log_loss", "neg_log_loss", "brier"])
@pytest.mark.parametrize("estimator", [LogisticRegression(max_iter=1000), DecisionTreeClassifier(random_state=0)],
                         ids=["logistic", "tree"])
def test_classification_losses(loss, estimator):
    # the tree gives probability 0 to the true class of its errors: the log-loss must still match scikit-learn's
    X, y = _classification()
    cv = StratifiedShuffleSplit(n_splits=6, test_size=0.2, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a ShuffleSplit-like cv does not warn
        res = cross_validate_until_redundant(estimator, X, y, cv=cv, loss=loss, tol=1e-9, return_estimator=True,
                                             return_indices=True)
    assert res["n_splits"] == 6 and res["monitor"].is_defined
    for est, test, value in zip(res["estimator"], res["indices"]["test"], res["test_loss"]):
        if loss in ("zero_one", "accuracy"):
            expected = 1 - accuracy_score(y[test], est.predict(X[test]))
        elif loss in ("log_loss", "neg_log_loss"):
            expected = sk_log_loss(y[test], est.predict_proba(X[test]), labels=est.classes_)
        else:
            expected = np.mean(np.sum((est.predict_proba(X[test]) - np.eye(3)[y[test]]) ** 2, axis=1))
        assert value == pytest.approx(expected, rel=1e-8)


def test_probability_losses():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    p = rng.uniform(0.01, 0.99, size=50)
    assert brier(y, p).mean() == pytest.approx(2 * brier_score_loss(y, p))
    assert log_loss(y, p).mean() == pytest.approx(sk_log_loss(y, p), rel=1e-9)
    assert log_loss(y, np.column_stack([1 - p, p])) == pytest.approx(log_loss(y, p))
    # boolean labels are the column positions 0 and 1, as in scikit-learn
    yb = y.astype(bool)
    for proba in (p, np.column_stack([1 - p, p])):
        assert log_loss(yb, proba).mean() == pytest.approx(sk_log_loss(yb, proba), rel=1e-9)
        assert np.array_equal(brier(yb, proba), brier(y, proba))
    # string labels through classes, and a class missing from the columns (absent from a training part)
    labels = np.array(["a", "b", "c"])[[0, 2, 1]]
    proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])
    eps = np.finfo(float).eps
    assert log_loss(labels, proba, classes=["a", "c"]) == pytest.approx([-np.log(0.7), -np.log(0.8), -np.log(eps)])
    # probabilities 0 and 1 are clipped to [eps, 1 - eps], as in scikit-learn
    hard = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert log_loss([0, 1, 1], hard).mean() == pytest.approx(sk_log_loss([0, 1, 1], hard), rel=1e-12)
    assert log_loss([1], [[1.0, 0.0]], eps=1e-12) == pytest.approx(-np.log(1e-12))
    assert brier(labels, proba, classes=["a", "c"]) == pytest.approx([0.18, 0.08, 1.5])
    assert zero_one(labels, ["a", "b", "b"]).tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        log_loss(labels, proba)  # string labels need classes
    # float32 probabilities with a 0: clipped at the float32 machine epsilon, as in scikit-learn
    p32 = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.3]], dtype=np.float32)
    assert log_loss([1, 1, 0], p32).mean() == pytest.approx(sk_log_loss([1, 1, 0], p32), rel=1e-6)
    # classes in any order, the columns following them
    unsorted = np.array([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]])
    assert log_loss(["a", "c", "a"], unsorted, classes=["c", "a"]) == pytest.approx(-np.log([0.8, 0.9, 0.4]))
    assert brier(["a", "c", "a"], unsorted, classes=["c", "a"]) == pytest.approx([0.08, 0.02, 0.72])


@pytest.mark.parametrize("labels", [[-1, 1], [1, 2], [1, 2, 3], ["cat", "dog", "eel"]],
                         ids=["minus_plus_one", "one_two", "one_to_three", "strings"])
def test_probability_losses_with_labels_that_are_not_positions(labels):
    rng = np.random.default_rng(1)
    classes = np.array(labels)
    y = classes[np.arange(60) % len(classes)]  # every class present
    proba = rng.dirichlet(np.ones(len(classes)), size=60)
    p = proba[:, 1] if len(classes) == 2 else proba  # a binary problem as P(classes[1])
    assert log_loss(y, p, classes=classes).mean() == pytest.approx(sk_log_loss(y, p, labels=classes), rel=1e-12)
    assert brier(y, p, classes=classes) == pytest.approx(_brier_expected(y, proba, classes), rel=1e-12)
    if len(classes) == 2:
        assert brier(y, p, classes=classes).mean() == pytest.approx(
            2 * brier_score_loss(y, p, pos_label=classes[1]), rel=1e-12)
    for func in (log_loss, brier):  # labels read as column positions would give silently wrong losses
        with pytest.raises(ValueError, match="column positions"):
            func(y, p)


@pytest.mark.parametrize("loss", ["log_loss", "brier", (log_loss, "predict_proba"), (brier, "predict_proba"),
                                  (user_log_loss, "predict_proba"), (user_brier, "predict_proba")],
                         ids=["log_loss", "brier", "log_loss_pair", "brier_pair", "user_log_loss_pair",
                              "user_brier_pair"])
@pytest.mark.parametrize("labels", [[-1, 1], [1, 2, 3], ["cat", "dog", "eel"]],
                         ids=["minus_plus_one", "one_to_three", "strings"])
def test_loop_with_labels_that_are_not_positions(labels, loss):
    X, y = _classification(n_classes=len(labels))
    y = np.array(labels)[y]
    cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=0)
    res = cross_validate_until_redundant(LogisticRegression(max_iter=1000), X, y, cv=cv, loss=loss, tol=1e-9,
                                         return_estimator=True, return_indices=True)
    log = (loss if isinstance(loss, str) else loss[0].__name__).endswith("log_loss")
    for est, test, value in zip(res["estimator"], res["indices"]["test"], res["test_loss"]):
        proba = est.predict_proba(X[test])
        if log:
            expected = sk_log_loss(y[test], proba, labels=est.classes_)
        else:
            expected = np.mean(_brier_expected(y[test], proba, est.classes_))
        assert value == pytest.approx(expected, rel=1e-10)
    assert resolve_loss(loss)[2]  # the pair's callable gets classes=estimator.classes_ too


def test_classes_only_for_probability_pairs():
    for func in (user_log_loss, user_brier):
        assert resolve_loss((func, "predict_proba"))[2]
        assert not resolve_loss((func, "predict"))[2] and not resolve_loss(func)[2]  # labels, not probabilities
    assert not resolve_loss((lambda y_true, proba: proba[:, 0], "predict_proba"))[2]  # no classes keyword
    assert not resolve_loss((user_log_loss, "decision_function"))[2]  # scores, not probabilities
    assert not resolve_loss((max, "predict_proba"))[2]  # a builtin without a signature

    def positional(y_true, proba, classes=None, /):
        return proba[:, 0]
    assert not resolve_loss((positional, "predict_proba"))[2]  # a positional-only classes cannot be passed by name


def test_label_and_prediction_shapes():
    p = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    for func in (log_loss, brier):
        # integer-valued floats are column positions (labels from pandas or torch); 0.5 is not
        assert np.array_equal(func([0.0, 1.0, 1.0], p), func([0, 1, 1], p))
        with pytest.raises(ValueError, match="column positions"):
            func([0.5, 1.0, 1.0], p)
        # a column of labels, as scikit-learn accepts; two columns are not labels
        assert np.array_equal(func(np.array([[0], [1], [1]]), p), func([0, 1, 1], p))
        assert np.array_equal(func(np.array([["a"], ["b"], ["b"]]), p, classes=["a", "b"]), func([0, 1, 1], p))
        with pytest.raises(ValueError, match="1-D"):
            func(np.array([[0, 1], [1, 0], [1, 1]]), p)
        with pytest.raises(ValueError, match="classes for"):  # as many classes as probability columns
            func(["a", "b"], np.full((2, 3), 1 / 3), classes=["a", "b"])
        with pytest.raises(ValueError, match="classes for"):  # and one class per column, not a 2-D array
            func(["a", "b"], p[:2], classes=[["a", "b"]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # an infinite label is rejected before any cast to int
            with pytest.raises(ValueError, match="column positions"):
                func([np.inf, 1.0, 0.0], p)
    assert log_loss([0], [[1.0, 0.0]])[0] == -np.log(1 - np.finfo(float).eps)  # clipped to [eps, 1 - eps]
    # (n, 1) predictions for (n,) targets
    assert squared_error([1, 2, 3], [[1.5], [2], [0]]).tolist() == [0.25, 0.0, 9.0]
    assert absolute_error([1, 2, 3], [[1.5], [2], [0]]).tolist() == [0.5, 0.0, 3.0]


def test_loop_with_a_class_missing_from_training_parts():
    # a class of 2 observations, absent from some training parts: probability 0, clipped as in scikit-learn
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 3))
    y = np.array(["cat"] * 29 + ["dog"] * 29 + ["eel"] * 2)
    X[y == "dog"] += 1.0
    cv = ShuffleSplit(n_splits=15, test_size=0.5, random_state=0)
    res = cross_validate_until_redundant(LogisticRegression(max_iter=1000), X, y, cv=cv, loss="log_loss", tol=1e-9,
                                         return_estimator=True, return_indices=True)
    all_classes = np.unique(y)
    n_missing = 0
    for est, test, value in zip(res["estimator"], res["indices"]["test"], res["test_loss"]):
        padded = np.zeros((len(test), 3))
        padded[:, np.searchsorted(all_classes, est.classes_)] = est.predict_proba(X[test])
        n_missing += len(est.classes_) < 3 and "eel" in y[test]
        assert value == pytest.approx(sk_log_loss(y[test], padded, labels=all_classes), rel=1e-8)
    assert n_missing > 0


def test_pandas_inputs():
    pd = pytest.importorskip("pandas")
    X, y = _regression(n=120, d=3, seed=3)
    order = np.random.default_rng(0).permutation(len(y))  # an index that is not the positions
    X_frame, y_series = pd.DataFrame(X, index=order, columns=["a", "b", "c"]), pd.Series(y, index=order)
    cv = ShuffleSplit(n_splits=6, test_size=0.2, random_state=0)
    array = cross_validate_until_redundant(Ridge(), X, y, cv=cv, tol=1e-9)
    frame = cross_validate_until_redundant(Ridge(), X_frame, y_series, cv=cv, tol=1e-9, return_estimator=True,
                                           return_indices=True)
    assert frame["n_splits"] == array["n_splits"] == 6
    assert frame["test_loss"] == pytest.approx(array["test_loss"], rel=1e-12)
    assert frame["monitor"].redundancy == pytest.approx(array["monitor"].redundancy, rel=1e-10)
    for est, test, value in zip(frame["estimator"], frame["indices"]["test"], frame["test_loss"]):
        assert value == pytest.approx(mean_squared_error(y_series.iloc[test], est.predict(X_frame.iloc[test])),
                                      rel=1e-12)


def test_regression_losses_with_several_outputs():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 3))
    Y = X @ rng.normal(size=(3, 2)) + rng.normal(size=(100, 2))
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for loss, metric in [("squared_error", mean_squared_error), ("absolute_error", mean_absolute_error),
                         ("mae", mean_absolute_error), ("mean_absolute_error", mean_absolute_error),
                         ("neg_mean_absolute_error", mean_absolute_error)]:
        res = cross_validate_until_redundant(Ridge(), X, Y, cv=cv, loss=loss, tol=1e-9, return_estimator=True,
                                             return_indices=True)
        for est, test, value in zip(res["estimator"], res["indices"]["test"], res["test_loss"]):
            assert value == pytest.approx(metric(Y[test], est.predict(X[test])), rel=1e-12), loss
    assert absolute_error([1.0, 2.0], [1.5, 0.0]).tolist() == [0.5, 2.0]
    assert absolute_error([[1.0, 2.0]], [[1.5, 0.0]]).tolist() == [1.25]  # averaged over the outputs


def test_splitter_checks():
    X, y = _regression(n=200)
    with pytest.warns(UserWarning) as record:  # one repeat of K-fold: never defined
        cross_validate_until_redundant(Ridge(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    messages = [str(w.message) for w in record]
    assert any("ShuffleSplit-like" in m for m in messages) and any("undefined" in m for m in messages)
    assert not any("pairs of splits have disjoint" in m for m in messages)  # flagged once, by the splitter check
    assert all(w.filename == __file__ for w in record)  # attributed to the caller
    with pytest.warns(UserWarning, match="GroupShuffleSplit draws the test sets by group"):
        cross_validate_until_redundant(Ridge(), X, y, cv=GroupShuffleSplit(5, test_size=0.2, random_state=0),
                                       groups=np.arange(len(y)) // 4)
    with pytest.raises(TypeError):
        cross_validate_until_redundant(Ridge(), X, y, cv=5)
    with pytest.raises(ValueError, match="exceeds"):
        cross_validate_until_redundant(Ridge(), X, y, cv=ShuffleSplit(n_splits=5), max_splits=6)


def test_loss_checks():
    X, y = _regression(n=200)
    with pytest.raises(ValueError, match="unknown loss"):
        cross_validate_until_redundant(Ridge(), X, y, loss="hinge")
    for name in ("mape", "mean_absolute_percentage_error", "neg_mean_absolute_percentage_error"):
        with pytest.raises(ValueError, match="unknown loss"):  # the absolute error is not a percentage error
            resolve_loss(name)
    for metric in ("r2", "neg_root_mean_squared_error", "roc_auc_ovr", "f1_macro"):
        with pytest.raises(ValueError, match="no redundancy"):
            cross_validate_until_redundant(Ridge(), X, y, loss=metric)
    with pytest.raises(TypeError, match="predict_proba"):  # before any fit
        cross_validate_until_redundant(Ridge(), X, y, loss="log_loss")
    Xc, yc = _classification()
    for name in ("squared_error", "mse", "neg_mean_squared_error"):  # the default, and its metric names
        with pytest.warns(UserWarning, match="on the predicted labels of a classifier"):
            cross_validate_until_redundant(LogisticRegression(), Xc, yc, loss=name, max_splits=3)
    for name in ("absolute_error", "mae"):  # a valid ordinal metric on labels: no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cross_validate_until_redundant(LogisticRegression(max_iter=1000), Xc, yc, loss=name, max_splits=4)
    with pytest.warns(UserWarning) as record:  # a 0-1 loss on a regressor: every loss is 1
        res = cross_validate_until_redundant(Ridge(), Xc, yc, loss="zero_one", max_splits=10)
    assert "but the estimator is a regressor" in str(record[0].message)
    assert any("constant" in str(w.message) for w in record)
    assert res["monitor"].redundancy == 0.0 and np.all(res["test_loss"] == 1.0)
    with warnings.catch_warnings(record=True) as record:  # both checks come before any fit
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="fit was called"):
            cross_validate_until_redundant(_Unfittable(), Xc, yc, loss="zero_one", max_splits=3)
    assert [w.filename for w in record if "but the estimator is a regressor" in str(w.message)] == [__file__]
    with pytest.raises(TypeError, match="predict_proba"):
        cross_validate_until_redundant(_Unfittable(), X, y, loss="brier")


def test_methods_that_exist_once_fitted():
    # a Stacking estimator's predict and predict_proba exist only once it is fitted: no error before the fit
    X, y = _regression()
    assert cross_validate_until_redundant(StackingRegressor([("r", Ridge())]), X, y, max_splits=3)["n_splits"] == 3
    Xc, yc = _classification()
    stacking = StackingClassifier([("l", LogisticRegression(max_iter=1000))])
    assert cross_validate_until_redundant(stacking, Xc, yc, loss="log_loss", max_splits=3)["n_splits"] == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a NotFittedError before the fit, no warning
        res = cross_validate_until_redundant(_ProbaOnceFitted(max_iter=1000), Xc, yc, loss="log_loss", max_splits=3)
    assert res["n_splits"] == 3
    # a method that a parameter or a wrapped estimator disables shows only once fitted: a TypeError after the first
    # fit, with scikit-learn's innermost reason
    with pytest.raises(TypeError, match=r"SVC has no predict_proba.*probability=False"):
        cross_validate_until_redundant(SVC(), Xc, yc, loss="log_loss", max_splits=3)
    with pytest.raises(TypeError, match=r"Pipeline has no predict_proba.*probability=False"):
        cross_validate_until_redundant(make_pipeline(StandardScaler(), SVC()), Xc, yc, loss="log_loss", max_splits=3)


def test_default_cv_and_budget():
    X, y = _regression(n=200)
    res = cross_validate_until_redundant(Ridge(), X, y)  # default: ShuffleSplit(100, test_size=0.2)
    assert res["monitor"].max_splits == 100 and res["monitor"].test_fraction == pytest.approx(0.2)
    assert res["stopped_early"] and res["n_splits"] < 50
    budget = cross_validate_until_redundant(Ridge(), X, y, max_splits=4)
    assert budget["n_splits"] == 4 and not budget["stopped_early"]
    larger = cross_validate_until_redundant(Ridge(), X, y, max_splits=150, tol=1e-9)
    assert larger["n_splits"] == 150
    # stratified for a classifier
    Xc, yc = _classification()
    res = cross_validate_until_redundant(LogisticRegression(), Xc, yc, loss="zero_one", max_splits=5,
                                         return_indices=True)
    expected = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0).split(Xc, yc)
    for test, (_, ref) in zip(res["indices"]["test"], expected):
        assert np.array_equal(test, ref)


def test_fit_params_and_result():
    X, y = _regression(seed=2)
    weights = np.random.default_rng(0).uniform(0.5, 2.0, size=len(y))
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=0)
    res = cross_validate_until_redundant(Ridge(), X, y, cv=cv, params={"sample_weight": weights},
                                         return_estimator=True, return_indices=True)
    for est, train in zip(res["estimator"], res["indices"]["train"]):
        assert est.coef_ == pytest.approx(Ridge().fit(X[train], y[train], sample_weight=weights[train]).coef_)
    pd = pytest.importorskip("pandas")
    res = cross_validate_until_redundant(Ridge(), X, y, cv=cv)
    assert "indices" not in res and len(pd.DataFrame(res)) == res["n_splits"]
