"""scikit-learn loop end to end, post hoc monitor, per-sample losses against scikit-learn's metrics."""
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss as sk_log_loss  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GroupShuffleSplit, KFold, ShuffleSplit, StratifiedShuffleSplit, cross_validate,
)
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

from cv_redundancy import (  # noqa: E402
    brier, cross_validate_until_redundant, log_loss, projected_gain, redundancy_from_cv_results, zero_one,
)


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


def test_splitter_checks():
    X, y = _regression(n=200)
    with pytest.warns(UserWarning) as record:  # one repeat of K-fold: never defined
        cross_validate_until_redundant(Ridge(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    messages = [str(w.message) for w in record]
    assert any("ShuffleSplit-like" in m for m in messages) and any("undefined" in m for m in messages)
    assert not any("pairs of splits have disjoint" in m for m in messages)  # flagged once, by the splitter check
    assert all(w.filename == __file__ for w in record)  # attributed to the caller
    with pytest.warns(UserWarning, match="GroupShuffleSplit"):
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
    for metric in ("r2", "neg_root_mean_squared_error", "roc_auc_ovr", "f1_macro"):
        with pytest.raises(ValueError, match="no redundancy"):
            cross_validate_until_redundant(Ridge(), X, y, loss=metric)
    with pytest.raises(TypeError, match="predict_proba"):  # before any fit
        cross_validate_until_redundant(Ridge(), X, y, loss="log_loss")
    Xc, yc = _classification()
    with pytest.warns(UserWarning, match="classifier"):
        cross_validate_until_redundant(LogisticRegression(), Xc, yc, max_splits=3)
    with pytest.warns(UserWarning) as record:  # a 0-1 loss on a regressor: every loss is 1
        res = cross_validate_until_redundant(Ridge(), Xc, yc, loss="zero_one", max_splits=10)
    assert "regressor" in str(record[0].message) and any("constant" in str(w.message) for w in record)
    assert res["monitor"].redundancy == 0.0 and np.all(res["test_loss"] == 1.0)


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
