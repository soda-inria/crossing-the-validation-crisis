"""Per-sample losses for the redundancy, on NumPy arrays.

The redundancy must be computed with the per-sample loss of the metric whose precision matters (Appendix D.2): the
mean of these losses over a test set is the metric on that set.

    loss name        metric                                  predictions needed
    squared_error    mean squared error                      predict
    zero_one         error rate, 1 - accuracy                predict (class labels)
    log_loss         log-loss (negative log-likelihood)      predict_proba
    brier            Brier score                             predict_proba

A metric that does not average over observations, such as R^2 or the ROC AUC, has no per-sample loss and no
redundancy (Appendix K of the paper).

The probability losses take ``classes``, the class of each column of the probability matrix (``estimator.classes_``
in scikit-learn); without it, labels are column positions. A 1-D probability vector is P(class 1) of a binary
problem. A true class that is not a column (absent from the training part of a split) has probability 0.
"""
import numpy as np


def squared_error(y_true, y_pred):
    """(y - prediction)^2; averaged over the outputs of a multi-output regression, as the MSE of scikit-learn."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_pred.shape != y_true.shape:
        if y_pred.size != y_true.size:
            raise ValueError(f"y_true of shape {y_true.shape} and predictions of shape {y_pred.shape}")
        y_pred = y_pred.reshape(y_true.shape)  # e.g. (n, 1) predictions for (n,) targets
    err = (y_true - y_pred) ** 2
    return err.reshape(len(err), -1).mean(axis=1) if err.ndim > 1 else err


def zero_one(y_true, y_pred):
    """1 where the predicted label differs from the true label, else 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_pred.shape != y_true.shape:
        if y_pred.size != y_true.size:
            raise ValueError(f"y_true of shape {y_true.shape} and predictions of shape {y_pred.shape}")
        y_pred = y_pred.reshape(y_true.shape)
    return (y_true != y_pred).astype(float)


def _proba_matrix(proba):
    p = np.asarray(proba, dtype=float)
    if p.ndim == 1:
        return np.column_stack([1.0 - p, p])
    if p.ndim != 2:
        raise ValueError(f"probabilities must be 1-D (binary) or 2-D (n, n_classes), got shape {p.shape}")
    return p


def _true_class_columns(y_true, n_columns, classes):
    """Column of each observation's class in the probability matrix, and whether that class is a column."""
    y = np.asarray(y_true)
    if classes is None:
        if not np.issubdtype(y.dtype, np.integer):
            if not (np.issubdtype(y.dtype, np.floating) and np.all(y == np.round(y))):
                raise ValueError("without classes, labels must be integer column positions; pass classes=")
            y = y.astype(int)
        present = (y >= 0) & (y < n_columns)
        return np.where(present, y, 0), present
    classes = np.asarray(classes)
    if classes.shape != (n_columns,):
        raise ValueError(f"{classes.size} classes for {n_columns} probability columns")
    sorter = np.argsort(classes)
    pos = np.clip(np.searchsorted(classes, y, sorter=sorter), 0, n_columns - 1)
    col = sorter[pos]
    return col, classes[col] == y


def log_loss(y_true, proba, classes=None, eps=None):
    """-log(p_true), the per-sample negative log-likelihood (cross-entropy) of predicted probabilities.

    As in ``sklearn.metrics.log_loss``, p_true is clipped to [eps, 1 - eps], with eps the machine epsilon of the
    probabilities' dtype by default, so that the mean over a test set is scikit-learn's log-loss even when a model
    gives probability 0 to the true class (trees, forests, nearest neighbours). The PCam and Yelp scripts of the paper
    use -log(p_true + 1e-12), which differs only when p_true < 1e-10.
    """
    if eps is None:
        dtype = np.asarray(proba).dtype
        eps = np.finfo(dtype if dtype in (np.float16, np.float32, np.float64) else np.float64).eps
    p = _proba_matrix(proba)
    col, present = _true_class_columns(y_true, p.shape[1], classes)
    if len(col) != len(p):
        raise ValueError(f"{len(col)} labels for {len(p)} rows of probabilities")
    p_true = np.where(present, p[np.arange(len(p)), col], 0.0)
    return -np.log(np.clip(p_true, eps, 1.0 - eps))


def brier(y_true, proba, classes=None):
    """sum_c (p_c - 1[y = c])^2, the multiclass Brier score of the PCam and Yelp scripts.

    For two classes it is twice ``sklearn.metrics.brier_score_loss``; the factor does not change the redundancy.
    """
    p = _proba_matrix(proba)
    col, present = _true_class_columns(y_true, p.shape[1], classes)
    if len(col) != len(p):
        raise ValueError(f"{len(col)} labels for {len(p)} rows of probabilities")
    onehot = np.zeros_like(p)
    rows = np.flatnonzero(present)
    onehot[rows, col[rows]] = 1.0
    return np.sum((p - onehot) ** 2, axis=1) + (~present)  # a missing true class adds (0 - 1)^2


# loss name -> (per-sample loss, prediction method of a scikit-learn estimator)
LOSSES = {
    "squared_error": (squared_error, "predict"),
    "zero_one": (zero_one, "predict"),
    "log_loss": (log_loss, "predict_proba"),
    "brier": (brier, "predict_proba"),
}

# metric names (scikit-learn scorer names included) -> loss name
METRIC_LOSSES = {
    "mse": "squared_error", "mean_squared_error": "squared_error", "neg_mean_squared_error": "squared_error",
    "accuracy": "zero_one", "error_rate": "zero_one",
    "nll": "log_loss", "neg_log_loss": "log_loss", "cross_entropy": "log_loss",
    "brier_score": "brier", "neg_brier_score": "brier",
}


# metrics that are not a mean of per-sample losses (prefixes, with or without "neg_")
_NOT_A_MEAN = ("r2", "explained_variance", "root_mean_squared", "rmse", "roc_auc", "average_precision", "f1",
               "precision", "recall", "jaccard", "balanced_accuracy", "matthews_corrcoef", "top_k_accuracy")


def resolve_loss(loss):
    """(per-sample loss function, prediction method, whether it takes ``classes``) for ``loss``.

    ``loss``: a name of ``LOSSES`` or ``METRIC_LOSSES``; a callable ``loss(y_true, y_pred)`` on the output of
    ``predict``, returning one value per sample; or a pair (callable, "predict" or "predict_proba" or
    "decision_function").
    """
    if isinstance(loss, str):
        name = METRIC_LOSSES.get(loss, loss)
        if name.removeprefix("neg_").startswith(_NOT_A_MEAN):
            raise ValueError(f"{loss!r} is not a mean of per-sample losses and has no redundancy (Appendix K of the "
                             "paper); for R^2 or the RMSE, use 'squared_error' (the MSE)")
        if name not in LOSSES:
            raise ValueError(f"unknown loss {loss!r}; use one of {sorted(LOSSES) + sorted(METRIC_LOSSES)}, "
                             "or a callable")
        func, method = LOSSES[name]
        return func, method, func in (log_loss, brier)
    if callable(loss):
        return loss, "predict", False
    if isinstance(loss, tuple) and len(loss) == 2 and callable(loss[0]):
        if loss[1] not in ("predict", "predict_proba", "decision_function"):
            raise ValueError(f"unknown prediction method {loss[1]!r}")
        return loss[0], loss[1], False
    raise TypeError(f"loss must be a name, a callable or a (callable, method) pair, got {loss!r}")
