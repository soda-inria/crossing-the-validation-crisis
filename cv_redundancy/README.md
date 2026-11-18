# cv_redundancy: when to stop splitting

A small helper to compute the *redundancy* of the paper in your own cross-validation, with scikit-learn or in a
PyTorch training loop, and to stop a Monte Carlo cross-validation (random train/test splits, as `ShuffleSplit`) once
more splits would add little precision.

> Célestin Eve, Gaël Varoquaux and Thomas Moreau (2026). *Crossing the Validation Crisis: Cross-Validation
> Reduces Benchmarking Variance Surprisingly Well.* Section 5.1 and Appendix D.2 (the redundancy). The stopping
> rule is this package's own (see below).

## Quick start

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from cv_redundancy import cross_validate_until_redundant

X, y = load_diabetes(return_X_y=True)
res = cross_validate_until_redundant(
    DecisionTreeRegressor(random_state=0), X, y,
    cv=ShuffleSplit(n_splits=200, test_size=0.2, random_state=0),  # the budget: at most 200 splits
    loss="squared_error",                                           # the per-sample loss of the metric of interest
)
print(res["n_splits"], res["stopped_early"])  # splits run, and whether the rule stopped before the budget
print(res["test_loss"].mean())                # the cross-validation estimate, here the MSE
print(res["monitor"])                         # the redundancy and its statistics after the last split
```

The unpruned tree stops after 79 of the 200 splits (redundancy 0.067). A `Ridge` regression, whose fit barely
changes from one split to the next, stops after 33. Its loss correlation is 0.996, so its redundancy, 0.2005, is just
under its maximum t = 89/442 = 0.2014: its splits err alike, and more of them would mostly repeat what the first ones
measured.

## Install

Nothing to build: the package is this folder. Start Python (or Jupyter) from the repository root, put the root on
`PYTHONPATH` (a script outside the root needs it, even when started from the root; the examples add it themselves), or
copy `cv_redundancy/` next to your code. The folder is under the BSD 3-Clause license of the repository
([LICENSE](../LICENSE), Copyright (c) 2026, Inria); keep that notice if you copy it.

The core needs NumPy only; `cross_validate_until_redundant` needs scikit-learn (`redundancy_from_cv_results`:
scikit-learn >= 1.3), and torch tensors are accepted when torch is installed. Tested with Python 3.12.11 (NumPy
2.2.6, scikit-learn 1.7.1, pandas 2.3.1, torch 2.8.0) and Python 3.11.5 (NumPy 2.4.6, scikit-learn 1.9.1, pandas
3.0.6, no torch).

## With scikit-learn

`cross_validate_until_redundant` fits a clone of the estimator on each split in turn and stops as soon as the rule
below says so. Its arguments follow `sklearn.model_selection.cross_validate`:

* `cv`: a `ShuffleSplit` or `StratifiedShuffleSplit`, whose `n_splits` is the budget. The default is
  `ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)`, stratified for a classifier. `max_splits=` sets the
  budget of the default `cv`, or lowers that of a given `cv` (at most its `n_splits`). Any other splitter runs with a
  warning.
* `loss`: the per-sample loss of the metric whose precision matters: `"squared_error"` (MSE), `"absolute_error"`
  (MAE), `"zero_one"` (accuracy), `"log_loss"`, `"brier"`, a metric name such as `"accuracy"` or `"neg_log_loss"`, a
  callable `loss(y_true, y_pred)` on `predict` that returns one value per sample, or a pair
  `(callable, "predict_proba")`, whose callable gets the `predict_proba` columns in the order of `estimator.classes_`
  (and `classes=estimator.classes_` if it takes a `classes` keyword, as the package's `log_loss` and `brier` do).
  Called directly, `log_loss` and `brier` need `classes=` unless the labels are the column positions 0..C-1.
* `tol` and `min_splits`: the stopping rule. `params`: passed to `fit`; entries with one value per observation, such
  as `sample_weight`, are restricted to the training part. `groups`, `return_estimator`, `return_indices`: as in
  scikit-learn.

It returns a dict like that of `cross_validate`, with `test_loss`, `fit_time`, `score_time` and `redundancy` per
split (`pandas.DataFrame(res)` gives one row per split, unless `return_indices=True`), plus `n_splits`,
`stopped_early` and the `monitor`.

After a run of scikit-learn's own `cross_validate` (X and y as above):

```python
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeRegressor
from cv_redundancy import redundancy_from_cv_results

cv_results = cross_validate(DecisionTreeRegressor(random_state=0), X, y,
                            cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0),
                            return_estimator=True, return_indices=True)
monitor = redundancy_from_cv_results(cv_results, X, y, loss="squared_error", max_splits=100)
print(monitor.remaining_gain())  # 1.77: going on to 100 splits would add 177% of gain, on the reference curve
```

`max_splits` defaults to the splits already run; pass a larger budget to ask whether to continue. Read
`remaining_gain()` rather than `should_stop()`: after 5 splits at t = 0.2, the rule cannot stop with a budget above 6
(with 100, it stops at split 28 at the earliest).

## With PyTorch, or any training loop

`RedundancyMonitor` only needs, after each split, the test indices and the per-sample losses on them (lists, NumPy
arrays or tensors). With your own `train` function and tensors `X`, `y`:

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
from cv_redundancy import RedundancyMonitor

n = len(y)
monitor = RedundancyMonitor(n_samples=n, max_splits=50)                   # tol=0.1, min_splits=3
for train_idx, test_idx in ShuffleSplit(n_splits=50, test_size=0.2).split(np.zeros(n)):
    model = train(X[train_idx], y[train_idx])
    with torch.no_grad():
        logits = model(X[test_idx])
    losses = F.cross_entropy(logits, y[test_idx], reduction="none")      # per-sample log-loss
    # for accuracy instead: losses = (logits.argmax(1) != y[test_idx]).float()
    monitor.add_split(test_idx, losses)
    if monitor.should_stop():
        break
print(monitor)
```

`n_samples` is the number of observations the test sets are drawn from (with a validation subset carved out of each
training part, it still counts in n). [`examples/torch_example.py`](examples/torch_example.py) is a complete loop that
runs in seconds on a CPU; [`examples/sklearn_example.py`](examples/sklearn_example.py) compares a stable and two less
stable regressors.

## The stopping rule

The paper validates the redundancy as a triage signal after 3 or 5 splits (Section 5.1, Appendix D.2, Table D.1; see
[What the redundancy is](#what-the-redundancy-is)). The rule below is this package's own way of acting on it. It uses
the shape of the reference curve (kappa = 0, see below), not an estimate of the gain. After k splits (k >= `min_splits`,
once the redundancy is defined), it projects the gain with G(K) = K / (1 + (K - 1) * max(omega, 0)) and stops when the
splits left in the budget K_max would add less than `tol` of gain:

    G(K_max) / G(k) - 1 < tol

With no budget (`max_splits=None` in `RedundancyMonitor` and `stopping_split`), the rule compares with the ceiling
1 / omega.

**Default `tol = 0.1`.** The test-side part of the variance of the cross-validation estimate is about that of a single
split divided by G (when the variance of a single split scales as 1 / n_te, Appendix D.2). So 10% more gain would
shrink that part of its standard error by less than 5% (1 - 1/sqrt(1.1)), and the total standard error, which also
carries the variation of the data at hand that no number of splits averages out, by even less. Raise `tol` when splits
are expensive, lower it when precision matters more than compute.

**Usually conservative with respect to the coupling.** The reference curve leaves out the train/test coupling kappa
(Appendix D.2; see [What the redundancy is](#what-the-redundancy-is)). It is mostly positive in the paper's simulated
settings, and largest at small training sizes, where the gains fall below the curve: the gain left is then smaller than
projected, and the rule continues longer than strictly needed. The coupling can also be negative (Appendix D.2), with
gains above the curve; the gain left at the stop can then be somewhat above `tol`. All this holds for a given
redundancy: a reading above the population value, through noise, makes the rule stop earlier, most visibly with large
budgets (see the caveat on noisy readings).

Split count at which the rule stops for a redundancy that stays at omega (`stopping_split`; tol = 0.1, min_splits =
3), with the gain on the reference curve at the stop and at the budget:

| omega | K_max = 20 | K_max = 200 | no budget |
|---|---|---|---|
| 0.01 | 18 (15.4 of 16.8) | 154 (60.9 of 66.9) | 991 (90.9 of 100) |
| 0.05 | 17 (9.4 of 10.3) | 93 (16.6 of 18.3) | 191 (18.2 of 20) |
| 0.1 | 16 (6.4 of 6.9) | 61 (8.7 of 9.6) | 91 (9.1 of 10) |
| 0.2 | 13 (3.8 of 4.2) | 33 (4.5 of 4.9) | 41 (4.6 of 5) |

At t = 0.2, omega = 0.2 is the maximum (rho = 1). With a small budget the rule saves few splits: even at the maximum,
going from 12 to 20 splits still adds 11% of gain. The savings come with large budgets, where a highly redundant run
stops after a sixth of 200 splits; hence the default budget of 100 splits.

**`min_splits`.** With the default `tol` and a test fraction up to 0.23, the rule never stops before split 5 unless
the budget is spent, so `min_splits=3` or `5` gives the same stop. It matters with a larger `tol` and a small budget,
or a test fraction well above 0.2 (`stopping_split(0.2, 10, tol=0.5)` and `stopping_split(0.5, 7)` are 4, and 5 with
`min_splits=5`): use 5 then on small data, where a reading after 3 splits is noisy.

## What the redundancy is

Monte Carlo cross-validation (MCCV: K random train/test splits) averages the test scores of its splits, and each split
costs one more fit. How much precision a new split buys depends on how differently the models fitted on different
splits err. The paper measures that precision with the *sample gain* G_K: K splits are worth a single split whose test
set is G_K times larger.

The redundancy reads this from the first splits of one run, on the data at hand alone (the n observations the test
sets are drawn from; no extra data). Two MCCV test sets share observations. On the observations that splits a and b
both hold out, take the per-sample losses of the two fitted models; average their covariance and their variance over
the pairs of splits, and take the ratio (the *loss correlation*, rho). The redundancy is

    omega = t * rho,    t = n_te / n, the test fraction

with n_te the realised mean test size. The redundancy lies between -t and t. When the fitted models err alike
(omega near t), more splits mostly repeat what the first ones measured; when they err independently (omega near 0),
each split still adds about as much as the first.

The paper relates the redundancy to the gain through the *reference curve*

    G(K) = K / (1 + (K - 1) * omega),

which tends to the ceiling 1 / omega as K grows. The curve leaves out the train/test coupling kappa, intrinsic to
cross-validation, which the data at hand cannot estimate: the redundancy is a triage signal, not an estimator of the
gain.

In the paper, after three splits and averaged over the runs of each setting, the redundancy orders 663 simulated
settings by gain G_20 (rank correlation -0.81) and 36 PCam settings (-0.90); both follow the curve, the simulated ones
falling below it at small training sizes (Figure 6, Table D.2). A single run's reading is noisier: after three splits it
correlates with log G_200 at -0.65 on the paper's pre-registered holdout and -0.62 on fresh seeds (rank correlations
-0.36 and -0.42; the holdout's are low mostly because 70% of its settings lie where the curve is flat). Interpret a
single reading asymmetrically: a high one flags runs where more splits are unlikely to bring large gains, a low one says
only that continuing is plausible (Appendix D.2, Table D.1).

The statistics are those of the paper's runs: `tests/test_reference.py` checks that `RedundancyMonitor` gives the
values of `_oof_intersection_icc` (`simulated/objective.py`) and `_pairwise_oof_icc`
(`pcam/pcam_deep_training_adapted.py`, `yelp/nlpfinetuning_aligned.py`).

## What the monitor reports

| Attribute | Meaning |
|---|---|
| `redundancy` | omega_hat = t * rho_hat, in [-t, t]; NaN while undefined |
| `loss_correlation` | rho_hat = mean pair covariance / mean pair variance (0 if the mean variance is 0) |
| `test_fraction` | t = mean test-set size / n_samples (the realised fraction) |
| `n_pairs`, `mean_intersection_size` | pairs of splits that share at least 2 test observations, and how many they share on average (about t n_te) |
| `mean_pair_covariance`, `mean_pair_variance` | the two terms of the ratio |
| `redundancy_se` | leave-one-split-out jackknife standard error of omega_hat (information only, not used by the rule) |
| `projected_gain(K)` | G(K) on the reference curve; `projected_gain()` at the splits run so far |
| `ceiling` | 1 / omega_hat, the limit of the curve |
| `remaining_gain()` | G(K_max) / G(k) - 1 |
| `should_stop()` | the rule; also True once the budget is spent |
| `summary()` | all of the above, as a dict |

`stopping_split(omega, max_splits, tol, min_splits)` gives the table above, `projected_gain(omega, K)` the curve.

## Caveats

* **Monte Carlo cross-validation only.** The factor t holds when the test sets are drawn independently (ShuffleSplit,
  StratifiedShuffleSplit). In K-fold, the splits of one repeat have disjoint test sets and t does not apply.
  `cross_validate_until_redundant` warns on any other splitter, and the monitor warns when its test sets are disjoint
  far more often than independent random test sets would be (one repeat of 5-fold is flagged after 2 to 5 splits from
  n = 42 on, after 2 from n = 271; on smaller data K-fold can go unnoticed). `GroupShuffleSplit` runs with a
  warning: the reference curve assumes independent observations, and the paper has no result on grouped splits.
* **The loss must match the metric** whose precision matters: squared error for the MSE, absolute error for the MAE,
  0-1 error for accuracy, negative log-likelihood for the log-loss, Brier score for the Brier score. A metric that is
  not a mean over observations, such as R^2 or the ROC AUC, has no redundancy (Appendix K); for R^2 or the RMSE, use
  the squared error. A loss that is constant on the shared test observations (a 0-1 loss on the predictions of a
  regressor, a perfect classifier) gives a loss correlation of 0 that carries no information; the monitor warns when a
  run ends on such a reading.
* **Single readings are noisy on small data.** On the PCam runs of Table D.2 (n_tr = 100 to 800, study sets of 125 to
  1,000 images), by our computation (the paper does not report it), the 10-90% range of single-run readings within a
  setting after three splits has a median of 0.11 to 0.14 over the settings of each metric (0.05 to 0.30 for single
  settings), wider than the spread between settings. The rule acts later: with the default `tol` and t = 0.2 it cannot
  stop before split 13 of 20 (33 of 200), and at 13 splits that median is down to 0.03 to 0.04 (0.016 to 0.126 for
  single settings). `redundancy_se` shows only the part of this noise due to the splits drawn, not that due to the data
  at hand.
* **Undefined at first.** The reading needs a pair of test sets that share at least two observations; with small
  test sets this can take more than two splits. Until then `should_stop()` returns False, unless the budget is spent,
  in which case it warns.

## Tests

From the repository root, with pytest installed, in under a minute (the torch tests are skipped without torch):

```bash
pytest -q cv_redundancy/tests
python cv_redundancy/examples/sklearn_example.py
python cv_redundancy/examples/torch_example.py
```

| File | Checks |
|---|---|
| `tests/test_reference.py` | agreement with the paper's three implementations (read from their scripts with `ast`) on random runs: overlapping Monte Carlo test sets, disjoint K-fold folds, tiny n with undefined pairs, constant and non-finite losses, at every number of splits (needs the full repository; skipped otherwise) |
| `tests/test_core.py` | incremental equals a direct batch computation; invariance to the order of the splits; jackknife, with equal and unequal test sizes, one split's losses 1e9 times larger, an undefined leave-one-out; the curve (G(1) = 1, G(K) = K at omega = 0, G -> 1/omega); the stopping rule against brute force, exact ties with `tol` included; input checks; K-fold detection and the other warnings |
| `tests/test_sklearn.py` | Ridge stops before an unpruned tree and 1-NN; post hoc equals incremental; per-sample losses against scikit-learn's metrics (class labels, missing classes, multi-output, MAE); pandas inputs; default cv and budget; splitter and loss checks |
| `tests/test_torch.py` | tensors (float32, bfloat16, boolean masks, attached to a graph, on a device) give the NumPy result |
