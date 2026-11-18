"""Stop a Monte Carlo cross-validation with the redundancy, in scikit-learn.

Three regressors on one simulated linear problem (300 observations, 10 features), with a budget of 200 ShuffleSplit
splits and a test fraction of 0.2:
* Ridge, whose fit barely changes from one split to the next: its split predictors err alike, the redundancy is
  near its maximum t = 0.2 and the run stops early;
* an unpruned decision tree, whose split predictors err far more independently: low redundancy, more splits pay;
* a 1-nearest-neighbour regressor, in between: its prediction for an observation changes only when the nearest
  neighbour of that observation leaves the training part.
Then the same statistics after the fact, from five splits of sklearn.model_selection.cross_validate. Runs in a few
seconds on a CPU, from the repository root:

    python cv_redundancy/examples/sklearn_example.py
"""
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# the repository root, so that the example runs from a clone without installing anything
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cv_redundancy import cross_validate_until_redundant, redundancy_from_cv_results  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 10))
y = X @ rng.normal(size=10) + rng.normal(size=300)
budget = 200
cv = ShuffleSplit(n_splits=budget, test_size=0.2, random_state=0)

print(f"Budget {budget} splits, tol 0.1 (stop once the remaining splits would add less than 10% of gain)")
print("Gains on the reference curve (kappa = 0): at the splits run, at the budget, and its ceiling\n")
at_budget = f"curve at {budget}"
print(f"{'learner':<14} {'splits':>6} {'MSE':>6} {'redundancy':>18} {'curve gain':>10} {at_budget:>12} {'ceiling':>8}")
for name, estimator in [("Ridge", Ridge(alpha=1.0)), ("DecisionTree", DecisionTreeRegressor(random_state=0)),
                        ("1-NN", KNeighborsRegressor(n_neighbors=1))]:
    res = cross_validate_until_redundant(estimator, X, y, cv=cv, loss="squared_error")
    m = res["monitor"]
    print(f"{name:<14} {res['n_splits']:>6} {res['test_loss'].mean():>6.2f} "
          f"{m.redundancy:>8.3f} +/- {m.redundancy_se:.3f} {m.projected_gain():>10.1f} "
          f"{m.projected_gain(budget):>12.1f} {m.ceiling:>8.1f}")

# After the fact: five splits of scikit-learn's cross_validate, and would a budget of 200 splits pay?
cv_results = cross_validate(DecisionTreeRegressor(random_state=0), X, y,
                            cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=1),
                            scoring="neg_mean_squared_error", return_estimator=True, return_indices=True)
monitor = redundancy_from_cv_results(cv_results, X, y, loss="squared_error", max_splits=budget)
print(f"\nDecision tree, after 5 splits of cross_validate:\n{monitor}")
