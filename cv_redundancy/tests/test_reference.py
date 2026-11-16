"""Agreement with the implementations behind the paper's numbers, on random inputs.

The reference functions are read from the paper's scripts with ``ast`` and executed with NumPy alone (no benchopt,
torch or transformers import): ``_oof_pair_stats`` and ``_oof_intersection_icc``, methods of the benchopt objective of
simulated/objective.py (in batch and in their cached incremental mode), and ``_pairwise_oof_icc`` of
pcam/pcam_deep_training_adapted.py and of yelp/nlpfinetuning_aligned.py. At every number of splits k >= 2, the
monitor must give the same loss correlation, number of pairs and mean intersection size, up to the order of the
floating-point sums over pairs. Skipped when ``cv_redundancy/`` is used outside the repository, without these scripts.
"""
import ast
import os
import textwrap
import warnings

import numpy as np
import pytest

from cv_redundancy import RedundancyMonitor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RTOL = 1e-12
SCRIPTS = ("simulated/objective.py", "pcam/pcam_deep_training_adapted.py", "yelp/nlpfinetuning_aligned.py")
if not all(os.path.exists(os.path.join(ROOT, path)) for path in SCRIPTS):
    pytest.skip("the paper's scripts are not next to cv_redundancy/", allow_module_level=True)


def _sources(path, names, class_name=None):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        body = ast.parse(f.read()).body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    found = {node.name: node for node in body if isinstance(node, ast.FunctionDef) and node.name in names}
    assert set(found) == set(names), f"{path}: missing {set(names) - set(found)}"
    return [ast.unparse(found[name]) for name in names]


def _simulated_reference():
    methods = _sources("simulated/objective.py", ["_oof_pair_stats", "_oof_intersection_icc"], "Objective")
    namespace = {"np": np}
    exec("class Reference:\n" + textwrap.indent("\n\n".join(methods), "    "), namespace)
    return namespace["Reference"]


def _script_reference(path):
    namespace = {"np": np}
    exec(_sources(path, ["_pairwise_oof_icc"])[0], namespace)
    return namespace["_pairwise_oof_icc"]


SIMULATED = _simulated_reference()
PCAM = _script_reference("pcam/pcam_deep_training_adapted.py")
YELP = _script_reference("yelp/nlpfinetuning_aligned.py")


# ----------------------------------------------------------------------------------------------------------------
# Random runs: test sets and per-sample losses with a sample effect shared across splits plus split noise.
def _monte_carlo(rng, n, n_te, k):
    return [rng.permutation(n)[:n_te] for _ in range(k)]


def _variable_sizes(rng, n, k):
    return [rng.permutation(n)[:rng.integers(2, n // 2)] for _ in range(k)]


def _kfold(rng, n, n_folds, n_repeats):
    return [fold for _ in range(n_repeats) for fold in np.array_split(rng.permutation(n), n_folds)]


def _losses(rng, tests, n, kind):
    effect = rng.normal(size=n)
    out = []
    for idx in tests:
        e = effect[idx] + rng.uniform(0.3, 2.0) * rng.normal(size=idx.size)
        if kind == "squared":
            out.append(e ** 2)
        elif kind == "zero_one":
            out.append((e > 0.8).astype(float))
        elif kind == "constant":
            out.append(np.full(idx.size, 0.5))
        elif kind == "nan":
            e = e ** 2
            e[rng.random(idx.size) < 0.05] = np.nan
            out.append(e)
        else:
            raise ValueError(kind)
    return out


SCENARIOS = {
    # name: (n, test sets, kind of loss)
    "monte_carlo": (80, lambda rng: _monte_carlo(rng, 80, 16, 12), "squared"),
    "monte_carlo_zero_one": (60, lambda rng: _monte_carlo(rng, 60, 12, 10), "zero_one"),
    "variable_test_sizes": (50, lambda rng: _variable_sizes(rng, 50, 9), "squared"),
    "repeated_kfold": (50, lambda rng: _kfold(rng, 50, 5, 3), "squared"),
    "single_kfold": (50, lambda rng: _kfold(rng, 50, 5, 1), "squared"),
    "tiny_n": (7, lambda rng: _monte_carlo(rng, 7, 2, 10), "squared"),
    "constant_losses": (40, lambda rng: _monte_carlo(rng, 40, 8, 6), "constant"),
    "non_finite_losses": (60, lambda rng: _monte_carlo(rng, 60, 15, 8), "nan"),
}


def _run(name, seed):
    n, make_tests, kind = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    tests = make_tests(rng)
    return n, tests, _losses(rng, tests, n, kind)


def _stacks(n, tests, losses, rng):
    """The simulated objective's layout: one row per split over all n observations, and the test masks."""
    q = rng.normal(size=(len(tests), n))  # values outside the test sets are never read
    masks = np.zeros((len(tests), n), dtype=bool)
    for a, (idx, loss) in enumerate(zip(tests, losses)):
        q[a, idx] = loss
        masks[a, idx] = True
    return q, masks


def _close(a, b):
    return a == b or abs(a - b) <= RTOL * max(abs(a), abs(b))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_agrees_with_paper_implementations(name, seed):
    n, tests, losses = _run(name, seed)
    q_stack, masks = _stacks(n, tests, losses, np.random.default_rng(seed + 100))
    simulated_incremental = SIMULATED()
    monitor = RedundancyMonitor(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # disjoint K-fold test sets and NaN losses warn
        for k in range(1, len(tests) + 1):
            monitor.add_split(tests[k - 1], losses[k - 1])
            if k < 2:
                continue
            pcam = PCAM(losses[:k], tests[:k])
            yelp = YELP(losses[:k], tests[:k])
            batch = SIMULATED()._oof_intersection_icc(q_stack[:k], masks[:k])
            incremental = simulated_incremental._oof_intersection_icc(q_stack[:k], masks[:k], cache_key="loss")
            if not monitor.is_defined:
                assert pcam[1] == 0 and np.isnan(pcam[0]) and np.isnan(pcam[2])
                assert yelp[1] == 0 and batch == (None,) * 5 and incremental == (None,) * 5
                assert np.isnan(monitor.loss_correlation) and np.isnan(monitor.redundancy)
                continue
            for ref in (pcam, yelp):
                assert _close(monitor.loss_correlation, ref[0]), (k, monitor.loss_correlation, ref[0])
                assert monitor.n_pairs == ref[1]
                assert _close(monitor.mean_intersection_size, ref[2])
            for ref in (batch, incremental):
                cov, var, rho, n_pairs, size = ref
                assert _close(monitor.mean_pair_covariance, cov)
                assert _close(monitor.mean_pair_variance, var)
                assert _close(monitor.loss_correlation, rho)
                assert monitor.n_pairs == n_pairs and _close(monitor.mean_intersection_size, size)
            t = np.mean([len(idx) for idx in tests[:k]]) / n
            assert _close(monitor.redundancy, t * monitor.loss_correlation)


def test_scenarios_reach_the_edge_cases():
    # undefined readings, pairs skipped for a small or non-finite intersection, a zero mean variance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {name: [RedundancyMonitor.from_splits(*_run(name, seed)) for seed in range(4)] for name in SCENARIOS}
    assert not any(m.is_defined for m in runs["single_kfold"])
    assert all(m.n_pairs < 45 for m in runs["tiny_n"]) and any(m.is_defined for m in runs["tiny_n"])
    assert all(m.mean_pair_variance == 0 and m.loss_correlation == 0.0 for m in runs["constant_losses"])
    assert all(0 < m.n_pairs < 28 for m in runs["non_finite_losses"])
    assert all(0 < m.n_pairs <= 3 * 25 for m in runs["repeated_kfold"])  # folds of different repeats only
