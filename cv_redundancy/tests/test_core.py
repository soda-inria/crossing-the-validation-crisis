"""RedundancyMonitor, projected_gain, stopping_split: batch formula, invariances, jackknife, stopping rule, inputs."""
import math
import re
import warnings

import numpy as np
import pytest

from cv_redundancy import DEFAULT_MIN_SPLITS, DEFAULT_TOL, RedundancyMonitor, projected_gain, stopping_split


def _same(s1, s2):
    """Equal summaries, NaN included."""
    return s1.keys() == s2.keys() and all(
        s1[key] == s2[key] or (isinstance(s1[key], float) and math.isnan(s1[key]) and math.isnan(s2[key]))
        for key in s1)


def _run(seed=0, n=120, n_te=24, k=15, noise=1.0):
    rng = np.random.default_rng(seed)
    effect = rng.normal(size=n)
    tests = [rng.permutation(n)[:n_te] for _ in range(k)]
    losses = [(effect[idx] + noise * rng.normal(size=n_te)) ** 2 for idx in tests]
    return n, tests, losses


def _variable_run(seed=5, n=120, k=8):
    """Test sets of different sizes, as in test_reference."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(size=n)
    tests = [rng.permutation(n)[:rng.integers(10, 40)] for _ in range(k)]
    return n, tests, [(effect[idx] + rng.normal(size=idx.size)) ** 2 for idx in tests]


def _batch(n, tests, losses):
    """Direct computation of Appendix D.2 on test masks over all pairs (independent of the monitor's code)."""
    covs, vars_ = [], []
    for a in range(len(tests)):
        for b in range(a + 1, len(tests)):
            ea, eb = np.full(n, np.nan), np.full(n, np.nan)
            ea[tests[a]], eb[tests[b]] = losses[a], losses[b]
            shared = ~np.isnan(ea) & ~np.isnan(eb)
            if shared.sum() < 2:
                continue
            x, y = ea[shared], eb[shared]
            covs.append(np.sum((x - x.mean()) * (y - y.mean())) / (x.size - 1))
            vars_.append((x.var(ddof=1) + y.var(ddof=1)) / 2)
    rho = np.mean(covs) / np.mean(vars_)
    return np.mean([len(idx) for idx in tests]) / n * rho, len(covs)


def test_incremental_equals_batch():
    n, tests, losses = _run()
    monitor = RedundancyMonitor(n)
    for k, (idx, loss) in enumerate(zip(tests, losses), start=1):
        monitor.add_split(idx, loss)
        if k >= 2:
            omega, n_pairs = _batch(n, tests[:k], losses[:k])
            assert monitor.redundancy == pytest.approx(omega, rel=1e-10) and monitor.n_pairs == n_pairs


def test_invariant_to_split_order():
    n, tests, losses = _run(seed=1)
    reference = RedundancyMonitor.from_splits(n, tests, losses)
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(len(tests))
        permuted = RedundancyMonitor.from_splits(n, [tests[i] for i in order], [losses[i] for i in order])
        assert permuted.n_pairs == reference.n_pairs
        for key in ("loss_correlation", "redundancy", "mean_intersection_size", "redundancy_se"):
            assert getattr(permuted, key) == pytest.approx(getattr(reference, key), rel=1e-12)


def test_invariant_to_order_within_a_split():
    n, tests, losses = _run(seed=2)
    rng = np.random.default_rng(0)
    shuffled = [rng.permutation(len(idx)) for idx in tests]
    a = RedundancyMonitor.from_splits(n, tests, losses)
    b = RedundancyMonitor.from_splits(n, [idx[p] for idx, p in zip(tests, shuffled)],
                                      [loss[p] for loss, p in zip(losses, shuffled)])
    assert _same(a.summary(), b.summary())


@pytest.mark.parametrize("make_run", [lambda: _run(seed=3, k=8), _variable_run], ids=["equal_sizes", "variable_sizes"])
def test_jackknife_matches_leave_one_split_out(make_run):
    n, tests, losses = make_run()
    monitor = RedundancyMonitor.from_splits(n, tests, losses)
    loo = [RedundancyMonitor.from_splits(n, tests[:j] + tests[j + 1:], losses[:j] + losses[j + 1:]).redundancy
           for j in range(len(tests))]
    assert monitor._leave_one_out() == pytest.approx(loo, rel=1e-10)
    k = len(loo)
    se = math.sqrt((k - 1) / k * np.sum((np.array(loo) - np.mean(loo)) ** 2))
    assert monitor.redundancy_se == pytest.approx(se, rel=1e-10)
    assert math.isnan(RedundancyMonitor.from_splits(n, tests[:2], losses[:2]).redundancy_se)


def test_bounds_and_strong_redundancy():
    # identical losses up to a split shift: loss correlation 1, redundancy t = 0.2
    n, tests, _ = _run(seed=4, n=50, n_te=10, k=6)
    effect = np.random.default_rng(0).normal(size=n)
    monitor = RedundancyMonitor.from_splits(n, tests, [effect[idx] + a for a, idx in enumerate(tests)])
    assert monitor.loss_correlation == pytest.approx(1.0) and monitor.redundancy == pytest.approx(0.2)
    assert monitor.ceiling == pytest.approx(5.0)
    for seed in range(5):
        m = RedundancyMonitor.from_splits(*_run(seed=seed, noise=3.0))
        assert -m.test_fraction <= m.redundancy <= m.test_fraction


def test_defaults():
    assert DEFAULT_TOL == 0.1 and DEFAULT_MIN_SPLITS == 3
    monitor = RedundancyMonitor(10)
    assert monitor.tol == DEFAULT_TOL and monitor.min_splits == DEFAULT_MIN_SPLITS and monitor.max_splits is None


def test_projected_gain():
    for omega in (0.0, 0.01, 0.1, 0.2, 0.5):
        assert projected_gain(omega, 1) == pytest.approx(1.0)
    assert projected_gain(0.0, 20) == pytest.approx(20.0)
    assert projected_gain(0.1, 20) == pytest.approx(20 / 2.9)
    for omega in (0.01, 0.05, 0.2):
        assert projected_gain(omega, 1e9) == pytest.approx(1 / omega, rel=1e-6)
        assert projected_gain(omega, math.inf) == pytest.approx(1 / omega)
    assert projected_gain(0.0, math.inf) == math.inf
    assert projected_gain(-0.05, 20) == pytest.approx(20.0)  # negative readings count as 0
    assert math.isnan(projected_gain(math.nan, 20))
    gains = projected_gain(np.array([0.0, 0.05, 0.2])[:, None], np.arange(1, 50)[None, :])
    assert gains.shape == (3, 49)
    assert np.all(np.diff(gains, axis=1) >= 0) and np.all(np.diff(gains, axis=0) <= 0)
    with pytest.raises(ValueError):
        projected_gain(0.1, 0)


def _brute_force_stop(omega, max_splits, tol, min_splits):
    for k in range(min_splits, max_splits + 1):
        if projected_gain(omega, max_splits) / projected_gain(omega, k) - 1 < tol:
            return k
    return max_splits


@pytest.mark.parametrize("max_splits", [5, 20, 200])
@pytest.mark.parametrize("tol", [0.02, 0.1, 0.5])
def test_stopping_split(max_splits, tol):
    omegas = [-0.1, 0.0, 0.01, 0.05, 0.1, 0.2]
    stops = [stopping_split(w, max_splits, tol) for w in omegas]
    assert stops == [_brute_force_stop(w, max_splits, tol, 3) for w in omegas]
    assert stops == sorted(stops, reverse=True)  # higher redundancy, earlier stop
    assert stops[0] == stops[1] == max(3, min(max_splits, math.floor(max_splits / (1 + tol)) + 1))


def test_stopping_split_without_budget():
    assert stopping_split(0.0, None) == math.inf
    for omega in (0.03, 0.07, 0.15):  # away from the ties of (1 - omega) / (tol omega) integer
        k = stopping_split(omega, None, 0.1)
        assert k == math.floor((1 - omega) / (0.1 * omega)) + 1  # G(inf) / G(k) - 1 = (1 - omega) / (k omega)
        assert stopping_split(omega, 100_000, 0.1) <= k
    for max_splits in (None, 20):
        for kwargs in ({"tol": 0}, {"tol": -0.1}, {"min_splits": 1}, {"max_splits": 0}):
            with pytest.raises(ValueError):
                stopping_split(0.1, **{"max_splits": max_splits, **kwargs})


def _redundant_run(n=50, n_te=12, k=60, seed=0):
    rng = np.random.default_rng(seed)
    effect = rng.normal(size=n)
    tests = [rng.permutation(n)[:n_te] for _ in range(k)]
    return n, tests, [effect[idx] + a for a, idx in enumerate(tests)]  # loss correlation 1: redundancy t = 0.24


@pytest.mark.parametrize("max_splits", [20, 200, None])
def test_should_stop_follows_the_rule(max_splits):
    n, tests, losses = _redundant_run(k=60 if max_splits is None else max_splits)
    expected = stopping_split(0.24, max_splits)
    monitor = RedundancyMonitor(n, max_splits=max_splits)
    for k, (idx, loss) in enumerate(zip(tests, losses), start=1):
        stop = monitor.add_split(idx, loss).should_stop()
        assert stop == (k >= expected), (k, expected, monitor)
        if stop:
            assert monitor.remaining_gain() < DEFAULT_TOL
            break
    assert monitor.n_splits == expected


def test_should_stop_waits_for_min_splits_and_a_defined_reading():
    n, tests, losses = _redundant_run()
    monitor = RedundancyMonitor(n, max_splits=20, tol=10.0, min_splits=4)  # any reading would stop
    for k in range(1, 5):
        monitor.add_split(tests[k - 1], losses[k - 1])
        assert monitor.should_stop() == (k >= 4)
    # disjoint test sets: undefined until the budget is spent, which then warns
    monitor = RedundancyMonitor(30, max_splits=3, tol=10.0)
    for idx in np.array_split(np.arange(30), 3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the disjoint test sets may warn
            monitor.add_split(idx, np.ones(idx.size))
        assert not monitor.is_defined and math.isnan(monitor.remaining_gain())
        if monitor.n_splits < 3:
            assert not monitor.should_stop()
    with pytest.warns(UserWarning, match="undefined"):
        assert monitor.should_stop()


def test_should_stop_warns_on_constant_losses():
    n, tests, _ = _run(k=20)
    monitor = RedundancyMonitor.from_splits(n, tests, [np.zeros(idx.size) for idx in tests], max_splits=20)
    assert monitor.is_defined and monitor.mean_pair_variance == 0 and monitor.redundancy == 0.0
    with pytest.warns(UserWarning, match="constant"):
        assert monitor.should_stop()


def test_summary_and_repr():
    n, tests, losses = _run(k=6)
    monitor = RedundancyMonitor(n, max_splits=20)
    assert "undefined" in repr(monitor)
    monitor = RedundancyMonitor.from_splits(n, tests, losses, max_splits=20)
    s = monitor.summary()
    assert s["redundancy"] == monitor.redundancy and s["n_pairs"] == monitor.n_pairs > 0
    assert s["projected_gain"] == pytest.approx(projected_gain(monitor.redundancy, 6))
    assert s["projected_gain_at_max_splits"] == pytest.approx(projected_gain(monitor.redundancy, 20))
    assert s["remaining_gain"] == pytest.approx(s["projected_gain_at_max_splits"] / s["projected_gain"] - 1)
    assert s["test_fraction"] == pytest.approx(0.2)
    text = repr(monitor)
    assert text.startswith("RedundancyMonitor(") and f"{monitor.redundancy:.3f}" in text and "splits=6 of 20" in text
    remaining = re.search(r"\(\+(\d+\.\d\d)% (<|>=) tol 10%\): (stop|continue)\)$", text)
    assert remaining and float(remaining[1]) == pytest.approx(100 * monitor.remaining_gain(), abs=0.005)
    assert (remaining[2] == "<") == (remaining[3] == "stop") == monitor.should_stop()
    text = repr(RedundancyMonitor.from_splits(n, tests, losses))
    assert "splits=6, no budget" in text and "ceiling" in text


def test_inputs():
    n, tests, losses = _run(k=3)
    reference = RedundancyMonitor.from_splits(n, tests, losses)
    # lists, (m, 1) losses, boolean masks (losses in index order)
    as_lists = RedundancyMonitor.from_splits(n, [list(i) for i in tests], [list(e) for e in losses])
    column = RedundancyMonitor.from_splits(n, tests, [e[:, None] for e in losses])
    masks, ordered = [], []
    for idx, loss in zip(tests, losses):
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        masks.append(mask)
        ordered.append(loss[np.argsort(idx)])
    as_masks = RedundancyMonitor.from_splits(n, masks, ordered)
    for other in (as_lists, column, as_masks):
        assert _same(other.summary(), reference.summary())
    one = RedundancyMonitor(10).add_split([3], 0.5)  # one test observation, a scalar loss
    assert one.n_splits == 1 and one.test_fraction == 0.1
    with pytest.raises(ValueError, match="not their mean"):
        RedundancyMonitor(10).add_split([1, 2, 3], 0.5)
    monitor = RedundancyMonitor(10)
    for idx, loss, error in [([1, 1, 2], [0, 1, 2], ValueError),   # duplicate index
                             ([0, 10], [0, 1], ValueError),         # out of range
                             ([-1, 3], [0, 1], ValueError),
                             ([0, 1, 2], [0, 1], ValueError),       # one loss per index
                             ([], [], ValueError),
                             ([0.0, 1.0], [0, 1], TypeError),       # float indices
                             (np.ones(5, dtype=bool), [0] * 5, ValueError),  # mask of the wrong length
                             ([0, 1], ["a", "b"], TypeError)]:
        with pytest.raises(error):
            monitor.add_split(idx, loss)
    assert monitor.n_splits == 0
    for kwargs in ({"n_samples": 1}, {"n_samples": 10, "max_splits": 0}, {"n_samples": 10, "tol": 0},
                   {"n_samples": 10, "min_splits": 1}):
        with pytest.raises(ValueError):
            RedundancyMonitor(**kwargs)
    with pytest.raises(ValueError):
        RedundancyMonitor.from_splits(n, tests, losses[:2])
    with pytest.raises(TypeError):
        RedundancyMonitor(10, 5)  # the rule's parameters are keyword-only


def _kfold(rng, n, n_folds, n_repeats):
    return [fold for _ in range(n_repeats) for fold in np.array_split(rng.permutation(n), n_folds)]


def test_warnings():
    # disjoint test sets of a K-fold-like run, where Monte Carlo sets would surely overlap
    folds = np.array_split(np.random.default_rng(0).permutation(500), 5)
    monitor = RedundancyMonitor(500)
    with pytest.warns(UserWarning, match="K-fold"):
        for idx in folds[:2]:
            monitor.add_split(idx, np.ones(idx.size))
    # repeated 5-fold on small data, where one disjoint pair is plausible under Monte Carlo CV but not three
    rng = np.random.default_rng(0)
    for n in (50, 125, 200):
        folds = _kfold(rng, n, 5, 2)
        with pytest.warns(UserWarning, match="K-fold"):
            RedundancyMonitor.from_splits(n, folds, [rng.normal(size=f.size) for f in folds])
    with pytest.warns(UserWarning, match="non-finite") as record:
        RedundancyMonitor(10).add_split([0, 1], [np.nan, 1.0])
    assert record[0].filename == __file__  # attributed to the caller, not to the package
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Monte Carlo runs raise no warning, even with many disjoint pairs
        RedundancyMonitor.from_splits(*_run())
        RedundancyMonitor.from_splits(*_run(n=125, n_te=25, k=20))
        many = RedundancyMonitor.from_splits(*_run(n=50, n_te=10, k=100))
    assert many._n_disjoint > 300  # about 0.08 of the 4,950 pairs
