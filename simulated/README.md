# Simulated benchmark

The [benchopt](https://benchopt.github.io/) benchmark of the simulated runs of *Crossing the Validation Crisis*
(overview and figure map in the [README at the repository root](../README.md)).

Each run draws a study set from a large simulated dataset, evaluates a learning algorithm with K splits of the study
set, and scores every fitted model on a held-out benchmarking set of 100,000 observations (the oracle) and on outer
chunks of fresh data. The objective also accumulates, split after split, the study-only statistics behind the
redundancy. The analysis scripts of `analysis/` and the figure scripts of [../figures/](../figures/README.md) compute
the sample gains, their bootstrap intervals and the redundancy from these scores.

## Install

Python 3.11 (the runs used 3.11.5), from this folder:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins **benchopt 1.9.0**. From benchopt 1.9.1 on, every cross-validation repetition gets its own copy
of the objective, so the statistics that the objective accumulates across the splits of a run (the `study_*` and
redundancy columns) come out empty. The other pins are the versions the paper's runs used. Install with pip rather
than `benchopt install`, which would also install the requirements of every solver.

## Run a configuration

From this folder:

```bash
# a one-seed check that runs in minutes
benchopt run . --config configs/preflight_derivation.yml --no-plot --no-cache --no-timeout -j 4
# one seed chunk (25 of the 1,000 seeds) of the main simulated experiment of Figures 4 and 5, on a 40-core node
benchopt run . --config configs/chunks_paper_main_mse/paper_main_gain_mse_k200__c00.yml \
    --no-plot --no-cache --no-timeout -j 40 --output paper_main_gain_mse_k200__c00
```

benchopt writes `outputs/<name>.parquet`: one row per (dataset, solver, objective parameters, split), with the
parameters in `p_dataset_*`, `p_solver_*` and `p_obj_*` columns and the scores and statistics in `objective_*`
columns. `python check_run_integrity.py outputs/<name>.parquet` checks the split bookkeeping of a result file.

## How the paper's runs were made

- **Chunked runs.** Most configurations were split into seed chunks (`make_chunks.py`); the chunks of every paper run
  and their manifests are in `configs/chunks_<design>/`. Each chunk ran as one task of a SLURM job array
  (`submit_<design>.slurm`, written for the CPU nodes of the Jean-Zay cluster: set the account, the array range and
  the environment, and `mkdir -p logs` first), with `benchopt run ... -j 40 --output <config>__cNN`.
  `merge_chunks.py` merges the chunk outputs per config and `check_run_integrity.py` checks them. Step by step:
  [docs/jean-zay.md](docs/jean-zay.md).
- **Single runs.** The ranking runs of Figure O.1 and the July 2026 runs (hyper-parameter search, frozen vs resampled
  splits, small budgets) ran as one `benchopt run . --config configs/<file>.yml` each on a 128-core machine. benchopt
  named their outputs `benchopt_run_<date>_<time>.parquet`, and the figure scripts read them under these names.

[configs/README.md](configs/README.md) lists every configuration, the launcher it belongs to, and where the paper
uses it. The [README at the repository root](../README.md) gives the data root under which the figure scripts expect
the outputs.

## Analyses of the redundancy runs

Run from this folder, in the environment above; they write their tables to `analysis/out_<name>/`.

| Script | Paper |
|---|---|
| `analysis/derivation_levers_analysis.py` | sample gains with bootstrap intervals, per configuration, of the redundancy designs (Figure 6, Table D.1) |
| `analysis/omega_gain_relation.py` | redundancy against sample gain, per configuration (the simulated points of Figure 6, `analysis/out_omega/`) |
| `analysis/redundancy_triage.py`, `analysis/redundancy_triage_paper_outputs.py` | single-run early stopping on the pre-registered holdout (Table D.1) |
| `analysis/testsize_factor_check.py` | test-fraction check of Appendix D.2 |
| `analysis/paper_appendix_redundancy_checks.py` | redundancy claims of Appendix G |
| `analysis/persample_r2_centering.py` | Appendix K: the R^2 gain of a heavily regularized Ridge, recomputed with the fold mean or the study mean |

The usage line of each script is at the top of its file; the [README at the repository root](../README.md) gives the
calls behind Table D.1.

## Layout

```
objective.py, datasets/, solvers/, benchmark_utils/   the benchopt benchmark
test_config.py, tests/      benchopt's test hook; tests of the objective's incremental statistics (pytest)
configs/                    run configurations, and the seed chunks of the chunked runs (chunks_<design>/)
submit_<design>.slurm       SLURM job arrays of the chunked runs (Jean-Zay)
make_chunks.py, merge_chunks.py, check_run_integrity.py
analysis/                   analyses of the redundancy runs
docs/jean-zay.md            running the chunked runs as SLURM arrays
```
