"""Where the analysis, figure and table scripts find their inputs.

Each location is set by an environment variable; a script's --data-root (or --figs-dir) argument, when it has one,
takes precedence over the variable. Relative values are resolved against the current directory.

DATA_ROOT   the run outputs (default: data/ at the root of this repository):
              outputs/      benchopt outputs of the chunked simulated runs, <config>__cNN.parquet
              merged/       the same runs merged per config by simulated/merge_chunks.py, <config>.parquet
              paper_runs/   single-file simulated runs (May and July 2026), benchopt_run_<date>.parquet
              ranking/      the 12 simulated runs of Figure O.1, benchopt_run_<date>.parquet
              pcam/         PatchCamelyon run outputs (results_output_pcam/, stratandnewmetrics/, ...)
              nlp/          Yelp run outputs (yelp_runs_a/, yelp_runs_b/, yelp_qwen3_0.6b/, yelp_aligned/)
            The README of the repository lists every sub-folder and the runs it holds.
FIGS_DIR    what the figure scripts write and the table scripts read back (default: figs/ at the root):
              sim/          paper_style_figs.py --out-dir (simulated gain figures and their tables/)
              real/         regen_real_data_gain_figures.py --out-dir (PCam and Yelp gain figures, tables/)
PCAM_DIR    the pcam/ folder (default: pcam/ of this repository): the outputs of its analysis scripts
            (analysis/out_*/) and the run outputs written next to it (results_output_pcam/).
YELP_DIR    the yelp/ folder (default: yelp/ of this repository): the run outputs written next to it.
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _location(env, default, value=None):
    return os.path.abspath(os.path.expanduser(value or os.environ.get(env) or default))


def data_root(root=None):
    """The data root: ``root`` if given, else $DATA_ROOT, else <repository>/data."""
    return _location("DATA_ROOT", os.path.join(REPO, "data"), root)


def data_path(*parts, root=None):
    """A path under the data root, e.g. data_path("pcam", "results_output_pcam")."""
    return os.path.join(data_root(root), *parts)


def figs_dir(root=None):
    """The figure-output root: ``root`` if given, else $FIGS_DIR, else <repository>/figs."""
    return _location("FIGS_DIR", os.path.join(REPO, "figs"), root)


def figs_path(*parts, root=None):
    """A path under the figure-output root, e.g. figs_path("sim", "tables")."""
    return os.path.join(figs_dir(root), *parts)


def sim_path(*parts):
    """A path inside simulated/, e.g. sim_path("analysis", "out_omega")."""
    return os.path.join(REPO, "simulated", *parts)


def pcam_path(*parts):
    """A path inside the pcam/ folder ($PCAM_DIR, else pcam/ of this repository)."""
    return os.path.join(_location("PCAM_DIR", os.path.join(REPO, "pcam")), *parts)


def yelp_path(*parts):
    """A path inside the yelp/ folder ($YELP_DIR, else yelp/ of this repository)."""
    return os.path.join(_location("YELP_DIR", os.path.join(REPO, "yelp")), *parts)
