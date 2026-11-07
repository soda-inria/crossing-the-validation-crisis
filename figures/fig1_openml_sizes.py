#!/usr/bin/env python3
"""Figure 1: number of samples of the 20% most used OpenML datasets.

The figure code is that of the notebook that drew the paper's figure (openml_datasets_distributions.py), without its
exploratory cells. What it does:

- lists every OpenML dataset and every OpenML task (openml.datasets.list_datasets and openml.tasks.list_tasks with
  output_format="dataframe" and no other argument) and keeps the datasets whose NumberOfInstances and NumberOfFeatures
  are both known;
- popularity of a dataset = its number of OpenML tasks (tasks["did"].value_counts(); 0 for a dataset without a task);
- subset = the ceil(n * top_percent / 100) datasets with the most tasks, out of the n datasets kept (get_popular_subset,
  --top-percent 20). Datasets tied at the cut are taken in the order that pandas' default sort leaves them in;
- histogram of their NumberOfInstances ("Number of samples"), capped at --max-value = 5,000: a larger dataset counts as
  5,001 and falls in the last bin, labelled "5000+". There are 10 bins: [0, 500), [500, 1000), ..., [4000, 4500) and
  [4500, 5001]. The first two, [0, 1000), hold the datasets with fewer than 1,000 samples and are drawn in Dark2
  colour 0 (green), the others in Dark2 colour 5 (yellow). Above each bar is its share of the subset ("<1%" below 1%),
  at font size 9 for bins 1-2, 8 for bins 3-4, 7 for bins 5-6 and 6 for bins 7-10. Figure size 4 x 2.3 in, seaborn
  despine, tight layout;
- centre text: the share of the subset with fewer than 1,000 samples (NumberOfInstances < 1000), that is, of the two
  green bins. The script computes and prints it (the notebook typed "55%").

The notebook's bins were 10 bins of width 500.1 from 0 to 5,001 (np.linspace(0, max_value + 1, 11)), so its two green
bins, [0, 1000.2), also counted the datasets with exactly 1,000 samples: their labels could add up to more than the
centre share. The bins above end the green bins at 1,000 exactly.

OpenML changes over time (datasets and tasks are added and deactivated), so a run today may differ slightly from the
paper's figure. The first version of the paper's figure (written by matplotlib 3.10.0 on 2026-01-28 with the
notebook's bins; the date of its OpenML query is not recorded) showed 1,276 datasets. Its bar heights give the bin
counts 558, 144, 41, 48, 18, 7, 19, 13, 8 and 420, so 702 datasets (55.0%) were in the two green bins. The present
figure was drawn with the bins above from the listings of 2026-11-05 15:15 UTC: 1,419 of the 7,092 datasets kept, bin
counts 627, 152, 90, 39, 31, 12, 19, 13, 8 and 428, so 779 datasets (54.9%) have fewer than 1,000 samples.

--cache FILE keeps the two listings in a local .npz file, which is not tracked. It holds the columns the figure reads
(the id, NumberOfInstances and NumberOfFeatures of each dataset, and the dataset id of each task) and the date of the
query. The first run queries OpenML and writes FILE; later runs read FILE and do not query OpenML again (delete FILE to
query again). Without --cache, every run queries OpenML.

Environment: the packages of requirements.txt, plus openml, which is needed only to query OpenML and not to
redraw from --cache. Run in figures/:
    python fig1_openml_sizes.py --out-dir ../figs/fig1 --cache ../figs/fig1/openml_listings.npz
The script writes fig1_openml_dataset_sizes.pdf, the paper's file name (top_<top-percent>_percent_histogram.pdf for
another --top-percent than 20), to --out-dir. It prints the date of the OpenML query, the size of the subset, its median number of samples and the centre
share.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
import paper_names as N  # noqa: E402

SMALL = 1000  # the centre text gives the share of the subset with fewer than 1,000 samples
# the columns of the two OpenML listings that the figure reads, and that --cache keeps
CACHE_COLUMNS = {"datasets": ["did", "NumberOfInstances", "NumberOfFeatures"], "tasks": ["did"]}


# -------------------------
# OpenML listings
# -------------------------

def query_openml():
    """The dataset and task listings of OpenML, queried as in the notebook (needs network access to OpenML)."""
    import openml  # needed only here: redrawing from --cache does not need openml

    datasets = openml.datasets.list_datasets(output_format="dataframe")
    tasks = openml.tasks.list_tasks(output_format="dataframe")
    return datasets, tasks


def save_cache(path, datasets, tasks, queried):
    """Write the columns CACHE_COLUMNS of the two listings, in their row order, and the date of the query to `path`."""
    arrays = {"queried": np.array(queried)}
    for name, frame in (("datasets", datasets), ("tasks", tasks)):
        for col in CACHE_COLUMNS[name]:
            if col == "did":
                arrays[f"{name}__{col}"] = frame[col].to_numpy(dtype=np.int64)
            else:  # the counts stay float, NaN marking a missing count (dropped later, as in the notebook)
                arrays[f"{name}__{col}"] = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:  # given a file object, np.savez does not append ".npz" to the name
        np.savez(f, **arrays)


def load_cache(path):
    """The two listings (columns CACHE_COLUMNS) and the date of the query, as save_cache wrote them."""
    with np.load(path, allow_pickle=False) as z:
        datasets = pd.DataFrame({col: z[f"datasets__{col}"] for col in CACHE_COLUMNS["datasets"]})
        tasks = pd.DataFrame({col: z[f"tasks__{col}"] for col in CACHE_COLUMNS["tasks"]})
        queried = str(z["queried"])
    return datasets, tasks, queried


def load_listings(cache=None):
    """The two OpenML listings and the date of the query: read from `cache` if that file exists, else queried from
    OpenML (and written to `cache` if it is given)."""
    if cache is not None and os.path.exists(cache):
        datasets, tasks, queried = load_cache(cache)
        print(f"OpenML listings read from {cache}: OpenML queried on {queried}, "
              f"{len(datasets)} datasets, {len(tasks)} tasks")
        return datasets, tasks, queried
    queried = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    datasets, tasks = query_openml()
    print(f"OpenML queried on {queried}: {len(datasets)} datasets, {len(tasks)} tasks")
    if cache is not None:
        save_cache(cache, datasets, tasks, queried)
        print(f"OpenML listings saved to {cache}")
    return datasets, tasks, queried


def datasets_with_popularity(cache=None):
    """The OpenML datasets whose NumberOfInstances and NumberOfFeatures are known, with their number of OpenML tasks
    (column num_tasks). This is the first cell of the notebook, without the exploratory collections."""
    datasets, tasks, _ = load_listings(cache)

    # Keep clean datasets only
    datasets = datasets.dropna(
        subset=["NumberOfInstances", "NumberOfFeatures"]
    )

    datasets["NumberOfInstances"] = datasets["NumberOfInstances"].astype(int)
    datasets["NumberOfFeatures"] = datasets["NumberOfFeatures"].astype(int)

    # --------------------------------
    # Dataset popularity (from tasks): number of OpenML tasks per dataset
    # --------------------------------
    task_counts = (
        tasks["did"]
        .value_counts()
        .rename("num_tasks")
        .reset_index()
        .rename(columns={"index": "did"})
    )

    datasets = datasets.merge(task_counts, on="did", how="left")
    datasets["num_tasks"] = datasets["num_tasks"].fillna(0)
    return datasets


# -------------------------
# Figure 1
# -------------------------

def get_popular_subset(popular_df, top_n=None, top_percent=None):
    """The top_n datasets with the most tasks, or the ceil(top_percent % of all) datasets with the most tasks."""
    if top_n is not None:
        subset = popular_df.sort_values("num_tasks", ascending=False).head(top_n)
    elif top_percent is not None:
        count = int(np.ceil(len(popular_df) * (top_percent / 100)))
        subset = popular_df.sort_values("num_tasks", ascending=False).head(count)
    else:
        raise ValueError("Specify either top_n or top_percent.")
    return subset


def histogram_bin_edges(max_value):
    """The 10 bins of the histogram: width max_value / 10 from 0 to max_value, the last one closed at max_value + 1 so
    that it holds the capped datasets (np.histogram closes its last bin). At max_value = 5000 the first two, the green
    ones, are [0, 500) and [500, 1000)."""
    return np.append(np.linspace(0, max_value, 11)[:-1], max_value + 1)


def figure_name(top_percent):
    """The paper's file name for the 20% most used datasets (Figure 1), a descriptive one otherwise."""
    return N.FIG1 if top_percent == 20 else f"top_{top_percent}_percent_histogram.pdf"


def plot_capped_kde_hist_for_top_x_percent(datasets, top_percent, max_value=10_000, center_text=None, out_dir="."):
    """Figure 1: histogram of the number of samples of the top_percent % most used datasets, capped at max_value, saved
    to out_dir/<figure_name(top_percent)> and then shown (plt.show(), as in the notebook). The notebook's
    name is kept, although the function draws no KDE."""
    top_x = get_popular_subset(datasets, top_percent=top_percent)
    capped_instances = top_x["NumberOfInstances"].copy()
    capped_instances[capped_instances > max_value] = max_value + 1

    # 10 bins, start at 0; the first two end at 1,000 exactly at max_value = 5000
    bin_edges = histogram_bin_edges(max_value)

    colors = sns.color_palette("Dark2")
    plt.figure(figsize=(4, 2.3))
    counts, bins, patches = plt.hist(
        capped_instances, bins=bin_edges, edgecolor='white', alpha=1, color=colors[5]
    )
    # Change color of first two bins
    if len(patches) >= 2:
        patches[0].set_facecolor(colors[0])
        patches[1].set_facecolor(colors[0])

    plt.xlim(0, max_value + 1)
    plt.xscale("linear")
    # plt.grid(axis='x', which='both')
    # x ticks every 1,000 samples, the last one (max_value) labelled "<max_value>+"
    xticks = np.arange(0, max_value + 2, 1000)
    xtick_labels = [str(x) if x != max_value else f"{max_value}+" for x in xticks]
    plt.xticks(xticks, xtick_labels)
    plt.xlabel("Number of samples")
    # plt.title(f"Histogram (Top {top_percent}%)")

    # Annotate proportions above bars with rotated text
    total = counts.sum()
    rotation_angle = 0  # Set your desired angle here
    for i, (count, patch) in enumerate(zip(counts, patches)):
        height = patch.get_height()
        if height > 0:
            proportion = count / total
            # Use larger fontsize for first two bins
            fontsize = 9 if i < 2 else 8 if i < 4 else 7 if i < 6 else 6
            plt.text(
                patch.get_x() + patch.get_width() / 2,
                height,
                f"{proportion:.0%}" if proportion >= 0.01 else "<1%",
                ha='center',
                va='bottom',
                fontsize=fontsize,
                rotation=rotation_angle
            )

    # Add text in the center of the plot if specified
    if center_text is not None:
        plt.gca().text(
            0.5, 0.5, center_text,
            transform=plt.gca().transAxes,
            ha='center', va='center',
            fontsize=12, color=colors[0], alpha=0.7, fontdict={'weight': 'bold'}
        )

    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(out_dir, figure_name(top_percent)), format="pdf")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Figure 1: number of samples of the most used OpenML datasets.")
    ap.add_argument("--top-percent", type=int, default=20,
                    help="share of the datasets with the most OpenML tasks, in percent (default 20)")
    ap.add_argument("--max-value", type=int, default=5000,
                    help="cap on the number of samples: larger datasets fall in the last bin, labelled "
                         "'<max-value>+' (default 5000)")
    ap.add_argument("--out-dir", default=".",
                    help="where to write the figure (default: the current directory)")
    ap.add_argument("--cache", metavar="FILE", default=None,
                    help="local .npz file (not tracked) of the two OpenML listings: read if it exists, else written "
                         "after the query, so that the figure can be redrawn without querying OpenML again")
    args = ap.parse_args()

    datasets = datasets_with_popularity(args.cache)

    # the subset that the plotting function selects again
    subset = get_popular_subset(datasets, top_percent=args.top_percent)
    n_samples = subset["NumberOfInstances"]
    # centre text: share of the subset with fewer than 1,000 samples (NumberOfInstances < 1000)
    share = (n_samples < SMALL).mean()
    # the two green bins: [0, 1000) at max_value = 5000, the same datasets as the centre share
    green_edge = histogram_bin_edges(args.max_value)[2]
    green = (n_samples < green_edge).mean()
    print(f"top {args.top_percent}% of the {len(datasets)} datasets by number of tasks: {len(subset)} datasets, "
          f"median {n_samples.median():g} samples")
    print(f"fewer than 1,000 samples: {share:.1%} (centre text: {share:.0%}); "
          f"two green bins, [0, {green_edge:g}): {green:.1%}")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_capped_kde_hist_for_top_x_percent(
        datasets, top_percent=args.top_percent, max_value=args.max_value,
        center_text=f"{share:.0%} have less than\n1,000 samples", out_dir=args.out_dir
    )
    print("wrote", os.path.join(args.out_dir, figure_name(args.top_percent)))


if __name__ == "__main__":
    main()
