"""Table M.1 PCam rows, every column: D-bar, s_D, t, p, single-split retrieval and K = 20 CV retrieval.

All the columns come from the 20-split MCCV (shuffle_split) runs of the three CNNs (pcam/configs/mccv_cnns.txt:
50 seeds, test fold a fifth), computed as the Yelp rows (code/nlp.py): the per-seed score is the benchmarking accuracy
averaged over the 20 splits; the paired t-test across the 50 seeds compares these scores (D-bar and s_D are the mean
and standard deviation of the per-seed differences) and orders each pair (higher mean first). The single-split
retrieval is the share of seeds in which the first split ranks that network first on test accuracy, the K = 20 CV
retrieval the same share with the test accuracy averaged over the 20 splits; ties count as failures. The study sets
of 30, 100, 300, 500 and 800 images give the table's n_tr = 24, 80, 240, 400, 640.

Input, under the data root (--data-root, else $DATA_ROOT, else data/ at the repository root; see code/datapaths.py):
pcam/results_output_pcam. Prints the table and its LaTeX rows.
"""
import argparse, glob, itertools, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import pandas as pd
from scipy import stats

K = 20
NETS = {"densenet121": "DenseNet121", "mobilenet_v2": "MobileNetV2", "wide_resnet101_2": "WideResNet101\\_2"}
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data-root", default=None,
                help="run outputs: pcam/results_output_pcam under it (default: $DATA_ROOT, else data/ at the repository root)")
a = ap.parse_args()
D = datapaths.data_path("pcam", "results_output_pcam", root=a.data_root)
files = [f for m in NETS for f in glob.glob(f"{D}/pcam_results_CV_shuffle_split_{m}_nsplits_{K}_seeds_*.parquet")]
d = pd.concat([pd.read_parquet(f, columns=["model", "study_size", "seed", "fold", "test_accuracy", "benchmark_accuracy"])
               for f in files], ignore_index=True)
d = d[d.fold <= K]
per = d.groupby(["study_size", "model", "seed"]).agg(bench=("benchmark_accuracy", "mean"), cv=("test_accuracy", "mean"))
per = per.join(d[d.fold == 1].set_index(["study_size", "model", "seed"])["test_accuracy"].rename("single"))
rows = []
for n in sorted(d.study_size.unique()):
    p = per.loc[n]
    for x, y in itertools.combinations(NETS, 2):
        X, Y = p.loc[x], p.loc[y]
        s = X.index.intersection(Y.index)
        X, Y = X.loc[s], Y.loc[s]
        hi, lo, H, L = (x, y, X, Y) if X.bench.mean() >= Y.bench.mean() else (y, x, Y, X)
        diff = H.bench - L.bench
        t, pv = stats.ttest_rel(H.bench, L.bench)
        rows.append(dict(n_tr=int(round(0.8 * n)), highest_mean=hi, lowest_mean=lo, mean_diff=diff.mean(),
                         std_diff=diff.std(ddof=1), t_stat=t, p_value=pv, n_seeds=len(s),
                         single_split_retrieval=(H.single > L.single).mean(), K20_cv_retrieval=(H.cv > L.cv).mean()))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print("\nLaTeX rows of Table M.1:")
for _, q in r.iterrows():
    pv = q.p_value
    ps = "\\textbf{$<$ 0.0001}" if pv < 1e-4 else (f"\\textbf{{{pv:.4f}}}" if pv < 0.05 else f"{pv:.4f}")
    print(f"{NETS[q.highest_mean]} & {NETS[q.lowest_mean]} & {q.n_tr} & {q.mean_diff:.5f} & {q.std_diff:.5f} & {q.t_stat:.3f} & {ps} & "
          f"{round(100 * q.single_split_retrieval)}\\% & {round(100 * q.K20_cv_retrieval)}\\% \\\\")
