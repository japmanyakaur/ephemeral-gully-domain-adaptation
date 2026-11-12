"""Cross-resolution test: how much does image detail change Pipeline A (Gemma 4 26B)?

Reads the runs made by notebooks/13_cross_resolution.ipynb and notebooks/13b_cross_resolution_repeats.ipynb
(in --dir) and the original runs (results/pipeline_a_full). Missing runs are skipped, so the script also works on
partial results.

Missouri (labels from the dataset): native 1 m (6 Nov run), reduced to 2.5 m and to 10 m, and a repeat of the
native images on the 150 test fields (control for run-to-run change). Measured on the 471 held-out fields with the
fixed cut-off score >= 3; AUC needs no cut-off. Paired bootstrap (2000 resamples) for the change against native.

Hisar (labels/hisar_labels_final.csv): super-resolved 2.5 m (6 Nov run), the same images again (control), and
plain 10 m without super-resolution. Paired exact McNemar test on the false alarms of the same no-gully fields.

Usage:
  python 13_cross_resolution_results.py --dir results/crossres --out results/tables
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

TAG = "pipeline_a_gemma4-26b"


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def scores(path, name):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)[["tile_id", "score"]].rename(columns={"score": name})
    d[name] = pd.to_numeric(d[name], errors="coerce")
    return d.dropna(subset=[name])


def boot(y, s_list, reps=2000, seed=0):
    """Bootstrap AUC and balanced accuracy (cut-off 3) for several score columns on the same fields."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(reps):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append([(roc_auc_score(y[i], s[i]), balanced_accuracy_score(y[i], (s[i] >= 3).astype(int))) for s in s_list])
    return np.array(out)          # reps x conditions x 2


def missouri(args):
    native = pd.read_csv(f"{args.a_dir}/{TAG}_missouri.csv")[["tile_id", "label", "score"]].rename(columns={"score": "native_1m"})
    M = native[native["label"].astype(str).isin(["0", "1"])].copy()
    M["y"] = M["label"].astype(int)
    for name, f in [("reduced_2.5m", "missouri_2.5m"), ("reduced_10m", "missouri_10m"),
                    ("native_1m_repeat", "missouri_native_test150")]:
        d = scores(f"{args.dir}/{TAG}_{f}.csv", name)
        if d is not None:
            M = M.merge(d, on="tile_id", how="left")
    t150 = set(pd.read_csv(args.test150)["tile_id"])
    rows = []
    for part, D in [("held-out 471", M[~M["tile_id"].isin(t150)]), ("test150", M[M["tile_id"].isin(t150)])]:
        conds = [c for c in ["native_1m", "reduced_2.5m", "reduced_10m", "native_1m_repeat"] if c in D]
        for c in conds:
            d = D.dropna(subset=[c])
            if len(d) < 20 or d["y"].nunique() < 2:
                continue
            both = d.dropna(subset=["native_1m"])
            y, s, s0 = both["y"].values, both[c].values, both["native_1m"].values
            b = boot(y, [s, s0])
            pred = (s >= 3).astype(int)
            rows.append(dict(region="Missouri", fields=part, condition=c, n=len(d), complete=len(d) == len(D),
                             AUC=roc_auc_score(y, s), AUC_lo=np.percentile(b[:, 0, 0], 2.5), AUC_hi=np.percentile(b[:, 0, 0], 97.5),
                             bal_acc=balanced_accuracy_score(y, pred), said_gully=pred.mean(),
                             found=pred[y == 1].mean(), specificity=1 - pred[y == 0].mean(),
                             AUC_minus_native=roc_auc_score(y, s) - roc_auc_score(y, s0),
                             diff_lo=np.percentile(b[:, 0, 0] - b[:, 1, 0], 2.5),
                             diff_hi=np.percentile(b[:, 0, 0] - b[:, 1, 0], 97.5),
                             native_AUC_same_fields=roc_auc_score(y, s0)))
    return pd.DataFrame(rows)


def hisar(args):
    lab = pd.read_csv(args.labels)[["tile_id", "label"]]
    H = lab[lab["label"].astype(str) != "skip"].copy()
    H["y"] = (H["label"].astype(str) == "1").astype(int)
    H = H.merge(pd.read_csv(f"{args.a_dir}/{TAG}_hisar.csv")[["tile_id", "score"]].rename(columns={"score": "superres_2.5m"}), on="tile_id")
    for name, f in [("superres_2.5m_repeat", "hisar_2.5m_rerun"), ("plain_10m", "hisar_10m")]:
        d = scores(f"{args.dir}/{TAG}_{f}.csv", name)
        if d is not None:
            H = H.merge(d, on="tile_id", how="left")
    rows, tests = [], []
    nneg, npos = int((H["y"] == 0).sum()), int((H["y"] == 1).sum())
    for c in [c for c in ["superres_2.5m", "superres_2.5m_repeat", "plain_10m"] if c in H]:
        d = H.dropna(subset=[c]); y, s = d["y"].values, d[c].values
        pred = (s >= 3).astype(int)
        fp, tp = int(((pred == 1) & (y == 0)).sum()), int(((pred == 1) & (y == 1)).sum())
        lo, hi = wilson(fp, int((y == 0).sum()))
        rows.append(dict(region="Hisar", condition=c, n=len(d), complete=len(d) == len(H), flagged=int(pred.sum()),
                         gullies_found=f"{tp}/{int(y.sum())}", false_alarms=fp, false_alarm_rate=fp / max((y == 0).sum(), 1),
                         far_lo=lo, far_hi=hi, AUC=roc_auc_score(y, s), bal_acc=balanced_accuracy_score(y, pred),
                         found_ids=" ".join(str(t) for t in d["tile_id"][(pred == 1) & (y == 1)])))
    for a, b in [("superres_2.5m", "plain_10m"), ("superres_2.5m", "superres_2.5m_repeat"), ("superres_2.5m_repeat", "plain_10m")]:
        if a in H and b in H:
            d = H.dropna(subset=[a, b]); neg = d["y"].values == 0
            fa, fb = (d[a].values >= 3) & neg, (d[b].values >= 3) & neg
            oa, ob = int((fa & ~fb).sum()), int((fb & ~fa).sum())
            p = binomtest(oa, oa + ob, 0.5).pvalue if oa + ob else 1.0
            tests.append(f"Hisar false alarms, {a} vs {b}: both {int((fa & fb).sum())}, only {a} {oa}, "
                         f"only {b} {ob}, exact McNemar p = {p:.3g}")
    return pd.DataFrame(rows), tests


def figure(mo, hi, tests, path):
    """Two panels: (a) Missouri AUC against pixel size; (b) Hisar false alarms with and without super-resolution.
    The repeat run is left out of the plot when it is identical to the original (it is reported in the tables)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, mark = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    d = mo[(mo["fields"] == "held-out 471") & mo["condition"].isin(["native_1m", "reduced_2.5m", "reduced_10m"])]
    d = d.assign(px=d["condition"].map({"native_1m": 1, "reduced_2.5m": 2.5, "reduced_10m": 10})).sort_values("px")
    a = ax[0]
    a.grid(axis="y", color=grid, lw=0.8); a.set_axisbelow(True)
    a.axhline(0.5, color=muted, lw=1, ls=(0, (4, 3)))
    a.text(15.5, 0.508, "coin toss", va="bottom", ha="right", color=muted, fontsize=8)
    a.errorbar(d["px"], d["AUC"], yerr=[d["AUC"] - d["AUC_lo"], d["AUC_hi"] - d["AUC"]], color=mark, lw=2,
               marker="o", ms=7, capsize=4, elinewidth=1.2, mec="white", mew=1.5)
    for r in d.itertuples():
        right = r.px < 5
        a.annotate(f"AUC {r.AUC:.2f}\n{r.found:.0%} of gullies found", (r.px, r.AUC),
                   xytext=(10, 10) if right else (-12, -34), ha="left" if right else "right",
                   textcoords="offset points", fontsize=8, color=ink)
    a.set(xscale="log", xlim=(0.8, 16), ylim=(0.4, 1.0), xticks=[1, 2.5, 10],
          xticklabels=["1 m\n(original)", "2.5 m", "10 m\n(Sentinel-2 level)"],
          xlabel="Pixel size of the images given to the model", ylabel="AUC (95% interval)")
    a.minorticks_off()
    a.set_title("(a) Missouri, 471 held-out fields", loc="left", fontsize=10, color=ink)

    b = ax[1]
    h = hi[hi["condition"].isin(["superres_2.5m", "plain_10m"])].copy()
    if len(h):
        h["lbl"] = h["condition"].map({"superres_2.5m": "Super-resolved\nto 2.5 m", "plain_10m": "Plain 10 m\n(no super-resolution)"})
        x = range(len(h))
        b.grid(axis="y", color=grid, lw=0.8); b.set_axisbelow(True)
        b.bar(x, h["false_alarm_rate"] * 100, width=0.5, color=mark, edgecolor="none")
        b.errorbar(x, h["false_alarm_rate"] * 100, yerr=[(h["false_alarm_rate"] - h["far_lo"]) * 100,
                   (h["far_hi"] - h["false_alarm_rate"]) * 100], fmt="none", ecolor=ink, elinewidth=1.2, capsize=4)
        for i, r in zip(x, h.itertuples()):
            found, total = r.gullies_found.split("/")
            b.text(i, r.far_hi * 100 + 0.4, f"{r.false_alarms} false alarms\n{found} of {total} gullies found",
                   ha="center", va="bottom", fontsize=8, color=ink)
        p = [t for t in tests if "superres_2.5m vs plain_10m" in t]
        top = h["far_hi"].max() * 100 + 3.6
        if p and len(h) == 2:
            b.plot([0, 0, 1, 1], [top - 0.4, top, top, top - 0.4], color=muted, lw=1)
            b.text(0.5, top + 0.2, f"paired test p = {float(p[0].split('p = ')[1]):.2f}", ha="center", va="bottom",
                   fontsize=8, color=muted)
        b.set_xticks(list(x)); b.set_xticklabels(h["lbl"])
        b.set(xlim=(-0.6, len(h) - 0.4), ylim=(0, top + 2.2), ylabel="False-alarm rate, 287 no-gully fields (%)")
        b.set_title("(b) Hisar, 297 fields", loc="left", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/crossres")
    ap.add_argument("--a-dir", default="results/pipeline_a_full")
    ap.add_argument("--test150", default="results/missouri/test150_ids.csv")
    ap.add_argument("--labels", default="labels/hisar_labels_final.csv")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--fig-dir", default=None, help="where to save crossres_results.png (default: --dir)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mo = missouri(args)
    hi, tests = hisar(args)
    mo.to_csv(f"{args.out}/crossres_missouri.csv", index=False)
    hi.to_csv(f"{args.out}/crossres_hisar.csv", index=False)
    with open(f"{args.out}/crossres_paired_tests.txt", "w") as f:
        f.write("\n".join(tests) + "\n")
    figure(mo, hi, tests, f"{args.fig_dir or args.dir}/crossres_results.png")
    pd.set_option("display.width", 250)
    print(mo.round(3).to_string(index=False))
    print(hi.drop(columns=["found_ids"]).round(3).to_string(index=False))
    print("\n".join(tests))


if __name__ == "__main__":
    main()
