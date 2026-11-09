"""Main results table for the paper: Pipelines A, B and C on the same fields.

Protocol (fixed before looking at Hisar):
  * Missouri test150 = the 150 locations (75 gully, 75 no gully, seed 42) used to compare models and
    to choose cut-offs. The cut-off for A and B is score >= 3. Pipeline C (logistic regression on the
    19 answers of Pipeline B) is trained on test150 only for the Missouri row.
  * Missouri held-out = the other 471 locations. Nothing was tuned on them.
  * Hisar = 297 labelled locations (10 gully, 3 skipped). For Hisar, C is reported twice:
    trained on test150 only, and trained on all 621 Missouri locations.

Confidence intervals: bootstrap (2000 stratified resamples) for AUC and balanced accuracy;
Wilson intervals for rates on Hisar (false-alarm rate, gullies found).

Usage:
  python 12_results_tables.py --a-dir results/pipeline_a_full --b-dir results/pipeline_b \
      --test150 results/missouri/pipeline_a_gemma4-26b_missouri.csv \
      --labels labels/hisar_labels_final.csv --out results/tables
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

Q = [f"q{i}" for i in range(1, 20)]


def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(y, s, pred, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs, bals = [], []
    for _ in range(reps):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[i], s[i])); bals.append(balanced_accuracy_score(y[i], pred[i]))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(bals, [2.5, 97.5])


def row(region, method, y, s, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    auc_ci, bal_ci = boot_ci(y, s, pred)
    far_ci, sens_ci = wilson(fp, fp + tn), wilson(tp, tp + fn)
    return dict(region=region, method=method, n=len(y), gully=int(y.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                AUC=roc_auc_score(y, s), AUC_lo=auc_ci[0], AUC_hi=auc_ci[1],
                bal_acc=balanced_accuracy_score(y, pred), bal_lo=bal_ci[0], bal_hi=bal_ci[1],
                found=tp / (tp + fn), found_lo=sens_ci[0], found_hi=sens_ci[1],
                false_alarm=fp / (fp + tn), fa_lo=far_ci[0], fa_hi=far_ci[1],
                flagged=(tp + fp) / len(y), precision=tp / max(tp + fp, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-dir", default="results/pipeline_a_full")
    ap.add_argument("--b-dir", default="results/pipeline_b")
    ap.add_argument("--test150", default="results/missouri/pipeline_a_gemma4-26b_missouri.csv",
                    help="any CSV whose tile_id column lists the 150 test locations")
    ap.add_argument("--labels", default="labels/hisar_labels_final.csv")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def table(region, label_from=None):
        ans = pd.read_csv(f"{args.b_dir}/pipeline_b_answers_{region}.csv").drop(columns="label")
        b = pd.read_csv(f"{args.b_dir}/pipeline_b_decisions_{region}.csv")[["tile_id", "label", "score"]].rename(columns={"score": "B"})
        a = pd.read_csv(f"{args.a_dir}/pipeline_a_gemma4-26b_{region}.csv")[["tile_id", "score"]].rename(columns={"score": "A"})
        d = ans.merge(b, on="tile_id").merge(a, on="tile_id")
        if label_from is not None:
            d = d.drop(columns="label").merge(label_from, on="tile_id")
        d = d[d["label"].astype(str).isin(["0", "1", "0.0", "1.0"])].copy()
        d["label"] = d["label"].astype(float).astype(int)
        return d.reset_index(drop=True)

    M = table("missouri")
    H = table("hisar", pd.read_csv(args.labels)[["tile_id", "label"]])
    t150 = set(pd.read_csv(args.test150)["tile_id"])
    T, R = M[M.tile_id.isin(t150)], M[~M.tile_id.isin(t150)]
    print(f"Missouri test150 {len(T)}, held-out {len(R)} ({R.label.sum()} gully); Hisar {len(H)} ({H.label.sum()} gully)")

    def clf(train, test, cols):
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        m.fit(train[cols].values.astype(float), train.label.values)
        return m.predict_proba(test[cols].values.astype(float))[:, 1]

    rows, preds = [], {}
    for region, D in [("Missouri held-out", R), ("Hisar", H)]:
        y = D.label.values
        for name, s, cut in [("A: zero-shot score", D.A.values, 3), ("B: 19 questions + reasoning", D.B.values, 3)]:
            preds[(region, name)] = (s >= cut).astype(int)
            rows.append(row(region, name, y, s, preds[(region, name)]))
        s = clf(T, D, Q); preds[(region, "C (trained on test150)")] = (s >= 0.5).astype(int)
        rows.append(row(region, "C: classifier on answers, trained on test150", y, s, preds[(region, "C (trained on test150)")]))
        if region == "Hisar":
            s = clf(M, D, Q); preds[(region, "C (trained on 621)")] = (s >= 0.5).astype(int)
            rows.append(row(region, "C: classifier on answers, trained on all 621 Missouri", y, s, preds[(region, "C (trained on 621)")]))
    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out}/main_results.csv", index=False)

    # paired test: false alarms of B vs C on Hisar (same fields)
    neg = H.label.values == 0
    lines = []
    for c_name in ["C (trained on 621)", "C (trained on test150)"]:
        b, c = preds[("Hisar", "B: 19 questions + reasoning")].astype(bool), preds[("Hisar", c_name)].astype(bool)
        only_b, only_c, both = int((b & ~c & neg).sum()), int((c & ~b & neg).sum()), int((b & c & neg).sum())
        p = binomtest(only_c, only_b + only_c, 0.5).pvalue if only_b + only_c else 1.0
        lines.append(f"Hisar false alarms, B vs {c_name}: both {both}, only B {only_b}, only C {only_c}, exact McNemar p = {p:.2g}")
    with open(f"{args.out}/paired_tests.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    pd.set_option("display.width", 250)
    show = out.copy()
    for c in ["AUC", "bal_acc"]:
        show[c] = show.apply(lambda r: f"{r[c]:.3f} ({r[c.split('_')[0] + '_lo' if c == 'AUC' else 'bal_lo']:.3f}-"
                                       f"{r['AUC_hi' if c == 'AUC' else 'bal_hi']:.3f})", axis=1)
    show["found"] = show.apply(lambda r: f"{r.TP}/{r.TP + r.FN} = {r.found:.2f} ({r.found_lo:.2f}-{r.found_hi:.2f})", axis=1)
    show["false_alarm"] = show.apply(lambda r: f"{100 * r.false_alarm:.1f}% ({100 * r.fa_lo:.1f}-{100 * r.fa_hi:.1f})", axis=1)
    print(show[["region", "method", "AUC", "bal_acc", "found", "false_alarm", "flagged", "precision"]].round(3).to_string(index=False))
    print("\n".join(lines))
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
