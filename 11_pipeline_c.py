"""Pipeline C: lightweight classifier on VLM-derived features, with limited local labels.

Features  : the 19 yes/no answers from Pipeline B (Gemma 4 26B), optionally plus the Pipeline A score.
Models    : logistic regression (class-balanced) and a small MLP (one hidden layer of 16, as in
            Tousi et al. 2025, trained on class-balanced resampled data).
Runs on a laptop CPU in about a minute; no GPU needed.

Experiments
  1. Missouri in-domain: 5-fold stratified cross-validation on the 621 Missouri locations, 5 repeats.
  2. Transfer, no local labels: train on all 621 Missouri locations, test on the 297 labelled Hisar locations.
  3. Limited local labels: k labelled Hisar locations (k = 10, 25, 50, 100, 150, 200; drawn at random,
     200 repeats) are used in four ways, and every variant is tested on the Hisar locations not drawn:
       a. missouri+local          : added to the Missouri training data (each location weighted equally)
       b. missouri+local_weighted : added with extra weight, so Hisar and Missouri carry equal total weight
       c. local_only              : a classifier trained on the k Hisar locations alone
       d. missouri_local_cutoff   : the Missouri-trained classifier is kept, and the k locations are used only
                                    to set its cut-off (aim: about 10% false alarms on the local no-gully fields)
     The same test locations are also scored with Pipeline A and Pipeline B for a fair comparison.
Experiments 3a-3d use logistic regression only (the MLP gave the same results in experiments 1 and 2).

Decision rule: probability >= 0.5 (classes balanced during training), except 3d.

Usage:
  python 11_pipeline_c.py --b-dir results/pipeline_b --labels labels/hisar_labels_final.csv \
      --a-dir results/pipeline_a_full --out results/pipeline_c
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
Q = [f"q{i}" for i in range(1, 20)]
KS = [10, 25, 50, 100, 150, 200]


def load(args):
    am = pd.read_csv(f"{args.b_dir}/pipeline_b_answers_missouri.csv")
    ah = pd.read_csv(f"{args.b_dir}/pipeline_b_answers_hisar.csv").drop(columns="label")
    bm = pd.read_csv(f"{args.b_dir}/pipeline_b_decisions_missouri.csv")[["tile_id", "score"]].rename(columns={"score": "b_score"})
    bh = pd.read_csv(f"{args.b_dir}/pipeline_b_decisions_hisar.csv")[["tile_id", "score"]].rename(columns={"score": "b_score"})
    pa_m = pd.read_csv(f"{args.a_dir}/pipeline_a_gemma4-26b_missouri.csv")[["tile_id", "score"]].rename(columns={"score": "a_score"})
    pa_h = pd.read_csv(f"{args.a_dir}/pipeline_a_gemma4-26b_hisar.csv")[["tile_id", "score"]].rename(columns={"score": "a_score"})
    lab = pd.read_csv(args.labels)[["tile_id", "label"]]
    M = am.merge(bm, on="tile_id").merge(pa_m, on="tile_id")
    H = ah.merge(bh, on="tile_id").merge(pa_h, on="tile_id").merge(lab, on="tile_id")
    H = H[H["label"].astype(str) != "skip"].copy()
    H["label"] = (H["label"].astype(str) == "1").astype(int)
    M["label"] = M["label"].astype(int)
    return M.reset_index(drop=True), H.reset_index(drop=True)


def balanced_resample(X, y, rng):
    """Oversample the smaller class so the MLP sees both classes equally."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    n = max(len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, n), rng.choice(neg, n)])
    return X[idx], y[idx]


def fit_predict(kind, Xtr, ytr, Xte, seed=0, weights=None):
    if len(set(ytr)) < 2:                       # only one class in training: predict that class
        return np.full(len(Xte), float(ytr[0]))
    if kind == "logreg":
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        m.fit(Xtr, ytr, sample_weight=weights)
        return m.predict_proba(Xte)[:, 1]
    rng = np.random.default_rng(seed)
    Xb, yb = balanced_resample(Xtr, ytr, rng)
    m = MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, max_iter=2000, random_state=seed)
    m.fit(Xb, yb)
    return m.predict_proba(Xte)[:, 1]


def wilson(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rates(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(flagged=pred.mean(), far=fp / max(fp + tn, 1),
                sens=(tp / (tp + fn)) if (tp + fn) else np.nan,
                auc=roc_auc_score(y, p) if len(set(y)) == 2 and len(set(p)) > 1 else np.nan,
                bal=balanced_accuracy_score(y, pred) if len(set(y)) == 2 else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-dir", default="results/pipeline_b")
    ap.add_argument("--a-dir", default="results/pipeline_a_full")
    ap.add_argument("--labels", default="labels/hisar_labels_final.csv")
    ap.add_argument("--out", default="results/pipeline_c")
    ap.add_argument("--repeats", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    M, H = load(args)
    print(f"Missouri {len(M)} ({M.label.sum()} gully); Hisar {len(H)} ({H.label.sum()} gully)")

    feature_sets = {"answers": Q, "answers+A": Q + ["a_score"]}
    rows = []

    # 1. Missouri in-domain cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    for fs, cols in feature_sets.items():
        for kind in ["logreg", "mlp"]:
            aucs, bals = [], []
            X, y = M[cols].values.astype(float), M.label.values
            for f, (tr, te) in enumerate(cv.split(X, y)):
                p = fit_predict(kind, X[tr], y[tr], X[te], seed=f)
                r = rates(y[te], p); aucs.append(r["auc"]); bals.append(r["bal"])
            rows.append(dict(experiment="1_missouri_cv", features=fs, model=kind, k=np.nan,
                             auc=np.mean(aucs), auc_sd=np.std(aucs), bal=np.mean(bals), bal_sd=np.std(bals)))

    # 2. Transfer with no local labels
    yH = H.label.values
    nneg, npos = int((yH == 0).sum()), int((yH == 1).sum())

    def add_ci(r, pred):
        fp = int(((pred == 1) & (yH == 0)).sum()); tp = int(((pred == 1) & (yH == 1)).sum())
        r.update(fp=fp, tp=tp, far_lo=wilson(fp, nneg)[0], far_hi=wilson(fp, nneg)[1],
                 sens_lo=wilson(tp, npos)[0], sens_hi=wilson(tp, npos)[1])
        return r

    for fs, cols in feature_sets.items():
        for kind in ["logreg", "mlp"]:
            p = np.mean([fit_predict(kind, M[cols].values.astype(float), M.label.values,
                                     H[cols].values.astype(float), seed=s) for s in range(5 if kind == "mlp" else 1)], axis=0)
            r = add_ci(rates(yH, p), (p >= 0.5).astype(int))
            rows.append(dict(experiment="2_transfer_k0", features=fs, model=kind, k=0, **r))
    for name, col, thr in [("pipeline_A", "a_score", 3), ("pipeline_B", "b_score", 3)]:
        s = H[col].values
        r = rates(yH, (s >= thr).astype(float))
        r["auc"] = roc_auc_score(yH, s)
        rows.append(dict(experiment="2_transfer_k0", features="-", model=name, k=0,
                         **add_ci(r, (s >= thr).astype(int))))

    # which questions the Missouri-trained classifier relies on
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(M[Q].values.astype(float), M.label.values)
    coef = pd.DataFrame({"question": Q, "weight": m.coef_[0],
                         "yes_missouri_gully": M[M.label == 1][Q].mean().values,
                         "yes_missouri_nogully": M[M.label == 0][Q].mean().values,
                         "yes_hisar_gully": H[H.label == 1][Q].mean().values,
                         "yes_hisar_nogully": H[H.label == 0][Q].mean().values})
    coef.sort_values("weight", key=abs, ascending=False).to_csv(f"{args.out}/pipeline_c_question_weights.csv", index=False)

    # 3 and 4. Limited local labels
    rng = np.random.default_rng(42)
    draws = []
    for k in KS:
        for rep in range(args.repeats):
            idx = rng.permutation(len(H))
            tr, te = idx[:k], idx[k:]
            if H.label.values[te].sum() == 0:      # keep at least one gully in the test part
                continue
            draws.append((k, rep, tr, te))
    rec = []
    base = {}   # Missouri-only classifier, trained once per feature set
    for fs, cols in feature_sets.items():
        base[fs] = fit_predict("logreg", M[cols].values.astype(float), M.label.values, H[cols].values.astype(float))
    for k, rep, tr, te in draws:
        yte = H.label.values[te]
        for fs, cols in feature_sets.items():
            Xm, ym = M[cols].values.astype(float), M.label.values
            Xh, yh = H[cols].values.astype(float), H.label.values
            Xtr, ytr = np.vstack([Xm, Xh[tr]]), np.concatenate([ym, yh[tr]])
            # a. pooled, equal weight per location
            p = fit_predict("logreg", Xtr, ytr, Xh[te])
            rec.append(dict(k=k, rep=rep, features=fs, model="logreg_missouri+local", **rates(yte, p)))
            # b. pooled, Hisar part carries the same total weight as Missouri
            w = np.concatenate([np.ones(len(ym)), np.full(k, len(ym) / k)])
            p = fit_predict("logreg", Xtr, ytr, Xh[te], weights=w)
            rec.append(dict(k=k, rep=rep, features=fs, model="logreg_missouri+local_weighted", **rates(yte, p)))
            # c. Hisar labels only
            p = fit_predict("logreg", Xh[tr], yh[tr], Xh[te])
            rec.append(dict(k=k, rep=rep, features=fs, model="logreg_local_only", **rates(yte, p)))
            # d. Missouri classifier, cut-off set on the local no-gully fields (target about 10% false alarms)
            neg = base[fs][tr][yh[tr] == 0]
            thr = np.quantile(neg, 0.9, method="higher") + 1e-9 if len(neg) else 0.5
            rec.append(dict(k=k, rep=rep, features=fs, model="logreg_missouri_local_cutoff",
                            **rates(yte, base[fs][te], thr=thr)))
        for name, col in [("pipeline_A", "a_score"), ("pipeline_B", "b_score")]:
            s = H[col].values[te]
            r = rates(yte, (s >= 3).astype(float)); r["auc"] = roc_auc_score(yte, s)
            rec.append(dict(k=k, rep=rep, features="-", model=name, **r))
    R = pd.DataFrame(rec)
    R.to_csv(f"{args.out}/pipeline_c_draws.csv", index=False)
    S = R.groupby(["model", "features", "k"]).agg(
        draws=("auc", "size"), auc=("auc", "mean"), auc_sd=("auc", "std"),
        far=("far", "mean"), far_sd=("far", "std"), sens=("sens", "mean"), sens_sd=("sens", "std"),
        flagged=("flagged", "mean")).reset_index()
    S.to_csv(f"{args.out}/pipeline_c_label_efficiency.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{args.out}/pipeline_c_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(S.round(3).to_string(index=False))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    show = [("logreg_missouri+local", "answers", "C: Missouri + k Hisar labels", "#1971c2", "-"),
            ("logreg_missouri+local_weighted", "answers", "C: Missouri + k Hisar labels, weighted", "#2f9e44", "-"),
            ("logreg_missouri_local_cutoff", "answers", "C: Missouri model, cut-off set on k Hisar labels", "#9c36b5", "-"),
            ("logreg_local_only", "answers", "C: k Hisar labels only", "#e8590c", "-"),
            ("pipeline_A", "-", "Pipeline A (no training)", "#495057", "--"),
            ("pipeline_B", "-", "Pipeline B (no training)", "#adb5bd", "--")]
    for model, fs, lab, col, ls in show:
        d = S[(S.model == model) & (S.features == fs)].sort_values("k")
        axes[0].plot(d.k, d.auc, ls, color=col, marker="o", ms=4, label=lab)
        axes[1].plot(d.k, d.far * 100, ls, color=col, marker="o", ms=4, label=lab)
        axes[2].plot(d.k, d.sens * 100, ls, color=col, marker="o", ms=4, label=lab)
    xl = "Labelled Hisar locations used (k)"
    axes[0].set(xlabel=xl, ylabel="AUC on the rest of Hisar", title="Ranking quality (higher is better)")
    axes[1].set(xlabel=xl, ylabel="False-alarm rate (%)", title="False alarms (lower is better)")
    axes[2].set(xlabel=xl, ylabel="Gullies found (%)", title="Gullies found (higher is better)")
    axes[0].axhline(0.5, color="#ced4da", lw=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Pipeline C: what limited local labels add (mean over random draws; features = 19 answers)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{args.out}/label_efficiency.png", dpi=200)
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
