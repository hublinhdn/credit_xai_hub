#!/usr/bin/env python3
"""Phase 7, step 9: what does a feature attribution actually depend on?

Comparing the attributions of two models tells you nothing until you know what
else differs between them. Three things can move a ranking of features:

  the attribution measure   a Shapley allocation against a local gradient
  the feature representation  whether the model sees the original columns or a
                              reduced space whose attributions must be mapped back
  the model family          linear, tree, neural

Three controls separate them. Each isolates one difference and holds the rest
fixed, so the comparison that remains is the one named.

A  measure         input-gradient against SHAP, same model, same features.
                   High agreement means the measure is not what separates
                   anything else in the paper.
B  representation  one family fitted on eight principal components, its SHAP
                   projected back through the loading matrix, against its own
                   SHAP on the original features. Low agreement means the
                   representation is doing the work.
C  model family    two families fitted inside the SAME eight components, both
                   projected back identically. Everything but the family is now
                   held fixed, and control A has bounded the measure.

Control C means nothing between models of unequal accuracy, so the precondition
is checked in the reduced space where the comparison is made, not inherited
from the original features.

    python scripts/15_attribution_controls.py --dataset german
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tabular.reduce import PCAProjector  # noqa: E402
from src.stats.tests import bootstrap_ci, holm, mcnemar  # noqa: E402
from src.tabular.datasets import DEFAULT_RAW, load  # noqa: E402
from src.tabular.splits import nx2_splits  # noqa: E402
from src.tabular.woe import WoEEncoder  # noqa: E402


def input_gradient(predict_proba, X: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Mean absolute central-difference gradient of P(adverse) per feature."""
    X = np.asarray(X, dtype=float)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        step = eps * max(float(np.std(X[:, j])), 1e-6)
        up, dn = X.copy(), X.copy()
        up[:, j] += step
        dn[:, j] -= step
        out[j] = float(np.abs((predict_proba(up) - predict_proba(dn)) / (2 * step)).mean())
    return out


def shap_values(clf, Z_tr: np.ndarray, Z_te: np.ndarray, linear: bool, seed: int) -> np.ndarray:
    import shap

    if linear:
        vals = shap.LinearExplainer(clf, Z_tr).shap_values(Z_te)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(Z_tr), size=min(200, len(Z_tr)), replace=False)
        ex = shap.KernelExplainer(lambda a: clf.predict_proba(a)[:, 1],
                                  shap.kmeans(Z_tr[idx], 10))
        vals = ex.shap_values(Z_te[: min(len(Z_te), 100)], silent=True)
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[..., -1]
    return np.abs(vals).mean(axis=0)


def rho(a: np.ndarray, b: np.ndarray) -> float:
    r = spearmanr(np.abs(a), np.abs(b)).statistic
    return float(r) if np.isfinite(r) else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="german")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=Path("results/credit/attribution_controls.json"))
    p.add_argument("--max-rows", type=int, default=None,
                   help="stratified subsample; the finite-difference gradient costs "
                        "two predictions per feature per row")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = load(args.dataset, args.raw_dir)
    if args.max_rows and args.max_rows < len(ds.y):
        rng = np.random.RandomState(args.seed)
        keep = np.concatenate([
            rng.choice(np.where(ds.y == c)[0],
                       size=max(1, int(round(args.max_rows * (ds.y == c).mean()))),
                       replace=False)
            for c in (0, 1)
        ])
        keep.sort()
        ds.X = ds.X.iloc[keep].reset_index(drop=True)
        ds.y = ds.y[keep]
        print(f"subsampled to n={len(ds.y)} (bad rate {ds.y.mean():.4f})")
    names = list(ds.X.columns)
    folds = list(nx2_splits(ds.y, n_repeats=max(1, args.n_folds // 2), seed=args.seed))[: args.n_folds]
    print(f"{args.dataset}: {len(folds)} folds, {len(names)} features, "
          f"{args.n_components} principal components\n")

    a_vals: dict[str, list[float]] = {"lr": [], "mlp": []}
    b_vals: dict[str, list[float]] = {"lr": [], "mlp": []}
    c_vals: list[float] = []
    c_auc: dict[str, list[float]] = {"lr": [], "mlp": []}
    c_mcnemar: list[float] = []

    for n, (r, half, tr, te) in enumerate(folds, 1):
        pca_back: dict[str, np.ndarray] = {}
        pca_pred: dict[str, np.ndarray] = {}
        pre = Pipeline([("woe", WoEEncoder(columns=ds.categorical)), ("scale", StandardScaler())])
        Z_tr = np.asarray(pre.fit_transform(ds.X.iloc[tr], ds.y[tr]), dtype=float)
        Z_te = np.asarray(pre.transform(ds.X.iloc[te]), dtype=float)

        enc = PCAProjector(n_components=args.n_components, seed=args.seed).fit(Z_tr)
        P_tr = enc.transform(Z_tr)
        P_te = enc.transform(Z_te)

        models = {
            "lr": (LogisticRegression(l1_ratio=0.0, solver="liblinear", max_iter=2000,
                                      random_state=args.seed), True),
            "mlp": (MLPClassifier(solver="lbfgs", hidden_layer_sizes=(32,), alpha=30.0,
                                  max_iter=3000, random_state=args.seed), False),
        }

        for key, (proto, linear) in models.items():
            raw = clone(proto).fit(Z_tr, ds.y[tr])
            shap_raw = shap_values(raw, Z_tr, Z_te, linear, args.seed)
            grad_raw = input_gradient(lambda a: raw.predict_proba(a)[:, 1], Z_te)
            a_vals[key].append(rho(shap_raw, grad_raw))

            on_pca = clone(proto).fit(P_tr, ds.y[tr])
            shap_pca = shap_values(on_pca, P_tr, P_te, linear, args.seed)
            pca_back[key] = enc.project_attribution_back(shap_pca)
            b_vals[key].append(rho(shap_raw, pca_back[key]))

            proba = on_pca.predict_proba(P_te)[:, 1]
            c_auc[key].append(float(roc_auc_score(ds.y[te], proba)))
            pca_pred[key] = (proba >= 0.5).astype(int)

        # Control C: two families inside the same principal component space,
        # both projected back the same way. This pairs directly with control B,
        # which changes the space and holds the family fixed.
        c_vals.append(rho(pca_back["lr"], pca_back["mlp"]))
        c_mcnemar.append(mcnemar(pca_pred["lr"], pca_pred["mlp"], ds.y[te]).p_value)

        print(f"  fold {n}/{len(folds)} done")

    def report(label: str, vals: list[float]) -> dict:
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return {}
        pt, lo, hi = bootstrap_ci(vals, seed=args.seed)
        ci = f"[{lo:.3f}, {hi:.3f}]" if len(vals) >= 3 else "CI needs 3+"
        print(f"  {label:<58} rho={pt:.3f} {ci}  (n={len(vals)})")
        return {"rho": pt, "lo": lo, "hi": hi, "n": len(vals)}

    print(f"\n{'=' * 78}\nA  measure: does input gradient agree with SHAP on the same model?\n{'=' * 78}")
    out = {"A": {k: report(f"{k}: SHAP vs input gradient, raw features", v) for k, v in a_vals.items()}}

    print(f"\n{'=' * 78}\nB  representation: does PCA back-projection preserve the ranking?\n{'=' * 78}")
    out["B"] = {k: report(f"{k}: raw SHAP vs PCA SHAP projected back", v) for k, v in b_vals.items()}

    print(f"\n{'=' * 78}\nC  model family: two families inside one representation\n{'=' * 78}")
    adj = holm({f"fold{i}": p for i, p in enumerate(c_mcnemar)})
    n_sep = sum(1 for v in adj.values() if v["significant_05"])
    print(f"  precondition: in the reduced space the two families reach AUC "
          f"{np.mean(c_auc['lr']):.4f} and {np.mean(c_auc['mlp']):.4f}; "
          f"McNemar with Holm separates them on {n_sep} of {len(c_mcnemar)} folds")
    out["C_precondition"] = {
        "auc_lr": float(np.mean(c_auc["lr"])),
        "auc_mlp": float(np.mean(c_auc["mlp"])),
        "n_separable": n_sep,
        "n_folds": len(c_mcnemar),
    }
    if n_sep > len(c_mcnemar) / 2:
        print("  WARNING: the pair is separable on most folds, so this control "
              "measures the accuracy gap and not the model family")
    out["C"] = report("linear vs neural, same components, both projected back", c_vals)

    print(f"\n{'=' * 78}\nverdict\n{'=' * 78}")
    a_mean = float(np.mean([v["rho"] for v in out["A"].values() if v]))
    b_mean = float(np.mean([v["rho"] for v in out["B"].values() if v]))
    c_mean = out["C"].get("rho", float("nan"))
    print(f"  measure, gradient vs SHAP        rho={a_mean:.3f}")
    print(f"  representation, raw vs projected rho={b_mean:.3f}")
    print(f"  model family, one representation rho={c_mean:.3f}")
    if np.isfinite(c_mean) and c_mean > b_mean:
        print(f"  -> changing the representation costs {c_mean - b_mean:.3f} more agreement "
              f"than changing the model family")
    else:
        print("  -> the model family matters at least as much as the representation here")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
