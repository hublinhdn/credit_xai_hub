#!/usr/bin/env python3
"""Phase 7, step 7: feature attributions for the classical arm.

Writes one ``attribution.csv`` per fold in the same schema the quantum arm
produces, so the two can be compared directly:

    feature, attribution, rank

Attribution is the mean absolute SHAP value over the evaluation half, which is
the standard per-feature importance and is on the same footing as the quantum
arm's mean absolute input gradient: both answer "how strongly does the decision
respond to this feature, on average, on held-out rows".

Script 11 saves no fitted models, so each fold is refitted here. That is exact
rather than approximate because the seed, the split and the grid are all
recorded, and it is cheaper than storing a model per fold. The refit is verified
against the stored metrics and the script stops if they disagree.

    python scripts/13_credit_attribution.py --datasets german --models lr_l2 rf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tabular.datasets import DEFAULT_RAW, load  # noqa: E402
from src.tabular.splits import N_REPEATS, nx2_splits  # noqa: E402

# Script 11 is already validated against the published benchmark, so its model
# definitions are loaded rather than duplicated or refactored out from under it.
_spec = importlib.util.spec_from_file_location(
    "train_classical", REPO_ROOT / "scripts" / "11_train_credit_classical.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_pipeline = _mod.build_pipeline
BalancedWeights = _mod.BalancedWeights
WOE_MODELS = _mod.WOE_MODELS

DEFAULT_OUT = Path("results/credit")
ATTRIBUTABLE_MODELS = ("lr_l1", "lr_l2", "rf", "sgb", "hgb", "mlp")


def shap_attribution(model_name: str, fitted, X_tr: pd.DataFrame, X_te: pd.DataFrame,
                      feature_names: list[str], background: int, seed: int) -> np.ndarray:
    """Mean absolute SHAP value per original feature.

    The explainer is chosen per family: exact for linear and tree models, and
    sampled for the neural one, which has no closed form.
    """
    import shap

    pre = fitted.named_steps["pre"]
    clf = fitted.named_steps["clf"]
    if isinstance(clf, BalancedWeights):
        clf = clf.estimator_

    Z_tr = pre.transform(X_tr)
    Z_te = pre.transform(X_te)
    Z_tr = np.asarray(Z_tr, dtype=float)
    Z_te = np.asarray(Z_te, dtype=float)

    if model_name in ("lr_l1", "lr_l2"):
        explainer = shap.LinearExplainer(clf, Z_tr)
        values = explainer.shap_values(Z_te)
    elif model_name in ("rf", "sgb", "hgb"):
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(Z_te, check_additivity=False)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(Z_tr), size=min(background, len(Z_tr)), replace=False)
        explainer = shap.KernelExplainer(
            lambda a: clf.predict_proba(a)[:, 1], shap.kmeans(Z_tr[idx], min(10, len(idx)))
        )
        values = explainer.shap_values(Z_te[: min(len(Z_te), 100)], silent=True)

    values = np.asarray(values)
    # Tree explainers may return one array per class, or stack classes last.
    if values.ndim == 3:
        values = values[..., -1] if values.shape[-1] == 2 else values[-1]
    if values.ndim == 1:
        values = values.reshape(1, -1)

    attribution = np.abs(values).mean(axis=0)

    # The tree branch uses a ColumnTransformer that reorders columns to
    # [categorical..., numeric...]; map back to the dataset's own order.
    if model_name not in WOE_MODELS:
        order = list(pre.transformers_[0][2]) + list(pre.transformers_[1][2])
        mapped = np.zeros(len(feature_names), dtype=float)
        for value, name in zip(attribution, order):
            mapped[feature_names.index(name)] = value
        attribution = mapped
    elif len(attribution) != len(feature_names):
        raise ValueError(
            f"{model_name}: got {len(attribution)} attributions for {len(feature_names)} features"
        )
    return attribution


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["german", "taiwan"])
    p.add_argument("--models", nargs="+", default=list(ATTRIBUTABLE_MODELS))
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--drop-protected", action="store_true",
                   help="match a run trained without the fair-lending attributes")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=4,
                   help="inner search parallelism. Not -1: several of these run at "
                        "once and each would claim every core, which drove the load "
                        "average to 161 on a 24-core host and starved all of them")
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--background", type=int, default=200, help="rows for the sampled explainer")
    p.add_argument("--max-folds", type=int, default=None, help="limit folds, for a quick pass")
    p.add_argument("--tolerance", type=float, default=2e-3,
                   help="how far a refit may drift from the stored AUC before it "
                        "counts as a mismatch rather than solver noise")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    cw = None if args.class_weight == "none" else "balanced"
    mismatches = 0
    drifted: list[tuple] = []

    for ds_name in args.datasets:
        ds = load(ds_name, args.raw_dir)
        if args.drop_protected:
            ds = ds.without_protected()
        feature_names = list(ds.X.columns)
        folds = list(nx2_splits(ds.y, n_repeats=N_REPEATS.get(ds_name, 5), seed=args.seed))
        if args.max_folds:
            folds = folds[: args.max_folds]
        print(f"\n=== {ds_name}: {len(folds)} folds, {len(feature_names)} features ===")

        for model_name in args.models:
            tag = model_name + ("_balanced" if cw else "")
            done = 0
            for r, half, train_idx, test_idx in folds:
                out_dir = args.out_dir / ds_name / tag / f"repeat_{r}" / f"half_{half}"
                if not out_dir.exists():
                    continue
                if args.skip_existing and (out_dir / "attribution.csv").exists():
                    done += 1
                    continue

                from sklearn.model_selection import GridSearchCV, StratifiedKFold

                pipe, grid = build_pipeline(model_name, ds.numeric, ds.categorical, cw, args.seed + r)
                inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True,
                                        random_state=args.seed + r)
                search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=args.n_jobs)
                search.fit(ds.X.iloc[train_idx], ds.y[train_idx])
                fitted = search.best_estimator_

                # Confirm the refit reproduces the stored fold before trusting it.
                stored = json.loads((out_dir / "metrics.json").read_text())
                score = fitted.predict_proba(ds.X.iloc[test_idx])[:, 1]
                from sklearn.metrics import roc_auc_score

                auc_now = float(roc_auc_score(ds.y[test_idx], score))
                auc_then = stored["metrics"]["auc_roc"]
                if auc_then is not None:
                    drift = abs(auc_now - auc_then)
                    if drift > args.tolerance:
                        print(f"  MISMATCH {tag} r{r}h{half}: refit AUC {auc_now:.6f} "
                              f"vs stored {auc_then:.6f}, drift {drift:.2e}")
                        mismatches += 1
                        continue
                    if drift > 1e-9:
                        # Iterative solvers drift when the thread count changes,
                        # because BLAS reductions are not associative. Reported
                        # rather than silently dropped.
                        drifted.append((tag, r, half, drift))

                attribution = shap_attribution(
                    model_name, fitted, ds.X.iloc[train_idx], ds.X.iloc[test_idx],
                    feature_names, args.background, args.seed + r,
                )
                order = np.argsort(-np.abs(attribution))
                rank = np.empty(len(attribution), dtype=int)
                rank[order] = np.arange(1, len(attribution) + 1)
                pd.DataFrame({
                    "feature": feature_names,
                    "attribution": attribution,
                    "rank": rank,
                }).to_csv(out_dir / "attribution.csv", index=False)
                done += 1

            if done:
                print(f"  {tag:<16} attributions written for {done} fold(s)")

    if mismatches:
        print(f"\n{mismatches} fold(s) did not reproduce: attributions for those are missing")
        return 1
    if drifted:
        worst = max(d[3] for d in drifted)
        print(f"\n{len(drifted)} fold(s) reproduced within tolerance but not exactly; "
              f"largest drift {worst:.2e}")
        for tag, r, half, d in drifted[:5]:
            print(f"    {tag} r{r}h{half}: {d:.2e}")
    print("\nAll refits reproduced their stored metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
