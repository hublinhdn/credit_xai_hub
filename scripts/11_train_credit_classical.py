#!/usr/bin/env python3
"""Phase 7, steps 2 and 3: the classical arm of the credit-risk comparison.

Runs N x 2 cross-validation (Dietterich 1998) exactly as the reference
benchmark does, selecting hyperparameters with an inner 5-fold cross-validation
on each training half, and writes one directory of artefacts per fold.

Step 2 is the validity gate: logistic regression on German Credit must land on
AUC 0.784 +/- 0.012, the value reported by Lessmann et al. (2015). Nothing
downstream is trustworthy until that reproduces, so run it first:

    python scripts/11_train_credit_classical.py --datasets german --models lr_l2 --gate

All preprocessing, weight-of-evidence coding included, is fitted inside the
pipeline and therefore inside the fold. Fitting WoE on the full dataset leaks
the label and is the most common evaluation error in this literature.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tabular.datasets import DEFAULT_RAW, load  # noqa: E402
from src.tabular.metrics import aggregate_credit, compute_credit_metrics  # noqa: E402
from src.tabular.splits import N_REPEATS, nx2_splits  # noqa: E402
from src.tabular.woe import WoEEncoder  # noqa: E402

DEFAULT_OUT = Path("results/credit")

# Published reference values for the validity gate (Lessmann et al. 2015,
# German Credit column). Used only to check our pipeline, never to tune it.
GATE_REFERENCE = {"lr_l2": (0.784, 0.012), "lr_l1": (0.784, 0.012)}
GATE_TOLERANCE_SD = 1.0  # must land within one published standard deviation

ALL_MODELS = ("majority", "lr_l1", "lr_l2", "rf", "sgb", "hgb", "mlp")
ALL_DATASETS = ("german", "taiwan")
# Linear and neural models need numeric inputs, trees take the mixed coding.
WOE_MODELS = {"lr_l1", "lr_l2", "mlp"}
# GradientBoostingClassifier and MLPClassifier take no class_weight argument;
# they are balanced through sample_weight instead (see BalancedWeights).
SAMPLE_WEIGHT_MODELS = {"sgb", "mlp"}


class BalancedWeights(BaseEstimator, ClassifierMixin):
    """Fit a classifier with balanced sample weights derived from its own ``y``.

    Wrapping the estimator rather than passing ``sample_weight`` as a fit
    parameter means each inner cross-validation fold computes weights from that
    fold's labels, with no metadata routing to get wrong. Without this, the two
    families that take no ``class_weight`` argument silently ignore the setting
    and the "balanced" results are identical to the unweighted ones.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        weights = compute_sample_weight("balanced", y)
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def build_pipeline(
    model: str, numeric: list[str], categorical: list[str], class_weight: str | None, seed: int
) -> tuple[Pipeline, dict]:
    """Return (pipeline, hyperparameter grid) for one model family."""
    if model in WOE_MODELS:
        pre = Pipeline(
            [
                ("woe", WoEEncoder(columns=categorical)),
                ("scale", StandardScaler()),
            ]
        )
    else:
        pre = ColumnTransformer(
            [
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
                ("num", "passthrough", numeric),
            ],
            remainder="drop",
        )

    # scikit-learn 1.8 deprecated `penalty`; l1_ratio=1 is lasso, 0 is ridge.
    if model == "lr_l1":
        clf = LogisticRegression(
            l1_ratio=1.0, solver="liblinear", max_iter=2000, class_weight=class_weight, random_state=seed
        )
        grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    elif model == "lr_l2":
        clf = LogisticRegression(
            l1_ratio=0.0, solver="liblinear", max_iter=2000, class_weight=class_weight, random_state=seed
        )
        grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    elif model == "rf":
        clf = RandomForestClassifier(
            # n_jobs=1: this sits inside a GridSearchCV that is itself
            # parallel, and -1 here multiplies against that.
            n_estimators=500, class_weight=class_weight, random_state=seed, n_jobs=1
        )
        grid = {"clf__max_depth": [None, 8, 16], "clf__min_samples_leaf": [1, 5]}
    elif model == "sgb":
        # Friedman's stochastic gradient boosting, the "SGB" row of the
        # reference benchmark (subsample < 1 is what makes it stochastic).
        clf = GradientBoostingClassifier(subsample=0.8, random_state=seed)
        grid = {
            "clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 6],
        }
    elif model == "hgb":
        clf = HistGradientBoostingClassifier(
            random_state=seed, class_weight=class_weight, early_stopping=False
        )
        grid = {"clf__max_iter": [100, 300], "clf__learning_rate": [0.05, 0.1]}
    elif model == "mlp":
        # Two deliberate choices, both needed for a fair dense-neural baseline:
        # (1) solver="lbfgs", which scikit-learn recommends for small datasets,
        #     and no internal early-stopping split, which on a 500-row training
        #     half would hold out too little data to be useful;
        # (2) an alpha range reaching far into heavy regularisation. A narrow
        #     range left this model at AUC 0.709 on German Credit against 0.791
        #     for the reference study's ANN; it needs alpha of order 10 to 100
        #     to be competitive, and an under-tuned dense baseline would flatter
        #     the quantum arm it is parameter-matched against.
        clf = MLPClassifier(solver="lbfgs", max_iter=3000, random_state=seed, early_stopping=False)
        grid = {
            "clf__hidden_layer_sizes": [(16,), (32,), (64,)],
            # Capped at 100: alpha=300 drove lbfgs to a degenerate all-negative
            # fit whose residual ranking was inverted (AUC 0.23 on two folds),
            # and the inner CV could not see it on 100-row inner folds.
            "clf__alpha": [0.01, 0.1, 1.0, 10.0, 30.0, 100.0],
        }
    else:
        raise ValueError(f"unknown model {model!r}")

    if model in SAMPLE_WEIGHT_MODELS and class_weight == "balanced":
        clf = BalancedWeights(estimator=clf)
        grid = {k.replace("clf__", "clf__estimator__"): v for k, v in grid.items()}

    return Pipeline([("pre", pre), ("clf", clf)]), grid


def count_params(model: str, fitted: Pipeline) -> int | None:
    """Trainable parameter count, for the parameter-efficiency axis."""
    clf = fitted.named_steps["clf"]
    if isinstance(clf, BalancedWeights):
        clf = clf.estimator_
    if isinstance(clf, LogisticRegression):
        return int(clf.coef_.size + clf.intercept_.size)
    if isinstance(clf, MLPClassifier):
        return int(sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    if isinstance(clf, RandomForestClassifier):
        return int(sum(e.tree_.node_count for e in clf.estimators_))
    if isinstance(clf, GradientBoostingClassifier):
        return int(sum(e.tree_.node_count for stage in clf.estimators_ for e in stage))
    if isinstance(clf, HistGradientBoostingClassifier):
        # Histogram boosting keeps its trees as predictor objects, not sklearn
        # Tree instances, so count their nodes through the predictor arrays.
        return int(
            sum(pred.nodes.shape[0] for stage in clf._predictors for pred in stage)
        )
    return None


def run_fold(
    model: str,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    class_weight: str | None,
    seed: int,
    cost_fn: float,
    cost_fp: float,
    inner_folds: int,
    n_jobs: int = 4,
) -> tuple[dict, pd.DataFrame]:
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_te, y_te = X.iloc[test_idx], y[test_idx]

    if model == "majority":
        # The trivial reference. Computed from the training half, never
        # hardcoded: both credit datasets are imbalanced, so 0.5 is wrong.
        major = int(np.bincount(y_tr, minlength=2).argmax())
        pred = np.full(len(y_te), major, dtype=np.int64)
        score = np.full(len(y_te), float(y_tr.mean()))
        best_params: dict = {}
        n_params = 1
    else:
        pipe, grid = build_pipeline(model, numeric, categorical, class_weight, seed)
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs, refit=True)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        score = best.predict_proba(X_te)[:, 1]
        pred = (score >= 0.5).astype(np.int64)
        best_params = {k: str(v) for k, v in search.best_params_.items()}
        n_params = count_params(model, best)

    m = compute_credit_metrics(y_te, pred, score, cost_fn=cost_fn, cost_fp=cost_fp)
    preds_df = pd.DataFrame(
        {
            "row_id": test_idx.astype(np.int64),
            "true_label": y_te.astype(np.int64),
            "pred": pred,
            "prob_pos": score,
        }
    )
    return {"metrics": m, "best_params": best_params, "n_params": n_params}, preds_df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS), choices=ALL_DATASETS)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    p.add_argument("--class-weight", choices=["none", "balanced", "both"], default="none")
    p.add_argument("--drop-protected", action="store_true",
                   help="drop fair-lending protected attributes before fitting")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=4,
                   help="inner search parallelism. Not -1: several of these run at "
                        "once and each would claim every core, which drove the load "
                        "average to 161 on a 24-core host and starved all of them")
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=None, help="override N of the N x 2 protocol")
    p.add_argument("--max-rows", type=int, default=None, help="stratified subsample, for smoke tests")
    p.add_argument("--gate", action="store_true", help="check the result against the published value")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    weights = {"none": [None], "balanced": ["balanced"], "both": [None, "balanced"]}[args.class_weight]
    gate_failures = 0

    for ds_name in args.datasets:
        ds = load(ds_name, args.raw_dir)
        if args.drop_protected:
            ds = ds.without_protected()
        X, y = ds.X, ds.y

        if args.max_rows and args.max_rows < len(y):
            rng = np.random.RandomState(args.seed)
            idx = np.concatenate([
                rng.choice(np.where(y == c)[0],
                           size=max(1, int(round(args.max_rows * (y == c).mean()))),
                           replace=False)
                for c in (0, 1)
            ])
            idx.sort()
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]
            print(f"[{ds_name}] subsampled to n={len(y)} (bad_rate={y.mean():.4f})")

        n_repeats = args.n_repeats or N_REPEATS.get(ds_name, 5)
        folds = list(nx2_splits(y, n_repeats=n_repeats, seed=args.seed))
        print(f"\n=== {ds_name}: n={len(y):,} bad_rate={y.mean():.4f} "
              f"protocol={n_repeats}x2 ({len(folds)} fits) ===")

        for model in args.models:
            for cw in weights:
                tag = f"{model}" + ("_balanced" if cw == "balanced" else "")
                run_root = args.out_dir / ds_name / tag
                agg_path = run_root / "aggregated.json"
                if args.skip_existing and agg_path.exists():
                    print(f"  {tag}: skipped (exists)")
                    continue

                t0 = time.time()
                per_fold = []
                n_params_seen: list[int] = []
                for r, half, train_idx, test_idx in folds:
                    out, preds = run_fold(
                        model, X, y, train_idx, test_idx, ds.numeric, ds.categorical,
                        cw, args.seed + r, ds.cost_fn, ds.cost_fp, args.inner_folds, args.n_jobs,
                    )
                    per_fold.append(out["metrics"])
                    if out["n_params"]:
                        n_params_seen.append(out["n_params"])

                    fold_dir = run_root / f"repeat_{r}" / f"half_{half}"
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    (fold_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "dataset": ds_name, "model": model, "class_weight": cw,
                                "repeat": r, "half": half, "seed": args.seed + r,
                                "n_params": out["n_params"], "best_params": out["best_params"],
                                "drop_protected": bool(args.drop_protected),
                                "metrics": out["metrics"].to_dict(),
                            },
                            indent=2,
                        )
                    )
                    preds.to_csv(fold_dir / "predictions.csv", index=False)

                agg = aggregate_credit(per_fold)
                agg.update({
                    "dataset": ds_name, "model": model, "class_weight": cw,
                    "n_params_median": int(np.median(n_params_seen)) if n_params_seen else None,
                    "protocol": f"{n_repeats}x2", "elapsed_s": round(time.time() - t0, 1),
                    "drop_protected": bool(args.drop_protected),
                })
                run_root.mkdir(parents=True, exist_ok=True)
                agg_path.write_text(json.dumps(agg, indent=2))

                auc_m, auc_s = agg["auc_roc_mean"], agg["auc_roc_std"]
                print(
                    f"  {tag:<16} AUC={auc_m:.4f}±{auc_s:.4f}  "
                    f"PCC={agg['pcc_mean']:.4f}  H={agg['h_measure_mean']:.4f}  "
                    f"BS={agg['brier_mean']:.4f}  cost={agg['expected_cost_at_half_mean']:.4f}  "
                    f"params={agg['n_params_median']}  [{agg['elapsed_s']}s]"
                )

                if args.gate and ds_name == "german" and model in GATE_REFERENCE and cw is None:
                    ref_m, ref_s = GATE_REFERENCE[model]
                    delta = abs(auc_m - ref_m)
                    ok = delta <= GATE_TOLERANCE_SD * ref_s
                    print(
                        f"    GATE vs Lessmann et al. (2015): published {ref_m:.3f}±{ref_s:.3f}, "
                        f"ours {auc_m:.4f}, |delta|={delta:.4f} -> {'PASS' if ok else 'FAIL'}"
                    )
                    if not ok:
                        gate_failures += 1

    if gate_failures:
        print(f"\n{gate_failures} validity gate(s) FAILED: do not proceed to the quantum arm")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
