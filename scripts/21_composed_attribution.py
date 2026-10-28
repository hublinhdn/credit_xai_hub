#!/usr/bin/env python3
"""Control B without the back-projection rule.

Control B compares a model's attribution on the original features against the
same model's attribution in the reduced space, carried back through the
absolute loading matrix. That rule is a choice, not the inverse map, so a
referee can fairly ask how much of Control B's number belongs to the rule
rather than to the representation.

This answers it by explaining the composed function, PCA followed by the
model, directly over the original features. No attribution is carried back, so
the rule cannot contribute. Three agreements are reported per family:

  published      original-feature SHAP against the back-projected SHAP
                 (this reproduces Control B)
  rule-free      original-feature SHAP against the composed function's SHAP
                 (the same question with the rule removed)
  rule cost      back-projected SHAP against the composed function's SHAP
                 (what the rule alone does to a ranking)

If rule-free is as low as published, the representation is doing the work. If
it is high, most of Control B belonged to the rule.

The linear arm is exact: with ``z = (x - mean) W^T``, a logit ``w z + b`` is
``(W^T w) x + (b - (W^T w) mean)``, so the composed model is a linear model
over the original features and the same explainer applies.

    python scripts/21_composed_attribution.py --dataset german
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tabular.reduce import PCAProjector  # noqa: E402
from src.stats.tests import bootstrap_ci  # noqa: E402
from src.tabular.datasets import DEFAULT_RAW, load  # noqa: E402
from src.tabular.splits import nx2_splits  # noqa: E402
from src.tabular.woe import WoEEncoder  # noqa: E402

# The comparison is only meaningful if both arms are explained exactly as
# Control B explains them, so the helpers come from the control script itself
# rather than being written a second time here.
_spec = importlib.util.spec_from_file_location(
    "attribution_controls", REPO_ROOT / "scripts" / "15_attribution_controls.py")
_ctl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ctl)
shap_values, rho = _ctl.shap_values, _ctl.rho


def composed_linear(on_pca, enc: PCAProjector) -> tuple[np.ndarray, float]:
    """The composed PCA-then-linear model, as a linear model over features."""
    w = np.asarray(on_pca.coef_, dtype=float).ravel()
    coef = enc.pca_.components_.T @ w
    intercept = float(on_pca.intercept_[0]) - float(coef @ enc.pca_.mean_)
    return coef, intercept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="german")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path,
                   default=Path("results/credit/composed_attribution.json"))
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import shap

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

    folds = list(nx2_splits(ds.y, n_repeats=max(1, args.n_folds // 2),
                            seed=args.seed))[: args.n_folds]
    print(f"{args.dataset}: {len(folds)} folds, {ds.X.shape[1]} features, "
          f"{args.n_components} components\n")

    vals: dict[str, dict[str, list[float]]] = {
        k: {"published": [], "rule_free": [], "rule_cost": []} for k in ("lr", "mlp")
    }

    for n, (r, half, tr, te) in enumerate(folds, 1):
        pre = Pipeline([("woe", WoEEncoder(columns=ds.categorical)),
                        ("scale", StandardScaler())])
        Z_tr = np.asarray(pre.fit_transform(ds.X.iloc[tr], ds.y[tr]), dtype=float)
        Z_te = np.asarray(pre.transform(ds.X.iloc[te]), dtype=float)
        enc = PCAProjector(n_components=args.n_components, seed=args.seed).fit(Z_tr)
        P_tr, P_te = enc.transform(Z_tr), enc.transform(Z_te)

        models = {
            "lr": (LogisticRegression(l1_ratio=0.0, solver="liblinear", max_iter=2000,
                                      random_state=args.seed), True),
            "mlp": (MLPClassifier(solver="lbfgs", hidden_layer_sizes=(32,), alpha=30.0,
                                  max_iter=3000, random_state=args.seed), False),
        }
        for key, (proto, linear) in models.items():
            raw = clone(proto).fit(Z_tr, ds.y[tr])
            s_orig = shap_values(raw, Z_tr, Z_te, linear, args.seed)

            on_pca = clone(proto).fit(P_tr, ds.y[tr])
            s_back = enc.project_attribution_back(
                shap_values(on_pca, P_tr, P_te, linear, args.seed))

            if linear:
                coef, intercept = composed_linear(on_pca, enc)
                v = shap.LinearExplainer((coef, intercept), Z_tr).shap_values(Z_te)
                s_composed = np.abs(np.asarray(v)).mean(axis=0)
            else:
                rng = np.random.RandomState(args.seed)
                idx = rng.choice(len(Z_tr), size=min(200, len(Z_tr)), replace=False)
                ex = shap.KernelExplainer(
                    lambda a: on_pca.predict_proba(enc.transform(a))[:, 1],
                    shap.kmeans(Z_tr[idx], 10))
                v = ex.shap_values(Z_te[: min(len(Z_te), 100)], silent=True)
                v = np.asarray(v)
                if v.ndim == 3:
                    v = v[..., -1]
                s_composed = np.abs(v).mean(axis=0)

            vals[key]["published"].append(rho(s_orig, s_back))
            vals[key]["rule_free"].append(rho(s_orig, s_composed))
            vals[key]["rule_cost"].append(rho(s_back, s_composed))
        print(f"  fold {n}/{len(folds)} done")

    def report(label: str, v: list[float]) -> dict:
        v = [x for x in v if np.isfinite(x)]
        pt, lo, hi = bootstrap_ci(v, seed=args.seed)
        print(f"  {label:<46} rho={pt:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(v)})")
        return {"rho": pt, "lo": lo, "hi": hi, "n": len(v)}

    print(f"\n{'=' * 78}\nControl B, with and without the back-projection rule\n{'=' * 78}")
    out: dict = {}
    for key in ("lr", "mlp"):
        out[key] = {
            "published": report(f"{key}: original vs back-projected (Control B)",
                                vals[key]["published"]),
            "rule_free": report(f"{key}: original vs composed function",
                                vals[key]["rule_free"]),
            "rule_cost": report(f"{key}: back-projected vs composed function",
                                vals[key]["rule_cost"]),
        }
    for name in ("published", "rule_free", "rule_cost"):
        m = float(np.mean([out[k][name]["rho"] for k in ("lr", "mlp")]))
        out[f"mean_{name}"] = m
        print(f"  mean {name:<12} {m:.3f}")

    # what the reduction keeps, which a referee is entitled to see beside the
    # agreements it produces
    ev = float(np.sum(enc.pca_.explained_variance_ratio_))
    out["explained_variance"] = ev
    print(f"  {'variance kept by the components':<46} {ev * 100:.1f}%")

    out["run"] = {"dataset": args.dataset, "n_rows": int(len(ds.y)),
                  "n_folds": len(folds), "n_components": args.n_components,
                  "max_rows": args.max_rows, "seed": args.seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
