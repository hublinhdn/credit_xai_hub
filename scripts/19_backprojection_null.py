#!/usr/bin/env python3
"""Is Control C's agreement the models, or the map they share?

Control C compares two model families fitted on the same principal components,
each attribution carried back to the original features through the same
loading matrix. Both sides pass through one shared linear map, so a high
agreement has two possible causes:

  the models agree    their attribution vectors over components are alike, and
                      the map preserves that
  the map dominates   the map concentrates on a few original features, so any
                      two component vectors land in nearly the same place

Nothing in the control separates those. This script does, two ways.

  component space   the same two attributions compared BEFORE the map. High
                    here means the models genuinely agree.
  permutation null  each model's component attribution has its entries
                    shuffled across components, keeping its magnitude profile
                    and destroying which component carries what, then carried
                    back and compared. High here means the map alone produces
                    agreement and Control C measures nothing.

Control C stands only if the observed value is far above the null.

    python scripts/19_backprojection_null.py --dataset german
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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from importlib import import_module  # noqa: E402

_controls = import_module("15_attribution_controls")
shap_values, rho = _controls.shap_values, _controls.rho


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="german")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--n-permutations", type=int, default=200)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=Path("results/credit/backprojection_null.json"))
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
        print(f"subsampled to n={len(ds.y)}")
    folds = list(nx2_splits(ds.y, n_repeats=max(1, args.n_folds // 2), seed=args.seed))[: args.n_folds]
    print(f"{args.dataset}: {len(folds)} folds, {args.n_components} components, "
          f"{args.n_permutations} permutations per fold\n")

    observed_back, observed_comp, null_back = [], [], []
    rng = np.random.RandomState(args.seed)

    for n, (_, _, tr, te) in enumerate(folds, 1):
        pre = Pipeline([("woe", WoEEncoder(columns=ds.categorical)), ("scale", StandardScaler())])
        Z_tr = np.asarray(pre.fit_transform(ds.X.iloc[tr], ds.y[tr]), dtype=float)
        Z_te = np.asarray(pre.transform(ds.X.iloc[te]), dtype=float)
        enc = PCAProjector(n_components=args.n_components, seed=args.seed).fit(Z_tr)
        P_tr, P_te = enc.transform(Z_tr), enc.transform(Z_te)

        comp: dict[str, np.ndarray] = {}
        for key, (proto, linear) in {
            "lr": (LogisticRegression(l1_ratio=0.0, solver="liblinear", max_iter=2000,
                                      random_state=args.seed), True),
            "mlp": (MLPClassifier(solver="lbfgs", hidden_layer_sizes=(32,), alpha=30.0,
                                  max_iter=3000, random_state=args.seed), False),
        }.items():
            m = clone(proto).fit(P_tr, ds.y[tr])
            comp[key] = shap_values(m, P_tr, P_te, linear, args.seed)

        observed_comp.append(rho(comp["lr"], comp["mlp"]))
        observed_back.append(rho(enc.project_attribution_back(comp["lr"]),
                                 enc.project_attribution_back(comp["mlp"])))

        # The null keeps each model's magnitude profile over components and
        # destroys only the assignment of magnitude to component. Anything the
        # map contributes survives; anything the models contribute does not.
        for _ in range(args.n_permutations):
            a = enc.project_attribution_back(rng.permutation(comp["lr"]))
            b = enc.project_attribution_back(rng.permutation(comp["mlp"]))
            null_back.append(rho(a, b))
        print(f"  fold {n}/{len(folds)} done")

    def rep(label: str, vals: list[float]) -> dict:
        vals = [v for v in vals if np.isfinite(v)]
        pt, lo, hi = bootstrap_ci(vals, seed=args.seed)
        print(f"  {label:<52} rho={pt:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(vals)})")
        return {"rho": pt, "lo": lo, "hi": hi, "n": len(vals)}

    print(f"\n{'=' * 78}\nis control C the models or the map?\n{'=' * 78}")
    out = {
        "observed_component_space": rep("observed, before the map (component space)", observed_comp),
        "observed_back_projected": rep("observed, after the map (control C)", observed_back),
        "permutation_null": rep("permutation null, after the map", null_back),
    }
    obs = out["observed_back_projected"]["rho"]
    null_hi = out["permutation_null"]["hi"]
    frac = float(np.mean([v >= obs for v in null_back]))
    out["null_exceeds_observed_fraction"] = frac
    print(f"\n  null draws reaching the observed value: {frac:.4f}")
    if obs > null_hi:
        print(f"  -> the observed {obs:.3f} sits above the null's upper bound "
              f"{null_hi:.3f}; the map alone does not produce it")
    else:
        print(f"  -> WARNING: the null reaches {null_hi:.3f} against an observed "
              f"{obs:.3f}; control C is measuring the shared map")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out["run"] = {"dataset": args.dataset, "n_rows": int(len(ds.y)),
                  "adverse_rate": float(ds.y.mean()), "n_folds": len(folds),
                  "n_components": args.n_components, "max_rows": args.max_rows,
                  "n_permutations": args.n_permutations, "seed": args.seed}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
