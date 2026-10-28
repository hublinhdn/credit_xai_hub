#!/usr/bin/env python3
"""What the encoding alone costs an attribution, for one model at a time.

Control B changes the representation by the most aggressive step available, a
principal component reduction. A referee can fairly object that credit scoring
rarely does that, while it does routinely choose between weight of evidence
codes and ordinal codes for nominal attributes.

That comparison needs no new fitting. The main grid gives the tree family
ordinal codes; the grid refitted for Section 4.3 gives every family weight of
evidence. The same model, the same fold, the same features by name, with only
the encoding of the nominal attributes different. The agreement between those
two attributions is the representation effect at the size a lender would
actually meet.

Linear and neural models receive weight of evidence in both grids, so they
have no encoding to change and serve here as the internal control: their
agreement should be one.

    python scripts/22_encoding_cost.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stats.tests import bootstrap_ci  # noqa: E402

TREES = ("rf", "hgb", "sgb")
CONTROL = ("lr_l2", "mlp")


def attributions(root: Path, dataset: str, model: str) -> dict[str, np.ndarray]:
    """One vector per fold, indexed by the fold's directory name."""
    out: dict[str, np.ndarray] = {}
    base = root / dataset / model
    if not base.exists():
        return out
    for f in sorted(base.rglob("attribution.csv")):
        df = pd.read_csv(f).sort_values("feature")
        out[str(f.parent.relative_to(base))] = np.abs(df["attribution"].to_numpy(float))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mixed-dir", type=Path, default=Path("results/credit"))
    p.add_argument("--woe-dir", type=Path, default=Path("results/credit_woe"))
    p.add_argument("--out", type=Path, default=Path("results/credit/encoding_cost.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out: dict = {}
    for dataset in ("german", "taiwan"):
        out[dataset] = {}
        print(f"\n{dataset}")
        for model in TREES + CONTROL:
            a = attributions(args.mixed_dir, dataset, model)
            b = attributions(args.woe_dir, dataset, model)
            shared = sorted(set(a) & set(b))
            vals = []
            for k in shared:
                if len(a[k]) != len(b[k]):
                    continue
                r = spearmanr(a[k], b[k]).statistic
                if np.isfinite(r):
                    vals.append(float(r))
            if len(vals) < 3:
                print(f"  {model:7} too few folds ({len(vals)})")
                continue
            pt, lo, hi = bootstrap_ci(vals, seed=args.seed)
            kind = "control" if model in CONTROL else "tree"
            print(f"  {model:7} {kind:8} rho={pt:.3f} [{lo:.3f}, {hi:.3f}]  ({len(vals)} folds)")
            out[dataset][model] = {"rho": pt, "lo": lo, "hi": hi,
                                   "n": len(vals), "kind": kind}
        trees = [v["rho"] for k, v in out[dataset].items() if v["kind"] == "tree"]
        ctrl = [v["rho"] for k, v in out[dataset].items() if v["kind"] == "control"]
        if trees:
            out[dataset]["mean_tree"] = float(np.mean(trees))
            print(f"  {'mean, encoding changed':17} {np.mean(trees):.3f}")
        if ctrl:
            out[dataset]["mean_control"] = float(np.mean(ctrl))
            print(f"  {'mean, unchanged':17} {np.mean(ctrl):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
