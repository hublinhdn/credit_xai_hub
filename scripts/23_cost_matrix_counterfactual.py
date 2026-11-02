#!/usr/bin/env python3
"""Whether class weighting helps: does the direction follow the cost matrix?

Section 4.6 gives each benchmark only the cost matrix it ships with. German
Credit carries a five to one matrix and weighting lowers expected cost there;
Taiwan carries none, symmetric costs are assumed, and weighting raises it. The
two benchmarks also differ in how imbalanced they are, and the more imbalanced
one is the one weighting hurts, which is what H7 claims.

With two datasets that argument cannot be made. German Credit has the
asymmetric matrix and the milder imbalance, Taiwan the symmetric matrix and
the heavier one, so the two candidate causes vary together and no measurement
separates them.

They separate without refitting anything. The cost matrix enters at evaluation
time, not at fitting time: expected cost is a function of the confusion matrix
and the matrix applied to it. Scoring every stored fold under both matrices
therefore gives a two by two design on each benchmark, in which the imbalance
is held fixed by the dataset and the matrix varies within it.

The counterfactual arm is a sensitivity analysis and not a claim about either
portfolio's real costs: Taiwan ships no matrix, and the five to one applied to
it here is German Credit's.

    python scripts/23_cost_matrix_counterfactual.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tabular.datasets import GERMAN_COST_FN, GERMAN_COST_FP  # noqa: E402
from src.tabular.metrics import expected_cost, min_expected_cost  # noqa: E402

MODELS = ("lr_l1", "lr_l2", "rf", "sgb", "hgb", "mlp")

# The matrix German Credit ships with, and the symmetric one Taiwan is scored
# under. Each is applied to both benchmarks.
MATRICES = {
    "asymmetric": (GERMAN_COST_FN, GERMAN_COST_FP),
    "symmetric": (1.0, 1.0),
}


def folds(root: Path, dataset: str, model: str) -> dict[str, pd.DataFrame]:
    """One prediction frame per fold, indexed by the fold's directory name."""
    base = root / dataset / model
    if not base.exists():
        return {}
    return {str(p.parent.relative_to(base)): pd.read_csv(p)
            for p in sorted(base.rglob("predictions.csv"))}


def costs(frame: pd.DataFrame, cost_fn: float, cost_fp: float) -> tuple[float, float]:
    """Expected cost at a threshold of 0.5, and at the cost-optimal threshold."""
    y = frame["true_label"].to_numpy(int)
    s = frame["prob_pos"].to_numpy(float)
    at_half = expected_cost(y, (s >= 0.5).astype(int), cost_fn, cost_fp)
    at_best, _ = min_expected_cost(y, s, cost_fn, cost_fp)
    return at_half, at_best


def check_against_stored(root: Path, dataset: str, model: str) -> str:
    """Confirm the recomputation matches what the training run stored.

    The stored figure uses the dataset's own matrix, so only that arm can be
    checked; it is the arm Section 4.6 quotes, which is what matters.
    """
    base = root / dataset / model
    fn, fp = MATRICES["asymmetric" if dataset == "german" else "symmetric"]
    worst = 0.0
    n = 0
    for p in sorted(base.rglob("metrics.json")):
        stored = json.loads(p.read_text())["metrics"].get("expected_cost_at_half")
        if stored is None:
            continue
        here, _ = costs(pd.read_csv(p.parent / "predictions.csv"), fn, fp)
        worst = max(worst, abs(here - float(stored)))
        n += 1
    return f"{n} folds, worst disagreement {worst:.2e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=Path("results/credit"))
    p.add_argument("--out", type=Path,
                   default=Path("results/credit/cost_matrix_counterfactual.json"))
    args = p.parse_args()

    out: dict = {}
    for dataset in ("german", "taiwan"):
        out[dataset] = {}
        print(f"\n{dataset}")
        print(f"  recomputation check: {check_against_stored(args.run_dir, dataset, 'lr_l2')}")
        for matrix, (cost_fn, cost_fp) in MATRICES.items():
            shipped = (matrix == "asymmetric") == (dataset == "german")
            cells = {}
            print(f"  {matrix} ({cost_fn:g}:{cost_fp:g})"
                  f"{'' if shipped else ', counterfactual'}")
            for model in MODELS:
                plain = folds(args.run_dir, dataset, model)
                weighted = folds(args.run_dir, dataset, f"{model}_balanced")
                shared = sorted(set(plain) & set(weighted))
                if len(shared) < 3:
                    print(f"    {model:7} too few shared folds ({len(shared)})")
                    continue
                half = [(costs(plain[k], cost_fn, cost_fp)[0],
                         costs(weighted[k], cost_fn, cost_fp)[0]) for k in shared]
                best = [(costs(plain[k], cost_fn, cost_fp)[1],
                         costs(weighted[k], cost_fn, cost_fp)[1]) for k in shared]
                u = float(np.mean([a for a, _ in half]))
                w = float(np.mean([b for _, b in half]))
                lowered = sum(1 for a, b in half if b < a)
                cells[model] = {
                    "at_half": {"unweighted": u, "weighted": w, "change": w - u,
                                "folds_lowered": lowered, "n_folds": len(half)},
                    "at_optimal": {
                        "unweighted": float(np.mean([a for a, _ in best])),
                        "weighted": float(np.mean([b for _, b in best])),
                        "change": float(np.mean([b - a for a, b in best])),
                    },
                }
                print(f"    {model:7} {u:8.4f} -> {w:8.4f}  {w - u:+8.4f}"
                      f"  {'lowers' if w < u else 'raises':6}"
                      f"  ({lowered}/{len(half)} folds lowered)")
            out[dataset][matrix] = {
                "cost_fn": cost_fn, "cost_fp": cost_fp,
                "shipped_with_dataset": shipped, "models": cells,
            }

    # The claim is about the matrix, so summarise by matrix across both
    # benchmarks: the design is only informative if the direction is constant
    # within a matrix and opposite between the two.
    summary = {}
    for matrix in MATRICES:
        models = [c for ds in out.values() for c in ds[matrix]["models"].values()]
        cells = [c["at_half"] for c in models]
        best = [c["at_optimal"] for c in models]
        summary[matrix] = {
            "cells": len(cells),
            "cells_lowered": sum(1 for c in cells if c["change"] < 0),
            "folds": sum(c["n_folds"] for c in cells),
            "folds_lowered": sum(c["folds_lowered"] for c in cells),
            "mean_change": float(np.mean([c["change"] for c in cells])),
            # At the cost-optimal threshold the question is whether the matrix
            # still decides the direction, so the cell count matters more than
            # the mean: one model can carry a mean on its own.
            "cells_lowered_at_optimal": sum(1 for c in best if c["change"] < 0),
            "mean_change_at_optimal": float(np.mean([c["change"] for c in best])),
        }
        s = summary[matrix]
        print(f"\n{matrix:11} at 0.5: {s['cells_lowered']}/{s['cells']} cells and "
              f"{s['folds_lowered']}/{s['folds']} folds lowered, "
              f"mean {s['mean_change']:+.4f}")
        print(f"{'':11} at the cost-optimal threshold: "
              f"{s['cells_lowered_at_optimal']}/{s['cells']} cells lowered, "
              f"mean {s['mean_change_at_optimal']:+.4f}")
    out["summary"] = summary
    out["run"] = {"models": list(MODELS), "matrices": MATRICES,
                  "run_dir": str(args.run_dir)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
