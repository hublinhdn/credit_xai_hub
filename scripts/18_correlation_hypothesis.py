#!/usr/bin/env python3
"""Does feature correlation explain when model families disagree on attributions?

Between-family attribution agreement is high on German Credit (Spearman 0.736
to 0.891) and low on Taiwan (0.265). The two datasets differ sharply in how
correlated their features are: German has no pair above |r| = 0.7, Taiwan has
twenty. The hypothesis is that correlated features make the attribution problem
underdetermined, so different models split the credit differently.

Comparing two datasets cannot establish that; two datasets differ in many ways
at once. This script tests it *within* each dataset, where everything else is
held fixed:

    H: a feature's disagreement across model families rises with how strongly
       that feature correlates with the others.

Disagreement for one feature is the standard deviation of its importance rank
across families, pooled over folds. Correlation is its maximum absolute
Pearson correlation with any other feature, measured in the space the models
actually see (weight of evidence coded, standardised).

A positive Spearman between the two, within a dataset, supports the
hypothesis. The test is reported per dataset and refuses to pool them.

    python scripts/18_correlation_hypothesis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stats.tests import bootstrap_ci  # noqa: E402
from src.tabular.datasets import DEFAULT_RAW, load  # noqa: E402
from src.tabular.woe import WoEEncoder  # noqa: E402

FAMILY = {"lr_l1": "linear", "lr_l2": "linear", "rf": "tree", "sgb": "tree",
          "hgb": "tree", "mlp": "neural"}


def feature_max_abs_corr(ds, seed: int = 0) -> pd.Series:
    """Each feature's strongest absolute correlation with any other feature.

    Computed on the weight-of-evidence coded, standardised matrix, which is
    what the linear and neural models see, rather than on the raw columns.
    """
    pre = Pipeline([("woe", WoEEncoder(columns=ds.categorical)), ("scale", StandardScaler())])
    Z = pre.fit_transform(ds.X, ds.y)
    Z = pd.DataFrame(np.asarray(Z, dtype=float), columns=list(ds.X.columns))
    C = np.array(Z.corr().abs().to_numpy(), copy=True)
    np.fill_diagonal(C, 0.0)
    return pd.Series(C.max(axis=1), index=Z.columns)


def per_feature_disagreement(classical_dir: Path, dataset: str) -> pd.Series:
    """Standard deviation of a feature's rank across families, pooled over folds."""
    by_fold: dict[tuple, dict[str, pd.Series]] = {}
    for agg in sorted(classical_dir.glob(f"{dataset}/*/aggregated.json")):
        name = agg.parent.name
        if name.endswith("_balanced") or name.replace("_balanced", "") not in FAMILY:
            continue
        fam = FAMILY[name]
        for path in sorted(agg.parent.rglob("attribution.csv")):
            key = path.relative_to(agg.parent).parts[:2]
            df = pd.read_csv(path).set_index("feature")["rank"]
            by_fold.setdefault(key, {}).setdefault(fam, []).append(df)

    spreads: list[pd.Series] = []
    for key, fams in by_fold.items():
        if len(fams) < 2:
            continue
        # One rank vector per family on this fold: average within family first,
        # so a family with more models does not dominate the spread.
        per_family = {f: pd.concat(v, axis=1).mean(axis=1) for f, v in fams.items()}
        mat = pd.DataFrame(per_family)
        spreads.append(mat.std(axis=1, ddof=0))
    if not spreads:
        return pd.Series(dtype=float)
    return pd.concat(spreads, axis=1).mean(axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["german", "taiwan"])
    p.add_argument("--classical-dir", type=Path, default=Path("results/credit"))
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--threshold", type=float, default=0.7, help="|r| above which a feature counts as correlated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/credit/correlation_hypothesis.json"))
    args = p.parse_args()

    summary: dict = {}
    for name in args.datasets:
        ds = load(name, args.raw_dir)
        corr = feature_max_abs_corr(ds, args.seed)
        dis = per_feature_disagreement(args.classical_dir, name)
        joined = pd.DataFrame({"max_abs_corr": corr, "rank_spread": dis}).dropna()
        if len(joined) < 6:
            print(f"\n{name}: only {len(joined)} features with both measures, skipping")
            continue

        rho = spearmanr(joined["max_abs_corr"], joined["rank_spread"])
        hi = joined[joined["max_abs_corr"] > args.threshold]["rank_spread"]
        lo = joined[joined["max_abs_corr"] <= args.threshold]["rank_spread"]

        print(f"\n{'=' * 72}\n{name}: {len(joined)} features\n{'=' * 72}")
        print(f"  features with max |r| > {args.threshold}: {len(hi)} of {len(joined)}")
        print(f"  Spearman(max |r|, rank spread) = {rho.statistic:+.3f}  p = {rho.pvalue:.4f}")

        if len(hi) >= 3 and len(lo) >= 3:
            u = mannwhitneyu(hi, lo, alternative="greater")
            p_hi, lo_hi, hi_hi = bootstrap_ci(list(hi), seed=args.seed)
            p_lo, lo_lo, hi_lo = bootstrap_ci(list(lo), seed=args.seed)
            print(f"  mean rank spread, correlated features   {p_hi:.2f} [{lo_hi:.2f}, {hi_hi:.2f}]")
            print(f"  mean rank spread, uncorrelated features {p_lo:.2f} [{lo_lo:.2f}, {hi_lo:.2f}]")
            print(f"  Mann-Whitney, correlated > uncorrelated: p = {u.pvalue:.4f}")
            verdict = "supported" if (rho.statistic > 0 and rho.pvalue < 0.05) else "not supported"
        else:
            print(f"  too few features on one side of the threshold for a group test "
                  f"({len(hi)} correlated, {len(lo)} uncorrelated)")
            verdict = "supported" if (rho.statistic > 0 and rho.pvalue < 0.05) else "not supported"

        print(f"  verdict for {name}: {verdict}")
        print("\n  most correlated features and their disagreement:")
        for f, r in joined.sort_values("max_abs_corr", ascending=False).head(5).iterrows():
            print(f"    {f:<22} max|r|={r['max_abs_corr']:.3f}  rank spread={r['rank_spread']:.2f}")

        summary[name] = {
            "n_features": int(len(joined)),
            "n_correlated": int(len(hi)),
            "spearman": float(rho.statistic),
            "p_value": float(rho.pvalue),
            "verdict": verdict,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    print("\nThe test is run inside each dataset, so it is not confounded by the "
          "many other ways the two datasets differ. Agreement between the two "
          "verdicts is what would make the explanation credible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
