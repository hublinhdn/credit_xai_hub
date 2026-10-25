#!/usr/bin/env python3
"""Sanity checks for the credit-risk pipeline.

Run before trusting any result and before pushing. Each check guards a mistake
that was actually made while building this pipeline, or one the literature
warns about.

    python scripts/11b_sanity_credit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tabular.datasets import load  # noqa: E402
from src.tabular.metrics import (  # noqa: E402
    compute_credit_metrics,
    expected_cost,
    h_measure,
    min_expected_cost,
)
from src.stats.tests import (  # noqa: E402
    bootstrap_ci,
    friedman,
    holm,
    mcnemar,
    nemenyi_critical_difference,
)
from src.tabular.splits import nx2_splits  # noqa: E402
from src.tabular.woe import WoEEncoder  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def check_splits() -> None:
    print("N x 2 cross-validation")
    y = np.array([0] * 700 + [1] * 300)
    folds = list(nx2_splits(y, n_repeats=10, seed=0))
    check("yields 2N fits", len(folds) == 20, f"got {len(folds)}")

    # The bug this guards: StratifiedKFold(n_splits=2) already emits both
    # orientations, so swapping again evaluates each repeat's first fold twice.
    r0 = [f for f in folds if f[0] == 0]
    a_train, a_test = r0[0][2], r0[0][3]
    b_train, b_test = r0[1][2], r0[1][3]
    check(
        "the two halves of a repeat are exchanged, not duplicated",
        set(a_train) == set(b_test) and set(a_test) == set(b_train),
    )
    check(
        "halves are disjoint and cover the dataset",
        len(set(a_train) & set(a_test)) == 0 and len(set(a_train) | set(a_test)) == len(y),
    )
    counts = np.zeros(len(y), dtype=int)
    for _, _, _, test_idx in folds:
        counts[test_idx] += 1
    check("every row is tested exactly N times", bool((counts == 10).all()), f"range {counts.min()}..{counts.max()}")
    rates = [y[f[3]].mean() for f in folds]
    check("test halves are stratified", max(rates) - min(rates) < 0.01, f"spread {max(rates)-min(rates):.4f}")


def check_woe() -> None:
    print("Weight-of-evidence coding")
    X = pd.DataFrame({"cat": ["good"] * 500 + ["bad"] * 500})
    y = np.array([0] * 450 + [1] * 50 + [0] * 50 + [1] * 450)
    maps = WoEEncoder(columns=["cat"]).fit(X, y).maps_["cat"]
    check("higher WoE means lower bad rate", maps["good"] > maps["bad"],
          f"good={maps['good']:.3f} bad={maps['bad']:.3f}")

    enc = WoEEncoder(columns=["cat"]).fit(X, y)
    unseen = enc.transform(pd.DataFrame({"cat": ["never_seen"]}))["cat"].iloc[0]
    check("unseen category carries no evidence", unseen == 0.0, f"got {unseen}")

    # Leakage guard: WoE fitted on one half must not reproduce the other half's
    # category statistics exactly.
    rng = np.random.RandomState(0)
    Xr = pd.DataFrame({"cat": rng.choice(list("abcd"), 1000)})
    yr = (rng.rand(1000) < 0.3).astype(int)
    m_half = WoEEncoder(columns=["cat"]).fit(Xr.iloc[:500], yr[:500]).maps_["cat"]
    m_full = WoEEncoder(columns=["cat"]).fit(Xr, yr).maps_["cat"]
    check("fitting on a half differs from fitting on the whole",
          any(abs(m_half[k] - m_full[k]) > 1e-9 for k in m_half))


def check_metrics() -> None:
    print("Performance measures")
    rng = np.random.RandomState(0)
    n = 2000
    y = (rng.rand(n) < 0.3).astype(int)
    perfect = y + rng.rand(n) * 1e-6
    check("H-measure of a perfect ranking is 1", abs(h_measure(y, perfect) - 1.0) < 1e-6)
    check("H-measure of a random ranking is near 0", abs(h_measure(y, rng.rand(n))) < 0.05)

    s = 1 / (1 + np.exp(-(1.2 * (2 * y - 1) + rng.randn(n))))
    h_raw = h_measure(y, s)
    h_rank = h_measure(y, pd.Series(s).rank().to_numpy())
    check("H-measure is invariant to monotone score transforms", abs(h_raw - h_rank) < 1e-9)

    hs = [h_measure(y, 1 / (1 + np.exp(-(k * (2 * y - 1) + rng.randn(n))))) for k in (0.0, 1.0, 2.0)]
    check("H-measure increases with signal strength", hs[0] < hs[1] < hs[2],
          " < ".join(f"{v:.3f}" for v in hs))

    c_min, _ = min_expected_cost(y, s, 5.0, 1.0)
    brute = min(expected_cost(y, (s >= t).astype(int), 5.0, 1.0) for t in np.unique(s))
    check("cost-optimal threshold matches a brute-force sweep", abs(c_min - brute) < 1e-9,
          f"{c_min:.6f} vs {brute:.6f}")

    m = compute_credit_metrics(y, (s >= 0.5).astype(int), s, cost_fn=5.0, cost_fp=1.0)
    check("minimum cost does not exceed cost at threshold 0.5",
          m.expected_cost_min <= m.expected_cost_at_half + 1e-12)

    # A model that predicts one class only must be visible, not silently scored.
    degenerate = compute_credit_metrics(y, np.zeros(n, dtype=int), np.full(n, 0.3))
    check("all-negative prediction is visible as pred_positive_rate 0",
          degenerate.pred_positive_rate == 0.0)


def check_datasets() -> None:
    print("Datasets")
    expected = {
        "german": (1000, 20, 300, 0.3000),
        "taiwan": (30000, 23, 6636, 0.2212),
    }
    for name, (n, feats, bad, rate) in expected.items():
        try:
            ds = load(name)
        except FileNotFoundError:
            check(f"{name} present", False, "run scripts/10_prepare_credit.py --download")
            continue
        check(f"{name} row/feature/label counts",
              len(ds) == n and ds.X.shape[1] == feats and int(ds.y.sum()) == bad,
              f"n={len(ds)} features={ds.X.shape[1]} bad={int(ds.y.sum())}")
        check(f"{name} bad rate", abs(ds.bad_rate - rate) < 5e-4, f"{ds.bad_rate:.4f}")
        check(f"{name} label 1 is the adverse class", ds.bad_rate < 0.5)
        check(f"{name} declares protected attributes", len(ds.protected) > 0,
              ", ".join(ds.protected))
        nodrop = ds.without_protected()
        check(f"{name} protected attributes can be dropped",
              nodrop.X.shape[1] == ds.X.shape[1] - len(ds.protected))


def check_stats() -> None:
    print("Statistical tests")
    from scipy.stats import chi2

    rng = np.random.RandomState(0)
    n = 500
    truth = (rng.rand(n) < 0.3).astype(int)
    check("McNemar on identical predictions is undefined, reported as p=1",
          mcnemar(truth, truth, truth).p_value == 1.0)

    a, b = truth.copy(), truth.copy()
    a[:40] = 1 - a[:40]
    b[:5] = 1 - b[:5]
    r = mcnemar(a, b, truth)
    check("McNemar p matches the chi-square survival function",
          abs(r.p_value - chi2.sf(r.statistic, 1)) < 1e-12, f"p={r.p_value:.3e}")
    check("McNemar counts discordant pairs the right way round", r.b > r.c,
          f"b={r.b} c={r.c}")

    # Worked example: raw 0.01/0.02/0.03/0.9 over a family of four.
    h = holm({"t1": 0.01, "t2": 0.02, "t3": 0.03, "t4": 0.9})
    check("Holm step-down arithmetic",
          abs(h["t1"]["p_holm"] - 0.04) < 1e-12
          and abs(h["t2"]["p_holm"] - 0.06) < 1e-12
          and abs(h["t3"]["p_holm"] - 0.06) < 1e-12
          and abs(h["t4"]["p_holm"] - 0.9) < 1e-12)
    check("Holm adjusted values never decrease",
          all(
              h[k1]["p_holm"] <= h[k2]["p_holm"] + 1e-12
              for k1, k2 in zip(["t1", "t2", "t3"], ["t2", "t3", "t4"])
          ))

    f = friedman({"good": [0.90, 0.88, 0.91, 0.89],
                  "mid": [0.80, 0.79, 0.82, 0.81],
                  "bad": [0.70, 0.68, 0.72, 0.69]})
    check("Friedman ranks the best classifier 1 (higher score is better)",
          f.average_ranks["good"] == 1.0 and f.average_ranks["bad"] == 3.0)
    check("Nemenyi critical difference is positive and finite",
          0 < nemenyi_critical_difference(3, 4) < 10,
          f"CD={nemenyi_critical_difference(3, 4):.3f}")

    pt, lo, hi = bootstrap_ci(list(rng.randn(200) * 0.1 + 0.8), seed=0)
    check("bootstrap interval brackets its point estimate", lo < pt < hi,
          f"{pt:.4f} in [{lo:.4f}, {hi:.4f}]")


def main() -> int:
    for fn in (check_splits, check_woe, check_metrics, check_stats, check_datasets):
        fn()
        print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: " + ", ".join(FAILS))
        return 1
    print("All sanity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
