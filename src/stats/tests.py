"""Paired comparison of classifiers, following Demsar (2006).

Three layers, in the order the analysis uses them:

1. ``mcnemar`` compares two classifiers on one evaluation fold, from their
   predictions on the same rows.
2. ``holm`` corrects a family of p-values for multiple comparison.
3. ``friedman`` with ``nemenyi_critical_difference`` compares many classifiers
   across many datasets, which is the protocol the reference credit-scoring
   benchmark uses.

Every function takes data and returns numbers. The first paper's analysis
transcribed p-values between scripts by hand, which is the one step of that
pipeline a reader could not reproduce; nothing here reads a hardcoded value.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range


@dataclass
class McNemarResult:
    """Outcome of one paired comparison on a single fold."""

    b: int          # a wrong, b right
    c: int          # a right, b wrong
    n_discordant: int
    statistic: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray) -> McNemarResult:
    """McNemar's test with Edwards' continuity correction.

    The two prediction vectors must be aligned to the same rows in the same
    order. With no discordant pairs the classifiers agree everywhere and the
    test is undefined, which is reported as ``p = 1``.
    """
    pred_a = np.asarray(pred_a).ravel()
    pred_b = np.asarray(pred_b).ravel()
    truth = np.asarray(truth).ravel()
    if not (len(pred_a) == len(pred_b) == len(truth)):
        raise ValueError("prediction and truth vectors must be the same length")

    a_right = pred_a == truth
    b_right = pred_b == truth
    b = int((~a_right & b_right).sum())
    c = int((a_right & ~b_right).sum())
    n = b + c
    if n == 0:
        return McNemarResult(b=b, c=c, n_discordant=0, statistic=0.0, p_value=1.0)

    stat = (abs(b - c) - 1) ** 2 / n
    # Survival function of chi-square with one degree of freedom.
    from math import erfc, sqrt

    return McNemarResult(b=b, c=c, n_discordant=n, statistic=float(stat),
                         p_value=float(erfc(sqrt(stat / 2.0))))


def holm(p_values: Mapping[str, float]) -> dict[str, dict[str, float | bool]]:
    """Holm step-down correction of a family of p-values.

    Sorts ascending, multiplies the k-th smallest by ``m - k``, then enforces
    monotonicity so an adjusted value never falls below an earlier one.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for k, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - k) * p))
        running = adj
        out[name] = {"p_raw": float(p), "p_holm": float(adj), "significant_05": bool(adj < 0.05)}
    return out


@dataclass
class FriedmanResult:
    """Friedman test across classifiers and datasets, plus the average ranks."""

    statistic: float
    p_value: float
    n_datasets: int
    n_classifiers: int
    average_ranks: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def friedman(scores: Mapping[str, Sequence[float]]) -> FriedmanResult:
    """Friedman test on a classifier-by-dataset score matrix.

    ``scores`` maps each classifier to its score on every dataset, in a
    consistent dataset order. Higher scores are better, so they are ranked
    descending: rank 1 is the best classifier on that dataset.
    """
    names = list(scores)
    matrix = np.asarray([list(scores[k]) for k in names], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("need at least two classifiers")
    if matrix.shape[1] < 2:
        raise ValueError("need at least two datasets")
    if not np.isfinite(matrix).all():
        raise ValueError("scores contain NaN or infinity")

    stat, p = friedmanchisquare(*matrix)
    # Rank per dataset, best = 1.
    ranks = np.apply_along_axis(lambda col: rankdata(-col), 0, matrix)
    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p),
        n_datasets=matrix.shape[1],
        n_classifiers=matrix.shape[0],
        average_ranks={n: float(r) for n, r in zip(names, ranks.mean(axis=1))},
    )


def nemenyi_critical_difference(n_classifiers: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Critical difference of average ranks for the Nemenyi post-hoc test.

    Two classifiers differ significantly when their average ranks are at least
    this far apart. Uses the studentized range at ``alpha``, divided by sqrt(2)
    as the test prescribes.
    """
    q = studentized_range.ppf(1 - alpha, n_classifiers, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(n_classifiers * (n_classifiers + 1) / (6.0 * n_datasets)))


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval. Returns (point, low, high)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
    stats = np.apply_along_axis(statistic, 1, draws)
    return (
        float(statistic(arr)),
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    )
