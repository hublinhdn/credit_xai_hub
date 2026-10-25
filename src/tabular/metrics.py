"""Credit-scoring performance measures.

Covers the four indicators used by Lessmann et al. (2015) so that results can
be placed on their scale (PCC, AUC, H-measure, Brier score) plus the measures a
credit application needs but that study does not report (balanced accuracy,
PR-AUC, the Kolmogorov-Smirnov statistic, and expected misclassification cost
under the asymmetric cost matrix shipped with German Credit).

``y = 1`` is the adverse outcome throughout, matching ``src.tabular.datasets``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _upper_convex_hull(fpr: np.ndarray, tpr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper convex hull of an ROC curve, as vertices ordered by FPR.

    ``fpr`` must be sorted ascending, as returned by ``sklearn.roc_curve``.
    """
    hull_x: list[float] = []
    hull_y: list[float] = []
    for x, y in zip(fpr, tpr):
        # Drop vertices that a new point makes non-concave.
        while len(hull_x) >= 2:
            x1, y1 = hull_x[-2], hull_y[-2]
            x2, y2 = hull_x[-1], hull_y[-1]
            # Cross product <= 0 means (x2, y2) is at or below the chord.
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                hull_x.pop()
                hull_y.pop()
            else:
                break
        hull_x.append(float(x))
        hull_y.append(float(y))
    return np.asarray(hull_x), np.asarray(hull_y)


def h_measure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> float:
    """H-measure of Hand (2009).

    The relative severity of the two error types is treated as unknown and
    integrated out under a ``Beta(alpha, beta)`` prior, which is what makes the
    measure independent of the classifier being scored (the criticism of AUC
    raised by Hand and Anagnostopoulos). ``Beta(2, 2)`` is the symmetric default;
    pass other values to weight one error type more heavily.

    Returns a value in ``[0, 1]``, higher is better.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n = len(y_true)
    n_bad = float(y_true.sum())
    if n_bad == 0 or n_bad == n:
        return float("nan")

    pi1 = n_bad / n          # prior of the adverse class
    pi0 = 1.0 - pi1

    fpr, tpr, _ = roc_curve(y_true, y_score)
    hx, hy = _upper_convex_hull(fpr, tpr)
    if len(hx) < 2:
        return float("nan")

    # Beta integrals: I0 = \int w, I1 = \int c w.
    mean_c = alpha / (alpha + beta)

    def i0(a: float, b: float) -> float:
        return float(beta_dist.cdf(b, alpha, beta) - beta_dist.cdf(a, alpha, beta))

    def i1(a: float, b: float) -> float:
        return float(
            mean_c * (beta_dist.cdf(b, alpha + 1, beta) - beta_dist.cdf(a, alpha + 1, beta))
        )

    # Hull segment slopes, decreasing. Vertex j is cost-optimal between the
    # cost parameters implied by its two adjacent slopes.
    dx = np.diff(hx)
    dy = np.diff(hy)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(dx > 0, dy / np.maximum(dx, 1e-300), np.inf)

    def c_of_slope(s: float) -> float:
        if not np.isfinite(s):
            return 1.0
        denom = pi0 + s * pi1
        return 0.0 if denom <= 0 else float(s * pi1 / denom)

    # Boundaries c_1 >= c_2 >= ... >= c_k, one per segment.
    cuts = [c_of_slope(float(s)) for s in slopes]

    loss = 0.0
    # Vertex 0 = (0, 0) is optimal for c in [cuts[0], 1].
    lo, hi = min(cuts[0], 1.0), 1.0
    loss += pi0 * hx[0] * i1(lo, hi) + pi1 * (1.0 - hy[0]) * (i0(lo, hi) - i1(lo, hi))
    # Interior vertices.
    for j in range(1, len(hx) - 1):
        hi_j = cuts[j - 1]
        lo_j = cuts[j]
        if lo_j > hi_j:
            lo_j, hi_j = hi_j, lo_j
        loss += pi0 * hx[j] * i1(lo_j, hi_j) + pi1 * (1.0 - hy[j]) * (i0(lo_j, hi_j) - i1(lo_j, hi_j))
    # Last vertex = (1, 1) is optimal for c in [0, cuts[-1]].
    lo, hi = 0.0, max(cuts[-1], 0.0)
    loss += pi0 * hx[-1] * i1(lo, hi) + pi1 * (1.0 - hy[-1]) * (i0(lo, hi) - i1(lo, hi))

    # Trivial classifier: predict all positive below c = pi1, all negative above.
    loss_max = pi0 * i1(0.0, pi1) + pi1 * (i0(pi1, 1.0) - i1(pi1, 1.0))
    if loss_max <= 0:
        return float("nan")
    return float(1.0 - loss / loss_max)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation, ``max(TPR - FPR)`` over all thresholds."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).ravel(), np.asarray(y_score).ravel())
    return float(np.max(tpr - fpr))


def expected_cost(
    y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float = 1.0, cost_fp: float = 1.0
) -> float:
    """Average misclassification cost per applicant.

    ``cost_fn`` is charged when an adverse case is predicted good (the costly
    error in lending), ``cost_fp`` when a good case is predicted adverse.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    return float((cost_fn * fn + cost_fp * fp) / len(y_true))


def min_expected_cost(
    y_true: np.ndarray, y_score: np.ndarray, cost_fn: float = 1.0, cost_fp: float = 1.0
) -> tuple[float, float]:
    """Lowest achievable average cost and the threshold that attains it."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    order = np.argsort(-y_score, kind="mergesort")
    ys = y_true[order]
    n_pos = float(ys.sum())
    n = len(ys)

    # Sweep the threshold from "predict none positive" downwards.
    tp = np.concatenate([[0.0], np.cumsum(ys)])
    fp = np.concatenate([[0.0], np.cumsum(1.0 - ys)])
    fn = n_pos - tp
    costs = (cost_fn * fn + cost_fp * fp) / n
    k = int(np.argmin(costs))
    thr = float(y_score[order][min(k, n - 1)]) if k > 0 else float(np.max(y_score) + 1.0)
    return float(costs[k]), thr


@dataclass
class CreditMetrics:
    """Performance of one fitted model on one evaluation fold."""

    n_samples: int
    bad_rate: float
    pred_positive_rate: float
    pcc: float                     # percentage correctly classified = accuracy
    balanced_accuracy: float
    f1_macro: float
    auc_roc: float | None
    pr_auc: float | None
    ks: float | None
    brier: float | None
    h_measure: float | None
    expected_cost_at_half: float
    expected_cost_min: float | None
    cost_optimal_threshold: float | None
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_credit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
    h_alpha: float = 2.0,
    h_beta: float = 2.0,
) -> CreditMetrics:
    """Compute the full measure set for one fold.

    ``y_score`` is the predicted probability of the adverse class. Score-based
    measures are reported as ``None`` when it is not supplied.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    auc = pr_auc = ks = brier = hm = None
    cost_min: float | None = None
    thr: float | None = None
    if y_score is not None:
        s = np.asarray(y_score).ravel()
        try:
            auc = float(roc_auc_score(y_true, s))
            pr_auc = float(average_precision_score(y_true, s))
            ks = ks_statistic(y_true, s)
            hm = h_measure(y_true, s, alpha=h_alpha, beta=h_beta)
            cost_min, thr = min_expected_cost(y_true, s, cost_fn, cost_fp)
        except ValueError:
            pass  # a fold with a single class present
        try:
            brier = float(brier_score_loss(y_true, s))
        except ValueError:
            brier = None

    return CreditMetrics(
        n_samples=int(len(y_true)),
        bad_rate=float(y_true.mean()),
        pred_positive_rate=float(y_pred.mean()),
        pcc=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        auc_roc=auc,
        pr_auc=pr_auc,
        ks=ks,
        brier=brier,
        h_measure=hm,
        expected_cost_at_half=expected_cost(y_true, y_pred, cost_fn, cost_fp),
        expected_cost_min=cost_min,
        cost_optimal_threshold=thr,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    )


def aggregate_credit(metrics: list[CreditMetrics]) -> dict[str, Any]:
    """Mean and population standard deviation of every numeric measure."""
    if not metrics:
        return {"n_folds": 0}
    out: dict[str, Any] = {"n_folds": len(metrics)}
    numeric_fields = [
        "pcc", "balanced_accuracy", "f1_macro", "auc_roc", "pr_auc", "ks",
        "brier", "h_measure", "expected_cost_at_half", "expected_cost_min",
        "pred_positive_rate",
    ]
    for field_name in numeric_fields:
        vals = [getattr(m, field_name) for m in metrics]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            out[f"{field_name}_mean"] = float(np.mean(vals))
            out[f"{field_name}_std"] = float(np.std(vals, ddof=0))
        else:
            out[f"{field_name}_mean"] = None
            out[f"{field_name}_std"] = None
    out["per_fold"] = [m.to_dict() for m in metrics]
    return out
