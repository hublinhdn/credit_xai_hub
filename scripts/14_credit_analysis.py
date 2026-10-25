#!/usr/bin/env python3
"""Phase 7, step 8: aggregate the results and answer the research questions.

Reads the results tree and writes every number the paper reports. Nothing is
transcribed by hand, which is what the first paper's analysis could not say.

    python scripts/14_credit_analysis.py --datasets german

Sections, in the order the questions depend on each other:

RQ4a  the validity gate against the published benchmark
RQ4b  what class weighting changes, and what it leaves alone
RQ1   attribution agreement between model families
RQ2   attribution stability of a single model across folds
RQ3   whether accuracy and attribution stability can be had together

RQ1 only compares pairs that McNemar cannot separate: two models that differ in
accuracy are expected to differ in reasoning, so that comparison says nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.stats.tests import (  # noqa: E402
    bootstrap_ci,
    friedman,
    holm,
    mcnemar,
    nemenyi_critical_difference,
)

CLASSICAL_DIR = Path("results/credit")
GATE = {"lr_l2": (0.784, 0.012), "lr_l1": (0.784, 0.012)}

FAMILY = {
    "lr_l1": "linear", "lr_l2": "linear",
    "rf": "tree", "sgb": "tree", "hgb": "tree",
    "mlp": "neural", "majority": "trivial",
}


def family_of(model: str) -> str:
    base = model.replace("_balanced", "")
    if base in FAMILY:
        return FAMILY[base]
    return "quantum"


def load_runs(classical_dir: Path, quantum_dir: Path | None, dataset: str) -> list[dict]:
    """One record per model configuration, with its fold directories.

    The quantum arm is opt-in. Holm corrects over the family of comparisons
    actually made, so loading configurations the paper does not compare would
    inflate that family and make every pair harder to separate.
    """
    runs = []
    for agg_path in sorted(classical_dir.glob(f"{dataset}/*/aggregated.json")):
        a = json.loads(agg_path.read_text())
        runs.append({"name": agg_path.parent.name, "family": family_of(a["model"]),
                     "arm": "classical", "agg": a, "dir": agg_path.parent})
    if quantum_dir is None:
        return runs
    for agg_path in sorted(quantum_dir.glob(f"{dataset}/*/aggregated.json")):
        a = json.loads(agg_path.read_text())
        runs.append({"name": agg_path.parent.name, "family": "quantum",
                     "arm": "quantum", "agg": a, "dir": agg_path.parent})
    return runs


def fold_frames(run_dir: Path, filename: str) -> dict[tuple, pd.DataFrame]:
    """Map a fold key to its predictions or attributions.

    Quantum runs add a seed level below the fold; classical runs do not.
    """
    out = {}
    for path in sorted(run_dir.rglob(filename)):
        parts = path.relative_to(run_dir).parts[:-1]
        out[parts] = pd.read_csv(path)
    return out


def section_gate(runs: list[dict], dataset: str) -> int:
    print(f"\n{'=' * 70}\nRQ4a  validity gate against Lessmann et al. (2015)\n{'=' * 70}")
    if dataset != "german":
        print("  not applicable: the published per-dataset figures cover German Credit")
        return 0
    failures = 0
    for r in runs:
        base = r["agg"].get("model")
        if base in GATE and r["agg"].get("class_weight") in (None, "None"):
            m, s = r["agg"]["auc_roc_mean"], r["agg"]["auc_roc_std"]
            ref_m, ref_s = GATE[base]
            ok = abs(m - ref_m) <= ref_s
            failures += not ok
            print(f"  {base:<8} ours {m:.4f}+/-{s:.4f}  published {ref_m:.3f}+/-{ref_s:.3f}  "
                  f"delta {abs(m - ref_m):.4f}  {'PASS' if ok else 'FAIL'}")
    return failures


def section_weighting(runs: list[dict]) -> None:
    print(f"\n{'=' * 70}\nRQ4b  what class weighting changes\n{'=' * 70}")
    by_name = {r["name"]: r for r in runs}
    print(f"  {'model':<10} {'AUC none':>9} {'AUC bal':>9} {'posrate none':>13} "
          f"{'posrate bal':>12} {'cost none':>10} {'cost bal':>9}")
    for name, r in sorted(by_name.items()):
        if name.endswith("_balanced"):
            continue
        b = by_name.get(f"{name}_balanced")
        if not b:
            continue
        a = r["agg"]
        bb = b["agg"]
        print(f"  {name:<10} {a['auc_roc_mean']:9.4f} {bb['auc_roc_mean']:9.4f} "
              f"{a['pred_positive_rate_mean']:13.4f} {bb['pred_positive_rate_mean']:12.4f} "
              f"{a['expected_cost_at_half_mean']:10.4f} {bb['expected_cost_at_half_mean']:9.4f}")
    print("  AUC is threshold-free, so weighting barely moves it; the operating "
          "point is where the change shows.")


def collapse_to_folds(frames: dict[tuple, pd.DataFrame]) -> dict[tuple, pd.DataFrame]:
    """Reduce every run to one prediction frame per fold.

    Quantum runs nest a seed level below the fold while classical runs do not,
    so their fold keys never intersect and every quantum-against-classical pair
    was silently skipped. Averaging the probabilities over a fold's seeds gives
    one representative vector per fold on both sides, which is what McNemar
    needs.
    """
    by_fold: dict[tuple, list[pd.DataFrame]] = {}
    for key, df in frames.items():
        by_fold.setdefault(key[:2], []).append(df)
    out: dict[tuple, pd.DataFrame] = {}
    for key, dfs in by_fold.items():
        if len(dfs) == 1:
            out[key] = dfs[0]
            continue
        base = dfs[0].sort_values("row_id").reset_index(drop=True)
        probs = np.mean(
            [d.sort_values("row_id")["prob_pos"].to_numpy() for d in dfs], axis=0
        )
        out[key] = pd.DataFrame({
            "row_id": base["row_id"],
            "true_label": base["true_label"],
            "pred": (probs >= 0.5).astype(np.int64),
            "prob_pos": probs,
        })
    return out


def paired_mcnemar(runs: list[dict]) -> tuple[dict, dict]:
    """McNemar on every shared fold, pooled per pair, then Holm-corrected."""
    preds = {r["name"]: collapse_to_folds(fold_frames(r["dir"], "predictions.csv"))
             for r in runs}
    raw: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for a, b in combinations(sorted(preds), 2):
        shared = sorted(set(preds[a]) & set(preds[b]))
        if not shared:
            continue
        tot_b = tot_c = 0
        for key in shared:
            fa, fb = preds[a][key], preds[b][key]
            if len(fa) != len(fb) or not (fa["row_id"].to_numpy() == fb["row_id"].to_numpy()).all():
                continue
            res = mcnemar(fa["pred"].to_numpy(), fb["pred"].to_numpy(), fa["true_label"].to_numpy())
            tot_b += res.b
            tot_c += res.c
        n = tot_b + tot_c
        if n == 0:
            continue
        from math import erfc, sqrt

        stat = (abs(tot_b - tot_c) - 1) ** 2 / n
        key = f"{a} vs {b}"
        raw[key] = float(erfc(sqrt(stat / 2.0)))
        detail[key] = {"b": tot_b, "c": tot_c, "n_folds": len(shared)}
    return raw, holm(raw) if raw else {}


def collapse_attributions(frames: dict[tuple, pd.DataFrame]) -> dict[tuple, pd.DataFrame]:
    """One attribution frame per fold, averaging a fold's seeds.

    Quantum runs carry several seeds per fold and classical runs carry one. Left
    uncollapsed, a quantum model's within-model pairs mix same-fold-different-seed
    comparisons with different-fold ones while a classical model's contain only
    the latter, so seed noise depresses the quantum figure and the two are not
    comparable. Collapsing first makes both sides fold-to-fold.
    """
    by_fold: dict[tuple, list[pd.DataFrame]] = {}
    for key, df in frames.items():
        by_fold.setdefault(key[:2], []).append(df)
    out: dict[tuple, pd.DataFrame] = {}
    for key, dfs in by_fold.items():
        if len(dfs) == 1:
            out[key] = dfs[0]
            continue
        base = dfs[0].sort_values("feature").reset_index(drop=True)
        vals = np.mean([d.sort_values("feature")["attribution"].to_numpy() for d in dfs], axis=0)
        order = np.argsort(-np.abs(vals))
        rank = np.empty(len(vals), dtype=int)
        rank[order] = np.arange(1, len(vals) + 1)
        out[key] = pd.DataFrame({"feature": base["feature"], "attribution": vals, "rank": rank})
    return out


def rank_agreement(x: pd.DataFrame, y: pd.DataFrame, top_k: int = 5) -> dict[str, float]:
    """Spearman, Kendall and top-k overlap between two attribution tables."""
    merged = x.merge(y, on="feature", suffixes=("_a", "_b"))
    if len(merged) < 3:
        return {}
    rho = spearmanr(merged["attribution_a"].abs(), merged["attribution_b"].abs()).statistic
    tau = kendalltau(merged["attribution_a"].abs(), merged["attribution_b"].abs()).statistic
    ta = set(merged.nlargest(top_k, "attribution_a")["feature"])
    tb = set(merged.nlargest(top_k, "attribution_b")["feature"])
    return {"spearman": float(rho), "kendall": float(tau),
            "jaccard_top_k": len(ta & tb) / len(ta | tb) if ta | tb else float("nan")}


def section_attribution(runs: list[dict], top_k: int, seed: int,
                        mcnemar_holm: dict | None = None) -> dict:
    print(f"\n{'=' * 70}\nRQ1 and RQ2  attribution agreement\n{'=' * 70}")
    attrs = {r["name"]: collapse_attributions(fold_frames(r["dir"], "attribution.csv"))
             for r in runs}
    attrs = {k: v for k, v in attrs.items() if v}
    if not attrs:
        print("  no attribution.csv found; run scripts/13_credit_attribution.py first")
        return {"within": {}}

    # RQ2: the same model on different folds.
    print(f"\n  RQ2  within-model stability across folds (Spearman, 95% bootstrap)")
    def summarise(vals: list[float]) -> dict:
        pt, lo, hi = bootstrap_ci(vals, seed=seed)
        return {"rho": pt, "lo": lo, "hi": hi, "n": len(vals)}

    within: dict[str, float] = {}
    within_ci: dict[str, dict] = {}
    for name, folds in sorted(attrs.items()):
        vals = [rank_agreement(folds[a], folds[b], top_k).get("spearman")
                for a, b in combinations(sorted(folds), 2)]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        pt, lo, hi = bootstrap_ci(vals, seed=seed)
        within[name] = pt
        within_ci[name] = {"rho": pt, "lo": lo, "hi": hi, "n": len(vals)}
        # A bootstrap over fewer than three values collapses onto its own
        # point, which looks like a tight interval and is not one.
        ci = f"[{lo:.3f}, {hi:.3f}]" if len(vals) >= 3 else "CI needs 3+ pairs"
        print(f"    {name:<24} family={family_of(name):<9} rho={pt:.3f} {ci}"
              f"  ({len(vals)} fold pairs)")

    # RQ1: different families on the same fold, but only where the two models
    # are not separable on accuracy. Two models that differ in discriminatory
    # power are expected to differ in reasoning, so including such a pair
    # measures the performance gap and reports it as disagreement. The method
    # has always said this; it was not enforced until now, and on Taiwan the
    # linear and tree families are 0.057 apart in AUC against 0.005 on German.
    def separable(a: str, b: str) -> bool:
        if not mcnemar_holm:
            return False
        for key, v in mcnemar_holm.items():
            parts = [x.strip() for x in key.split(" vs ")]
            if a in parts and b in parts:
                return bool(v["significant_05"])
        return False

    print(f"\n  RQ1  between-family agreement, pairs McNemar cannot separate")
    skipped = 0
    between: dict[tuple[str, str], list[float]] = {}
    for a, b in combinations(sorted(attrs), 2):
        fa, fb = family_of(a), family_of(b)
        if fa == fb:
            continue
        if separable(a, b):
            skipped += 1
            continue
        shared = sorted(set(attrs[a]) & set(attrs[b]))
        if not shared:
            continue
        vals = [rank_agreement(attrs[a][k], attrs[b][k], top_k).get("spearman") for k in shared]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            between.setdefault(tuple(sorted((fa, fb))), []).extend(vals)

    if mcnemar_holm:
        print(f"    excluded {skipped} pair(s) separable on accuracy")
    for (fa, fb), vals in sorted(between.items()):
        pt, lo, hi = bootstrap_ci(vals, seed=seed)
        ci = f"[{lo:.3f}, {hi:.3f}]" if len(vals) >= 3 else "CI needs 3+ comparisons"
        print(f"    {fa:<9} vs {fb:<9} rho={pt:.3f} {ci}  ({len(vals)} comparisons)")

    detail: dict = {"within": within, "within_ci": within_ci, "excluded_pairs": skipped,
                    "between": {f"{a} vs {b}": summarise(v) for (a, b), v in between.items()}}

    if within and between:
        w = float(np.mean(list(within.values())))
        bvals = [v for vals in between.values() for v in vals]
        b = float(np.mean(bvals))
        detail["mean_within"] = w
        detail["mean_between"] = b
        print(f"\n    mean within-model  rho = {w:.3f}  ({len(within)} models)")
        print(f"    mean between-family rho = {b:.3f}  ({len(bvals)} comparisons)")
        verdict = "supported" if b < w else "NOT supported"
        if len(within) < 3 or len(bvals) < 10:
            verdict += " (too few comparisons to rest anything on)"
        print(f"    H1 predicts between < within: {verdict}")

    # The same pass with the restriction lifted. Section 5.1 of the paper
    # quotes this to show what omitting the restriction costs, so it is
    # generated here rather than recomputed by hand.
    if mcnemar_holm:
        unrestricted: dict[tuple[str, str], list[float]] = {}
        for a, b in combinations(sorted(attrs), 2):
            fa, fb = family_of(a), family_of(b)
            if fa == fb:
                continue
            shared = sorted(set(attrs[a]) & set(attrs[b]))
            vals = [rank_agreement(attrs[a][k], attrs[b][k], top_k).get("spearman")
                    for k in shared]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                unrestricted.setdefault(tuple(sorted((fa, fb))), []).extend(vals)
        detail["between_unrestricted"] = {
            f"{a} vs {b}": summarise(v) for (a, b), v in unrestricted.items()}
        print(f"\n  the same pairs with the restriction lifted, for Section 5.1")
        for (fa, fb), vals in sorted(unrestricted.items()):
            pt, lo, hi = bootstrap_ci(vals, seed=seed)
            print(f"    {fa:<9} vs {fb:<9} rho={pt:.3f} [{lo:.3f}, {hi:.3f}]"
                  f"  ({len(vals)} comparisons)")
    return detail


def section_friedman(runs: list[dict], alpha: float = 0.05) -> dict:
    """Friedman across folds, with the Nemenyi critical difference.

    Demsar (2006) blocks on datasets, but two datasets give the test no power.
    Folds are used as the blocking factor instead: each fold ranks every model,
    which is the same construction with twenty blocks rather than two.
    """
    print(f"\n{'=' * 70}\nFriedman across folds, Nemenyi post-hoc\n{'=' * 70}")
    per_model: dict[str, dict[tuple, float]] = {}
    for r in runs:
        folds = collapse_to_folds(fold_frames(r["dir"], "predictions.csv"))
        scores = {}
        for key, df in folds.items():
            try:
                from sklearn.metrics import roc_auc_score

                scores[key] = float(roc_auc_score(df["true_label"], df["prob_pos"]))
            except ValueError:
                pass
        if scores:
            per_model[r["name"]] = scores

    shared = set.intersection(*(set(v) for v in per_model.values())) if per_model else set()
    shared = sorted(shared)
    if len(shared) < 3 or len(per_model) < 3:
        print(f"  not enough shared folds ({len(shared)}) or models ({len(per_model)})")
        return {}

    matrix = {name: [per_model[name][k] for k in shared] for name in sorted(per_model)}
    res = friedman(matrix)
    cd = nemenyi_critical_difference(res.n_classifiers, res.n_datasets, alpha)
    print(f"  {res.n_classifiers} models over {res.n_datasets} folds: "
          f"chi2={res.statistic:.2f}, p={res.p_value:.3e}")
    print(f"  Nemenyi critical difference at alpha={alpha}: {cd:.3f} rank units")
    ranked = sorted(res.average_ranks.items(), key=lambda kv: kv[1])
    print(f"  {'model':<24} {'avg rank':>9}  separated from the best?")
    best = ranked[0][1]
    for name, rk in ranked:
        sep = "" if name == ranked[0][0] else ("yes" if rk - best > cd else "no")
        print(f"  {name:<24} {rk:9.2f}  {sep}")
    return {"statistic": res.statistic, "p_value": res.p_value,
            "critical_difference": cd, "average_ranks": res.average_ranks}


def section_protected(classical_dir: Path, noprot_dir: Path, dataset: str, seed: int) -> dict:
    """Does dropping the fair-lending protected attributes change anything?

    Two questions, because they can disagree: whether discrimination falls, and
    whether the explanation changes.
    """
    print(f"\n{'=' * 70}\nfair-lending protected attributes: with against without\n{'=' * 70}")
    out: dict = {}
    print(f"  {'model':<10} {'AUC with':>9} {'AUC without':>12} {'delta':>8}  "
          f"{'attribution rho':>16}")
    for agg_with in sorted(classical_dir.glob(f"{dataset}/*/aggregated.json")):
        name = agg_with.parent.name
        agg_without = noprot_dir / dataset / name / "aggregated.json"
        if not agg_without.exists():
            continue
        a = json.loads(agg_with.read_text())
        b = json.loads(agg_without.read_text())
        if a.get("auc_roc_mean") is None or b.get("auc_roc_mean") is None:
            continue

        # Attribution agreement over the features both versions still have.
        attr_a = fold_frames(agg_with.parent, "attribution.csv")
        attr_b = fold_frames(agg_without.parent, "attribution.csv")
        rhos = []
        for key in sorted(set(attr_a) & set(attr_b)):
            r = rank_agreement(attr_a[key], attr_b[key]).get("spearman")
            if r is not None and np.isfinite(r):
                rhos.append(r)
        rho_txt = f"{np.mean(rhos):.3f} (n={len(rhos)})" if rhos else "no attributions"
        print(f"  {name:<10} {a['auc_roc_mean']:9.4f} {b['auc_roc_mean']:12.4f} "
              f"{b['auc_roc_mean'] - a['auc_roc_mean']:+8.4f}  {rho_txt:>16}")
        out[name] = {"auc_with": a["auc_roc_mean"], "auc_without": b["auc_roc_mean"],
                     "attribution_rho": float(np.mean(rhos)) if rhos else None}
    if out:
        d = [v["auc_without"] - v["auc_with"] for v in out.values()]
        print(f"\n  mean AUC change from dropping them: {np.mean(d):+.4f}")
        print("  the protected attributes carry no incremental discriminatory value"
              if abs(np.mean(d)) < 0.01 else "  dropping them changes discrimination materially")
    return out


def section_rq3(runs: list[dict], within: dict[str, float]) -> None:
    """RQ3 as designed: can accuracy and attribution stability be had together?

    H3 says no configuration sits on the Pareto frontier of both axes. The
    frontier is computed rather than eyeballed.
    """
    print(f"\n{'=' * 70}\nRQ3  accuracy against attribution stability\n{'=' * 70}")
    pts = []
    for r in runs:
        rho = within.get(r["name"])
        auc = r["agg"].get("auc_roc_mean")
        if rho is None or auc is None:
            continue
        pts.append({"name": r["name"], "family": r["family"], "auc": auc, "rho": rho})
    if len(pts) < 3:
        print("  needs attributions for at least three models")
        return

    frontier = [
        p for p in pts
        if not any(q["auc"] >= p["auc"] and q["rho"] >= p["rho"]
                   and (q["auc"] > p["auc"] or q["rho"] > p["rho"]) for q in pts)
    ]
    best_auc = max(pts, key=lambda p: p["auc"])
    best_rho = max(pts, key=lambda p: p["rho"])
    print(f"  most accurate   {best_auc['name']:<24} AUC={best_auc['auc']:.4f} rho={best_auc['rho']:.3f}")
    print(f"  most stable     {best_rho['name']:<24} AUC={best_rho['auc']:.4f} rho={best_rho['rho']:.3f}")
    print(f"\n  Pareto frontier ({len(frontier)} of {len(pts)} configurations):")
    for p in sorted(frontier, key=lambda p: -p["auc"]):
        print(f"    {p['name']:<24} {p['family']:<9} AUC={p['auc']:.4f} rho={p['rho']:.3f}")
    both = best_auc["name"] == best_rho["name"]
    print(f"\n  H3 says no configuration is best on both axes: "
          f"{'NOT supported, one model wins both' if both else 'supported'}")


def section_tradeoff(runs: list[dict]) -> None:
    print(f"\n{'=' * 70}\naccuracy against parameter count\n{'=' * 70}")
    rows = []
    for r in runs:
        a = r["agg"]
        if a.get("auc_roc_mean") is None:
            continue
        rows.append({"name": r["name"], "family": r["family"], "arm": r["arm"],
                     "auc": a["auc_roc_mean"], "sd": a["auc_roc_std"],
                     "params": a.get("n_params") or a.get("n_params_median")})
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(f"  {'model':<24} {'arm':<10} {'family':<9} {'AUC':>8} {'sd':>7} {'params':>9}")
    for _, r in df.iterrows():
        p = "n/a" if r["params"] is None or (isinstance(r["params"], float) and np.isnan(r["params"])) else f"{int(r['params']):,}"
        print(f"  {r['name']:<24} {r['arm']:<10} {r['family']:<9} {r['auc']:8.4f} {r['sd']:7.4f} {p:>9}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["german"])
    p.add_argument("--classical-dir", type=Path, default=CLASSICAL_DIR)
    p.add_argument("--quantum-dir", type=Path, default=None,
                   help="opt-in second arm; omitted, only the classical grid is analysed")
    p.add_argument("--noprot-dir", type=Path, default=Path("results/credit_noprot"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/credit/analysis.json"))
    args = p.parse_args()

    summary: dict = {}
    gate_failures = 0
    for dataset in args.datasets:
        runs = load_runs(args.classical_dir, args.quantum_dir, dataset)
        if not runs:
            print(f"no results for {dataset}")
            continue
        print(f"\n{'#' * 70}\n# {dataset}: {len(runs)} configurations\n{'#' * 70}")
        gate_failures += section_gate(runs, dataset)
        section_weighting(runs)
        section_tradeoff(runs)

        raw, corrected = paired_mcnemar(runs)
        print(f"\n{'=' * 70}\npaired McNemar with Holm correction ({len(raw)} pairs)\n{'=' * 70}")
        indistinguishable = [k for k, v in corrected.items() if not v["significant_05"]]
        print(f"  pairs a reader cannot separate at 0.05: {len(indistinguishable)} of {len(raw)}")
        for k in sorted(corrected, key=lambda k: corrected[k]["p_holm"])[:10]:
            v = corrected[k]
            print(f"    {k:<46} p={v['p_raw']:.3e} holm={v['p_holm']:.3e} "
                  f"{'*' if v['significant_05'] else ''}")

        attribution = section_attribution(runs, args.top_k, args.seed, corrected)
        section_rq3(runs, attribution["within"])
        fried = section_friedman(runs)
        prot = section_protected(args.classical_dir, args.noprot_dir, dataset, args.seed)
        summary[dataset] = {
            "attribution": attribution,
            "friedman": fried,
            "protected": prot,
            "n_configs": len(runs),
            "mcnemar_holm": corrected,
            "indistinguishable_pairs": indistinguishable,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 1 if gate_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
