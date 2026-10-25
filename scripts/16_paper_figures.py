#!/usr/bin/env python3
"""Phase 7, step 10: every figure and table the paper uses.

Reads the results tree and writes PNG figures plus LaTeX table fragments, so no
number in the manuscript is typed by hand. Run it again after any experiment
changes and the manuscript updates with it.

    python scripts/16_paper_figures.py --out-dir report-credit/figures \
        --tex-dir report-credit/tables

House style follows the first paper so the two look like one body of work:
the primary series in blue, the second benchmark in amber, reference lines in
red. Resolution is raised from that paper's 140 dpi, which is thin for print.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DPI = 300
C_CLASSICAL = "#3a7ca5"
C_RED = "#d9534f"
C_BASELINE = "#888888"
C_ACCENT = "#5cb85c"
C_WARN = "#f0ad4e"


def load_aggregates(classical: Path, dataset: str) -> list[dict]:
    rows = []
    for f in sorted(classical.glob(f"{dataset}/*/aggregated.json")):
        a = json.loads(f.read_text())
        if a.get("auc_roc_mean") is None:
            continue
        rows.append({
            "name": f.parent.name,
            "auc": a["auc_roc_mean"],
            "sd": a.get("auc_roc_std") or 0.0,
            "pcc": a.get("pcc_mean"),
            "posrate": a.get("pred_positive_rate_mean"),
            "cost": a.get("expected_cost_at_half_mean"),
            "h": a.get("h_measure_mean"),
        })
    return rows


def fig_controls(controls: dict[str, dict], out: Path) -> None:
    """The paper's key figure: the three controls, both datasets.

    Shows at a glance that the measure is innocent, the projection is not, and
    that two model families inside one feature space agree closely.
    """
    labels = ["A. measure\n(gradient vs SHAP)",
              "B. representation\n(PCA round trip)",
              "C. model family\n(two families, one space)"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.36
    x = np.arange(3)

    for i, (ds, col) in enumerate((("german", C_CLASSICAL), ("taiwan", C_WARN))):
        c = controls.get(ds)
        if not c:
            continue
        vals, los, his = [], [], []
        for key in ("A", "B", "C"):
            node = c.get(key, {})
            if key in ("A", "B"):
                sub = [v for v in node.values() if isinstance(v, dict) and "rho" in v]
                v = float(np.mean([s["rho"] for s in sub])) if sub else np.nan
                lo = float(np.mean([s["lo"] for s in sub])) if sub else np.nan
                hi = float(np.mean([s["hi"] for s in sub])) if sub else np.nan
            else:
                v, lo, hi = node.get("rho", np.nan), node.get("lo", np.nan), node.get("hi", np.nan)
            vals.append(v)
            los.append(v - lo if np.isfinite(lo) else 0)
            his.append(hi - v if np.isfinite(hi) else 0)
        tops = [v + h if np.isfinite(v) else np.nan for v, h in zip(vals, his)]
        pos = x + (i - 0.5) * width
        ax.bar(pos, vals, width, yerr=[los, his], capsize=4,
               color=col, alpha=0.8, edgecolor="white", linewidth=0.6,
               label=f"{ds.capitalize()} Credit" if ds == "german" else "Taiwan default")
        # Overlay the per-model values. Averaging hides the most telling case:
        # on Taiwan, control B is -0.290 for the linear model and +0.331 for the
        # neural one, and the mean of 0.022 conceals that the ranking inverts.
        for xi, key in zip(pos, ("A", "B", "C")):
            node = c.get(key, {})
            if key == "C":
                continue
            per = [(k, v["rho"]) for k, v in node.items()
                   if isinstance(v, dict) and "rho" in v]
            for k, v in per:
                ax.plot(xi - width * 0.26, v, marker="o", color="black", markersize=5,
                        markerfacecolor="white", markeredgewidth=1.2, zorder=4)
                ax.annotate(k, (xi - width * 0.26, v), textcoords="offset points",
                            xytext=(8, -3), fontsize=7.5, color="black")
        # No value labels: Table~\ref{tab:controls} carries the numbers, and
        # printing them here put the same six figures on the page twice.

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Spearman rank correlation of feature attributions")
    ax.set_title("What a feature attribution actually depends on")
    ax.set_ylim(-0.45, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")



def fig_weighting(rows: list[dict], out: Path, bad_rate: float, dataset: str) -> None:
    """Class weighting moves the operating point and leaves AUC alone."""
    pairs = []
    by = {r["name"]: r for r in rows}
    for name, r in sorted(by.items()):
        if name.endswith("_balanced") or f"{name}_balanced" not in by:
            continue
        pairs.append((name, r, by[f"{name}_balanced"]))
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = [p[0] for p in pairs]
    x = np.arange(len(pairs))
    w = 0.36

    axes[0].bar(x - w / 2, [p[1]["auc"] for p in pairs], w, color=C_CLASSICAL,
                alpha=0.85, label="unweighted")
    axes[0].bar(x + w / 2, [p[2]["auc"] for p in pairs], w, color=C_ACCENT,
                alpha=0.85, label="balanced")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Discrimination barely moves")
    axes[0].set_ylim(0.45, 0.85)

    axes[1].bar(x - w / 2, [p[1]["posrate"] for p in pairs], w, color=C_CLASSICAL, alpha=0.85)
    axes[1].bar(x + w / 2, [p[2]["posrate"] for p in pairs], w, color=C_ACCENT, alpha=0.85)
    axes[1].axhline(bad_rate, color=C_RED, linestyle="--", linewidth=1.3)
    axes[1].text(len(pairs) - 0.4, bad_rate + 0.012, f"true rate {bad_rate:.3f}",
                 color=C_RED, fontsize=9, ha="right")
    axes[1].set_ylabel("Predicted positive rate")
    axes[1].set_title("The operating point does")

    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        a.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Class weighting on {dataset.capitalize()}", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")



def tex_table(rows: list[str], header: str, caption: str, label: str, spec: str) -> str:
    body = "\n".join(rows)
    return (
        "\\begin{table}[h!]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def table_discrimination(rows: list[dict], mcnemar: dict, out: Path) -> None:
    """Which models may enter the attribution comparison, and why.

    The last column is the paper's precondition made visible: a pair McNemar
    separates is a pair whose attributions should not be compared.
    """
    base = [r for r in rows if not r["name"].endswith("_balanced")
            and r["name"] != "majority"]
    ordered = sorted(base, key=lambda r: -r["auc"])
    top = ordered[0] if ordered else None
    lines = []
    for r in ordered:
        verdict = "--"
        if top and r["name"] != top["name"]:
            for k, v in mcnemar.items():
                parts = [x.strip() for x in k.split(" vs ")]
                if top["name"] in parts and r["name"] in parts:
                    verdict = "yes" if v["significant_05"] else "no"
                    break
        name = r["name"].replace("_", "\\_")
        bold = "\\textbf{" + name + "}" if r is top else name
        lines.append(f"{bold} & {r['auc']:.4f} & {r['sd']:.4f} & {verdict} \\\\")
    top_name = top["name"].replace("_", "\\_") if top else "the best model"
    out.write_text(tex_table(
        lines,
        "Model & AUC & SD & Separable from the best?",
        "Discrimination on German Credit under ten by two cross-validation, "
        f"unweighted models only. The last column reports paired McNemar with "
        f"Holm correction against \\texttt{{{top_name}}}; a pair it separates is "
        "excluded from the attribution comparison.",
        "tab:discrimination", "lrrc"))
    print(f"  wrote {out}")


def table_controls(controls: dict[str, dict], nulls: dict[str, dict], out: Path) -> None:
    """The four controls in one table.

    Control D used to have a table of its own, which repeated C's value as one
    of its rows. One table, with a rule between the controls and the
    diagnostics that interpret C, says the same thing once.
    """
    names = {"A": "A. Measure (input gradient vs SHAP)",
             "B": "B. Representation (PCA round trip)",
             "C": "C. Model family (two families, one space)"}
    lines = []
    for key in ("A", "B", "C"):
        cells = []
        for ds in ("german", "taiwan"):
            node = controls.get(ds, {}).get(key, {})
            if key in ("A", "B"):
                sub = [v for v in node.values() if isinstance(v, dict) and "rho" in v]
                cells.append(f"{np.mean([s_['rho'] for s_ in sub]):.3f}" if sub else "--")
            else:
                r, lo, hi = node.get("rho"), node.get("lo"), node.get("hi")
                cells.append(f"{r:.3f} [{lo:.3f}, {hi:.3f}]" if r is not None else "--")
        lines.append(f"{names[key]} & {cells[0]} & {cells[1]} \\\\")

    if nulls:
        lines.append("\\midrule")
        lines.append("\\multicolumn{3}{l}{\\emph{D. Is C the models, or the map "
                     "they share?}} \\\\")
        for label, key, ci in (
            ("\\quad C's two attributions, before the map", "observed_component_space", False),
            ("\\quad Permutation null, after the map", "permutation_null", True),
        ):
            cells = []
            for ds in ("german", "taiwan"):
                node = nulls.get(ds, {}).get(key)
                if not node:
                    cells.append("--")
                elif ci:
                    cells.append(f"{node['rho']:.3f} [{node['lo']:.3f}, {node['hi']:.3f}]")
                else:
                    cells.append(f"{node['rho']:.3f}")
            lines.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
        frac = []
        for ds in ("german", "taiwan"):
            v = nulls.get(ds, {}).get("null_exceeds_observed_fraction")
            frac.append(f"{100 * v:.2f}\\%" if v is not None else "--")
        lines.append(f"\\quad Null draws reaching C & {frac[0]} & {frac[1]} \\\\")

    out.write_text(tex_table(
        lines, "Control & German Credit & Taiwan default",
        "The four controls. A, B and C each separate one thing from the model "
        "family; D asks whether C measured the models or the loading matrix "
        "both its attributions pass through, by comparing them before that "
        "matrix is applied and against a null that reassigns each model's "
        "component magnitudes at random. C is interpretable only where it "
        "clears that null. Spearman rank correlation throughout.",
        "tab:controls", "lcc"))
    print(f"  wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classical-dir", type=Path, default=Path("results/credit"))
    p.add_argument("--out-dir", type=Path, default=Path("report-credit/figures"))
    p.add_argument("--tex-dir", type=Path, default=Path("report-credit/tables"))
    p.add_argument("--analysis", type=Path, default=Path("results/credit/analysis.json"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tex_dir.mkdir(parents=True, exist_ok=True)

    controls, nulls = {}, {}
    for target, files in ((controls, (("german", "attribution_controls.json"),
                                      ("taiwan", "attribution_controls_taiwan.json"))),
                          (nulls, (("german", "backprojection_null.json"),
                                   ("taiwan", "backprojection_null_taiwan.json")))):
        for ds, fn in files:
            f = args.classical_dir / fn
            if f.exists():
                target[ds] = json.loads(f.read_text())

    print("Figures")
    if controls:
        fig_controls(controls, args.out_dir / "02_attribution_controls.png")

    for ds, bad in (("german", 0.3000), ("taiwan", 0.2212)):
        rows = load_aggregates(args.classical_dir, ds)
        if rows:
            fig_weighting(rows, args.out_dir / f"04_class_weighting_{ds}.png", bad, ds)

    print("Tables")
    mcn = {}
    if args.analysis.exists():
        mcn = json.loads(args.analysis.read_text()).get("german", {}).get("mcnemar_holm", {})
    german = load_aggregates(args.classical_dir, "german")
    if german:
        table_discrimination(german, mcn, args.tex_dir / "tab_discrimination.tex")
    if controls:
        table_controls(controls, nulls, args.tex_dir / "tab_controls.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
