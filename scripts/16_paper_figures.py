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
    """What each change costs, which is the sentence the paper actually makes.

    The first version of this plotted agreement, while every claim in the text
    is about agreement lost, so a reader had to invert it in their head. It
    also drew a bar for Taiwan's control C that the next paragraph withdraws.
    Both are fixed here: the axis is cost, and the withdrawn value is marked
    as not measurable rather than drawn and then disowned.
    """
    def mean_rho(node: dict) -> float:
        sub = [v["rho"] for v in node.values() if isinstance(v, dict) and "rho" in v]
        return float(np.mean(sub)) if sub else np.nan

    rows = [
        ("the attribution measure", "A", "SHAP against an input gradient"),
        ("the model family", "C", "two families, one representation"),
        ("the representation", "B", "one family, two representations"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    h = 0.34
    for i, (label, key, _) in enumerate(rows):
        for j, (ds, col, name) in enumerate((("german", C_CLASSICAL, "German Credit"),
                                             ("taiwan", C_WARN, "Taiwan default"))):
            c = controls.get(ds, {})
            y = i + (0.5 - j) * h
            # Taiwan's control C does not clear its own null, so there is no
            # value to draw. Saying so on the axis beats drawing a bar the
            # text takes back.
            if key == "C" and ds == "taiwan":
                ax.text(0.012, y, "not measurable on this benchmark (Control D)",
                        va="center", fontsize=9, color="#8a6d3b", style="italic")
                continue
            node = c.get(key, {})
            rho = node.get("rho") if key == "C" else mean_rho(node)
            if rho is None or not np.isfinite(rho):
                continue
            cost = 1.0 - rho
            ax.barh(y, cost, height=h, color=col, alpha=0.85,
                    edgecolor="white", linewidth=0.6,
                    label=name if i == 0 else None)
            ax.text(cost + 0.012, y, f"{cost:.3f}", va="center", fontsize=10,
                    fontweight="bold", color="#333333")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{l}\n({d})" for l, _, d in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Agreement lost when that one thing changes  "
                  "(1 \u2212 Spearman rank correlation)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # The paper's claim is the ratio, so draw it rather than leave it to be
    # computed from two bar lengths.
    g = controls.get("german", {})
    if g.get("C", {}).get("rho") is not None:
        fam, rep = 1 - g["C"]["rho"], 1 - mean_rho(g.get("B", {}))
        if np.isfinite(rep) and fam > 0:
            ax.annotate(f"{rep / fam:.0f} times the cost of changing the model",
                        xy=(rep, 2 - h / 2), xytext=(0.42, 1.52),
                        fontsize=10, color="#333333",
                        arrowprops=dict(arrowstyle="->", color="#777777", lw=1.1))
    ax.legend(loc="upper right", fontsize=9, frameon=False)
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
