#!/usr/bin/env python3
"""Check the manuscript's numbers against the stored results.

The paper claims no number in it is typed by hand. That claim is only worth
something if something checks it, because prose numbers do not update when an
experiment is re-run. Four stale Taiwan values survived a rewrite before this
script existed.

    python scripts/17_check_manuscript.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX = REPO_ROOT / "report-credit" / "main.tex"
TABLES = REPO_ROOT / "report-credit" / "tables"
RESULTS = REPO_ROOT / "results" / "credit"


def mean_rho(node: dict) -> float:
    vals = [v["rho"] for v in node.values() if isinstance(v, dict) and "rho" in v]
    return sum(vals) / len(vals) if vals else float("nan")


def collect_claims() -> list[tuple[str, float]]:
    claims: list[tuple[str, float]] = []
    for ds, fn in (("german", "attribution_controls.json"),
                   ("taiwan", "attribution_controls_taiwan.json")):
        f = RESULTS / fn
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        for key in ("A", "B"):
            if key in c:
                claims.append((f"{ds} control {key} mean", round(mean_rho(c[key]), 3)))
        if c.get("C", {}).get("rho") is not None:
            claims.append((f"{ds} control C", round(c["C"]["rho"], 3)))
            for b in ("lo", "hi"):
                claims.append((f"{ds} control C {b}", round(c["C"][b], 3)))
        preb = c.get("B_precondition") or {}
        for fam, v in preb.items():
            claims.append((f"{ds} control B AUC raw, {fam}", round(v["auc_raw"], 4)))
            claims.append((f"{ds} control B AUC components, {fam}", round(v["auc_pca"], 4)))

        pre = c.get("C_precondition")
        if pre:
            claims.append((f"{ds} control C AUC linear", round(pre["auc_lr"], 4)))
            claims.append((f"{ds} control C AUC neural", round(pre["auc_mlp"], 4)))

    # The baseline Control C is read against, measured where the control is.
    f = RESULTS / "attribution_controls_replication.json"
    if f.exists():
        base = json.loads(f.read_text()).get("C_baseline") or {}
        for fam, label in (("lr", "linear"), ("mlp", "neural")):
            if fam in base:
                claims.append((f"german control C baseline, {label}",
                               round(base[fam]["rho"], 3)))

    for ds, fn in (("german", "backprojection_null.json"),
                   ("taiwan", "backprojection_null_taiwan.json")):
        f = RESULTS / fn
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        claims.append((f"{ds} control D component space",
                       round(c["observed_component_space"]["rho"], 3)))
        claims.append((f"{ds} control D null", round(c["permutation_null"]["rho"], 3)))
        for b in ("lo", "hi"):
            claims.append((f"{ds} control D null {b}", round(c["permutation_null"][b], 3)))
        claims.append((f"{ds} control D null fraction",
                       round(100 * c["null_exceeds_observed_fraction"], 2)))

    # Only the values the manuscript actually quotes are checked. Requiring
    # every stored number to appear would force the paper to print all of
    # them; the point of the check is that a quoted number is not stale.
    f = RESULTS / "analysis.json"
    if f.exists():
        a = json.loads(f.read_text())
        for ds, node in a.items():
            att = node.get("attribution") or {}
            within = att.get("within") or {}
            if within:
                claims.append((f"{ds} lowest within-model", round(min(within.values()), 3)))
                claims.append((f"{ds} highest within-model", round(max(within.values()), 3)))
            if att.get("mean_between") is not None:
                claims.append((f"{ds} between-family, restricted",
                               round(att["mean_between"], 3)))
            lt = (att.get("between_unrestricted") or {}).get("linear vs tree")
            if lt:
                claims.append((f"{ds} linear vs tree, unrestricted", round(lt["rho"], 3)))
            rhos = [v["attribution_rho"] for v in (node.get("protected") or {}).values()
                    if v.get("attribution_rho") is not None]
            if rhos:
                claims.append((f"{ds} protected, lowest rho", round(min(rhos), 3)))
                claims.append((f"{ds} protected, highest rho", round(max(rhos), 3)))
    f = RESULTS / "analysis_woe.json"
    g = RESULTS / "analysis.json"
    if f.exists() and g.exists():
        woe, mix = json.loads(f.read_text()), json.loads(g.read_text())
        for ds in woe:
            a = ((mix.get(ds) or {}).get("attribution") or {}).get("between_unrestricted") or {}
            b = (woe[ds].get("attribution") or {}).get("between_unrestricted") or {}
            for pair in a.keys() & b.keys():
                claims.append((f"{ds} one encoding, {pair}", round(b[pair]["rho"], 3)))
                claims.append((f"{ds} encoding change, {pair}",
                               round(b[pair]["rho"] - a[pair]["rho"], 3)))

    return claims


def check_freshness() -> list[str]:
    """Stored results that predate the artefacts they were computed from.

    The value checks below confirm a stored number appears in the manuscript.
    They cannot tell that the stored number is itself out of date, which is how
    a stale correlation result reached the paper: the file was written two
    hours before the attributions it summarises were regenerated.
    """
    stale: list[str] = []
    # Every summary at the top of the results tree, found rather than listed:
    # a named list would not have covered the one file that went stale next.
    # The controls and the null refit from the raw data instead of summarising
    # the per-fold tree, so their timestamps say nothing and they are excluded.
    derived = [f for f in sorted(RESULTS.glob("*.json"))
               if not f.name.startswith(("attribution_controls", "backprojection_null"))]
    sources = sorted(RESULTS.rglob("attribution.csv")) + sorted(RESULTS.rglob("aggregated.json"))
    if not sources or not derived:
        return stale
    newest = max(p.stat().st_mtime for p in sources)
    for f in derived:
        if f.stat().st_mtime < newest:
            stale.append(f"{f.name} is older than the artefacts it summarises")
    return stale


def main() -> int:
    global TEX, TABLES
    # The manuscript is not part of the released repository, so the path is an
    # argument there rather than a constant.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", type=Path, default=TEX,
                    help="the manuscript to check (default: the one in this repo)")
    ap.add_argument("--tables", type=Path, default=TABLES,
                    help="directory of generated LaTeX tables")
    args = ap.parse_args()
    TEX, TABLES = args.tex, args.tables
    if not TEX.exists():
        print(f"{TEX} not found")
        return 1
    # Generated tables are \input into the manuscript, so a number that moved
    # out of the prose and into a table is still in the paper.
    stale = check_freshness()
    for note in stale:
        print(f"  STALE {note}")
    tex = TEX.read_text() + "".join(f.read_text() for f in sorted(TABLES.glob("*.tex")))
    for name in ("tab_discrimination", "tab_controls"):
        if f"\\input{{tables/{name}}}" not in tex:
            print(f"warning: {name} is generated but not included by the manuscript")

    # Compare numerically rather than as text: the manuscript writes 0.15
    # where the stored value rounds to 0.150, and both are the same claim.
    found_numbers = [float(t) for t in re.findall(r"-?\d+\.\d+", tex)]

    missing = 0
    print(f"{'claim':<40} {'value':>9}  present in manuscript?")
    for label, value in collect_claims():
        found = any(abs(f - value) < 5e-5 for f in found_numbers)
        print(f"  {label:<38} {value:9.4f}  {'yes' if found else 'NOT FOUND'}")
        if not found:
            missing += 1

    print()
    if stale:
        print(f"{len(stale)} stored result(s) are older than their inputs; "
              f"regenerate before trusting the values above")
        return 1
    if missing:
        print(f"{missing} stored value(s) do not appear in the manuscript. Either the "
              f"prose is stale or the claim was deliberately dropped; check each.")
        return 1
    print("every stored value appears in the manuscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
