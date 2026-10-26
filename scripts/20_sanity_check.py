#!/usr/bin/env python3
"""Static checks for the wiring that breaks when something moves.

Every failure this catches has already happened in this project at least
once, which is why each check exists rather than being a plausible idea:

  imports        a symbol moved module and a caller still named the old one
  flags          a command in a README or Makefile passed a flag the script
                 does not define, or omitted one it requires
  make           a Makefile rule depended on a file nothing produces any more
  latex          main.tex pulled in a file that is no longer at that path
  release        the release tree drifted from the code that produced the
                 published results
  contents       the release is missing a file it documents, or carries a
                 generated table the manuscript no longer uses

None of this runs an experiment or needs data. It is meant to be cheap enough
to run before every push.

    python scripts/20_sanity_check.py
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "release" / "credit_xai_hub"
REPORT = REPO / "report-credit"

failures: list[str] = []
checks = 0


def fail(check: str, detail: str) -> None:
    failures.append(f"{check}: {detail}")


def module_path(dotted: str) -> Path | None:
    p = REPO / Path(*dotted.split(".")).with_suffix(".py")
    return p if p.exists() else None


def exported_names(path: Path) -> set[str]:
    """Top-level names a module defines, including its own imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def py_files() -> list[Path]:
    return sorted(
        [p for p in (REPO / "scripts").glob("*.py")]
        + [p for p in (REPO / "src").rglob("*.py")]
    )


def check_imports() -> None:
    """Every `from src.x import y` names something src/x actually defines."""
    global checks
    for path in py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            fail("imports", f"{path.relative_to(REPO)} does not parse: {e}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("src."):
                continue
            checks += 1
            target = module_path(node.module)
            if target is None:
                fail("imports", f"{path.relative_to(REPO)} imports from "
                                f"{node.module}, which does not exist")
                continue
            available = exported_names(target)
            for alias in node.names:
                if alias.name != "*" and alias.name not in available:
                    fail("imports", f"{path.relative_to(REPO)} imports "
                                    f"{alias.name!r} from {node.module}, "
                                    f"which does not define it")


def declared_flags(script: Path) -> set[str]:
    src = script.read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))


def check_documented_flags() -> None:
    """Commands written in docs and Makefiles use flags that exist."""
    global checks
    docs = [RELEASE / "README.md", REPORT / "TODO.md", REPORT / "Makefile"]
    docs += sorted(REPO.glob("*.md"))
    pattern = re.compile(r"scripts/(\d+[a-z]*_[a-z_0-9]+)\.py((?:\s+--[a-z0-9-]+(?:[= ][^\s\\]+)?)*)")
    for doc in docs:
        if not doc.exists():
            continue
        for m in pattern.finditer(doc.read_text(encoding="utf-8")):
            script = REPO / "scripts" / f"{m.group(1)}.py"
            if not script.exists():
                fail("flags", f"{doc.name} names scripts/{m.group(1)}.py, "
                              f"which does not exist")
                continue
            declared = declared_flags(script)
            for flag in re.findall(r"--[a-z0-9-]+", m.group(2) or ""):
                checks += 1
                if flag not in declared:
                    fail("flags", f"{doc.name} runs {script.name} with {flag}, "
                                  f"which it does not define")


def check_make_prerequisites() -> None:
    """Every file a Makefile rule needs either exists or has a rule."""
    global checks
    mk = REPORT / "Makefile"
    if not mk.exists():
        return
    text = mk.read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z0-9_./$()-]+):", text, re.M))
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_./$()-]+):\s*(.*)$", line)
        if not m or line.startswith("\t"):
            continue
        for prereq in m.group(2).split():
            if "$" in prereq or prereq in targets or prereq.startswith("."):
                continue
            if "/" not in prereq and "." not in prereq:
                continue  # a phony name
            checks += 1
            if not (REPORT / prereq).exists():
                fail("make", f"rule {m.group(1)!r} needs {prereq}, which is "
                             f"absent and has no rule")


def check_latex_inputs() -> None:
    """Every \\input and \\includegraphics in the manuscript resolves."""
    global checks
    tex = REPORT / "main.tex"
    if not tex.exists():
        return
    src = tex.read_text(encoding="utf-8")
    for ref in re.findall(r"\\input\{([^}]+)\}", src):
        checks += 1
        if not (REPORT / ref).exists() and not (REPORT / f"{ref}.tex").exists():
            fail("latex", f"main.tex inputs {ref}, which is not on disk")
    for ref in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", src):
        checks += 1
        if not (REPORT / ref).exists():
            fail("latex", f"main.tex includes {ref}, which is not on disk")


# One file is deliberately different in the two trees: the top-level package
# docstring names this repository's several studies, while the release ships
# only two of those packages and says so. Listed here so the difference is
# declared rather than silently tolerated.
PARITY_EXCEPTIONS = {Path("src/__init__.py")}


def check_release_parity() -> None:
    """The release tree still carries the code that produced the results."""
    global checks
    if not RELEASE.exists():
        return
    for rel in sorted(list((RELEASE / "src").rglob("*.py"))
                      + list((RELEASE / "scripts").glob("*.py"))):
        inner = rel.relative_to(RELEASE)
        if inner in PARITY_EXCEPTIONS:
            continue
        dev = REPO / inner
        checks += 1
        if not dev.exists():
            fail("release", f"{inner} is in the release and not in the repo")
        elif hashlib.sha256(dev.read_bytes()).digest() != \
                hashlib.sha256(rel.read_bytes()).digest():
            fail("release", f"{inner} differs between the repo and the release")


def check_release_contents() -> None:
    """The release holds what it documents, and no table the paper dropped.

    Parity compares the files that are in both trees, so a file missing from
    the release passes it silently. That is how the release came to ship a
    pre-merge table, to lack the table of the paper's Table 5, and to omit the
    manuscript checker the paper's availability statement promises.
    """
    global checks
    if not RELEASE.exists():
        return
    readme = RELEASE / "README.md"
    if readme.exists():
        for name in sorted(set(re.findall(r"scripts/(\d+[a-z]*_[a-z_0-9]+)\.py",
                                          readme.read_text(encoding="utf-8")))):
            checks += 1
            if not (RELEASE / "scripts" / f"{name}.py").exists():
                fail("contents", f"the release README runs scripts/{name}.py, "
                                 f"which the release does not ship")
    tex = REPORT / "main.tex"
    if tex.exists() and (RELEASE / "tables").exists():
        wanted = {f"{m}.tex" for m in
                  re.findall(r"\\input\{tables/([^}]+)\}", tex.read_text(encoding="utf-8"))}
        shipped = {p.name for p in (RELEASE / "tables").glob("*.tex")}
        for name in sorted(wanted - shipped):
            checks += 1
            fail("contents", f"the manuscript inputs tables/{name}, which the "
                             f"release does not ship")
        for name in sorted(shipped - wanted):
            checks += 1
            fail("contents", f"the release ships tables/{name}, which the "
                             f"manuscript does not use")


def main() -> int:
    for fn in (check_imports, check_documented_flags, check_make_prerequisites,
               check_latex_inputs, check_release_parity, check_release_contents):
        fn()

    name = {"imports": "symbol imports", "flags": "documented flags",
            "make": "make prerequisites", "latex": "latex inputs",
            "release": "release parity", "contents": "release contents"}
    print(f"{checks} assertions over six areas\n")
    if not failures:
        for k in name.values():
            print(f"  ok   {k}")
        print("\nnothing to fix")
        return 0
    for f in failures:
        print(f"  FAIL {f}")
    print(f"\n{len(failures)} problem(s); fix before running or pushing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
