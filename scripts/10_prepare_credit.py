#!/usr/bin/env python3
"""Phase 7, step 1: fetch and verify the two public credit-risk benchmarks.

Downloads the archives from the UCI repository, checks them against recorded
SHA-256 digests, extracts them, and prints the dataset characteristics so they
can be compared against the published figures before any model is fitted.

    python scripts/10_prepare_credit.py --download
    python scripts/10_prepare_credit.py --verify-only

The raw data is deliberately NOT committed: it is public, it is reproducible
from these URLs, and keeping it out of the repository keeps the reproducibility
bundle small and the provenance explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tabular.datasets import DEFAULT_RAW, excel_engine, load_german, load_taiwan  # noqa: E402

SOURCES = {
    "german.zip": {
        "url": "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
        "sha256": "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b",
        "member": "german.data",
        "member_sha256": "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871",
    },
    "taiwan.zip": {
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "sha256": "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602",
        "member": "default of credit card clients.xls",
        "member_sha256": "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933",
    },
}

# What the raw files must contain. Counted directly from the files, not taken
# from the dataset documentation.
EXPECTED = {
    "german": {"n": 1000, "features": 20, "n_bad": 300, "bad_rate": 0.3000},
    "taiwan": {"n": 30000, "features": 23, "n_bad": 6636, "bad_rate": 0.2212},
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, spec in SOURCES.items():
        dest = raw_dir / fname
        if dest.exists() and sha256_of(dest) == spec["sha256"]:
            print(f"  {fname}: already present, digest matches")
        else:
            print(f"  {fname}: downloading from {spec['url']}")
            urllib.request.urlopen(spec["url"])  # fail fast on a dead endpoint
            urllib.request.urlretrieve(spec["url"], dest)
            got = sha256_of(dest)
            if got != spec["sha256"]:
                raise SystemExit(
                    f"digest mismatch for {fname}\n  expected {spec['sha256']}\n  got      {got}"
                )
            print(f"  {fname}: downloaded, digest matches")

        member = spec["member"]
        out = raw_dir / member
        if not out.exists():
            with zipfile.ZipFile(dest) as z:
                z.extract(member, raw_dir)
            print(f"  extracted {member}")
        got = sha256_of(out)
        if got != spec["member_sha256"]:
            raise SystemExit(f"digest mismatch for extracted {member}: got {got}")

    # Excel reading needs xlrd; emit a CSV once so the rest of the pipeline and
    # the remote machine do not need that dependency.
    xls = raw_dir / SOURCES["taiwan.zip"]["member"]
    csv = raw_dir / "taiwan_default.csv"
    if not csv.exists():
        try:
            import pandas as pd

            pd.read_excel(xls, header=1, engine=excel_engine()).to_csv(csv, index=False)
            print(f"  wrote {csv.name} for xlrd-free loading")
        except ImportError:
            print("  note: xlrd missing, skipping CSV export (loader will read the .xls)")


def verify(raw_dir: Path) -> int:
    failures = 0
    for name, loader in (("german", load_german), ("taiwan", load_taiwan)):
        exp = EXPECTED[name]
        try:
            ds = loader(raw_dir)
        except FileNotFoundError as e:
            print(f"  {name}: MISSING ({e})")
            failures += 1
            continue

        print(f"  {ds.summary()}")
        checks = [
            ("n", len(ds), exp["n"]),
            ("features", ds.X.shape[1], exp["features"]),
            ("n_bad", int(ds.y.sum()), exp["n_bad"]),
        ]
        for label, got, want in checks:
            if got != want:
                print(f"    FAIL {label}: got {got}, expected {want}")
                failures += 1
        if abs(ds.bad_rate - exp["bad_rate"]) > 5e-4:
            print(f"    FAIL bad_rate: got {ds.bad_rate:.4f}, expected {exp['bad_rate']:.4f}")
            failures += 1
        if ds.protected:
            print(f"    protected attributes present: {', '.join(ds.protected)}")
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--download", action="store_true", help="fetch and extract the archives")
    p.add_argument("--verify-only", action="store_true", help="only check what is already on disk")
    args = p.parse_args()

    if args.download and not args.verify_only:
        print("Downloading raw data")
        download(args.raw_dir)
        print()

    print("Verifying dataset characteristics")
    failures = verify(args.raw_dir)
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("All dataset checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
