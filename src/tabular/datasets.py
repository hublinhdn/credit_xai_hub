"""Loaders for the two public credit-risk benchmarks.

Label convention (both datasets): ``y = 1`` is the adverse outcome (bad risk /
default), ``y = 0`` is the good outcome. This matches the Taiwan file and
inverts the German file, whose native coding is ``1 = good, 2 = bad``.

Verified against the raw files on 2026-10-23:
    German Credit  1,000 rows, 20 features,    300 bad  (rate 0.3000)
    Taiwan default 30,000 rows, 23 features, 6,636 bad  (rate 0.2212)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RAW = Path("data/raw/credit")

# --- German Credit (UCI id 144) --------------------------------------------
# Attribute order follows german.doc. Numeric attributes are 2, 5, 8, 11, 13,
# 16 and 18 (1-based); the remaining thirteen are qualitative A-codes.
GERMAN_COLUMNS = [
    "checking_status",          # A11..A14
    "duration_months",          # numeric
    "credit_history",           # A30..A34
    "purpose",                  # A40..A410
    "credit_amount",            # numeric
    "savings_status",           # A61..A65
    "employment_since",         # A71..A75
    "installment_rate",         # numeric
    "personal_status_sex",      # A91..A95  -- encodes SEX, see PROTECTED
    "other_parties",            # A101..A103
    "residence_since",          # numeric
    "property_magnitude",       # A121..A124
    "age_years",                # numeric
    "other_payment_plans",      # A141..A143
    "housing",                  # A151..A153
    "existing_credits",         # numeric
    "job",                      # A171..A174
    "num_dependents",           # numeric
    "telephone",                # A191..A192
    "foreign_worker",           # A201..A202 -- national origin, see PROTECTED
]

GERMAN_NUMERIC = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "num_dependents",
]

# Attributes that fair-lending rules treat as protected characteristics.
# They are part of the standard benchmark and are therefore kept by default so
# that results stay comparable with Lessmann et al. (2015); scripts expose a
# switch to drop them and report the comparison both ways.
GERMAN_PROTECTED = ["personal_status_sex", "foreign_worker", "age_years"]

# Misclassification costs shipped with the dataset (german.doc section 8):
# predicting "good" for an actual "bad" costs 5, the reverse costs 1.
GERMAN_COST_FN = 5.0
GERMAN_COST_FP = 1.0

# --- Taiwan default of credit card clients (UCI id 350) --------------------
TAIWAN_CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
TAIWAN_PROTECTED = ["SEX", "MARRIAGE", "AGE"]
TAIWAN_TARGET = "default payment next month"


@dataclass
class CreditDataset:
    """A loaded credit-risk dataset in its raw (un-encoded) form."""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    numeric: list[str]
    categorical: list[str]
    protected: list[str]
    cost_fn: float = 1.0
    cost_fp: float = 1.0
    notes: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def bad_rate(self) -> float:
        return float(self.y.mean())

    def summary(self) -> str:
        return (
            f"{self.name}: n={len(self):,} features={self.X.shape[1]} "
            f"(numeric={len(self.numeric)}, categorical={len(self.categorical)}) "
            f"bad_rate={self.bad_rate:.4f} cost_fn:cost_fp={self.cost_fn:g}:{self.cost_fp:g}"
        )

    def without_protected(self) -> "CreditDataset":
        """Return a copy with the protected attributes dropped."""
        keep = [c for c in self.X.columns if c not in self.protected]
        return CreditDataset(
            name=f"{self.name}_noprot",
            X=self.X[keep].copy(),
            y=self.y.copy(),
            numeric=[c for c in self.numeric if c in keep],
            categorical=[c for c in self.categorical if c in keep],
            protected=[],
            cost_fn=self.cost_fn,
            cost_fp=self.cost_fp,
            notes={**self.notes, "dropped_protected": list(self.protected)},
        )


def load_german(raw_dir: Path = DEFAULT_RAW) -> CreditDataset:
    """Load UCI Statlog German Credit from ``german.data``."""
    path = Path(raw_dir) / "german.data"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/10_prepare_credit.py --download"
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS + ["label"])
    if len(df) != 1000 or df.shape[1] != 21:
        raise ValueError(f"unexpected german.data shape {df.shape}, expected (1000, 21)")

    y = (df["label"].to_numpy() == 2).astype(np.int64)  # 2 = bad -> positive class
    X = df.drop(columns=["label"])
    categorical = [c for c in GERMAN_COLUMNS if c not in GERMAN_NUMERIC]
    for c in GERMAN_NUMERIC:
        X[c] = pd.to_numeric(X[c])

    return CreditDataset(
        name="german",
        X=X,
        y=y,
        numeric=list(GERMAN_NUMERIC),
        categorical=categorical,
        protected=list(GERMAN_PROTECTED),
        cost_fn=GERMAN_COST_FN,
        cost_fp=GERMAN_COST_FP,
        notes={"source": "UCI id 144", "label_coding": "1=good,2=bad -> y=1 is bad"},
    )


def load_taiwan(raw_dir: Path = DEFAULT_RAW) -> CreditDataset:
    """Load UCI Default of Credit Card Clients.

    Reads the ``.xls`` when ``xlrd`` is available and falls back to a ``.csv``
    produced by ``scripts/10_prepare_credit.py`` otherwise, so the pipeline does
    not depend on a binary Excel reader being installed.
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "taiwan_default.csv"
    xls_path = raw_dir / "default of credit card clients.xls"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xls_path.exists():
        df = pd.read_excel(xls_path, header=1)
    else:
        raise FileNotFoundError(
            f"neither {csv_path} nor {xls_path} found. "
            "Run: python scripts/10_prepare_credit.py --download"
        )

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if TAIWAN_TARGET not in df.columns:
        raise ValueError(f"target column {TAIWAN_TARGET!r} missing; got {list(df.columns)[:5]}...")

    y = df[TAIWAN_TARGET].to_numpy().astype(np.int64)
    X = df.drop(columns=[TAIWAN_TARGET])
    if X.shape[1] != 23:
        raise ValueError(f"expected 23 features, got {X.shape[1]}")

    categorical = [c for c in TAIWAN_CATEGORICAL if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    # Categorical codes are integers in the file; make them explicit strings so
    # the WoE encoder and one-hot encoder do not treat them as ordered.
    for c in categorical:
        X[c] = X[c].astype(str)

    return CreditDataset(
        name="taiwan",
        X=X,
        y=y,
        numeric=numeric,
        categorical=categorical,
        protected=[c for c in TAIWAN_PROTECTED if c in X.columns],
        cost_fn=1.0,
        cost_fp=1.0,
        notes={"source": "UCI id 350", "label_coding": "1 = default next month"},
    )


LOADERS = {"german": load_german, "taiwan": load_taiwan}


def load(name: str, raw_dir: Path = DEFAULT_RAW) -> CreditDataset:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(LOADERS)}")
    return LOADERS[name](raw_dir)
