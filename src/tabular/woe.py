"""Weight-of-evidence coding for categorical credit attributes.

Implemented as a scikit-learn transformer so that it can only ever be fitted
inside a pipeline on a fold's training half. Fitting WoE on the full dataset
leaks the label into the features and is the most common evaluation error in
the credit-scoring literature, so the fold-safe form is the only one provided.

For a category *i*:

    WoE_i = ln( (good_i / good_total) / (bad_i / bad_total) )

with ``y = 1`` the adverse (bad) outcome, matching ``src.tabular.datasets``.
Laplace smoothing keeps the logarithm finite for categories that contain only
one class, and categories unseen at fit time map to 0.0, i.e. "no evidence".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Replace each categorical column with its weight of evidence.

    Parameters
    ----------
    columns:
        Columns to encode. ``None`` encodes every non-numeric column seen at fit
        time.
    smoothing:
        Laplace count added to both the good and the bad tally of each category.
    """

    def __init__(self, columns: list[str] | None = None, smoothing: float = 0.5):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WoEEncoder":
        X = pd.DataFrame(X).copy()
        y = np.asarray(y).ravel()
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if not np.isin(np.unique(y), [0, 1]).all():
            raise ValueError("WoEEncoder expects a binary 0/1 target")

        cols = self.columns
        if cols is None:
            cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.columns_ = list(cols)

        n_bad = float(y.sum())
        n_good = float(len(y) - n_bad)
        if n_bad == 0 or n_good == 0:
            raise ValueError("WoEEncoder needs both classes present in the training fold")

        s = float(self.smoothing)
        self.maps_: dict[str, dict[object, float]] = {}
        for c in self.columns_:
            table: dict[object, float] = {}
            for value, idx in X.groupby(c, observed=True).groups.items():
                yi = y[X.index.get_indexer(idx)]
                bad = float(yi.sum()) + s
                good = float(len(yi) - yi.sum()) + s
                table[value] = float(np.log((good / (n_good + 2 * s)) / (bad / (n_bad + 2 * s))))
            self.maps_[c] = table

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "maps_")
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            if c not in X.columns:
                raise ValueError(f"column {c!r} missing at transform time")
            # Unseen categories carry no evidence either way.
            X[c] = X[c].map(self.maps_[c]).astype(float).fillna(0.0)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)
