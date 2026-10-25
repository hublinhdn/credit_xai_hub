"""Principal component reduction, and carrying an attribution back from it.

The attribution analysis needs a reduction whose inverse is explicit. PCA is
linear, so an importance over components maps back to the original features
through the loading matrix, and the map is the same for every model fitted in
that space. That property is what makes a like-for-like comparison between
model families possible at all, and it is also what has to be controlled for,
because two attributions that share a map can agree because of the map.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted


class PCAProjector(BaseEstimator, TransformerMixin):
    """Fit ``n_components`` principal components and project attributions back."""

    def __init__(self, n_components: int = 8, seed: int = 0):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y=None) -> "PCAProjector":
        X = np.asarray(X, dtype=float)
        k = min(self.n_components, X.shape[1], X.shape[0])
        self.pca_ = PCA(n_components=k, random_state=self.seed).fit(X)
        self.n_features_in_ = X.shape[1]
        self.n_components_ = k
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "pca_")
        return self.pca_.transform(np.asarray(X, dtype=float))

    def project_attribution_back(self, attribution: np.ndarray) -> np.ndarray:
        """Carry a per-component attribution back to the original features.

        With components ``z = (x - mean) W^T``, a change in component j moves
        with ``W[j]``, so a magnitude ``a_j`` on component j contributes
        ``|a_j| * |W[j, i]|`` to feature i. Magnitudes are used because the
        quantity being carried back is an importance, which has no sign.
        """
        check_is_fitted(self, "pca_")
        a = np.abs(np.asarray(attribution, dtype=float)).ravel()
        if len(a) != self.pca_.components_.shape[0]:
            raise ValueError(
                f"attribution has {len(a)} entries, expected {self.pca_.components_.shape[0]}"
            )
        return a @ np.abs(self.pca_.components_)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        check_is_fitted(self, "pca_")
        return self.pca_.explained_variance_ratio_
