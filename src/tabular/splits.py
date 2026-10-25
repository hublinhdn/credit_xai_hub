"""N x 2 cross-validation, the protocol used by the reference benchmark.

Lessmann et al. (2015), following Dietterich (1998), split a dataset in half,
train on one half and evaluate on the other, swap the roles, and repeat the
whole procedure N times. N is chosen by dataset size: N = 10 for the small
sets (including German Credit) and N = 5 for the large ones.

Each fold is drawn with stratification so that the bad rate of both halves
matches the full dataset, which matters here because both benchmarks are
imbalanced.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Repetition counts used by the reference study, keyed by our dataset names.
N_REPEATS = {"german": 10, "taiwan": 5}


def nx2_splits(
    y: np.ndarray, n_repeats: int, seed: int = 0
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat, half, train_idx, test_idx)`` for N x 2 cross-validation.

    ``StratifiedKFold(n_splits=2)`` already emits both orientations of the same
    halving: its second split is the first one with train and test exchanged.
    The indices are therefore yielded as-is. Swapping them again in the caller
    would cancel that out and silently evaluate each repeat's first fold twice,
    halving the protocol.

    A full pass yields ``2 * n_repeats`` fits in which every row is a test case
    exactly ``n_repeats`` times.
    """
    y = np.asarray(y).ravel()
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + r)
        for half, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield r, half, train_idx, test_idx


def n_fits(n_repeats: int) -> int:
    return 2 * n_repeats
