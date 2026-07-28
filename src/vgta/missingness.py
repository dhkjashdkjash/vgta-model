from __future__ import annotations

import numpy as np


def apply_mcar_mask(observed: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    observed = np.asarray(observed, dtype=bool)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    available = np.flatnonzero(observed)
    count = int(np.rint(len(available) * fraction))
    result = np.zeros_like(observed)
    if count:
        selected = np.random.default_rng(seed).choice(available, size=count, replace=False)
        result[selected] = True
    return result


def apply_block_mask(observed: np.ndarray, *, length: int, seed: int) -> np.ndarray:
    observed = np.asarray(observed, dtype=bool)
    if length < 1:
        raise ValueError("length must be positive")
    starts = [
        start
        for start in range(max(0, observed.size - length + 1))
        if observed[start : start + length].all()
    ]
    result = np.zeros_like(observed)
    if starts:
        start = int(np.random.default_rng(seed).choice(starts))
        result[start : start + length] = True
    return result
