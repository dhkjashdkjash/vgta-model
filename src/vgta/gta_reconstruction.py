from __future__ import annotations

import numpy as np


def merge_observed_and_reconstructed(observed: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    if observed.shape != reconstructed.shape:
        raise ValueError("observed and reconstructed arrays must share a shape")
    return np.where(np.isfinite(observed), observed, reconstructed)


def run_gta(args) -> int:
    if args.input is None:
        raise ValueError("GTA requires --input pointing to prepared sequences")
    from vgta.pipeline import run_gta_stage

    return int(run_gta_stage(args) or 0)
