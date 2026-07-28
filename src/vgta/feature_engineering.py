from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


FEATURE_NAMES = (
    "Oa03", "Oa04", "Oa05", "Oa06", "Oa08", "Oa09", "Oa10", "Oa11",
    "OTCI", "OTCI2", "OTCI3", "sqrt_OTCI", "log_OTCI", "NDRI", "GBNR",
    "BSR", "FLLin", "TBI", "GIFAPAR", "IWV",
)
SOURCE_NAMES = FEATURE_NAMES[:8] + FEATURE_NAMES[-2:]


@dataclass(frozen=True)
class FeatureResult:
    values: np.ndarray
    valid: np.ndarray


def _safe_div(numerator: np.ndarray, denominator: np.ndarray, epsilon: float) -> np.ndarray:
    return numerator / (denominator + epsilon)


def compute_features(
    arrays: Mapping[str, np.ndarray], epsilon: float = 1e-8
) -> FeatureResult:

    missing = [name for name in SOURCE_NAMES if name not in arrays]
    if missing:
        raise KeyError(", ".join(missing))
    source = {name: np.asarray(arrays[name], dtype=float) for name in SOURCE_NAMES}
    shape = source[SOURCE_NAMES[0]].shape
    if any(value.shape != shape for value in source.values()):
        raise ValueError("all OLCI source arrays must have the same shape")
    oa03, oa04, oa05, oa06, oa08, oa09, oa10, oa11 = (
        source[name] for name in FEATURE_NAMES[:8]
    )
    otci = _safe_div(oa11 - oa10, oa10 - oa08, epsilon)
    values = np.stack(
        (
            oa03, oa04, oa05, oa06, oa08, oa09, oa10, oa11,
            otci, otci ** 2, otci ** 3, np.sqrt(np.abs(otci)), np.log1p(np.abs(otci)),
            _safe_div(oa11 - oa08, oa11 + oa08, epsilon),
            _safe_div(oa06 - oa04, oa06 + oa04, epsilon),
            _safe_div(oa04, oa03, epsilon),
            _safe_div(oa10 - oa08, oa09 - oa08, epsilon),
            _safe_div(oa05 - oa03, oa06, epsilon),
            source["GIFAPAR"], source["IWV"],
        ),
        axis=-1,
    )
    valid = np.isfinite(values).all(axis=-1)
    return FeatureResult(values=values, valid=valid)
