from __future__ import annotations

from pathlib import Path


BANDS = ("Oa03", "Oa04", "Oa05", "Oa06", "Oa08", "Oa09", "Oa10", "Oa11")


def validate_evidence_products(level1: Path, level2: Path) -> None:

    level1 = Path(level1)
    level2 = Path(level2)
    missing_bands = [band for band in BANDS if not (level1 / f"{band}_radiance.nc").is_file()]
    if missing_bands:
        raise ValueError("Level-1 OLCI bands are missing for: " + ", ".join(missing_bands))
    missing_auxiliary = [name for name in ("gifapar.nc", "iwv.nc", "otci.nc")
                         if not (level2 / name).is_file()]
    if missing_auxiliary:
        raise ValueError("Level-2 auxiliary files are missing: " + ", ".join(missing_auxiliary))
    if not (level2 / "lqsf.nc").is_file():
        raise ValueError("Level-2 quality flags file lqsf.nc is missing")
