from __future__ import annotations

from pathlib import Path

import pandas as pd


TIME_COLUMN = "Monitoring Time"
VALUE_COLUMN = "Chlorophyll"


def normalize_chla_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse, sort, and median-aggregate duplicate Chl-a timestamps."""
    missing = [name for name in (TIME_COLUMN, VALUE_COLUMN) if name not in frame]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    clean = pd.DataFrame(
        {
            "time": pd.to_datetime(frame[TIME_COLUMN], errors="coerce"),
            "chlorophyll_mg_l": pd.to_numeric(frame[VALUE_COLUMN], errors="coerce"),
        }
    )
    if clean["time"].isna().any():
        raise ValueError("Monitoring Time contains invalid timestamps")
    if (clean["chlorophyll_mg_l"].dropna() < 0).any():
        raise ValueError("negative Chlorophyll values are not allowed")
    grouped = clean.groupby("time", sort=True, as_index=False).agg(
        chlorophyll_mg_l=("chlorophyll_mg_l", "median"),
        duplicate_count=("time", "size"),
    )
    return grouped


def load_chla_csv(path: Path) -> pd.DataFrame:
    return normalize_chla_frame(pd.read_csv(path))


def regularize_four_hourly(frame: pd.DataFrame) -> pd.DataFrame:

    if frame.empty:
        raise ValueError("cannot regularize an empty Chl-a series")
    indexed = frame.set_index("time")
    index = pd.date_range(indexed.index.min(), indexed.index.max(), freq="4h")
    grid = indexed.reindex(index).rename_axis("time").reset_index()
    grid["observed"] = grid["chlorophyll_mg_l"].notna()
    grid["duplicate_count"] = grid["duplicate_count"].fillna(0).astype(int)
    return grid
