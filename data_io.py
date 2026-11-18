# Dataset loading with pickle cache and derived-column parsing.
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


_DRY_RE = re.compile(r"dry_days:\s*(\d+)")
_FLOOD_RE = re.compile(r"flood_days:\s*(\d+)")


def _parse_counts(series: pd.Series) -> pd.DataFrame:
    """Parse 'dry_days:8; flood_days:1' into integer columns."""
    dry = series.astype(str).str.extract(_DRY_RE)[0]
    flood = series.astype(str).str.extract(_FLOOD_RE)[0]
    return pd.DataFrame({
        "dry_days": pd.to_numeric(dry, errors="coerce"),
        "flood_days": pd.to_numeric(flood, errors="coerce"),
    })


def load_dataset(cache: bool = True) -> pd.DataFrame:
    """Load the unified CSV once, cache as pickle for fast subsequent runs."""
    cache_path = cfg.PROCESSED_DIR / "dataset.pkl"
    if cache and cache_path.exists():
        src = cfg.find_dataset()
        if cache_path.stat().st_mtime >= src.stat().st_mtime:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    src = cfg.find_dataset()
    df = pd.read_csv(src, low_memory=False, encoding="utf-8")

    # Derived columns from compound fields
    counts = _parse_counts(df[cfg.C_DRYFLOOD])
    df["dry_days"] = counts["dry_days"]
    df["flood_days"] = counts["flood_days"]

    # Basic hygiene: drop rows without a crop label or season
    n0 = len(df)
    df = df[df[cfg.C_CROP].notna() & (df[cfg.C_CROP].astype(str).str.strip() != "")]
    df = df[df[cfg.C_SEASON].notna()]
    df = df.drop_duplicates()
    print(f"[data_io] loaded {len(df):,} rows (raw {n0:,}, dropped {n0 - len(df):,})")

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f, protocol=4)
    return df


def recent(df: pd.DataFrame, years: int | None = None) -> pd.DataFrame:
    """Filter to the most recent `years` years of data."""
    years = years or cfg.RECENT_YEARS
    ymax = df[cfg.C_YEAR].max()
    return df[df[cfg.C_YEAR] >= ymax - years + 1]
