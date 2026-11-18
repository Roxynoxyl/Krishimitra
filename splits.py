# Stage B: leakage-safe splits.
# PRIMARY: group-by-district holdout. SECONDARY: random stratified (optimistic).
# Split IDs persisted so every experiment uses identical splits.
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import hashlib


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def district_split(df: pd.DataFrame, seed: int = cfg.SEED, test_size: float = cfg.TEST_SIZE):
    """Primary split: hold out whole districts (all their rows)."""
    districts = (
        df[[cfg.C_STATE, cfg.C_DIST]]
        .drop_duplicates()
        .sort_values([cfg.C_STATE, cfg.C_DIST])
        .reset_index(drop=True)
    )
    rng = _rng(seed)
    idx = rng.permutation(len(districts))
    n_test = max(1, int(round(len(districts) * test_size)))
    test_d = districts.iloc[idx[:n_test]]
    test_keys = set(zip(test_d[cfg.C_STATE], test_d[cfg.C_DIST]))
    is_test = df.apply(lambda r: (r[cfg.C_STATE], r[cfg.C_DIST]) in test_keys, axis=1)
    return ~is_test, is_test, sorted(list(test_keys))


def stratified_random_split(df: pd.DataFrame, seed: int = cfg.SEED, test_size: float = cfg.TEST_SIZE):
    """Secondary split: row-level stratified by crop label (optimistic benchmark)."""
    rng = _rng(seed)
    is_test = pd.Series(False, index=df.index)
    for _, sub in df.groupby(cfg.C_CROP):
        n_test = max(1, int(round(len(sub) * test_size))) if len(sub) > 4 else 0
        if n_test:
            chosen = rng.choice(sub.index.to_numpy(), size=n_test, replace=False)
            is_test.loc[chosen] = True
    return ~is_test, is_test


def save_split_ids(path: Path, seed: int, test_districts: list, split: str = "district_holdout",
                   n_train: int | None = None, n_test: int | None = None) -> None:
    payload = {
        "seed": seed,
        "split": split,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "n_test_districts": len(test_districts) if test_districts else None,
        "test_districts": [{"state": s, "district": d} for s, d in (test_districts or [])],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def save_random_split_ids(path: Path, seed: int, train_mask: pd.Series, test_mask: pd.Series) -> None:
    payload = {
        "seed": seed,
        "split": "stratified_random",
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "note": "Row-level stratified split; optimistic secondary benchmark.",
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
