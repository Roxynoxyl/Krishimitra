"""Reproducible KrishiMitra training entry point.

Run from krishimitra/: python -m src.train [--model-selection macro_f1|top3]
The primary deployment model is trained/evaluated on a district holdout. A
secondary stratified random benchmark is also written for comparison; it is
not used to claim geographic generalization.
"""
import argparse
import json
from datetime import datetime, timezone

import pandas as pd

import config as cfg
from src.data_io import load_dataset
from src.evaluate import run
from src.splits import stratified_random_split, save_random_split_ids


def _random_split_summary(df: pd.DataFrame) -> dict:
    """Record the secondary random split without replacing the primary artifact."""
    work = df.dropna(subset=[cfg.C_CROP, cfg.C_SEASON]).copy()
    if cfg.MAX_TRAIN_ROWS and len(work) > cfg.MAX_TRAIN_ROWS:
        work = work.sample(cfg.MAX_TRAIN_ROWS, random_state=cfg.SEED)
    train_mask, test_mask = stratified_random_split(work)
    save_random_split_ids(cfg.OUT_DIR / "random_split_ids.json", cfg.SEED, train_mask, test_mask)
    return {
        "split": "stratified_random",
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "warning": "optimistic secondary benchmark; not used for deployment selection",
    }


def main():
    parser = argparse.ArgumentParser(description="Train KrishiMitra models")
    parser.add_argument("--model-selection", choices=["macro_f1", "top3"], default="macro_f1")
    parser.add_argument("--with-yield", action="store_true", help="reserved for the separate yield module")
    args = parser.parse_args()

    df = load_dataset()
    metric = "val_macro_f1" if args.model_selection == "macro_f1" else "val_top3_accuracy"
    report = run(df, selection_metric=metric)
    manifest = {
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classification_selection": args.model_selection,
        "classification_best_model": report["best_model"],
        "classification_results": report["E1_E2_model_baselines_no_filter"],
        "primary_split": report["split"],
        "secondary_split": _random_split_summary(df),
        "yield": "run separately with src.yield_model; --with-yield retained for CLI compatibility",
    }
    (cfg.REPORTS_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
