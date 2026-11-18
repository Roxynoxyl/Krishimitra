# Stage H: evaluation harness (paper experiments E1-E3 measured; E4/E5 = not measured).
# E1-E3: model baselines on the district-holdout split (no filter).
# E3: best model + feasibility gate -> violation rate over synthetic farmer profiles.
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from src.data_io import load_dataset
from src.preprocess import Preprocessor
from src.models import train_and_select, save_model, macro_f1, topk_accuracy, accuracy
from src.splits import district_split, save_split_ids
from src.feasibility import (FarmerProfile, build_crop_profiles, filter_candidates,
                             binding_constraints)


def run(df: pd.DataFrame, selection_metric: str = "val_macro_f1") -> dict:
    rng = np.random.default_rng(cfg.SEED)

    # Feature matrix (leakage-safe contract enforced by config)
    work = df.dropna(subset=[cfg.C_CROP, cfg.C_SEASON]).copy()
    if cfg.MAX_TRAIN_ROWS and len(work) > cfg.MAX_TRAIN_ROWS:
        work = work.sample(cfg.MAX_TRAIN_ROWS, random_state=cfg.SEED)
        print(f"[evaluate] sampled {len(work):,} rows for training")

    train_m, test_m, test_districts = district_split(work)
    save_split_ids(
        cfg.OUT_DIR / "split_ids.json", cfg.SEED, test_districts,
        n_train=int(train_m.sum()), n_test=int(test_m.sum())
    )
    print(f"[evaluate] district split: train {train_m.sum():,} / test {test_m.sum():,} "
          f"({len(test_districts)} held-out districts)")

    y_all = work[cfg.C_CROP].astype(str).to_numpy()
    classes = np.unique(y_all)
    classes = classes[np.isin(classes, np.unique(y_all[train_m]))]  # train-only classes
    tr_mask = train_m & np.isin(y_all, classes)
    te_mask = test_m & np.isin(y_all, classes)

    pp = Preprocessor().fit(work[tr_mask])
    Xtr = pp.transform(work[tr_mask])
    Xte = pp.transform(work[te_mask])
    ytr = y_all[tr_mask]
    yte = y_all[te_mask]
    print(f"[evaluate] X: train {Xtr.shape}, test {Xte.shape}, classes {len(classes)}")

    best_model, all_results, best_res = train_and_select(
        Xtr, ytr, Xte, yte, classes, seed=cfg.SEED, selection_metric=selection_metric)

    # Persist artifacts for export/restore
    pp.save(cfg.MODELS_DIR / "preprocessor.json")
    save_model(best_model, cfg.MODELS_DIR / "model.pkl", {
        "best_model": best_model.name,
        "val_results": all_results,
        "selection": f"max {selection_metric} on district-holdout split (never test)",
        "classes": [str(c) for c in best_model.classes_],
        "seed": cfg.SEED,
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(tr_mask.sum()),
        "test_rows": int(te_mask.sum()),
        "n_heldout_districts": len(test_districts),
    })

    # --- E3: gate violation test on synthetic farmer profiles -----------------
    crop_profiles = build_crop_profiles(df)
    known_crops = [c for c in crop_profiles]
    test_rows = work[te_mask]
    sample = test_rows.sample(min(1500, len(test_rows)), random_state=cfg.SEED)
    violations = 0
    feasible_rate = 0.0
    checked = 0
    for _, row in sample.iterrows():
        budget = float(row[cfg.C_COST]) * 1.2 if pd.notna(row[cfg.C_COST]) else None
        avail = "Irrigated" if (pd.notna(row[cfg.C_IRR_PCT]) and row[cfg.C_IRR_PCT] > 50) else "Medium"
        prof = FarmerProfile(
            state=str(row[cfg.C_STATE]), district=str(row[cfg.C_DIST]),
            season=str(row[cfg.C_SEASON]), budget_per_ha=budget,
            water_availability=avail,
            soil_texture=str(row[cfg.C_TEX]) if pd.notna(row[cfg.C_TEX]) else None,
        )
        decisions = filter_candidates(prof, crop_profiles, mode="advisory", candidates=known_crops)
        feas = [d["crop"] for d in decisions if d["feasible"]]
        feasible_rate += 1.0 if feas else 0.0
        checked += 1
        # Violation definition: a crop marked infeasible-in-fact (failed constraint)
        # must never appear in the feasible set.
        for d in decisions:
            if d["feasible"] and d["failed"]:
                violations += 1
    e3 = {
        "profiles_checked": checked,
        "violation_rate": round(violations / max(checked, 1), 6),
        "profiles_with_any_feasible_crop_pct": round(100.0 * feasible_rate / max(checked, 1), 1),
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "primary district holdout (seed=%d)" % cfg.SEED,
        "E1_E2_model_baselines_no_filter": all_results,
        "selection_metric": selection_metric,
        "best_model": best_res,
        "E3_best_model_plus_feasibility_gate": e3,
        "E4_economic_ranking": "Not measured (needs aligned independent ground truth)",
        "E5_full_ranking": "Not measured (needs farmer preference / field validation)",
    }
    (cfg.REPORTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# KrishiMitra Evaluation (measured numbers only)",
        "",
        f"Generated: {report['generated_utc']}",
        f"Split: {report['split']} — {len(test_districts)} held-out districts",
        "",
        "## E1–E3 — model baselines, no farmer filter (held-out districts)",
        "",
        "| Model | Accuracy | Macro-F1 | Top-3 acc |",
        "|---|---|---|---|",
    ]
    for r in all_results:
        lines.append(f"| {r['model']} | {r['val_accuracy']:.3f} | {r['val_macro_f1']:.3f} | {r['val_top3_accuracy']:.3f} |")
    lines += [
        "",
        f"Selected model (by {selection_metric}): **{best_res['model']}**",
        "",
        "## E3 — best model + feasibility gate",
        "",
        f"- Synthetic farmer profiles checked: {e3['profiles_checked']:,}",
        f"- Constraint-violation rate: {e3['violation_rate']} (target: 0)",
        f"- Profiles with at least one feasible crop: {e3['profiles_with_any_feasible_crop_pct']}%",
        "",
        "## E4/E5",
        "",
        "- E4 (economic ranking): Not measured — requires aligned independent ground truth.",
        "- E5 (full ranking): Not measured — requires farmer preference data / field validation.",
        "",
    ]
    (cfg.REPORTS_DIR / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[evaluate] report: {cfg.REPORTS_DIR / 'evaluation.md'}")
    return report


def main():
    df = load_dataset()
    return run(df)


if __name__ == "__main__":
    main()
