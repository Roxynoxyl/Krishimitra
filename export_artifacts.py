# Export artifacts for the web app.
# Precomputes model suitability P(crop | district-season features) for every
# (State, District, Season) in recent data, so the web app is 100% static
# (Vercel-friendly) and the heavy model runs offline.
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from src.data_io import load_dataset, recent
from src.preprocess import Preprocessor
from src.models import load_model
from src.feasibility import build_crop_profiles
from src.economics import build_economics


def _geo_tree(df: pd.DataFrame) -> dict:
    geo = {}
    for state, sub in df.groupby(cfg.C_STATE):
        geo[str(state)] = sorted(sub[cfg.C_DIST].dropna().astype(str).unique().tolist())
    return geo


def _district_season_features(df: pd.DataFrame, pp: Preprocessor) -> dict:
    """Median decision-time features per (State, District, Season) from recent years."""
    feats = {}
    group_cols = [cfg.C_STATE, cfg.C_DIST, cfg.C_SEASON]
    for (state, dist, season), sub in recent(df).groupby(group_cols):
        row = sub[cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES].median(numeric_only=True)
        # categorical: mode
        cat_vals = {}
        for c in cfg.CATEGORICAL_FEATURES:
            m = sub[c].dropna().astype(str).mode()
            cat_vals[c] = m.iloc[0] if len(m) else None
        rec = {}
        for c in cfg.NUMERIC_FEATURES:
            v = row.get(c, np.nan)
            rec[c] = float(v) if pd.notna(v) else None
        rec.update(cat_vals)
        feats[f"{state}|{dist}|{season}"] = rec
    return feats


def _score_all(model, pp: Preprocessor, feat_map: dict, candidate_map: dict) -> dict:
    """For each district-season key, predict crop probabilities over candidate crops."""
    classes = list(model.classes_)
    out = {}
    rows, keys = [], []
    for key, feats in feat_map.items():
        rows.append(feats)
        keys.append(key)
    # Build one dataframe per key would be slow; batch: repeat each row per candidate?
    # Simpler: predict_proba over ALL classes once per key, then subset candidates.
    X = pp.transform(pd.DataFrame(rows))
    proba = model.predict_proba(X)          # (n_keys, n_classes)
    for i, key in enumerate(keys):
        cands = candidate_map.get(key, classes)
        idx = [classes.index(c) for c in cands if c in classes]
        if not idx:
            idx = list(range(len(classes)))
            cands = classes
        p = proba[i, idx]
        s = p.sum()
        p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        out[key] = {c: round(float(v), 4) for c, v in zip(cands, p)}
    return out


def _candidate_crops(df: pd.DataFrame, recent_years: int = cfg.RECENT_YEARS) -> dict:
    """Candidate crops per (State, District, Season): crops actually grown in the
    district in recent years; fall back to state-level if district has none."""
    d = recent(df, recent_years)
    per_district = {}
    for (state, dist, season), sub in d.groupby([cfg.C_STATE, cfg.C_DIST, cfg.C_SEASON]):
        per_district[f"{state}|{dist}|{season}"] = sorted(sub[cfg.C_CROP].astype(str).unique().tolist())
    per_state = {}
    for (state, season), sub in d.groupby([cfg.C_STATE, cfg.C_SEASON]):
        per_state[f"{state}|{season}"] = sorted(sub[cfg.C_CROP].astype(str).unique().tolist())
    return {"district": per_district, "state": per_state}


def main():
    df = load_dataset()
    print(f"[export] dataset {len(df):,} rows")

    bundle = load_model(cfg.MODELS_DIR / "model.pkl")
    model, meta = bundle["model"], bundle["meta"]
    pp = Preprocessor.load(cfg.MODELS_DIR / "preprocessor.json")
    print(f"[export] model={meta['best_model']} selection={meta['selection']}")

    crop_profiles = build_crop_profiles(df)
    econ = build_economics(df)
    geo = _geo_tree(df)
    feats = _district_season_features(df, pp)
    cand = _candidate_crops(df)

    # candidates per key with state fallback
    cand_map = {}
    for key in feats:
        state, dist, season = key.split("|", 2)
        cands = cand["district"].get(key) or cand["state"].get(f"{state}|{season}") or []
        cand_map[key] = cands

    suit = _score_all(model, pp, feats, cand_map)
    print(f"[export] scored {len(suit):,} district-season combinations")

    web = cfg.WEB_DATA_DIR
    artifacts = cfg.ARTIFACTS_DIR

    def dump(name: str, obj) -> int:
        text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        for d in (web, artifacts):
            (d / name).write_text(text, encoding="utf-8")
        return len(text)

    sizes = {}
    sizes["geo.json"] = dump("geo.json", geo)
    sizes["crops.json"] = dump("crops.json", crop_profiles)
    sizes["economics.json"] = dump("economics.json", econ)
    sizes["suitability.json"] = dump("suitability.json", suit)
    meta_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_rows": int(len(df)),
        "dataset_years": [int(df[cfg.C_YEAR].min()), int(df[cfg.C_YEAR].max())],
        "model": meta,
        "ranking_weights": cfg.RANKING_WEIGHTS,
        "district_seasons_scored": len(suit),
        "n_crops": len(crop_profiles),
        "n_districts": sum(len(v) for v in geo.values()),
        "artifact_kb": {k: round(v / 1024, 1) for k, v in sizes.items()},
        "water_levels": ["Low", "Medium", "High", "Irrigated"],
    }
    sizes["meta.json"] = dump("meta.json", meta_out)
    print(f"[export] artifacts: {meta_out['artifact_kb']} KB -> {web}")
    return meta_out


if __name__ == "__main__":
    main()
