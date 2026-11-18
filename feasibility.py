# Stage D: feasibility gate (the paper's core contribution).
# Feasible(c) = Season(c) AND Soil(c) AND Water(c) AND Budget(c)
# UNKNOWN never defaults to feasible. Strict mode excludes unknowns;
# advisory mode retains them with a warning. Never silently relaxes constraints.
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg

WATER_ORDER = {"Low": 0, "Medium": 1, "High": 2}          # crop need side
AVAIL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Irrigated": 3}  # farmer side

STATUS_PASS, STATUS_FAIL, STATUS_UNKNOWN = "pass", "fail", "unknown"


@dataclass
class FarmerProfile:
    state: str
    district: str
    season: str
    budget_per_ha: float | None = None      # INR/ha the farmer can spend
    water_availability: str | None = None   # Low / Medium / High / Irrigated
    soil_texture: str | None = None         # e.g. Loamy, Black, Red, Sandy
    area_ha: float | None = None
    labor_available_hours: float | None = None
    mechanization: str | None = None       # None / Low / Medium / High
    health_status: str | None = None       # optional self-reported context
    risk_preference: str = "Balanced"      # Conservative / Balanced / Growth
    market_preference: str = "Balanced"    # Local / Balanced / Contract
    # Optional measured observations used for transparency and future live inference.
    rainfall_mm: float | None = None
    mean_temperature_c: float | None = None
    humidity_pct: float | None = None
    soil_n: float | None = None
    soil_p: float | None = None
    soil_k: float | None = None
    soil_ph: float | None = None
    irrigation_source: str | None = None


# ---------------------------------------------------------------------------
# Crop profiles derived from the dataset (requirements side of the gate)
# ---------------------------------------------------------------------------
def build_crop_profiles(df: pd.DataFrame) -> dict:
    """Per-crop: compatible seasons, soil textures, water-need category, cost/ha."""
    profiles = {}
    d = df.copy()
    d["_rain"] = pd.to_numeric(d[cfg.C_RAIN], errors="coerce")
    d["_cost"] = pd.to_numeric(d[cfg.C_COST], errors="coerce")

    # crop -> median rainfall of the records it is grown in (water-need proxy)
    crop_rain = d.groupby(cfg.C_CROP)["_rain"].median()

    # tercile cuts across crops -> Low / Medium / High water need
    cuts = crop_rain.quantile([1 / 3, 2 / 3]).to_numpy()
    water_need = {}
    for crop, v in crop_rain.items():
        if np.isnan(v):
            water_need[crop] = None
        elif v <= cuts[0]:
            water_need[crop] = "Low"
        elif v <= cuts[1]:
            water_need[crop] = "Medium"
        else:
            water_need[crop] = "High"

    for crop, sub in d.groupby(cfg.C_CROP):
        n = len(sub)
        seasons = (
            sub[cfg.C_SEASON].astype(str).value_counts(normalize=True)
            .pipe(lambda s: sorted(s[s >= 0.01].index.tolist()))
        )
        textures = (
            sub[cfg.C_TEX].dropna().astype(str).value_counts(normalize=True)
            .pipe(lambda s: sorted(s[s >= 0.05].index.tolist()))
        )
        def med(col):
            s = pd.to_numeric(sub[col], errors="coerce")
            return float(s.median()) if s.notna().any() else None

        profiles[crop] = {
            "seasons": seasons,
            "soil_textures": textures,
            "water_need": water_need.get(crop),
            "cost_per_ha": float(sub["_cost"].median()) if sub["_cost"].notna().any() else None,
            "median_rainfall_mm": med(cfg.C_RAIN),
            "median_temperature_c": med(cfg.C_MEANT),
            "median_humidity_pct": med(cfg.C_HUM),
            "median_soil_n": med(cfg.C_N),
            "median_soil_p": med(cfg.C_P),
            "median_soil_k": med(cfg.C_K),
            "median_soil_ph": med(cfg.C_PH),
            "n_records": int(n),
        }
    return profiles


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------
def _check_season(season: str, crop_profile: dict) -> tuple[str, str]:
    seasons = crop_profile.get("seasons") or []
    if not seasons:
        return STATUS_UNKNOWN, "no season coverage for this crop in the dataset"
    if str(season) in seasons:
        return STATUS_PASS, f"grown in {season} season"
    return STATUS_FAIL, f"not a {season} crop (dataset shows: {', '.join(seasons)})"


def _check_soil(soil_texture: str | None, crop_profile: dict) -> tuple[str, str]:
    if soil_texture is None:
        return STATUS_UNKNOWN, "farmer soil texture not provided"
    texs = crop_profile.get("soil_textures") or []
    if not texs:
        return STATUS_UNKNOWN, "no soil-texture coverage for this crop"
    if str(soil_texture) in texs:
        return STATUS_PASS, f"grows in {soil_texture} soil"
    return STATUS_FAIL, f"poor fit for {soil_texture} soil (dataset shows: {', '.join(texs)})"


def _check_water(avail: str | None, crop_profile: dict) -> tuple[str, str]:
    if avail is None:
        return STATUS_UNKNOWN, "farmer water availability not provided"
    need = crop_profile.get("water_need")
    if need is None:
        return STATUS_UNKNOWN, "no rainfall coverage for this crop"
    if avail == "Irrigated":
        return STATUS_PASS, "irrigation covers water need"
    if AVAIL_ORDER.get(avail, -1) >= WATER_ORDER[need]:
        return STATUS_PASS, f"water {avail} >= crop need {need}"
    return STATUS_FAIL, f"water {avail} below crop need {need}"


def _check_budget(budget: float | None, crop_profile: dict) -> tuple[str, str]:
    cost = crop_profile.get("cost_per_ha")
    if budget is None:
        return STATUS_UNKNOWN, "farmer budget not provided"
    if cost is None:
        return STATUS_UNKNOWN, "cultivation cost unknown in dataset"
    if float(budget) >= cost:
        return STATUS_PASS, f"budget {budget:,.0f} >= cost {cost:,.0f} INR/ha"
    return STATUS_FAIL, f"cost {cost:,.0f} INR/ha exceeds budget {budget:,.0f}"


def evaluate_crop(profile: FarmerProfile, crop: str, crop_profile: dict) -> dict:
    checks = {
        "season": _check_season(profile.season, crop_profile),
        "soil": _check_soil(profile.soil_texture, crop_profile),
        "water": _check_water(profile.water_availability, crop_profile),
        "budget": _check_budget(profile.budget_per_ha, crop_profile),
    }
    statuses = {k: v[0] for k, v in checks.items()}
    reasons = {k: v[1] for k, v in checks.items()}
    advisories = []
    if profile.labor_available_hours is None:
        advisories.append("labor capacity not provided; labor feasibility is advisory only")
    if profile.health_status and profile.health_status.lower() not in {"good", "healthy", "no limitations"}:
        advisories.append("health context is self-reported; consult an agronomist/health professional")
    return {"crop": crop, "statuses": statuses, "reasons": reasons, "advisories": advisories}


def decide(decision: dict, mode: str = "strict") -> dict:
    """Apply mode semantics: strict -> any non-pass excludes; advisory -> only fails exclude."""
    st = decision["statuses"]
    if mode not in ("strict", "advisory"):
        raise ValueError(f"unknown mode: {mode}")
    failed = [k for k, v in st.items() if v == STATUS_FAIL]
    unknown = [k for k, v in st.items() if v == STATUS_UNKNOWN]
    if mode == "strict":
        feasible = not failed and not unknown
    else:
        feasible = not failed
    decision["feasible"] = feasible
    decision["failed"] = failed
    decision["unknown"] = unknown
    decision["warnings"] = [
        f"{k}: {decision['reasons'][k]}" for k in unknown
    ]
    return decision


def filter_candidates(profile: FarmerProfile, crop_profiles: dict, mode: str = "strict",
                      candidates: list[str] | None = None) -> list[dict]:
    """Evaluate + decide for every candidate crop. Sorted: feasible first."""
    out = []
    for crop in (candidates or sorted(crop_profiles)):
        d = decide(evaluate_crop(profile, crop, crop_profiles[crop]), mode=mode)
        out.append(d)
    out.sort(key=lambda d: (not d["feasible"], d["crop"]))
    return out


def binding_constraints(decisions: list[dict]) -> list[tuple[str, int]]:
    """When nothing is feasible: which constraints excluded the most crops."""
    counts: dict[str, int] = {}
    for d in decisions:
        for k in d["failed"]:
            counts[k] = counts.get(k, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])
