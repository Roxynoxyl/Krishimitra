# KrishiMitra unit tests (no pytest dependency — run: python tests/run_tests.py)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import config as cfg
from src.feasibility import (FarmerProfile, build_crop_profiles, decide, filter_candidates,
                             evaluate_crop, binding_constraints, STATUS_PASS, STATUS_FAIL,
                             STATUS_UNKNOWN)
from src.ranking import rank_candidates, compute_risk, sensitivity_check
from src.models import GaussianNB, accuracy, macro_f1, topk_accuracy

PASSED, FAILED = [], []


def check(name, cond):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


# --- feature contract -------------------------------------------------------
def test_feature_contract():
    all_feats = set(cfg.NUMERIC_FEATURES) | set(cfg.CATEGORICAL_FEATURES)
    leaked = all_feats & set(cfg.FORBIDDEN_PREDICTORS)
    check("no forbidden predictor in feature contract", not leaked)
    check("target not a feature", cfg.TARGET not in all_feats)


# --- feasibility gate -------------------------------------------------------
FAKE_PROFILES = {
    "Rice": {"seasons": ["Kharif"], "soil_textures": ["Loamy", "Clayey"],
             "water_need": "High", "cost_per_ha": 20000.0, "n_records": 100},
    "Wheat": {"seasons": ["Rabi"], "soil_textures": ["Loamy"],
              "water_need": "Low", "cost_per_ha": 15000.0, "n_records": 100},
    "Mystery": {"seasons": [], "soil_textures": [], "water_need": None,
                "cost_per_ha": None, "n_records": 0},
}


def _profile(**kw):
    base = dict(state="X", district="Y", season="Kharif")
    base.update(kw)
    return FarmerProfile(**base)


def test_strict_unknown_never_feasible():
    d = decide(evaluate_crop(_profile(), "Rice", FAKE_PROFILES["Rice"]), mode="strict")
    check("strict: unknown soil excludes Rice", (not d["feasible"]) and "soil" in d["unknown"])


def test_advisory_retains_unknown_with_warning():
    d = decide(evaluate_crop(_profile(), "Rice", FAKE_PROFILES["Rice"]), mode="advisory")
    check("advisory: Rice feasible with warnings", d["feasible"] and len(d["warnings"]) > 0)


def test_season_fail():
    d = decide(evaluate_crop(_profile(season="Rabi"), "Rice", FAKE_PROFILES["Rice"]),
               mode="advisory")
    check("season mismatch fails even in advisory", (not d["feasible"]) and "season" in d["failed"])


def test_water_fail():
    d = decide(evaluate_crop(_profile(water_availability="Low", soil_texture="Loamy"),
                             "Rice", FAKE_PROFILES["Rice"]), mode="advisory")
    check("low water fails high-need crop", "water" in d["failed"])


def test_budget_fail():
    d = decide(evaluate_crop(_profile(budget_per_ha=10000, water_availability="Irrigated",
                                      soil_texture="Loamy"), "Rice", FAKE_PROFILES["Rice"]),
               mode="advisory")
    check("budget below cost fails", "budget" in d["failed"])


def test_unknown_defaults_never_true():
    d = decide(evaluate_crop(_profile(), "Mystery", FAKE_PROFILES["Mystery"]), mode="strict")
    check("all-unknown crop excluded in strict", not d["feasible"])


def test_binding_constraints():
    prof = _profile(season="Rabi", water_availability="Low")
    decisions = filter_candidates(prof, FAKE_PROFILES, mode="advisory")
    bc = dict(binding_constraints(decisions))
    check("binding constraints reported for empty/failed sets", bc.get("season", 0) >= 1)


def test_impossible_profile_returns_nothing_feasible():
    # Summer matches no known crop: fails for known crops, unknown for Mystery.
    # Strict mode excludes both -> zero feasible crops.
    prof = _profile(season="Summer")
    decisions = filter_candidates(prof, FAKE_PROFILES, mode="strict")
    check("impossible profile -> zero feasible crops (strict)",
          all(not d["feasible"] for d in decisions))
    # Advisory retains only the all-unknown crop, with warnings (paper semantics).
    adv = filter_candidates(prof, FAKE_PROFILES, mode="advisory")
    feas_adv = [d["crop"] for d in adv if d["feasible"]]
    check("advisory retains only unknown-status crops", feas_adv == ["Mystery"])


# --- ranking ----------------------------------------------------------------
def _econ_lookup(crop):
    table = {
        "A": ({"yield_t_ha": 2.0, "price_inr_t": 20000, "msp_inr_t": 20000,
               "yield_cv": 0.1, "price_cv": 0.1, "cost_inr_ha": 10000,
               "profit_inr_ha": 30000}, "district"),
        "B": ({"yield_t_ha": 4.0, "price_inr_t": 10000, "msp_inr_t": 9000,
               "yield_cv": 0.4, "price_cv": 0.4, "cost_inr_ha": 10000,
               "profit_inr_ha": 30000}, "district"),
        "C": (None, "none"),
    }
    return table.get(crop, (None, "none"))


def test_ranking_order_and_scores():
    cands = [{"crop": "A", "suitability": 0.8}, {"crop": "B", "suitability": 0.8}]
    ranked = rank_candidates(cands, _econ_lookup)
    check("ranked best-first", ranked[0]["score"] >= ranked[1]["score"])
    check("components present", "components" in ranked[0])


def test_risk_penalizes_volatile():
    cands = [{"crop": "A", "suitability": 0.8}, {"crop": "B", "suitability": 0.8}]
    ranked = rank_candidates(cands, _econ_lookup)
    a = next(r for r in ranked if r["crop"] == "A")
    b = next(r for r in ranked if r["crop"] == "B")
    check("volatile crop scores lower", a["score"] > b["score"])


def test_unknown_econ_neutral():
    r, note = compute_risk(None)
    check("unknown risk -> None + note", r is None and "unknown" in note)


def test_sensitivity_runs():
    cands = [{"crop": "A", "suitability": 0.8}, {"crop": "B", "suitability": 0.8}]
    res = sensitivity_check(cands, _econ_lookup, cfg.RANKING_WEIGHTS)
    check("sensitivity check returns stability flag", "stable_top1" in res)


# --- model ------------------------------------------------------------------
def test_gnb_learns_synthetic():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1, (200, 5))
    X1 = rng.normal(3, 1, (200, 5))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array(["a"] * 200 + ["b"] * 200)
    m = GaussianNB().fit(X, y)
    acc = accuracy(y, m.predict(X))
    check(f"GaussianNB learns synthetic (acc={acc:.2f})", acc > 0.95)
    proba = m.predict_proba(X[:5])
    check("proba rows sum to 1", np.allclose(proba.sum(axis=1), 1.0))
    check("top3 accuracy works", topk_accuracy(y, m.predict_proba(X), m.classes_, 3) > 0.95)
    check("macro_f1 range", 0 <= macro_f1(y, m.predict(X), m.classes_) <= 1)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
