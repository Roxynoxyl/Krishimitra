# Stage F: multi-objective ranking among FEASIBLE crops only.
# R(c) = w_s*Suit + w_y*Ynorm + w_p*Pnorm + w_m*Mnorm - w_r*Risknorm
# Objectives normalized on the candidate set; weights disclosed project parameters.
import numpy as np

import config as cfg


def _minmax(values: list[float | None]) -> list[float | None]:
    known = [v for v in values if v is not None]
    if not known:
        return [None] * len(values)
    lo, hi = min(known), max(known)
    if hi - lo < 1e-9:
        return [0.5 if v is not None else None for v in values]
    return [(v - lo) / (hi - lo) if v is not None else None for v in values]


def compute_risk(econ_record: dict | None) -> tuple[float | None, str]:
    """Risk = 0.6*yield_cv + 0.4*price_cv; neutral 0.5 when unknown (flagged)."""
    if not econ_record:
        return None, "unknown: no economics for this crop in this location"
    ycv = econ_record.get("yield_cv")
    pcv = econ_record.get("price_cv")
    if ycv is None and pcv is None:
        return None, "unknown: insufficient year-over-year variation data"
    parts = [(0.6, ycv), (0.4, pcv)]
    wsum = sum(w for w, v in parts if v is not None)
    risk = sum(w * v for w, v in parts if v is not None) / wsum
    return float(min(risk, 2.0)), "estimated from historical yield/price variability"


def rank_candidates(candidates: list[dict], econ_lookup, weights: dict | None = None) -> list[dict]:
    """candidates: list of feasible decision dicts with 'crop' key.
    econ_lookup: callable crop -> (record|None, scope).
    Adds 'score' and component values; returns sorted best-first."""
    w = weights or cfg.RANKING_WEIGHTS
    crops = [d["crop"] for d in candidates]
    records = [econ_lookup(c) for c in crops]

    y_vals = [r[0]["yield_t_ha"] if r[0] else None for r in records]
    p_vals = [r[0]["profit_inr_ha"] if r[0] else None for r in records]
    m_vals = [r[0]["price_inr_t"] if r[0] else None for r in records]
    risks = [compute_risk(r[0])[0] if r[0] else None for r in records]

    Yn, Pn, Mn, Rn = _minmax(y_vals), _minmax(p_vals), _minmax(m_vals), _minmax(risks)

    out = []
    for i, d in enumerate(candidates):
        rec, scope = records[i]
        suit = d.get("suitability", 0.5)      # model P(crop|features); 0.5 if none
        yn = Yn[i] if Yn[i] is not None else 0.0
        pn = Pn[i] if Pn[i] is not None else 0.0
        mn = Mn[i] if Mn[i] is not None else 0.0
        rn = Rn[i] if Rn[i] is not None else 0.5  # neutral risk penalty when unknown
        score = (
            w["suitability"] * suit
            + w["yield"] * yn
            + w["profit"] * pn
            + w["market"] * mn
            - w["risk"] * rn
        )
        r = dict(d)
        r.update({
            "score": round(float(score), 4),
            "components": {
                "suitability": round(float(suit), 4),
                "yield_norm": None if Yn[i] is None else round(float(Yn[i]), 3),
                "profit_norm": None if Pn[i] is None else round(float(Pn[i]), 3),
                "market_norm": None if Mn[i] is None else round(float(Mn[i]), 3),
                "risk_norm": None if Rn[i] is None else round(float(Rn[i]), 3),
            },
            "econ_scope": scope,
            "econ": rec,
        })
        out.append(r)
    out.sort(key=lambda r: -r["score"])
    return out


def sensitivity_check(candidates: list[dict], econ_lookup, base_weights: dict,
                      delta: float = 0.2) -> dict:
    """Rank stability under ±delta weight perturbations (paper 5.3)."""
    base = rank_candidates(candidates, econ_lookup, base_weights)
    base_order = [c["crop"] for c in base]
    stable = True
    for key in base_weights:
        for sign in (+1, -1):
            w = dict(base_weights)
            w[key] = w[key] * (1 + sign * delta)
            order = [c["crop"] for c in rank_candidates(candidates, econ_lookup, w)]
            if order[:1] != base_order[:1]:
                stable = False
    return {"stable_top1": stable, "delta": delta}
