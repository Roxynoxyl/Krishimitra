# Stage E: economic estimation (only where data align).
# revenue = yield x price; profit = revenue - cost - transport.
# All values are HISTORICAL MEANS -> labeled "estimate" (paper 5.3).
import numpy as np
import pandas as pd

import config as cfg


def build_economics(df: pd.DataFrame) -> dict:
    """Per (State, District, Crop) economics + (State, Crop) fallback."""
    g = df.copy()
    g["_yield"] = pd.to_numeric(g[cfg.C_YIELD], errors="coerce")
    g["_price"] = pd.to_numeric(g[cfg.C_PRICE], errors="coerce")
    g["_cost"] = pd.to_numeric(g[cfg.C_COST], errors="coerce")
    g["_trans"] = pd.to_numeric(g[cfg.C_TRANS], errors="coerce")
    g["_msp"] = pd.to_numeric(g[cfg.C_MSP], errors="coerce")

    def agg(sub: pd.DataFrame) -> dict | None:
        y = sub["_yield"].dropna()
        p = sub["_price"].dropna()
        if len(y) == 0:
            return None
        yield_mean = float(y.mean())
        price_mean = float(p.mean()) if len(p) else None
        cost_mean = float(sub["_cost"].dropna().mean()) if sub["_cost"].notna().any() else None
        trans_mean = float(sub["_trans"].dropna().mean()) if sub["_trans"].notna().any() else None
        msp_mean = float(sub["_msp"].dropna().mean()) if sub["_msp"].notna().any() else None
        profit = None
        if price_mean is not None and cost_mean is not None:
            profit = yield_mean * price_mean - cost_mean - (trans_mean or 0.0)
        return {
            "yield_t_ha": round(yield_mean, 3),
            "yield_cv": round(float(y.std() / y.mean()), 3) if len(y) > 1 and y.mean() else None,
            "price_inr_t": round(price_mean, 1) if price_mean is not None else None,
            "price_cv": round(float(p.std() / p.mean()), 3) if len(p) > 1 and p.mean() else None,
            "cost_inr_ha": round(cost_mean, 1) if cost_mean is not None else None,
            "msp_inr_t": round(msp_mean, 1) if msp_mean is not None else None,
            "transport_inr_ha": round(trans_mean, 1) if trans_mean is not None else None,
            "profit_inr_ha": round(profit, 1) if profit is not None else None,
            "n_years": int(sub[cfg.C_YEAR].nunique()),
            "last_year": int(sub[cfg.C_YEAR].max()),
        }

    econ: dict = {"district": {}, "state": {}}
    for (state, dist, crop), sub in g.groupby([cfg.C_STATE, cfg.C_DIST, cfg.C_CROP]):
        r = agg(sub)
        if r:
            econ["district"][f"{state}|{dist}|{crop}"] = r
    for (state, crop), sub in g.groupby([cfg.C_STATE, cfg.C_CROP]):
        r = agg(sub)
        if r:
            econ["state"][f"{state}|{crop}"] = r
    return econ


def get_economics(econ: dict, state: str, district: str, crop: str) -> tuple[dict | None, str]:
    """District-level economics with state-level fallback. Returns (record, scope)."""
    r = econ["district"].get(f"{state}|{district}|{crop}")
    if r:
        return r, "district"
    r = econ["state"].get(f"{state}|{crop}")
    if r:
        return r, "state"
    return None, "none"
