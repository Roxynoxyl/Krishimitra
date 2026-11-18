# Stage A: data validation and profiling.
# Produces outputs/reports/data_profile.md and data_profile.json (paper Phase 1 gate).
import json

import numpy as np
import pandas as pd

import config as cfg
from src.data_io import load_dataset


def profile(df: pd.DataFrame) -> dict:
    n = len(df)
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    profile_dict = {
        "rows": n,
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "n_crops": int(df[cfg.C_CROP].nunique()),
        "n_states": int(df[cfg.C_STATE].nunique()),
        "n_districts": int(df[cfg.C_DIST].nunique()),
        "state_district_pairs": int(df.groupby([cfg.C_STATE, cfg.C_DIST]).ngroups),
        "year_min": int(df[cfg.C_YEAR].min()),
        "year_max": int(df[cfg.C_YEAR].max()),
        "seasons": sorted(map(str, df[cfg.C_SEASON].dropna().unique().tolist())),
        "soil_textures": sorted(map(str, df[cfg.C_TEX].dropna().unique().tolist())),
        "irrigation_sources": sorted(map(str, df[cfg.C_IRR_SRC].dropna().unique().tolist())),
        "missing_counts": missing,
        "missing_pct": {c: round(100.0 * v / n, 2) for c, v in missing.items()},
        "target_distribution_top10": {
            str(k): int(v) for k, v in df[cfg.C_CROP].value_counts().head(10).items()
        },
        "target_distribution_min": int(df[cfg.C_CROP].value_counts().min()),
        "coverage": {
            "price_pct": round(100.0 * df[cfg.C_PRICE].notna().mean(), 2),
            "cost_pct": round(100.0 * df[cfg.C_COST].notna().mean(), 2),
            "msp_pct": round(100.0 * df[cfg.C_MSP].notna().mean(), 2),
            "soil_npk_pct": round(100.0 * df[cfg.C_N].notna().mean(), 2),
            "rainfall_pct": round(100.0 * df[cfg.C_RAIN].notna().mean(), 2),
            "soil_texture_pct": round(100.0 * df[cfg.C_TEX].notna().mean(), 2),
            "irrigation_pct": round(100.0 * df[cfg.C_IRR_SRC].notna().mean(), 2),
        },
    }
    return profile_dict


def write_report(p: dict) -> None:
    with open(cfg.REPORTS_DIR / "data_profile.json", "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)

    lines = [
        "# KrishiMitra Data Profile",
        "",
        f"- Rows: {p['rows']:,} | Columns: {p['columns']} | Duplicate rows: {p['duplicates']:,}",
        f"- Crops: {p['n_crops']} | States: {p['n_states']} | Districts: {p['n_districts']} "
        f"({p['state_district_pairs']} state-district pairs)",
        f"- Years: {p['year_min']}–{p['year_max']}",
        f"- Seasons: {', '.join(p['seasons'])}",
        f"- Soil textures: {', '.join(p['soil_textures'])}",
        f"- Irrigation sources: {', '.join(p['irrigation_sources'])}",
        "",
        "## Feature coverage (decision-time vs economic)",
        "",
    ]
    for k, v in p["coverage"].items():
        lines.append(f"- {k}: {v}%")
    lines += ["", "## Missingness (columns with any missing)", ""]
    for c, pct in sorted(p["missing_pct"].items(), key=lambda kv: -kv[1]):
        if pct > 0:
            lines.append(f"- `{c}`: {pct}% ({p['missing_counts'][c]:,} rows)")
    lines += ["", "## Target (crop) distribution — top 10", ""]
    for k, v in p["target_distribution_top10"].items():
        lines.append(f"- {k}: {v:,}")
    rarer = [k for k in p["target_distribution_top10"]]
    lines += ["", f"Rarest crop has {p['target_distribution_min']:,} rows.", ""]
    (cfg.REPORTS_DIR / "data_profile.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_dataset()
    p = profile(df)
    write_report(p)
    print(f"[validate] profile written: {cfg.REPORTS_DIR / 'data_profile.md'}")
    return p


if __name__ == "__main__":
    main()
