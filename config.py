# KrishiMitra pipeline configuration.
# Source of truth for paths, seeds, feature contract, and ranking weights.
from pathlib import Path

ROOT = Path(__file__).resolve().parent              # krishimitra/ (code package)
WORKSPACE = ROOT.parent                              # project root (D:\A\KRISHIMITRA)

DATASET_CANDIDATES = [
    WORKSPACE / "KRISHIMITRA DATASET.csv",
    Path(r"D:\A\KRISHIMITRA\KRISHIMITRA DATASET.csv"),
    ROOT / "data" / "raw" / "KRISHIMITRA DATASET.csv",
]

def find_dataset() -> Path:
    for p in DATASET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "KRISHIMITRA DATASET.csv not found. Place it in the workspace root or set the path in config.py."
    )

# Reproducibility
SEED = 42
TEST_SIZE = 0.2                 # district-holdout fraction (primary split)
MAX_TRAIN_ROWS = 300_000        # cap for training speed (sampled with SEED); set None for full data
RECENT_YEARS = 15               # years used for district-level aggregates / economics

# Output locations
OUT_DIR = WORKSPACE / "outputs"
REPORTS_DIR = OUT_DIR / "reports"
MODELS_DIR = OUT_DIR / "models"
ARTIFACTS_DIR = OUT_DIR / "artifacts"
PROCESSED_DIR = OUT_DIR / "processed"
WEB_DATA_DIR = WORKSPACE / "web" / "data"
for _d in (REPORTS_DIR, MODELS_DIR, ARTIFACTS_DIR, PROCESSED_DIR, WEB_DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset schema (exact column names from KRISHIMITRA DATASET.csv)
# ---------------------------------------------------------------------------
C_STATE = "State"
C_DIST = "District"
C_CROP = "Crop"
C_YEAR = "Crop_Year"
C_SEASON = "Season"
C_AREA = "Area (hectares)"
C_PROD = "Production (tonnes)"
C_YIELD = "Yield (t/ha)"
C_RAIN = "Seasonal cumulative rainfall"
C_DRYFLOOD = "Dry-spell / flood-day counts"
C_MEANT = "Mean growing-season temperature"
C_MAXT = "Maximum temperature"
C_HEAT = "Heat-stress degree days (flowering)"
C_HUM = "Humidity"
C_SUN = "Sunshine hours"
C_N = "Soil Nitrogen (N)"
C_P = "Soil Phosphorus (P)"
C_K = "Soil Potassium (K)"
C_PH = "Soil pH"
C_SOC = "Soil organic carbon"
C_TEX = "Soil texture class"
C_IRR_SRC = "Irrigation source (canal / well / tube well / rainfed)"
C_IRR_PCT = "% net sown area irrigated"
C_GW = "Groundwater depth"
C_FERT = "Fertilizer applied (kg/ha N, P, K)"
C_HYV = "Seed variety / HYV share"
C_SOW = "Sowing date"
C_PEST = "Pest & disease incidence"
C_PRICE = "Mandi modal price (district \u00d7 crop \u00d7 month)"
C_MSP = "MSP (crop \u00d7 year)"
C_COST = "Cost of cultivation per hectare"
C_TRANS = "Transport / market-access cost"

# ---------------------------------------------------------------------------
# Feature contract (leakage controls, paper section 4.3)
# Predictors must be knowable AT RECOMMENDATION TIME.
# EXCLUDED: Area/Production/Yield (target-derived outcomes), Fertilizer/HYV/sowing
# (post-decision farmer actions), Pest (post-outcome), Price/MSP/Cost/Transport
# (post-outcome economics -> used in ranking only, never as classifiers inputs).
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    C_RAIN, "dry_days", "flood_days", C_MEANT, C_MAXT, C_HEAT, C_HUM, C_SUN,
    C_N, C_P, C_K, C_PH, C_SOC, C_IRR_PCT, C_GW,
]
# State is known at recommendation time and can generalize to unseen districts.
# District itself is deliberately excluded because the primary test holds out
# entire districts and an unseen district has no train-fitted vocabulary.
CATEGORICAL_FEATURES = [C_STATE, C_SEASON, C_TEX, C_IRR_SRC]
TARGET = C_CROP

FORBIDDEN_PREDICTORS = [
    C_AREA, C_PROD, C_YIELD, C_FERT, C_HYV, C_SOW, C_PEST,
    C_PRICE, C_MSP, C_COST, C_TRANS, C_YEAR,
]

# Ranking weights (paper: w_s, w_y, w_p, w_m, w_r) -- disclosed project parameters
RANKING_WEIGHTS = {
    "suitability": 0.35,
    "yield": 0.15,
    "profit": 0.30,
    "market": 0.10,
    "risk": 0.10,
}
