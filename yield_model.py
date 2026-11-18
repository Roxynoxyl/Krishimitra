# Optional economic component: yield prediction with a temporal holdout.
# This is separate from crop classification: historical Yield is an outcome,
# never a crop-classifier predictor.
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from src.preprocess import Preprocessor


def _metrics(y_true, y_pred):
    err = y_true - y_pred
    return {
        "mae_t_ha": round(float(np.mean(np.abs(err))), 4),
        "rmse_t_ha": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "n": int(len(y_true)),
    }


def train_yield_model(df: pd.DataFrame, report_path=None):
    """Train RF regressor on earlier years, test on latest years.

    Crop is included as a one-hot input because yield is conditional on crop;
    area/production are excluded to prevent direct outcome leakage.
    """
    from sklearn.ensemble import RandomForestRegressor

    d = df.copy()
    d["__target"] = pd.to_numeric(d[cfg.C_YIELD], errors="coerce")
    d = d.dropna(subset=["__target", cfg.C_YEAR, cfg.C_CROP])
    d["__crop_feature"] = d[cfg.C_CROP].astype(str)
    original_cats = cfg.CATEGORICAL_FEATURES
    cfg.CATEGORICAL_FEATURES = original_cats + ["__crop_feature"]
    try:
        cutoff = int(d[cfg.C_YEAR].quantile(0.8))
        train = d[d[cfg.C_YEAR] < cutoff]
        test = d[d[cfg.C_YEAR] >= cutoff]
        pp = Preprocessor().fit(train)
        X_train, X_test = pp.transform(train), pp.transform(test)
        y_train, y_test = train["__target"].to_numpy(), test["__target"].to_numpy()
        model = RandomForestRegressor(
            n_estimators=150, max_depth=20, min_samples_leaf=5,
            n_jobs=-1, random_state=cfg.SEED,
        ).fit(X_train, y_train)
        pred = model.predict(X_test)
        report = {
            "target": cfg.C_YIELD,
            "train_year_max": int(train[cfg.C_YEAR].max()),
            "test_year_min": int(test[cfg.C_YEAR].min()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "metrics": _metrics(y_test, pred),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "Temporal holdout; economic estimates remain separate from classifier evaluation.",
        }
        if report_path:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return model, pp, report
    finally:
        cfg.CATEGORICAL_FEATURES = original_cats
