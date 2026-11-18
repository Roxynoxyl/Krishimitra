# Stage B: leakage-safe preprocessing.
# Numeric values are median-imputed with explicit missingness indicators.
# Categorical values are one-hot encoded. All vocabularies/statistics are fit
# on training data only (Project_Paper.docx section 4.3).
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


class Preprocessor:
    """Train-fitted numeric imputation + missing flags + one-hot encoding."""

    def __init__(self):
        self.num_medians: dict[str, float] = {}
        self.cat_vocab: dict[str, list[str]] = {}
        self.feature_names: list[str] = []
        self.num_mean: np.ndarray | None = None
        self.num_std: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, standardize: bool = True) -> "Preprocessor":
        self.num_medians = {
            c: float(pd.to_numeric(df[c], errors="coerce").median())
            if pd.to_numeric(df[c], errors="coerce").notna().any() else 0.0
            for c in cfg.NUMERIC_FEATURES
        }
        self.cat_vocab = {
            c: sorted(df[c].dropna().astype(str).unique().tolist())
            for c in cfg.CATEGORICAL_FEATURES
        }
        self.feature_names = self._feature_names()
        X = self._transform_raw(df, standardize=False)
        if standardize:
            # Standardize only the continuous value columns and leave missing
            # flags/one-hot columns interpretable and bounded.
            n_num = len(cfg.NUMERIC_FEATURES)
            self.num_mean = X[:, :n_num].mean(axis=0)
            self.num_std = X[:, :n_num].std(axis=0)
            self.num_std[self.num_std == 0] = 1.0
        return self

    def _feature_names(self) -> list[str]:
        names = []
        for c in cfg.NUMERIC_FEATURES:
            names.extend([c, f"{c}__missing"])
        for c in cfg.CATEGORICAL_FEATURES:
            names.extend([f"{c}={v}" for v in self.cat_vocab.get(c, [])])
            names.append(f"{c}=__unknown__")
        return names

    def _transform_raw(self, df: pd.DataFrame, standardize: bool = True) -> np.ndarray:
        cols = []
        for j, c in enumerate(cfg.NUMERIC_FEATURES):
            raw = pd.to_numeric(df[c], errors="coerce")
            missing = raw.isna().to_numpy(np.float64)
            values = raw.fillna(self.num_medians[c]).to_numpy(np.float64)
            if standardize and self.num_mean is not None:
                values = (values - self.num_mean[j]) / self.num_std[j]
            cols.extend([values, missing])

        for c in cfg.CATEGORICAL_FEATURES:
            values = df[c].astype("string")
            vocab = self.cat_vocab.get(c, [])
            for v in vocab:
                cols.append((values == v).to_numpy(np.float64))
            cols.append((~values.isin(vocab) | values.isna()).to_numpy(np.float64))
        return np.column_stack(cols).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._transform_raw(df, standardize=True)

    def save(self, path: Path) -> None:
        payload = {
            "num_medians": self.num_medians,
            "cat_vocab": self.cat_vocab,
            "num_mean": self.num_mean.tolist() if self.num_mean is not None else None,
            "num_std": self.num_std.tolist() if self.num_std is not None else None,
            "feature_names": self.feature_names,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        pp = cls()
        pp.num_medians = payload["num_medians"]
        pp.cat_vocab = payload["cat_vocab"]
        pp.num_mean = np.array(payload["num_mean"]) if payload["num_mean"] else None
        pp.num_std = np.array(payload["num_std"]) if payload["num_std"] else None
        pp.feature_names = payload["feature_names"]
        return pp
