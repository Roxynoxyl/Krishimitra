# Stage C: suitability models.
# Baselines per the paper: Random Forest and XGBoost when available (sklearn/xgboost
# importable); otherwise a pure-numpy Gaussian Naive Bayes fallback ("gaussiannb")
# so the pipeline always runs. Model selection is by validation evidence only.
import pickle
from pathlib import Path

import numpy as np

import config as cfg


# ---------------------------------------------------------------------------
# Metrics (pure numpy so no sklearn dependency)
# ---------------------------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> float:
    f1s = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int = 3) -> float:
    k = min(k, proba.shape[1])
    top = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    hits = [y_true[i] in classes[top[i]] for i in range(len(y_true))]
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# Gaussian Naive Bayes (pure numpy baseline)
# ---------------------------------------------------------------------------
class GaussianNB:
    name = "gaussiannb"

    def __init__(self, var_smoothing: float = 1e-6):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None   # (C, F) per-class means
        self.var_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        classes = np.unique(y)
        self.classes_ = classes
        C, F = len(classes), X.shape[1]
        self.theta_ = np.zeros((C, F))
        self.var_ = np.zeros((C, F))
        self.class_prior_ = np.zeros(C)
        eps = self.var_smoothing * X.var(axis=0).max()
        for i, c in enumerate(classes):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + eps
            self.class_prior_[i] = len(Xc) / len(X)
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        logprior = np.log(self.class_prior_)
        # per-class Gaussian log-likelihood, vectorized over chunks
        out = np.empty((X.shape[0], len(self.classes_)), dtype=np.float64)
        for i in range(len(self.classes_)):
            t, v = self.theta_[i], self.var_[i]
            ll = -0.5 * np.sum(np.log(2 * np.pi * v)) - 0.5 * (((X - t) ** 2) / v).sum(axis=1)
            out[:, i] = ll + logprior[i]
        # log-sum-exp normalization
        m = out.max(axis=1, keepdims=True)
        lse = m + np.log(np.exp(out - m).sum(axis=1, keepdims=True))
        return out - lse

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Optional sklearn / xgboost models (used automatically when importable)
# ---------------------------------------------------------------------------
def _try_sklearn_rf(seed: int):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        return None
    return RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )


class RandomForestWrapper:
    name = "random_forest"

    def __init__(self, seed: int = cfg.SEED):
        self.inner = _try_sklearn_rf(seed)
        if self.inner is None:
            raise ImportError("sklearn not available")

    def fit(self, X, y):
        self.inner.fit(X, y)
        return self

    @property
    def classes_(self):
        return self.inner.classes_

    def predict_proba(self, X):
        return self.inner.predict_proba(X)

    def predict(self, X):
        return self.inner.predict(X)


def _try_xgboost(seed: int):
    """Create a deliberately bounded XGBoost multiclass baseline.

    It uses the same train/test split and feature matrix as RF/NB. The small
    depth and histogram tree method keep this first comparison practical on
    the 300k-row development cap; tuning belongs to a later experiment.
    """
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    return XGBClassifier(
        n_estimators=120,
        max_depth=8,
        learning_rate=0.10,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )


class XGBoostWrapper:
    name = "xgboost"

    def __init__(self, seed: int = cfg.SEED):
        self.inner = _try_xgboost(seed)
        if self.inner is None:
            raise ImportError("xgboost not available")
        self._classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # XGBoost requires integer labels for multiclass objectives. Keep the
        # original crop names at the wrapper boundary for a common interface.
        self._classes = np.unique(y)
        encoded = np.searchsorted(self._classes, y).astype(np.int32)
        self.inner.set_params(num_class=len(self._classes))
        self.inner.fit(X, encoded)
        return self

    @property
    def classes_(self):
        return self._classes

    def predict_proba(self, X):
        return np.asarray(self.inner.predict_proba(X))

    def predict(self, X):
        indices = np.asarray(self.inner.predict(X), dtype=np.int64)
        return self._classes[indices]


def available_models(seed: int = cfg.SEED) -> list:
    models = [GaussianNB()]
    rf = _try_sklearn_rf(seed)
    if rf is not None:
        models.append(RandomForestWrapper(seed))
    xgb = _try_xgboost(seed)
    if xgb is not None:
        models.append(XGBoostWrapper(seed))
    return models


def train_and_select(Xtr, ytr, Xva, yva, classes, seed: int = cfg.SEED,
                     selection_metric: str = "val_macro_f1"):
    """Train every available model and select by a declared validation metric."""
    if selection_metric not in {"val_macro_f1", "val_top3_accuracy", "val_accuracy"}:
        raise ValueError(f"unsupported selection metric: {selection_metric}")
    results = []
    fitted = []
    for m in available_models(seed):
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xva)
        pred = m.classes_[np.argmax(proba, axis=1)]
        res = {
            "model": m.name,
            "val_accuracy": round(accuracy(yva, pred), 4),
            "val_macro_f1": round(macro_f1(yva, pred, classes), 4),
            "val_top3_accuracy": round(topk_accuracy(yva, proba, classes, 3), 4),
        }
        results.append(res)
        fitted.append(m)
        print(f"[models] {m.name}: {res}")
    best = max(range(len(results)), key=lambda i: results[i][selection_metric])
    return fitted[best], results, results[best]


def save_model(model, path: Path, meta: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "meta": meta}, f)


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
