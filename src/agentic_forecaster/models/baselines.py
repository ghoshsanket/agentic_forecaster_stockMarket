"""Classical baselines: Random Forest, Logistic Regression, Majority."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class RandomForestBaseline:
    def __init__(self, n_estimators: int = 300, max_depth: int = 8,
                 min_samples_leaf: int = 5, class_weight: str = "balanced",
                 seed: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestBaseline:
        self.model.fit(X.reshape(len(X), -1), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X.reshape(len(X), -1))


class LogisticRegressionBaseline:
    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 class_weight: str = "balanced", seed: int = 42):
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, class_weight=class_weight, random_state=seed
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionBaseline:
        self.model.fit(X.reshape(len(X), -1), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X.reshape(len(X), -1))


class MajorityBaseline:
    """Predicts the majority class from training data with calibrated prob."""

    def __init__(self):
        self.majority_class: int = 0
        self.prob: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> MajorityBaseline:
        counts = np.bincount(y.astype(int), minlength=2)
        self.majority_class = int(np.argmax(counts))
        self.prob = float(counts[self.majority_class] / counts.sum())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        proba = np.zeros((n, 2), dtype=np.float64)
        proba[:, self.majority_class] = self.prob
        proba[:, 1 - self.majority_class] = 1.0 - self.prob
        return proba
