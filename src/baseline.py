"""Baseline: a normal classifier whose low confidence is used as the novelty score."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class ConfidenceBaseline:
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y):
        self.model = HistGradientBoostingClassifier(random_state=self.seed).fit(X, y)
        return self

    def predict(self, X, normal_code):
        """Return (novelty s(x) = 1 - max probability, predicted class code, 1 - P(Normal))."""
        proba = self.model.predict_proba(X)
        p_normal = proba[:, list(self.model.classes_).index(normal_code)]
        return 1.0 - proba.max(axis=1), self.model.classes_[proba.argmax(axis=1)], 1.0 - p_normal

    def info(self):
        return {"n_iter": int(self.model.n_iter_), "classes": [int(c) for c in self.model.classes_]}
