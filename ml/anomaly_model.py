"""
Anomaly detection model for equipment sensor readings.

Uses an IsolationForest trained per-fleet on normalized sensor features.
Training runs are tracked in MLflow (params, metrics, and the model artifact).
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENSOR_FEATURES, ANOMALY_MODEL_PATH, ANOMALY_CONTAMINATION


class AnomalyModel:
    """Wraps a StandardScaler + IsolationForest pair as a single artifact."""

    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        X = df[SENSOR_FEATURES].values
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.is_fitted = True
        return self

    def score(self, readings: dict) -> dict:
        """
        readings: dict of {feature_name: value} for a single point-in-time reading.
        Returns anomaly score in [0, 1] (higher = more anomalous) plus raw decision info.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyModel is not fitted / loaded yet.")

        x = np.array([[readings.get(f, 0.0) for f in SENSOR_FEATURES]])
        xs = self.scaler.transform(x)

        # decision_function: higher = more normal. We invert + normalize to [0,1].
        raw_score = self.model.decision_function(xs)[0]
        is_outlier = self.model.predict(xs)[0] == -1

        # squash raw_score (~[-0.5, 0.5]) into a 0-1 "risk" score
        risk_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))

        return {
            "risk_score": round(risk_score, 4),
            "is_anomalous": bool(is_outlier),
            "raw_decision_function": round(float(raw_score), 4),
        }

    def save(self, path: str = ANOMALY_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model, "is_fitted": self.is_fitted}, path)

    @classmethod
    def load(cls, path: str = ANOMALY_MODEL_PATH) -> "AnomalyModel":
        obj = cls()
        data = joblib.load(path)
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        obj.is_fitted = data["is_fitted"]
        return obj


_MODEL_CACHE = None


def get_model() -> AnomalyModel:
    """Lazily load (and cache) the trained anomaly model for use by agents/API."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if os.path.exists(ANOMALY_MODEL_PATH):
            _MODEL_CACHE = AnomalyModel.load(ANOMALY_MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"No trained model found at {ANOMALY_MODEL_PATH}. Run `python ml/train.py` first."
            )
    return _MODEL_CACHE
