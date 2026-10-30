"""
Train the IsolationForest anomaly model on historical sensor data and
log the run (params, metrics, artifact) to MLflow.

Usage:
    python ml/train.py --data data/sample_sensor_data.csv
"""
import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SENSOR_FEATURES,
    ANOMALY_MODEL_PATH,
    ANOMALY_CONTAMINATION,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from ml.anomaly_model import AnomalyModel


def main(data_path: str, contamination: float):
    df = pd.read_csv(data_path)
    missing = [c for c in SENSOR_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    mlflow = None
    try:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow_enabled = True
    except ImportError:
        print("[warn] mlflow is not installed. Training without experiment tracking.")
        mlflow_enabled = False
    except Exception as e:
        print(f"[warn] Could not reach MLflow tracking server ({e}). Training without tracking.")
        mlflow_enabled = False

    def _train_and_log():
        model = AnomalyModel(contamination=contamination)
        model.fit(df)

        # quick internal eval: score the training set itself
        preds = [model.score(row) for row in df[SENSOR_FEATURES].to_dict(orient="records")]
        anomaly_rate = sum(p["is_anomalous"] for p in preds) / len(preds)
        avg_risk = sum(p["risk_score"] for p in preds) / len(preds)

        if mlflow_enabled:
            mlflow.log_param("contamination", contamination)
            mlflow.log_param("n_estimators", model.model.n_estimators)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("features", ",".join(SENSOR_FEATURES))
            mlflow.log_metric("train_anomaly_rate", anomaly_rate)
            mlflow.log_metric("train_avg_risk_score", avg_risk)

        model.save(ANOMALY_MODEL_PATH)

        if mlflow_enabled:
            mlflow.log_artifact(ANOMALY_MODEL_PATH, artifact_path="model")

        print(f"Trained on {len(df)} rows | anomaly_rate={anomaly_rate:.3f} | "
              f"avg_risk={avg_risk:.3f}")
        print(f"Model saved to {ANOMALY_MODEL_PATH}")

    if mlflow_enabled:
        with mlflow.start_run(run_name="isolation_forest_train"):
            _train_and_log()
    else:
        _train_and_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_sensor_data.csv")
    parser.add_argument("--contamination", type=float, default=ANOMALY_CONTAMINATION)
    args = parser.parse_args()
    main(args.data, args.contamination)
