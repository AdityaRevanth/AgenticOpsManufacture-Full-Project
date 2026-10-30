"""
Lightweight sensor ingestion helpers.

In production this would subscribe to an MQTT/OPC-UA/Kafka feed from the
plant historian. For this system it normalizes incoming readings (dict,
CSV row, or batch DataFrame) into the schema the agents expect.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SENSOR_FEATURES


def normalize_reading(raw: dict) -> dict:
    """Coerce an incoming reading dict to the canonical feature schema, filling gaps with None."""
    out = {"equipment_id": raw.get("equipment_id"), "timestamp": raw.get("timestamp")}
    for f in SENSOR_FEATURES:
        out[f] = float(raw[f]) if f in raw and raw[f] is not None else None
    return out


def load_batch_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SENSOR_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required sensor columns: {missing}")
    return df


def latest_reading_for_equipment(df: pd.DataFrame, equipment_id: str) -> dict:
    """Pull the most recent row for a given equipment_id from a batch DataFrame."""
    sub = df[df["equipment_id"] == equipment_id]
    if sub.empty:
        raise ValueError(f"No readings found for equipment_id={equipment_id}")
    sub = sub.sort_values("timestamp")
    row = sub.iloc[-1].to_dict()
    return normalize_reading(row)
