"""
FastAPI service exposing the same multi-agent pipeline over plain HTTP,
for the Streamlit dashboard and any other non-MCP client.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.graph import analyze_equipment
from agents.tools import score_anomaly
from ingestion.failure_records import search_similar_failures
from config import API_HOST, API_PORT

app = FastAPI(
    title="Plant Predictive Maintenance API",
    description="Multi-agent pipeline for equipment anomaly detection and maintenance planning.",
    version="1.0.0",
)


class SensorReading(BaseModel):
    equipment_id: str
    equipment_type: str
    temperature_c: float
    vibration_mm_s: float
    pressure_psi: float
    rpm: float = Field(..., description="Shaft speed in rotations per minute")
    current_amps: float


class HistoryQuery(BaseModel):
    query: str
    equipment_type: str | None = None
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(reading: SensorReading):
    readings = reading.model_dump(exclude={"equipment_id", "equipment_type"})
    try:
        result = analyze_equipment(reading.equipment_id, reading.equipment_type, readings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result["readings"] = readings
    return result


@app.post("/anomaly_score")
def anomaly_score(reading: SensorReading):
    readings = reading.model_dump(exclude={"equipment_id", "equipment_type"})
    try:
        return score_anomaly(readings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/failure_history")
def failure_history(q: HistoryQuery):
    try:
        return search_similar_failures(q.query, equipment_type=q.equipment_type, top_k=q.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
