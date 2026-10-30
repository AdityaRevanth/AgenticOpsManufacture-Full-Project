"""
Tool functions shared across the LangGraph nodes and CrewAI agents.
Keeping these as plain functions (rather than baking API calls into agent
classes) makes them independently testable and reusable by the MCP server.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, GROQ_MODEL, SENSOR_FEATURES
from ml.anomaly_model import get_model
from ingestion.failure_records import search_similar_failures

_GROQ_CLIENT = None


def get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        from groq import Groq
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
    return _GROQ_CLIENT


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
    """Single call to Llama 3.3 on Groq's free-tier inference API."""
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def score_anomaly(readings: dict) -> dict:
    """Run the trained IsolationForest model against a single sensor reading."""
    model = get_model()
    clean = {f: readings.get(f) for f in SENSOR_FEATURES}
    return model.score(clean)


def retrieve_historical_matches(equipment_type: str, readings: dict, anomaly_info: dict, top_k: int = 3) -> list[dict]:
    """
    Build a natural-language description of the current anomalous state and use it
    to semantically search the Qdrant failure-record collection (RAG retrieval).
    """
    reading_desc = ", ".join(f"{k}={v}" for k, v in readings.items() if k in SENSOR_FEATURES and v is not None)
    query_text = (
        f"Equipment type {equipment_type} showing readings: {reading_desc}. "
        f"Anomaly risk score {anomaly_info.get('risk_score')}."
    )
    return search_similar_failures(query_text, equipment_type=equipment_type, top_k=top_k)


def format_readings_for_prompt(readings: dict) -> str:
    return "\n".join(f"- {f}: {readings.get(f)}" for f in SENSOR_FEATURES)


def format_history_for_prompt(matches: list[dict]) -> str:
    if not matches:
        return "No closely matching historical failure records were found."
    lines = []
    for m in matches:
        lines.append(
            f"- [{m.get('similarity')} similarity] {m.get('title')} "
            f"(root cause: {m.get('root_cause')}, severity: {m.get('severity')}) "
            f"— symptoms: {', '.join(m.get('symptoms', []))}"
        )
    return "\n".join(lines)
