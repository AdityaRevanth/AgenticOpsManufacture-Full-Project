"""
Central configuration for the Plant Predictive Maintenance system.
All values are read from environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq / LLM model ---
# NOTE: llama-3.3-70b-versatile was deprecated by Groq (announced 2026-06-17,
# decommissioned 2026-08-16). Default is now openai/gpt-oss-120b, Groq's
# recommended replacement. Override via GROQ_MODEL in .env if Groq deprecates
# this one too — no code changes needed elsewhere, everything reads from here.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# LiteLLM-style model string used by CrewAI
CREWAI_LLM_MODEL = f"groq/{GROQ_MODEL}"

# --- Qdrant ---
# Embedded/local mode by default (no server, no Docker needed): Qdrant runs
# in-process and persists to this folder. Set QDRANT_URL to a real server
# ("http://localhost:6333" or a Qdrant Cloud URL) to use a server instead.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_local")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "failure_records")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "plant-anomaly-detection")

# --- Anomaly model ---
ANOMALY_MODEL_PATH = os.getenv("ANOMALY_MODEL_PATH", "ml/artifacts/isolation_forest.joblib")
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
ANOMALY_SCORE_THRESHOLD = float(os.getenv("ANOMALY_SCORE_THRESHOLD", "0.55"))

# --- FastAPI ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Sensor feature columns used across ingestion / ml / agents ---
SENSOR_FEATURES = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_psi",
    "rpm",
    "current_amps",
]
