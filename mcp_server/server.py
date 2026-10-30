"""
FastMCP server exposing the plant predictive-maintenance pipeline as MCP tools,
callable directly from Claude Desktop, Claude Code, or Cursor.

Run:
    python mcp_server/server.py                # stdio transport (default, for Claude Desktop/Cursor)
    python mcp_server/server.py --transport sse --port 8765   # SSE transport, for remote clients
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from ingestion.sensor_ingest import load_batch_csv, latest_reading_for_equipment
from ingestion.failure_records import search_similar_failures, seed as seed_qdrant
from agents.graph import analyze_equipment
from agents.tools import score_anomaly
from config import SENSOR_FEATURES

mcp = FastMCP(
    name="plant-predictive-maintenance",
    instructions=(
        "Tools for diagnosing plant equipment health from live sensor readings. "
        "Use `analyze_equipment` for a full anomaly-detection + RAG + multi-agent "
        "diagnosis pass. Use `get_anomaly_score` for a quick risk check only. "
        "Use `query_failure_history` to search past incident records directly."
    ),
)


@mcp.tool()
def analyze_equipment_readings(
    equipment_id: str,
    equipment_type: str,
    temperature_c: float,
    vibration_mm_s: float,
    pressure_psi: float,
    rpm: float,
    current_amps: float,
) -> dict:
    """
    Run the full multi-agent pipeline on a single equipment sensor reading:
    anomaly scoring -> historical failure retrieval (RAG) -> root-cause diagnosis
    -> prioritized maintenance action plan.

    Returns a dict with: status, anomaly (risk score), historical_matches,
    diagnosis, recommended_actions, priority.
    """
    readings = {
        "temperature_c": temperature_c,
        "vibration_mm_s": vibration_mm_s,
        "pressure_psi": pressure_psi,
        "rpm": rpm,
        "current_amps": current_amps,
    }
    result = analyze_equipment(equipment_id, equipment_type, readings)
    result.pop("readings", None)  # avoid echoing input back verbatim
    return result


@mcp.tool()
def analyze_equipment_from_csv(equipment_id: str, equipment_type: str, csv_path: str = "data/sample_sensor_data.csv") -> dict:
    """
    Pull the most recent sensor reading for `equipment_id` from a CSV batch file
    (e.g. an export from the plant historian) and run the full diagnostic pipeline on it.
    """
    df = load_batch_csv(csv_path)
    reading = latest_reading_for_equipment(df, equipment_id)
    readings = {f: reading[f] for f in SENSOR_FEATURES}
    result = analyze_equipment(equipment_id, equipment_type, readings)
    result.pop("readings", None)
    return result


@mcp.tool()
def get_anomaly_score(
    temperature_c: float,
    vibration_mm_s: float,
    pressure_psi: float,
    rpm: float,
    current_amps: float,
) -> dict:
    """
    Quick anomaly check only (no historical retrieval or diagnosis) — returns a
    risk_score in [0,1] and an is_anomalous flag from the IsolationForest model.
    Useful for lightweight polling/dashboard checks.
    """
    readings = {
        "temperature_c": temperature_c,
        "vibration_mm_s": vibration_mm_s,
        "pressure_psi": pressure_psi,
        "rpm": rpm,
        "current_amps": current_amps,
    }
    return score_anomaly(readings)


@mcp.tool()
def query_failure_history(query: str, equipment_type: str = "", top_k: int = 3) -> list:
    """
    Semantic search over historical failure records stored in Qdrant.
    `query` can be a free-text description of symptoms (e.g. "rising vibration and bearing heat").
    Optionally filter by `equipment_type` (e.g. "Centrifugal Pump").
    """
    return search_similar_failures(query, equipment_type=equipment_type or None, top_k=top_k)


@mcp.tool()
def seed_failure_history(records_path: str = "data/sample_failure_records.json") -> str:
    """
    Admin/setup tool: embeds and upserts failure records from a JSON file into the
    Qdrant `failure_records` collection. Run once before using the other tools.
    """
    seed_qdrant(records_path)
    return f"Seeded failure records from {records_path} into Qdrant."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, port=args.port)
