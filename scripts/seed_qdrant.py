"""
Convenience script to bootstrap the system:
  1. Trains the anomaly model on sample sensor data (tracked in MLflow)
  2. Embeds and upserts historical failure records into Qdrant

Run once after `docker-compose up -d qdrant mlflow`:
    python scripts/seed_qdrant.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train import main as train_main
from ingestion.failure_records import seed as seed_failures

if __name__ == "__main__":
    print("== Step 1/2: training anomaly model ==")
    train_main("data/sample_sensor_data.csv", contamination=0.05)

    print("\n== Step 2/2: seeding Qdrant with historical failure records ==")
    seed_failures("data/sample_failure_records.json")

    print("\nSetup complete. You can now run:")
    print("  uvicorn api.main:app --reload          # HTTP API")
    print("  streamlit run dashboard/streamlit_app.py # dashboard")
    print("  python mcp_server/server.py              # MCP server (stdio)")
