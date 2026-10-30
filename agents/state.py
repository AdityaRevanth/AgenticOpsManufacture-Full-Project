"""
Shared state object passed between LangGraph nodes in the plant-maintenance pipeline.
"""
from typing import TypedDict, Optional


class PlantState(TypedDict, total=False):
    equipment_id: str
    equipment_type: str
    readings: dict                 # raw sensor reading dict
    anomaly: dict                  # output of score_anomaly()
    historical_matches: list       # output of retrieve_historical_matches()
    diagnosis: str                 # Diagnostician agent's root-cause narrative
    recommended_actions: list      # Maintenance Planner agent's structured actions
    priority: str                  # "low" | "medium" | "high" | "critical"
    status: str                    # "normal" | "anomalous"
    error: Optional[str]
