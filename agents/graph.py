"""
LangGraph StateGraph that orchestrates the end-to-end pipeline:

  ingest -> anomaly_detect -> [conditional] -> retrieve_history -> diagnose_and_plan -> finalize
                                    (normal branch skips straight to finalize)

The heavy multi-agent reasoning (diagnosis + planning) is delegated to a
CrewAI crew inside the `diagnose_and_plan` node, so LangGraph handles control
flow / branching while CrewAI handles collaborative reasoning.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.graph import StateGraph, END

from agents.state import PlantState
from agents.tools import score_anomaly, retrieve_historical_matches
from agents.crew import run_diagnosis_crew
from config import ANOMALY_SCORE_THRESHOLD


def node_ingest(state: PlantState) -> PlantState:
    if not state.get("readings"):
        return {**state, "error": "No sensor readings provided."}
    return state


def node_anomaly_detect(state: PlantState) -> PlantState:
    try:
        anomaly = score_anomaly(state["readings"])
    except FileNotFoundError as e:
        return {**state, "error": str(e)}
    status = "anomalous" if (anomaly["is_anomalous"] or anomaly["risk_score"] >= ANOMALY_SCORE_THRESHOLD) else "normal"
    return {**state, "anomaly": anomaly, "status": status}


def node_retrieve_history(state: PlantState) -> PlantState:
    matches = retrieve_historical_matches(
        equipment_type=state.get("equipment_type", "Unknown"),
        readings=state["readings"],
        anomaly_info=state["anomaly"],
        top_k=3,
    )
    return {**state, "historical_matches": matches}


def node_diagnose_and_plan(state: PlantState) -> PlantState:
    result = run_diagnosis_crew(
        equipment_id=state["equipment_id"],
        equipment_type=state.get("equipment_type", "Unknown"),
        readings=state["readings"],
        anomaly=state["anomaly"],
        historical_matches=state.get("historical_matches", []),
    )
    plan = result["plan"]
    return {
        **state,
        "diagnosis": result["diagnosis"],
        "recommended_actions": plan.get("actions", []),
        "priority": plan.get("priority", "medium"),
    }


def node_finalize_normal(state: PlantState) -> PlantState:
    return {
        **state,
        "diagnosis": "Readings are within normal operating envelope. No anomaly detected.",
        "recommended_actions": [],
        "priority": "none",
    }


def route_after_anomaly(state: PlantState) -> str:
    if state.get("error"):
        return "end"
    return "anomalous" if state.get("status") == "anomalous" else "normal"


def build_graph():
    graph = StateGraph(PlantState)

    graph.add_node("ingest", node_ingest)
    graph.add_node("anomaly_detect", node_anomaly_detect)
    graph.add_node("retrieve_history", node_retrieve_history)
    graph.add_node("diagnose_and_plan", node_diagnose_and_plan)
    graph.add_node("finalize_normal", node_finalize_normal)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "anomaly_detect")

    graph.add_conditional_edges(
        "anomaly_detect",
        route_after_anomaly,
        {
            "anomalous": "retrieve_history",
            "normal": "finalize_normal",
            "end": END,
        },
    )

    graph.add_edge("retrieve_history", "diagnose_and_plan")
    graph.add_edge("diagnose_and_plan", END)
    graph.add_edge("finalize_normal", END)

    return graph.compile()


_COMPILED_GRAPH = None


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def analyze_equipment(equipment_id: str, equipment_type: str, readings: dict) -> dict:
    """Main entry point used by both the MCP server and the FastAPI layer."""
    graph = get_graph()
    initial_state: PlantState = {
        "equipment_id": equipment_id,
        "equipment_type": equipment_type,
        "readings": readings,
    }
    final_state = graph.invoke(initial_state)
    return dict(final_state)
