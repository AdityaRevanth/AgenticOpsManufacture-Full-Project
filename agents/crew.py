"""
CrewAI crew that collaborates on root-cause diagnosis and maintenance planning
once the LangGraph pipeline has determined an equipment reading is anomalous.

Two agents:
  1. Diagnostician  - correlates live sensor deviations with retrieved historical
                       failure records to propose a root-cause hypothesis.
  2. Maintenance Planner - turns the diagnosis into concrete, prioritized
                       maintenance actions with an urgency rating.
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CREWAI_LLM_MODEL
from agents.tools import format_readings_for_prompt, format_history_for_prompt

from crewai import Agent, Task, Crew, Process, LLM


def _build_llm() -> LLM:
    # CrewAI's LLM wrapper uses LiteLLM under the hood; "groq/<model>" routes to Groq.
    return LLM(model=CREWAI_LLM_MODEL, temperature=0.2)


def build_crew(equipment_id: str, equipment_type: str, readings: dict, anomaly: dict, historical_matches: list):
    llm = _build_llm()

    diagnostician = Agent(
        role="Reliability Diagnostician",
        goal="Determine the most likely root cause of an equipment anomaly using live sensor "
             "deviations and similar historical failure records.",
        backstory="A senior reliability engineer with two decades of rotating-equipment "
                   "failure analysis experience across pumps, motors, compressors, and fans.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    planner = Agent(
        role="Maintenance Planner",
        goal="Convert a root-cause diagnosis into a concrete, prioritized maintenance action plan "
             "that a plant engineer can execute immediately.",
        backstory="A CMMS-savvy maintenance planner who translates diagnoses into work orders, "
                   "parts lists, and urgency ratings.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    readings_block = format_readings_for_prompt(readings)
    history_block = format_history_for_prompt(historical_matches)

    diagnose_task = Task(
        description=(
            f"Equipment {equipment_id} ({equipment_type}) has been flagged as anomalous.\n\n"
            f"Live sensor readings:\n{readings_block}\n\n"
            f"Anomaly model output: risk_score={anomaly.get('risk_score')}, "
            f"is_anomalous={anomaly.get('is_anomalous')}\n\n"
            f"Similar historical failure records retrieved from the maintenance archive:\n{history_block}\n\n"
            "Write a concise (4-6 sentence) root-cause hypothesis. Explicitly reference which "
            "sensor deviations and which historical record(s), if any, support your hypothesis. "
            "If no historical record is a strong match, say so and reason from first principles."
        ),
        expected_output="A short root-cause diagnosis paragraph.",
        agent=diagnostician,
    )

    plan_task = Task(
        description=(
            "Using the diagnosis from the previous task, produce a maintenance action plan.\n"
            "Respond ONLY with valid JSON (no markdown fences, no preamble) matching this schema:\n"
            "{\n"
            '  "priority": "low" | "medium" | "high" | "critical",\n'
            '  "estimated_time_to_failure": "<free text estimate>",\n'
            '  "actions": [\n'
            '    {"action": "<specific task>", "reason": "<why>", "parts_needed": ["..."]}\n'
            "  ],\n"
            '  "recommended_response_window": "<e.g. within 24 hours>"\n'
            "}\n"
            "Provide 2-4 actions, ordered by execution sequence."
        ),
        expected_output="A JSON object matching the schema exactly.",
        agent=planner,
        context=[diagnose_task],
    )

    crew = Crew(
        agents=[diagnostician, planner],
        tasks=[diagnose_task, plan_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, diagnose_task, plan_task


def run_diagnosis_crew(equipment_id: str, equipment_type: str, readings: dict, anomaly: dict, historical_matches: list) -> dict:
    """Kicks off the crew and returns a structured dict with diagnosis + plan."""
    crew, diagnose_task, plan_task = build_crew(equipment_id, equipment_type, readings, anomaly, historical_matches)
    crew.kickoff()

    diagnosis_text = str(diagnose_task.output.raw) if diagnose_task.output else ""
    plan_raw = str(plan_task.output.raw) if plan_task.output else "{}"

    # Defensive parsing: strip accidental markdown fences, fall back gracefully.
    cleaned = plan_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        plan = {
            "priority": "medium",
            "estimated_time_to_failure": "unknown",
            "actions": [{"action": "Manual review required — planner output was not valid JSON.",
                         "reason": cleaned[:300], "parts_needed": []}],
            "recommended_response_window": "as soon as possible",
        }

    return {"diagnosis": diagnosis_text, "plan": plan}
