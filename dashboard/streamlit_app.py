"""
Streamlit dashboard for the Plant Predictive Maintenance system.

Uses the Model Context Protocol (MCP) to communicate with the
FastMCP server and invoke the equipment diagnosis pipeline.

Run:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys

# Add the project root to Python's import path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

import streamlit as st

from mcp_client.client import analyze_equipment


st.set_page_config(
    page_title="Plant Predictive Maintenance",
    layout="wide",
)

st.title("🏭 Plant Predictive Maintenance")

st.caption(
    "Multi-agent anomaly detection, historical failure retrieval, "
    "and maintenance planning via Model Context Protocol."
)

with st.expander("🔌 MCP Connection", expanded=True):

    col1, col2, col3 = st.columns(3)

    col1.success("🟢 Connected")
    col2.info("Protocol: Model Context Protocol")
    col3.info("Server: FastMCP")

    st.caption(
        "The dashboard communicates with the diagnostic pipeline "
        "through an MCP client and FastMCP server."
    )


EQUIPMENT_PRESETS = {
    "PUMP-101 (Centrifugal Pump)": (
        "PUMP-101",
        "Centrifugal Pump",
        62.0,
        2.4,
        105.0,
        1780.0,
        14.0,
    ),
    "MTR-204 (Induction Motor)": (
        "MTR-204",
        "Induction Motor",
        58.0,
        2.1,
        100.0,
        1600.0,
        18.0,
    ),
    "CMP-317 (Rotary Compressor)": (
        "CMP-317",
        "Rotary Compressor",
        65.0,
        2.6,
        95.0,
        1500.0,
        20.0,
    ),
    "FAN-088 (Exhaust Fan)": (
        "FAN-088",
        "Exhaust Fan",
        60.0,
        2.0,
        90.0,
        1700.0,
        12.0,
    ),
}


with st.sidebar:

    st.header("Sensor Input")

    preset_label = st.selectbox(
        "Equipment",
        list(EQUIPMENT_PRESETS.keys()),
    )

    (
        eq_id,
        eq_type,
        dt,
        dv,
        dp,
        drpm,
        dc,
    ) = EQUIPMENT_PRESETS[preset_label]

    equipment_id = st.text_input(
        "Equipment ID",
        value=eq_id,
    )

    equipment_type = st.text_input(
        "Equipment Type",
        value=eq_type,
    )

    temperature_c = st.slider(
        "Temperature (°C)",
        20.0,
        130.0,
        dt,
        0.5,
    )

    vibration_mm_s = st.slider(
        "Vibration (mm/s)",
        0.0,
        15.0,
        dv,
        0.1,
    )

    pressure_psi = st.slider(
        "Pressure (psi)",
        0.0,
        200.0,
        dp,
        1.0,
    )

    rpm = st.slider(
        "RPM",
        0.0,
        3600.0,
        drpm,
        10.0,
    )

    current_amps = st.slider(
        "Current (A)",
        0.0,
        60.0,
        dc,
        0.5,
    )

    run_btn = st.button(
        "Run Diagnostic Pipeline",
        type="primary",
        use_container_width=True,
    )


if run_btn:

    with st.spinner(
        "Calling MCP server and running diagnostic pipeline..."
    ):

        try:

            result = analyze_equipment(
                equipment_id,
                equipment_type,
                temperature_c,
                vibration_mm_s,
                pressure_psi,
                rpm,
                current_amps,
            )

        except Exception as e:

            st.error(f"MCP pipeline call failed: {e}")
            st.stop()


    anomaly = result.get("anomaly", {})
    status = result.get("status", "unknown")


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Status",
        status.upper(),
    )

    col2.metric(
        "Risk Score",
        f"{anomaly.get('risk_score', 0):.2f}",
    )

    col3.metric(
        "Priority",
        result.get("priority", "n/a").upper(),
    )


    st.divider()


    if status == "anomalous":

        st.subheader("🔍 Diagnosis")

        st.write(
            result.get(
                "diagnosis",
                "No diagnosis returned.",
            )
        )


        st.subheader("📚 Similar Historical Failures")

        matches = result.get(
            "historical_matches",
            [],
        )


        if matches:

            for match in matches:

                title = match.get(
                    "title",
                    "Historical failure",
                )

                similarity = match.get(
                    "similarity",
                    0,
                )

                with st.expander(
                    f"{title} — similarity {similarity}"
                ):

                    st.write(
                        f"**Root cause:** "
                        f"{match.get('root_cause', '—')}"
                    )

                    st.write(
                        f"**Severity:** "
                        f"{match.get('severity', '—')}"
                    )

                    symptoms = match.get(
                        "symptoms",
                        [],
                    )

                    st.write(
                        f"**Symptoms:** "
                        f"{', '.join(symptoms)}"
                    )

                    st.write(
                        match.get(
                            "description",
                            "",
                        )
                    )

        else:

            st.info(
                "No closely matching historical "
                "records found."
            )


        st.subheader(
            "🛠️ Recommended Maintenance Actions"
        )

        actions = result.get(
            "recommended_actions",
            [],
        )


        if actions:

            for index, action in enumerate(
                actions,
                1,
            ):

                st.markdown(
                    f"**{index}. "
                    f"{action.get('action', '—')}**"
                )

                st.caption(
                    f"Reason: "
                    f"{action.get('reason', '—')}"
                )

                parts = action.get(
                    "parts_needed",
                    [],
                )

                if parts:

                    st.caption(
                        f"Parts needed: "
                        f"{', '.join(parts)}"
                    )

        else:

            st.info(
                "No structured actions returned."
            )


    else:

        st.success(
            "Equipment readings are within the "
            "normal operating envelope. "
            "No action needed."
        )


    with st.expander("Raw MCP pipeline output"):

        st.json(result)


else:

    st.info(
        "Adjust sensor values in the sidebar "
        "and click **Run Diagnostic Pipeline**."
    )