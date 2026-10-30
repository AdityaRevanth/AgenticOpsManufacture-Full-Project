# Plant Predictive Maintenance Copilot

## An Autonomous Multi-Agent System for Equipment Anomaly Diagnosis Exposed via Model Context Protocol

An intelligent predictive-maintenance system for rotating industrial equipment such as pumps, motors, compressors, and fans.

The system combines **Isolation Forest anomaly detection**, **Qdrant Retrieval-Augmented Generation (RAG)**, **LangGraph orchestration**, **CrewAI multi-agent reasoning**, **Groq LLM inference**, and the **Model Context Protocol (MCP)**.

The system can receive live equipment sensor readings, determine whether the equipment is operating abnormally, retrieve similar historical failures, identify a probable root cause, and generate a prioritized maintenance action plan.

---

## Architecture

The primary human-facing application uses MCP:

```text
┌───────────────────────────────┐
│       Streamlit Dashboard     │
│                               │
│  Sensor Input / Results       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Python MCP Client      │
│     mcp_client/client.py      │
└───────────────┬───────────────┘
                │
                │ MCP over stdio
                ▼
┌───────────────────────────────┐
│          FastMCP Server       │
│     mcp_server/server.py      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│        Predictive Maintenance Pipeline    │
│                                           │
│  Isolation Forest → Anomaly Detection     │
│              ↓                            │
│  LangGraph → Pipeline Orchestration       │
│              ↓                            │
│  Qdrant → Historical Failure Retrieval    │
│              ↓                            │
│  CrewAI → Diagnosis + Maintenance Plan    │
│              ↓                            │
│  Groq LLM → Agent Reasoning               │
└───────────────────────────────────────────┘
```

### Alternative REST API

The project also contains a conventional FastAPI interface:

```text
HTTP Client
     │
     │ HTTP
     ▼
FastAPI
     │
     ▼
Shared Diagnostic Pipeline
```

FastAPI is an alternative interface and is **not required to run the Streamlit dashboard**.

---

## MCP Implementation

The project implements both sides of an MCP workflow:

### MCP Server

The MCP server is implemented using **FastMCP** in:

```text
mcp_server/server.py
```

It exposes the following tools:

| Tool                         | Purpose                                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| `analyze_equipment_readings` | Runs the complete diagnostic pipeline on sensor readings              |
| `analyze_equipment_from_csv` | Reads the latest equipment reading from a CSV file and runs diagnosis |
| `get_anomaly_score`          | Performs a lightweight Isolation Forest anomaly check                 |
| `query_failure_history`      | Performs semantic search over historical failures stored in Qdrant    |
| `seed_failure_history`       | Seeds historical failure records into Qdrant                          |

### MCP Client

The reusable MCP client is implemented in:

```text
mcp_client/client.py
```

The client starts the local FastMCP server using **stdio transport**, establishes an MCP session, and invokes MCP tools.

The Streamlit dashboard uses this client rather than directly calling the FastAPI endpoint.

### MCP Communication

The current implementation uses local MCP communication over standard input/output:

```text
Streamlit
    │
    ▼
mcp_client/client.py
    │
    │ starts:
    │ python mcp_server/server.py
    │
    ▼
FastMCP Server
    │
    │ stdin / stdout
    ▼
MCP Client
```

The MCP server therefore runs locally as a Python process when the MCP client establishes the connection. A separate network port is not required for this stdio-based MCP transport.

---

## Diagnostic Pipeline

### 1. Sensor Input

The system accepts five sensor features:

* Temperature
* Vibration
* Pressure
* RPM
* Motor current

Example:

```text
Equipment: PUMP-101
Type: Centrifugal Pump

Temperature: 95 °C
Vibration: 9 mm/s
Pressure: 70 psi
RPM: 1500
Current: 22 A
```

### 2. Anomaly Detection

An **Isolation Forest** model determines whether the equipment reading is anomalous.

The model uses:

```text
temperature_c
vibration_mm_s
pressure_psi
rpm
current_amps
```

The output includes:

```text
risk_score
is_anomalous
raw_decision_function
```

Normal readings avoid unnecessary downstream reasoning.

### 3. Conditional Branching

The diagnostic workflow is orchestrated using LangGraph.

```text
                 Sensor Reading
                       │
                       ▼
                Isolation Forest
                       │
                 Is anomaly?
                  /         \
                NO           YES
                │             │
                ▼             ▼
             Normal        Qdrant RAG
                              │
                              ▼
                        Historical Cases
                              │
                              ▼
                           CrewAI
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
               Diagnostician     Maintenance
                                  Planner
                     │                 │
                     └────────┬────────┘
                              ▼
                    Diagnosis + Actions
```

This conditional design prevents expensive retrieval and LLM reasoning when the equipment is operating normally.

---

## Historical Failure Retrieval

Historical failures are stored in **Qdrant** as vectorized records.

The system uses semantic search to find failures that resemble the current equipment symptoms.

For example:

```text
Query:
high vibration and bearing overheating

Equipment:
Centrifugal Pump
```

can retrieve:

```text
Bearing wear leading to high vibration and overheating
Similarity: 0.6428
```

The historical records provide:

* Failure title
* Description
* Symptoms
* Root cause
* Corrective action
* Severity
* Similarity score

---

## Multi-Agent Reasoning

When an anomaly is detected, the system uses CrewAI agents to produce a more useful maintenance response.

### Diagnostician Agent

Analyzes:

* Current sensor deviations
* Retrieved historical failures
* Equipment type
* Failure symptoms

and proposes the most likely root cause.

### Maintenance Planner Agent

Converts the diagnosis into a structured maintenance plan containing:

* Recommended action
* Reason
* Required parts
* Priority

---

## Example Anomalous Case

Example input:

```text
Equipment ID: PUMP-101
Equipment Type: Centrifugal Pump
Temperature: 95 °C
Vibration: 9 mm/s
Pressure: 70 psi
RPM: 1500
Current: 22 A
```

Example result:

```text
Risk Score: 0.582
Anomalous: true
Priority: critical
Status: anomalous
```

Historical retrieval can identify bearing wear as the strongest matching failure pattern.

The system can then produce a diagnosis such as:

```text
Likely root cause:
Bearing wear caused by lubrication breakdown.
```

and generate recommended maintenance actions such as:

```text
1. Shut down and isolate the equipment
2. Inspect bearing condition
3. Restore or replace lubrication
4. Replace damaged bearings if required
5. Realign and verify equipment operation
```

---

## Normal Operation

Normal readings follow a lightweight path.

Example:

```text
Sensor Reading
      │
      ▼
Isolation Forest
      │
      ▼
No anomaly
      │
      ▼
Normal result
```

The system does not unnecessarily perform historical retrieval or multi-agent diagnosis for a normal reading.

---

## Project Structure

```text
D:\Mini Project
│
├── agents/
│   ├── graph.py
│   ├── crew.py
│   └── tools.py
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── data/
│   ├── sample_failure_records.json
│   └── sample_sensor_data.csv
│
├── ingestion/
│   └── failure_records.py
│
├── mcp_client/
│   ├── __init__.py
│   └── client.py
│
├── mcp_server/
│   └── server.py
│
├── ml/
│   ├── train.py
│   ├── anomaly_model.py
│   └── artifacts/
│
├── scripts/
│   └── seed_qdrant.py
│
├── .env
├── .env.example
├── config.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Technologies Used

| Technology            | Role                               |
| --------------------- | ---------------------------------- |
| Python                | Core implementation                |
| Streamlit             | Human-facing dashboard             |
| MCP                   | Standardized tool communication    |
| FastMCP               | MCP server implementation          |
| Isolation Forest      | Unsupervised anomaly detection     |
| Qdrant                | Vector database and historical RAG |
| Sentence Transformers | Text embeddings                    |
| LangGraph             | Workflow orchestration             |
| CrewAI                | Multi-agent reasoning              |
| Groq                  | LLM inference                      |
| FastAPI               | Optional REST API                  |
| MLflow                | Model experiment tracking          |

---

## Setup

### 1. Create and activate the virtual environment

PowerShell:

```powershell
cd "D:\Mini Project"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure the Groq API key

Create or edit:

```text
.env
```

and add:

```text
GROQ_API_KEY=your_api_key_here
```

The project reads the LLM configuration from the environment/configuration layer.

---

## Initial Data and Model Setup

Run this once after installation:

```powershell
cd "D:\Mini Project"
.venv\Scripts\Activate.ps1
python scripts/seed_qdrant.py
```

This performs the initial setup of the anomaly model and historical failure records.

It trains the Isolation Forest and seeds the historical failure records into Qdrant.

After setup, the seed command does **not** need to be run every time the dashboard is started.

---

# Running the Project

## Recommended: Streamlit + MCP

The Streamlit dashboard now communicates with the diagnostic system through MCP.

Run only:

```powershell
cd "D:\Mini Project"
.venv\Scripts\Activate.ps1
streamlit run dashboard/streamlit_app.py
```

No Uvicorn server is required for the Streamlit + MCP workflow.

The MCP client automatically starts the local FastMCP server when the diagnostic pipeline is invoked.

---

## MCP Command-Line Client

For direct MCP testing:

```powershell
cd "D:\Mini Project"
.venv\Scripts\Activate.ps1
python mcp_client.py
```

The client provides an interactive menu for testing MCP capabilities.

Available operations include:

```text
1. Analyze equipment
2. Get anomaly score
3. Search failure history
4. Analyze equipment from CSV
5. List MCP tools
6. Exit
```

---

## FastAPI REST Interface

The FastAPI interface remains available as an alternative HTTP interface.

Start it with:

```powershell
cd "D:\Mini Project"
.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The API provides endpoints for:

```text
GET  /health
POST /analyze
POST /anomaly_score
POST /failure_history
```

FastAPI is independent of the Streamlit + MCP execution path.

---

## CSV Analysis

The project includes sample sensor data:

```text
data/sample_sensor_data.csv
```

The MCP tool:

```text
analyze_equipment_from_csv
```

can retrieve the most recent reading for a specified equipment ID and run the diagnostic pipeline.

Example equipment ID:

```text
PUMP-101
```

---

## Testing the MCP Implementation

The MCP implementation can be tested directly using:

```powershell
cd "D:\Mini Project"
.venv\Scripts\Activate.ps1
python mcp_client.py
```

The client establishes an MCP session and discovers the tools exposed by the FastMCP server.

The expected tools are:

```text
analyze_equipment_readings
analyze_equipment_from_csv
get_anomaly_score
query_failure_history
seed_failure_history
```

The full diagnostic path has been tested using anomalous sensor readings and has successfully returned:

* Isolation Forest anomaly score
* Anomaly status
* Historical Qdrant matches
* Root-cause diagnosis
* Maintenance recommendations
* Priority
* Final status

---

## Important Implementation Detail

MCP is the **interface layer**, not the anomaly-detection algorithm itself.

The responsibilities are separated as follows:

```text
MCP
│
└── Provides standardized tool communication

FastMCP
│
└── Exposes project capabilities as MCP tools

Isolation Forest
│
└── Detects anomalous sensor readings

Qdrant
│
└── Retrieves similar historical failures

LangGraph
│
└── Controls the diagnostic workflow

CrewAI
│
└── Performs diagnosis and maintenance planning

Groq
│
└── Provides LLM inference
```

This separation allows the predictive-maintenance pipeline to be exposed to different clients without rewriting the underlying diagnostic logic.

---

## Limitations

* The current training and failure-history data are synthetic.
* The historical failure archive is small and intended for demonstration.
* Diagnosis is based on a single sensor reading rather than a long temporal sequence.
* The system depends on the configured Groq LLM provider for agent reasoning.
* Maintenance actions are recommendations only and are not automatically executed on industrial equipment.
* The current MCP implementation uses local stdio transport.
* The Streamlit dashboard uses MCP as its diagnostic communication interface, while FastAPI remains available as a separate REST interface.

---

## Future Improvements

Possible extensions include:

* Real-time SCADA/IoT sensor ingestion
* Larger historical failure archives
* Persistent equipment telemetry
* Time-series anomaly detection
* More specialized maintenance agents
* Authentication and authorization for MCP tools
* Remote MCP deployment
* Automated maintenance-ticket generation
* CMMS integration
* Maintenance history feedback loops
* Continuous model retraining
* Production monitoring and observability

---

## Project Summary

The Plant Predictive Maintenance Copilot demonstrates how classical machine learning, retrieval-augmented generation, multi-agent systems, workflow orchestration, and the Model Context Protocol can be combined into a single predictive-maintenance system.

The system does more than detect an anomaly. When abnormal behavior is detected, it retrieves relevant historical failures, reasons about the likely root cause, and produces a structured maintenance plan.

The MCP layer makes these capabilities available as standardized tools that can be invoked by MCP clients, while the Streamlit dashboard provides a human-friendly interface for interacting with the same system.
