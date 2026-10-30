import pandas as pd
import numpy as np
import json

np.random.seed(42)
equipment = [
    ("PUMP-101", "Centrifugal Pump", "Building A - Cooling Loop"),
    ("MTR-204", "Induction Motor", "Building B - Line 2"),
    ("CMP-317", "Rotary Compressor", "Utility Room 3"),
    ("FAN-088", "Exhaust Fan", "Building A - HVAC"),
]

rows = []
start = pd.Timestamp("2026-06-01")
for eq_id, eq_type, loc in equipment:
    base_temp = np.random.uniform(55, 70)
    base_vib = np.random.uniform(1.5, 3.0)
    base_pressure = np.random.uniform(80, 120)
    base_rpm = np.random.uniform(1450, 1800)
    base_current = np.random.uniform(10, 25)
    for i in range(500):
        ts = start + pd.Timedelta(minutes=15 * i)
        drift = (i / 500) * np.random.choice([0, 0, 0, 1])  # some units drift toward failure
        temp = base_temp + drift * 25 + np.random.normal(0, 1.2)
        vib = base_vib + drift * 6 + np.random.normal(0, 0.15)
        pressure = base_pressure - drift * 20 + np.random.normal(0, 2)
        rpm = base_rpm - drift * 150 + np.random.normal(0, 10)
        current = base_current + drift * 8 + np.random.normal(0, 0.5)
        rows.append([eq_id, eq_type, loc, ts.isoformat(), round(temp, 2), round(vib, 3),
                     round(pressure, 2), round(rpm, 1), round(current, 2)])

df = pd.DataFrame(rows, columns=["equipment_id", "equipment_type", "location", "timestamp",
                                  "temperature_c", "vibration_mm_s", "pressure_psi", "rpm", "current_amps"])
df.to_csv("data/sample_sensor_data.csv", index=False)
print(df.shape)

failure_records = [
    {
        "id": 1, "equipment_type": "Centrifugal Pump",
        "title": "Bearing wear leading to high vibration and overheating",
        "description": "Progressive bearing degradation caused vibration to climb from 2.5 mm/s to over 9 mm/s over 3 weeks, accompanied by a 20C temperature rise. Root cause was lubrication breakdown from a missed PM cycle. Resulted in unplanned 14-hour outage.",
        "symptoms": ["rising vibration", "bearing housing overheating", "audible grinding noise", "gradual current increase"],
        "root_cause": "Lubrication breakdown / bearing wear",
        "corrective_action": "Replace drive-end bearing, flush and re-lubricate housing, verify shaft alignment.",
        "severity": "high",
    },
    {
        "id": 2, "equipment_type": "Induction Motor",
        "title": "Winding insulation degradation from thermal cycling",
        "description": "Motor windings showed insulation resistance decline after repeated thermal cycling from frequent start-stop operation. Current draw increased 30% above baseline before a phase-to-ground fault tripped the breaker.",
        "symptoms": ["current draw creeping up", "winding temperature above rating", "intermittent breaker trips"],
        "root_cause": "Insulation breakdown from thermal cycling",
        "corrective_action": "Rewind or replace motor, review duty cycle, add soft-start controller.",
        "severity": "critical",
    },
    {
        "id": 3, "equipment_type": "Rotary Compressor",
        "title": "Discharge pressure loss from worn rotor seals",
        "description": "Discharge pressure dropped steadily over two weeks while RPM held steady, indicating internal leakage past worn rotor seals. Efficiency loss increased energy consumption 18% before failure.",
        "symptoms": ["declining discharge pressure", "steady rpm", "rising energy consumption", "oil carryover"],
        "root_cause": "Worn rotor seals / internal leakage",
        "corrective_action": "Replace rotor seal kit, inspect oil separator, verify pressure setpoints.",
        "severity": "medium",
    },
    {
        "id": 4, "equipment_type": "Exhaust Fan",
        "title": "Impeller imbalance from debris accumulation",
        "description": "Vibration spiked sharply within 48 hours due to debris buildup on fan blades causing imbalance. Motor current fluctuated erratically and bearing temperature rose quickly, requiring emergency shutdown.",
        "symptoms": ["sudden vibration spike", "erratic current", "rapid bearing temp rise"],
        "root_cause": "Impeller imbalance from debris",
        "corrective_action": "Shut down and clean impeller blades, rebalance rotor, inspect bearings for damage.",
        "severity": "high",
    },
    {
        "id": 5, "equipment_type": "Centrifugal Pump",
        "title": "Cavitation from suction line restriction",
        "description": "Partially clogged suction strainer caused cavitation, evident as erratic pressure readings and a distinctive rattling noise. Left unaddressed, cavitation pitted the impeller within a month.",
        "symptoms": ["erratic pressure", "rattling noise", "reduced flow", "mild vibration increase"],
        "root_cause": "Suction strainer clog causing cavitation",
        "corrective_action": "Clean/replace suction strainer, inspect impeller for cavitation pitting.",
        "severity": "medium",
    },
    {
        "id": 6, "equipment_type": "Induction Motor",
        "title": "Rotor bar fault causing torque ripple",
        "description": "Broken rotor bar produced characteristic current sidebands and torque ripple, detected as periodic RPM fluctuation and a low-frequency vibration harmonic. Progressed to full stall over 10 days.",
        "symptoms": ["periodic rpm fluctuation", "low-frequency vibration harmonic", "current sidebands"],
        "root_cause": "Broken rotor bar",
        "corrective_action": "Replace rotor or motor, perform current signature analysis on fleet.",
        "severity": "high",
    },
]
with open("data/sample_failure_records.json", "w") as f:
    json.dump(failure_records, f, indent=2)
print("wrote failure records:", len(failure_records))
