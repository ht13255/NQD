from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SIGNOFF_VERSION = "2.0.0"

GATE_DEFINITIONS: list[dict[str, Any]] = [
    {"gate_id": "gate_drc", "gate_name": "Design Rule Check", "category": "drc", "severity": "critical"},
    {"gate_id": "gate_design_vs_layout", "gate_name": "Design vs Layout Cross-validation", "category": "design_vs_layout", "severity": "critical"},
    {"gate_id": "gate_consistency", "gate_name": "Layout Consistency Closure", "category": "consistency", "severity": "critical"},
    {"gate_id": "gate_area_agreement", "gate_name": "Die Area Agreement", "category": "design_vs_layout", "severity": "critical"},
    {"gate_id": "gate_lvs", "gate_name": "Layout vs Schematic (Light)", "category": "integrity", "severity": "major"},
    {"gate_id": "gate_qec", "gate_name": "QEC Plan Compliance", "category": "consistency", "severity": "critical"},
]

SIGNOFF_SCHEMA = {
    "signoff_version": SIGNOFF_VERSION,
    "handoff_policy": {
        "critical_gate_failure": "quarantine_bundle",
        "quarantine_flag": "QUARANTINE",
    },
    "gate_definitions": GATE_DEFINITIONS,
}


def write_signoff_schema(output_dir: Path) -> dict[str, str]:
    path = output_dir / "signoff_schema.json"
    path.write_text(json.dumps(SIGNOFF_SCHEMA, indent=2), encoding="utf-8")
    return {"signoff_schema": str(path)}
