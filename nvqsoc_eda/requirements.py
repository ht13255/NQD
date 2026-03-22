from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .spec import DesignSpec


OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"


@dataclass(slots=True)
class RequirementLayers:
    hard_constraints: list[str] = field(default_factory=list)
    soft_targets: list[str] = field(default_factory=list)
    physical_context: list[str] = field(default_factory=list)
    logical_context: list[str] = field(default_factory=list)
    layout_directives: list[str] = field(default_factory=list)
    simulation_directives: list[str] = field(default_factory=list)
    extracted_literals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RequirementBundle:
    raw_requirements: str
    normalized_spec: dict[str, Any]
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    architecture_hypotheses: list[str] = field(default_factory=list)
    simulation_focus: list[str] = field(default_factory=list)
    layout_focus: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    source: str = "heuristic"
    model: str = "fallback"
    ollama_used: bool = False
    team_profile_name: str = "default_nv_team"
    team_notes: str = ""
    parse_confidence: float = 0.5
    layers: RequirementLayers = field(default_factory=RequirementLayers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TeamProfile:
    name: str
    application_aliases: dict[str, list[str]]
    routing_aliases: dict[str, list[str]]
    emphasis_aliases: dict[str, list[str]]
    default_layout_focus: list[str]
    default_simulation_focus: list[str]
    template_hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_TEAM_PROFILE = TeamProfile(
    name="default_nv_team",
    application_aliases={
        "quantum_repeater": ["repeater", "network", "entanglement", "node", "link", "spin-photon", "리피터", "네트워크 노드", "링크"],
        "magnetometer": ["magnetometer", "magnetometry", "sensor", "odmr", "자기 센서", "field imaging"],
        "processor": ["processor", "logic", "compute", "연산", "제어 중심"],
        "memory": ["memory", "buffer", "storage", "quantum memory", "메모리"],
        "imager": ["imager", "imaging", "microscope", "이미징", "camera"],
    },
    routing_aliases={
        "compact": ["compact", "dense", "고밀도", "small area", "tight pitch"],
        "balanced": ["balanced", "균형", "general purpose"],
        "low_loss": ["low loss", "저손실", "wide routing", "signal integrity"],
    },
    emphasis_aliases={
        "fidelity": ["high fidelity", "fidelity", "게이트 정확도", "error rate"],
        "coherence": ["coherence", "t2", "결맞음", "long-lived"],
        "yield": ["yield", "수율", "manufacturing"],
        "power": ["low power", "전력", "cryo power"],
        "latency": ["low latency", "latency", "지연", "fast loop"],
        "area": ["small area", "compact", "면적"],
        "scalability": ["scalable", "확장", "many qubits"],
        "robustness": ["robust", "margin", "stability", "안정성"],
        "placement": ["placement", "dense layout", "tile placement", "pad ring", "power mesh"],
        "crosstalk": ["crosstalk", "cross-talk", "interference", "shield fence", "em coupling", "간섭"],
        "logical": ["logical qubit", "logical qubits", "logical", "논리 큐비트", "fault tolerant"],
        "qec": ["qec", "surface code", "color code", "bacon shor", "error correction", "syndrome", "decoder"],
        "schedule": ["schedule", "timeline", "latency chain", "pipeline", "throughput", "스케줄"],
        "world_model": ["world model", "predictive", "rollout", "planning model"],
    },
    default_layout_focus=[
        "Pad ring, seal ring, shielding fence, power mesh, and segmented macros",
        "Tile-level keepout, via farms, resonator spines, and redundant optical trunks",
    ],
    default_simulation_focus=[
        "Spin-photon-control analytical stack",
        "Dense placement, cross-talk, control-loop, phase-noise, and power-grid realism",
    ],
    template_hints=[
        "Prefer conservative cryogenic assumptions when the user is vague.",
        "Treat shield fence, pad ring, redundant photonic routing, and control mesh as layout-critical keywords.",
    ],
)


def _load_team_profile(team_profile_path: str | Path | None) -> TeamProfile:
    if not team_profile_path:
        return DEFAULT_TEAM_PROFILE
    data = json.loads(Path(team_profile_path).read_text(encoding="utf-8"))
    merged = DEFAULT_TEAM_PROFILE.to_dict() | data
    return TeamProfile(**merged)


def _application_from_text(text: str, team_profile: TeamProfile) -> str:
    lowered = text.lower()
    scores = {key: sum(lowered.count(token.lower()) for token in tokens) for key, tokens in team_profile.application_aliases.items()}
    best = max(scores.items(), key=lambda item: item[1])
    return best[0] if best[1] > 0 else "general"


def _extract_number(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _extract_target_qubits(text: str) -> int | None:
    patterns = [
        r"([0-9]+)\s*(?:physical qubits?|physical q|물리 큐비트)",
        r"([0-9]+)\s*(?:qubits?|큐비트)",
    ]
    value = _extract_number(patterns, text)
    return int(value) if value is not None else None


def _extract_percent_value(text: str, keywords: list[str]) -> float | None:
    for keyword in keywords:
        pattern = rf"{keyword}[^\n0-9%]*([0-9]+(?:\.[0-9]+)?)\s*%"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
        pattern = rf"{keyword}[^\n0-9]*([0-9]\.[0-9]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_t2_us(text: str) -> float | None:
    patterns = [r"t2[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*(ns|us|µs|ms|s)", r"coherence[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*(ns|us|µs|ms|s)"]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            scale = {"ns": 1e-3, "us": 1.0, "µs": 1.0, "ms": 1e3, "s": 1e6}[unit]
            return value * scale
    return None


def _extract_magnetic_field_mT(text: str) -> float | None:
    patterns = [r"magnetic[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mt|t|gauss|g)", r"field[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mt|t|gauss|g)"]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == "t":
                value *= 1000.0
            elif unit in {"gauss", "g"}:
                value *= 0.1
            return value
    return None


def _extract_temperature_k(text: str) -> float | None:
    patterns = [r"([0-9]+(?:\.[0-9]+)?)\s*k\b", r"temperature[^\n0-9]*([0-9]+(?:\.[0-9]+)?)"]
    return _extract_number(patterns, text)


def _extract_fabrication_node(text: str) -> int | None:
    value = _extract_number([r"([0-9]+)\s*nm\s*(?:node|process)?", r"node[^\n0-9]*([0-9]+)"], text)
    if value is None:
        return None
    return int(value)


def _extract_isotopic_purity(text: str) -> float | None:
    return _extract_number([r"isotopic[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*ppm", r"purity[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*ppm"], text)


def _extract_wavelength_nm(text: str) -> float | None:
    return _extract_number([r"([0-9]+(?:\.[0-9]+)?)\s*nm\s*(?:laser|wavelength|optical)", r"wavelength[^\n0-9]*([0-9]+(?:\.[0-9]+)?)"], text)


def _extract_cryo_stage_count(text: str) -> int | None:
    value = _extract_number([r"([0-9]+)\s*(?:stage|stages)", r"cryo[^\n0-9]*([0-9]+)\s*stage"], text)
    return int(value) if value is not None else None


def _extract_logical_qubits(text: str) -> int | None:
    value = _extract_number([r"([0-9]+)\s*(?:logical qubits?|logical q|논리 큐비트)", r"logical[^\n0-9]*([0-9]+)\s*(?:qubits?|q)?"], text)
    return int(value) if value is not None else None


def _extract_logical_error_rate(text: str) -> float | None:
    value = _extract_number([r"logical error[^\n0-9]*([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)", r"logical ber[^\n0-9]*([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)"], text)
    return float(value) if value is not None else None


def _extract_qec_code(text: str) -> str | None:
    lowered = text.lower()
    if "auto qec" in lowered or "adaptive qec" in lowered or "dynamic qec" in lowered:
        return "auto"
    if "surface code" in lowered or "surface-code" in lowered:
        return "surface_code"
    if "color code" in lowered or "color-code" in lowered:
        return "color_code"
    if "bacon shor" in lowered or "bacon-shor" in lowered or "baconshor" in lowered:
        return "bacon_shor"
    return None


def _infer_weights(text: str, team_profile: TeamProfile) -> dict[str, float]:
    lowered = text.lower()
    weights = {
        "fidelity": 0.22,
        "coherence": 0.16,
        "yield": 0.13,
        "power": 0.09,
        "latency": 0.09,
        "area": 0.08,
        "scalability": 0.07,
        "robustness": 0.05,
        "placement": 0.06,
        "crosstalk": 0.05,
        "logical": 0.05,
        "qec": 0.05,
        "schedule": 0.04,
        "world_model": 0.03,
    }
    for key, tokens in team_profile.emphasis_aliases.items():
        weights[key] += 0.03 * sum(lowered.count(token) for token in tokens)
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _extract_requirement_layers(requirements_text: str, normalized_spec: dict[str, Any]) -> RequirementLayers:
    lowered = requirements_text.lower()
    clauses = [clause.strip() for clause in re.split(r"[\n\.;]", requirements_text) if clause.strip()]
    hard_constraints: list[str] = []
    soft_targets: list[str] = []
    physical_context: list[str] = []
    logical_context: list[str] = []
    layout_directives: list[str] = []
    simulation_directives: list[str] = []
    for clause in clauses:
        clause_low = clause.lower()
        if any(token in clause_low for token in ["under", "below", "less than", "must", "<=", "at most", "no more than"]):
            hard_constraints.append(clause)
        elif any(token in clause_low for token in ["high", "strong", "maximize", "prioritize", "prefer", "very", "improve"]):
            soft_targets.append(clause)
        if any(token in clause_low for token in ["t2", "coherence", "temperature", "magnetic", "field", "power", "nm", "ppm"]):
            physical_context.append(clause)
        if any(token in clause_low for token in ["logical", "qec", "error correction", "syndrome", "decoder", "fault tolerant"]):
            logical_context.append(clause)
        if any(token in clause_low for token in ["layout", "pad ring", "shield", "routing", "bus", "tile", "macro", "area", "dense"]):
            layout_directives.append(clause)
        if any(token in clause_low for token in ["simulate", "simulation", "cross-talk", "entanglement", "noise", "thermal", "latency"]):
            simulation_directives.append(clause)
    if not hard_constraints:
        hard_constraints = [
            f"Die area <= {normalized_spec['max_die_area_mm2']} mm2",
            f"Power <= {normalized_spec['max_power_mw']} mW",
            f"Latency <= {normalized_spec['max_latency_ns']} ns",
        ]
    extracted_literals = {
        "application": normalized_spec.get("application"),
        "target_qubits": normalized_spec.get("target_qubits"),
        "target_logical_qubits": normalized_spec.get("target_logical_qubits"),
        "qec_code": normalized_spec.get("qec_code"),
        "routing_preference": normalized_spec.get("routing_preference"),
    }
    return RequirementLayers(
        hard_constraints=hard_constraints,
        soft_targets=soft_targets,
        physical_context=physical_context,
        logical_context=logical_context,
        layout_directives=layout_directives,
        simulation_directives=simulation_directives,
        extracted_literals=extracted_literals,
    )


def _heuristic_requirements(requirements_text: str, team_profile: TeamProfile, team_notes: str) -> RequirementBundle:
    application = _application_from_text(requirements_text, team_profile)
    qubits = float(_extract_target_qubits(requirements_text) or 64)
    derived_area = max(24.0, 20.0 + 0.14 * qubits + 6.0 * max(1, int(_extract_logical_qubits(requirements_text) or max(1, int(qubits) // 48))))
    die_area = _extract_number([r"([0-9]+(?:\.[0-9]+)?)\s*mm(?:\^?2|²)", r"area[^\n0-9]*([0-9]+(?:\.[0-9]+)?)"], requirements_text) or derived_area
    power_value = _extract_number([r"([0-9]+(?:\.[0-9]+)?)\s*mw\b", r"power[^\n0-9]*([0-9]+(?:\.[0-9]+)?)"], requirements_text)
    temp_value = _extract_temperature_k(requirements_text) or 4.2
    field_value = _extract_magnetic_field_mT(requirements_text) or 18.0
    t2_value = _extract_t2_us(requirements_text) or 1200.0
    node_value = _extract_fabrication_node(requirements_text) or 90
    purity_value = _extract_isotopic_purity(requirements_text) or 30.0
    wavelength_value = _extract_wavelength_nm(requirements_text) or 637.0
    stage_count = _extract_cryo_stage_count(requirements_text) or (3 if temp_value <= 20.0 else 1)
    logical_qubits = _extract_logical_qubits(requirements_text) or max(1, int(qubits) // 48)
    logical_error_rate = _extract_logical_error_rate(requirements_text) or 1e-3
    qec_code = _extract_qec_code(requirements_text) or "auto"
    gate_fidelity = _extract_percent_value(requirements_text, ["gate", "게이트"]) or 0.995
    readout_fidelity = _extract_percent_value(requirements_text, ["readout", "판독", "측정"]) or 0.965
    latency_value = _extract_number([r"([0-9]+(?:\.[0-9]+)?)\s*ns\b", r"latency[^\n0-9]*([0-9]+(?:\.[0-9]+)?)"], requirements_text) or 180.0
    yield_value = _extract_percent_value(requirements_text, ["yield", "수율"]) or 0.72
    optical_budget = _extract_number([r"optical[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*mw", r"laser[^\n0-9]*([0-9]+(?:\.[0-9]+)?)\s*mw"], requirements_text) or 140.0

    routing_preference = "balanced"
    lowered = requirements_text.lower()
    for routing_key, aliases in team_profile.routing_aliases.items():
        if any(token.lower() in lowered for token in aliases):
            routing_preference = routing_key
            break

    design_name = re.sub(r"[^a-z0-9_]+", "_", f"{application}_{int(qubits)}q_from_nl".lower()).strip("_")
    spec_dict = {
        "design_name": design_name,
        "application": application,
        "target_qubits": int(qubits),
        "target_logical_qubits": int(logical_qubits),
        "max_die_area_mm2": float(die_area),
        "max_power_mw": float(power_value or 480.0),
        "operating_temp_k": float(temp_value),
        "magnetic_field_mT": float(field_value),
        "optical_wavelength_nm": float(wavelength_value),
        "target_t2_us": float(t2_value),
        "target_gate_fidelity": float(gate_fidelity),
        "target_readout_fidelity": float(readout_fidelity),
        "target_logical_error_rate": float(logical_error_rate),
        "max_latency_ns": float(latency_value),
        "min_yield": float(yield_value),
        "fabrication_node_nm": int(node_value),
        "diamond_thickness_um": 25.0,
        "isotopic_purity_ppm": float(purity_value),
        "optical_power_budget_mw": float(optical_budget),
        "cryo_stage_count": int(stage_count),
        "qec_code": qec_code,
        "qec_enabled": True,
        "syndrome_cycle_ns": 120.0,
        "decoder_margin_ns": 60.0,
        "logical_connectivity": "grid",
        "routing_preference": routing_preference,
        "objective_weights": _infer_weights(requirements_text + "\n" + team_notes, team_profile),
    }
    normalized = DesignSpec.from_dict(spec_dict).to_dict()
    goals = [
        f"Target {normalized['target_qubits']} qubits for {normalized['application']}",
        f"Expose {normalized['target_logical_qubits']} logical qubits with `{normalized['qec_code']}`",
        f"Hit gate fidelity >= {normalized['target_gate_fidelity']:.4f}",
        f"Hit readout fidelity >= {normalized['target_readout_fidelity']:.4f}",
    ]
    constraints = [
        f"Die area <= {normalized['max_die_area_mm2']} mm2",
        f"Power <= {normalized['max_power_mw']} mW",
        f"Latency <= {normalized['max_latency_ns']} ns",
        f"Logical error rate <= {normalized['target_logical_error_rate']}",
    ]
    architecture_hypotheses = [
        f"Primary application maps to `{normalized['application']}` architecture priors",
        f"Routing preference inferred as `{normalized['routing_preference']}`",
    ]
    simulation_focus = list(team_profile.default_simulation_focus)
    simulation_focus.append("QEC threshold, logical error-rate, and syndrome/decoder timing analysis")
    layout_focus = list(team_profile.default_layout_focus)
    layout_focus.append("Logical patch overlays, syndrome lanes, and ancilla-rich stabilizer structure")
    assumptions = [
        "Unspecified quantities use repository defaults",
        "Natural language ambiguities are resolved toward safe balanced cryogenic operation",
    ]
    if team_notes.strip():
        assumptions.append(f"Team notes applied: {team_notes.strip()}")
    layers = _extract_requirement_layers(requirements_text, normalized)
    return RequirementBundle(
        raw_requirements=requirements_text,
        normalized_spec=normalized,
        goals=goals,
        constraints=constraints,
        architecture_hypotheses=architecture_hypotheses,
        simulation_focus=simulation_focus,
        layout_focus=layout_focus,
        assumptions=assumptions,
        source="heuristic",
        model="fallback",
        ollama_used=False,
        team_profile_name=team_profile.name,
        team_notes=team_notes,
        parse_confidence=0.66,
        layers=layers,
    )


def _ollama_prompt(requirements_text: str, team_profile: TeamProfile, team_notes: str) -> str:
    return f"""
You are an expert NV-center quantum SoC EDA planner.
Convert the user's natural-language requirements into strict JSON.

Team profile name: {team_profile.name}

Team application aliases:
{json.dumps(team_profile.application_aliases, ensure_ascii=False, indent=2)}

Team routing aliases:
{json.dumps(team_profile.routing_aliases, ensure_ascii=False, indent=2)}

Team emphasis aliases:
{json.dumps(team_profile.emphasis_aliases, ensure_ascii=False, indent=2)}

Team template hints:
{json.dumps(team_profile.template_hints, ensure_ascii=False, indent=2)}

Extra team notes:
{team_notes or "(none)"}

Return JSON only with this schema:
{{
  "normalized_spec": {{
    "design_name": string,
    "application": string,
    "target_qubits": integer,
    "target_logical_qubits": integer,
    "max_die_area_mm2": number,
    "max_power_mw": number,
    "operating_temp_k": number,
    "magnetic_field_mT": number,
    "optical_wavelength_nm": number,
    "target_t2_us": number,
    "target_gate_fidelity": number,
    "target_readout_fidelity": number,
    "target_logical_error_rate": number,
    "max_latency_ns": number,
    "min_yield": number,
    "fabrication_node_nm": integer,
    "diamond_thickness_um": number,
    "isotopic_purity_ppm": number,
    "optical_power_budget_mw": number,
    "cryo_stage_count": integer,
    "qec_code": string,
    "qec_enabled": boolean,
    "syndrome_cycle_ns": number,
    "decoder_margin_ns": number,
    "logical_connectivity": string,
    "routing_preference": string,
    "objective_weights": object
  }},
  "goals": [string],
  "constraints": [string],
  "architecture_hypotheses": [string],
  "simulation_focus": [string],
  "layout_focus": [string],
  "assumptions": [string]
}}

Use physically reasonable defaults for missing values.
Applications must be one of: quantum_repeater, magnetometer, processor, memory, imager, general.
QEC codes must be one of: auto, surface_code, color_code, bacon_shor.
Routing preference must be one of: compact, balanced, low_loss.
Make objective_weights sum approximately to 1.
Infer not only numeric limits but also design intent, likely architecture, simulation emphasis, and layout emphasis.
Prefer conservative physically realistic assumptions over optimistic ones.
If the user mixes multiple intents, choose the primary application and mention secondary intents in assumptions.
Infer logical-qubit intent and QEC code whenever the text mentions logical qubits, error correction, syndrome, decoder, or fault tolerance.

User requirements:
{requirements_text}
""".strip()


def _ollama_parse(requirements_text: str, model: str, timeout_s: int, team_profile: TeamProfile, team_notes: str) -> RequirementBundle | None:
    payload = json.dumps(
        {
            "model": model,
            "prompt": _ollama_prompt(requirements_text, team_profile, team_notes),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")
    request = urllib.request.Request(OLLAMA_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8", "ignore"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    raw_json = result.get("response", "")
    if not raw_json:
        return None
    try:
        parsed = json.loads(raw_json)
        normalized_spec = DesignSpec.from_dict(parsed.get("normalized_spec", {})).to_dict()
    except Exception:
        return None
    return RequirementBundle(
        raw_requirements=requirements_text,
        normalized_spec=normalized_spec,
        goals=[str(item) for item in parsed.get("goals", [])],
        constraints=[str(item) for item in parsed.get("constraints", [])],
        architecture_hypotheses=[str(item) for item in parsed.get("architecture_hypotheses", [])],
        simulation_focus=[str(item) for item in parsed.get("simulation_focus", [])],
        layout_focus=[str(item) for item in parsed.get("layout_focus", [])],
        assumptions=[str(item) for item in parsed.get("assumptions", [])],
        source="ollama",
        model=model,
        ollama_used=True,
        team_profile_name=team_profile.name,
        team_notes=team_notes,
        parse_confidence=0.84,
        layers=_extract_requirement_layers(requirements_text, normalized_spec),
    )


def parse_design_requirements(
    requirements_text: str,
    model: str = "llama3.1:8b",
    timeout_s: int = 120,
    team_profile_path: str | Path | None = None,
    team_notes: str = "",
) -> RequirementBundle:
    requirements_text = requirements_text.strip()
    if not requirements_text:
        raise ValueError("requirements_text must not be empty")
    team_profile = _load_team_profile(team_profile_path)
    bundle = _ollama_parse(requirements_text, model=model, timeout_s=timeout_s, team_profile=team_profile, team_notes=team_notes)
    if bundle is not None:
        return bundle
    return _heuristic_requirements(requirements_text, team_profile=team_profile, team_notes=team_notes)


def write_requirement_artifacts(output_dir: Path, bundle: RequirementBundle, include_markdown: bool = False) -> dict[str, str]:
    json_path = output_dir / "requirements_analysis.json"
    json_path.write_text(json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if not include_markdown:
        return {"requirements_json": str(json_path)}
    md_path = output_dir / "requirements_analysis.md"
    lines = [
        "# Requirements Analysis",
        "",
        f"- Source: `{bundle.source}`",
        f"- Model: `{bundle.model}`",
        f"- Ollama used: `{bundle.ollama_used}`",
        f"- Team profile: `{bundle.team_profile_name}`",
        "",
        "## Goals",
        "",
    ]
    lines.extend(f"- {item}" for item in bundle.goals)
    lines.extend(["", "## Constraints", ""])
    lines.extend(f"- {item}" for item in bundle.constraints)
    lines.extend(["", "## Architecture Hypotheses", ""])
    lines.extend(f"- {item}" for item in bundle.architecture_hypotheses)
    lines.extend(["", "## Simulation Focus", ""])
    lines.extend(f"- {item}" for item in bundle.simulation_focus)
    lines.extend(["", "## Layout Focus", ""])
    lines.extend(f"- {item}" for item in bundle.layout_focus)
    lines.extend(["", "## Assumptions", ""])
    lines.extend(f"- {item}" for item in bundle.assumptions)
    lines.extend(["", "## Normalized Spec", "", "```json", json.dumps(bundle.normalized_spec, indent=2, ensure_ascii=False), "```", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"requirements_json": str(json_path), "requirements_md": str(md_path)}
