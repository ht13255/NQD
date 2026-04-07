from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

import numpy as np

from .compute_backend import get_torch_module, resolve_compute_backend
from .dense_placement import FastDenseSignals
from .papers import PaperKnowledge
from .requirements import RequirementBundle
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec


_RNG = np.random.default_rng(314159)
_INPUT_FEATURES = 53
_W1 = _RNG.normal(0.0, 0.22, size=(_INPUT_FEATURES * 3, 48))
_B1 = _RNG.normal(0.0, 0.05, size=(48,))
_W2 = _RNG.normal(0.0, 0.18, size=(48, 24))
_B2 = _RNG.normal(0.0, 0.04, size=(24,))
_GRAPH_SELF = _RNG.normal(0.0, 0.30, size=(8, 8))
_GRAPH_MSG = _RNG.normal(0.0, 0.24, size=(8, 8))
_GRAPH_BIAS = _RNG.normal(0.0, 0.04, size=(8,))
_GRAPH_ADJ = np.array(
    [
        [1.0, 0.4, 0.2, 0.1, 0.3, 0.2],
        [0.4, 1.0, 0.3, 0.2, 0.5, 0.2],
        [0.2, 0.3, 1.0, 0.4, 0.2, 0.4],
        [0.1, 0.2, 0.4, 1.0, 0.3, 0.4],
        [0.3, 0.5, 0.2, 0.3, 1.0, 0.4],
        [0.2, 0.2, 0.4, 0.4, 0.4, 1.0],
    ],
    dtype=float,
)
_GRAPH_ADJ = _GRAPH_ADJ / _GRAPH_ADJ.sum(axis=1, keepdims=True)
_OUT_W = _RNG.normal(0.0, 0.24, size=(32, 10))
_OUT_B = _RNG.normal(0.0, 0.04, size=(10,))
_TORCH_CACHE: dict[str, dict[str, Any]] = {}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


def _torch_constants(device: str) -> dict[str, Any] | None:
    torch = get_torch_module()
    if torch is None:
        return None
    cached = _TORCH_CACHE.get(device)
    if cached is None:
        cached = {
            "W1": torch.as_tensor(_W1, dtype=torch.float32, device=device),
            "B1": torch.as_tensor(_B1, dtype=torch.float32, device=device),
            "W2": torch.as_tensor(_W2, dtype=torch.float32, device=device),
            "B2": torch.as_tensor(_B2, dtype=torch.float32, device=device),
            "GRAPH_SELF": torch.as_tensor(_GRAPH_SELF, dtype=torch.float32, device=device),
            "GRAPH_MSG": torch.as_tensor(_GRAPH_MSG, dtype=torch.float32, device=device),
            "GRAPH_BIAS": torch.as_tensor(_GRAPH_BIAS, dtype=torch.float32, device=device),
            "GRAPH_ADJ": torch.as_tensor(_GRAPH_ADJ, dtype=torch.float32, device=device),
            "OUT_W": torch.as_tensor(_OUT_W, dtype=torch.float32, device=device),
            "OUT_B": torch.as_tensor(_OUT_B, dtype=torch.float32, device=device),
        }
        _TORCH_CACHE[device] = cached
    return cached


@dataclass(slots=True)
class FrozenNeuralSignals:
    design_quality: float
    architecture_affinity: float
    latency_risk: float
    crosstalk_risk: float
    yield_risk: float
    routing_risk: float
    confidence: float
    prototype_affinity: float
    pitch_delta_um: float
    optical_bus_delta: int
    microwave_line_delta: int
    shielding_delta: int
    mux_delta: int
    cavity_scale: float
    power_delta_dbm: float
    latent_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _routing_code(preference: str) -> list[float]:
    mapping = {
        "compact": [1.0, 0.0, 0.0],
        "balanced": [0.0, 1.0, 0.0],
        "low_loss": [0.0, 0.0, 1.0],
    }
    return mapping.get(preference.lower(), [0.0, 1.0, 0.0])


def _application_code(application: str) -> list[float]:
    mapping = {
        "quantum_repeater": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "magnetometer": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "processor": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "memory": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "imager": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "general": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    return mapping.get(application.lower(), mapping["general"])


def _paper_alignments(candidate: CandidateDesign, paper_knowledge: PaperKnowledge | None) -> tuple[float, float, float]:
    if paper_knowledge is None:
        return 0.5, 0.5, 0.5
    q_alignment = 1.0 - min(abs(math.log10(max(candidate.cavity_q, 1.0) / max(paper_knowledge.recommended_cavity_q, 1.0))) / 1.2, 1.0)
    width_alignment = 1.0 - min(abs(candidate.waveguide_width_um - paper_knowledge.recommended_waveguide_width_um) / 0.25, 1.0)
    pitch_alignment = 1.0 - min(abs(candidate.cell_pitch_um - paper_knowledge.recommended_pitch_um) / 20.0, 1.0)
    return max(q_alignment, 0.0), max(width_alignment, 0.0), max(pitch_alignment, 0.0)


def _feature_vector(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    dense_signals: FastDenseSignals,
    paper_knowledge: PaperKnowledge | None,
    requirements_bundle: RequirementBundle | None,
) -> np.ndarray:
    q_align, width_align, pitch_align = _paper_alignments(candidate, paper_knowledge)
    requirement_weights = requirements_bundle.normalized_spec.get("objective_weights", {}) if requirements_bundle else spec.objective_weights
    req_source = 1.0 if requirements_bundle and requirements_bundle.ollama_used else 0.0
    goals_count = len(requirements_bundle.goals) if requirements_bundle else 0
    constraints_count = len(requirements_bundle.constraints) if requirements_bundle else 0
    values = [
        spec.target_qubits / 192.0,
        spec.max_die_area_mm2 / 64.0,
        spec.max_power_mw / 800.0,
        spec.operating_temp_k / 100.0,
        spec.magnetic_field_mT / 120.0,
        spec.target_t2_us / 3000.0,
        spec.target_gate_fidelity,
        spec.target_readout_fidelity,
        spec.max_latency_ns / 400.0,
        spec.min_yield,
        candidate.qubits / max(spec.target_qubits, 1),
        candidate.cell_pitch_um / 64.0,
        candidate.optical_bus_count / max(candidate.qubits, 1),
        candidate.microwave_line_count / max(candidate.qubits, 1),
        candidate.resonator_count / max(candidate.qubits, 1),
        candidate.metal_layers / 8.0,
        candidate.shielding_layers / 5.0,
        candidate.waveguide_width_um / 0.9,
        math.log10(max(candidate.cavity_q, 1.0)) / 7.0,
        candidate.implant_dose_scale / 1.5,
        candidate.anneal_quality,
        candidate.microwave_power_dbm / 7.0,
        candidate.control_mux_factor / 8.0,
        candidate.amplifier_gain_db / 30.0,
        candidate.amplifier_noise_temp_k / 12.0,
        candidate.detector_qe,
        metrics.gate_fidelity / max(spec.target_gate_fidelity, 1e-6),
        metrics.readout_fidelity / max(spec.target_readout_fidelity, 1e-6),
        metrics.t2_us / max(spec.target_t2_us, 1e-6),
        spec.max_latency_ns / max(metrics.latency_ns, 1e-6),
        spec.max_die_area_mm2 / max(metrics.die_area_mm2, 1e-6),
        spec.max_power_mw / max(metrics.power_mw, 1e-6),
        metrics.yield_estimate / max(spec.min_yield, 1e-6),
        metrics.routing_capacity,
        metrics.scalability_score,
        metrics.robustness_margin,
        dense_signals.placement_score,
        dense_signals.min_spacing_um / max(candidate.cell_pitch_um, 1e-6),
        dense_signals.mean_offset_um / max(candidate.cell_pitch_um, 1e-6),
        dense_signals.optical_load_std,
        dense_signals.microwave_load_std,
        dense_signals.effective_crosstalk_linear,
        dense_signals.spectral_radius,
        dense_signals.p95_active_interference,
        q_align,
        width_align,
        pitch_align,
        requirement_weights.get("latency", spec.objective_weights.get("latency", 0.0)),
        requirement_weights.get("crosstalk", spec.objective_weights.get("crosstalk", 0.0)),
        requirement_weights.get("placement", spec.objective_weights.get("placement", 0.0)),
        req_source,
        goals_count / 8.0,
        constraints_count / 8.0,
    ]
    return np.asarray(values, dtype=float)


def _input_projection(features: np.ndarray) -> np.ndarray:
    if features.shape[0] != _INPUT_FEATURES:
        raise ValueError(f"expected {_INPUT_FEATURES} features, got {features.shape[0]}")
    backend = resolve_compute_backend()
    constants = _torch_constants(backend.device) if backend.engine == "torch" and backend.gpu_enabled else None
    torch = get_torch_module() if constants is not None else None
    if torch is not None and constants is not None:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=backend.device)
        enriched = torch.cat([tensor, torch.sin(torch.pi * tensor), torch.cos(torch.pi * tensor)], dim=0)
        hidden = torch.tanh(enriched @ constants["W1"] + constants["B1"])
        hidden = torch.tanh(hidden @ constants["W2"] + constants["B2"])
        return hidden.detach().cpu().numpy()
    enriched = np.concatenate([features, np.sin(math.pi * features), np.cos(math.pi * features)], axis=0)
    hidden = np.tanh(enriched @ _W1 + _B1)
    hidden = np.tanh(hidden @ _W2 + _B2)
    return hidden


def _graph_projection(features: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    backend = resolve_compute_backend()
    constants = _torch_constants(backend.device) if backend.engine == "torch" and backend.gpu_enabled else None
    torch = get_torch_module() if constants is not None else None
    if torch is not None and constants is not None:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=backend.device)
        hidden_tensor = torch.as_tensor(hidden, dtype=torch.float32, device=backend.device)
        nodes = torch.zeros((6, 8), dtype=torch.float32, device=backend.device)
        nodes[0] = torch.stack([feature_tensor[10], feature_tensor[12], feature_tensor[13], feature_tensor[14], hidden_tensor[0], hidden_tensor[1], hidden_tensor[2], hidden_tensor[3]])
        nodes[1] = torch.stack([feature_tensor[17], feature_tensor[18], feature_tensor[33], feature_tensor[44], hidden_tensor[4], hidden_tensor[5], hidden_tensor[6], hidden_tensor[7]])
        nodes[2] = torch.stack([feature_tensor[5], feature_tensor[26], feature_tensor[28], feature_tensor[35], hidden_tensor[8], hidden_tensor[9], hidden_tensor[10], hidden_tensor[11]])
        nodes[3] = torch.stack([feature_tensor[3], feature_tensor[31], feature_tensor[24], feature_tensor[34], hidden_tensor[12], hidden_tensor[13], hidden_tensor[14], hidden_tensor[15]])
        nodes[4] = torch.stack([feature_tensor[36], feature_tensor[37], feature_tensor[38], feature_tensor[39], hidden_tensor[16], hidden_tensor[17], hidden_tensor[18], hidden_tensor[19]])
        nodes[5] = torch.stack([feature_tensor[32], feature_tensor[40], feature_tensor[41], feature_tensor[42], hidden_tensor[20], hidden_tensor[21], hidden_tensor[22], hidden_tensor[23]])
        for _ in range(3):
            messages = constants["GRAPH_ADJ"] @ nodes @ constants["GRAPH_MSG"]
            self_term = nodes @ constants["GRAPH_SELF"]
            nodes = torch.tanh(messages + self_term + constants["GRAPH_BIAS"])
        return nodes.detach().cpu().numpy()
    nodes = np.zeros((6, 8), dtype=float)
    nodes[0] = np.array([features[10], features[12], features[13], features[14], hidden[0], hidden[1], hidden[2], hidden[3]])
    nodes[1] = np.array([features[17], features[18], features[33], features[44], hidden[4], hidden[5], hidden[6], hidden[7]])
    nodes[2] = np.array([features[5], features[26], features[28], features[35], hidden[8], hidden[9], hidden[10], hidden[11]])
    nodes[3] = np.array([features[3], features[31], features[24], features[34], hidden[12], hidden[13], hidden[14], hidden[15]])
    nodes[4] = np.array([features[36], features[37], features[38], features[39], hidden[16], hidden[17], hidden[18], hidden[19]])
    nodes[5] = np.array([features[32], features[40], features[41], features[42], hidden[20], hidden[21], hidden[22], hidden[23]])
    for _ in range(3):
        messages = _GRAPH_ADJ @ nodes @ _GRAPH_MSG
        self_term = nodes @ _GRAPH_SELF
        nodes = np.tanh(messages + self_term + _GRAPH_BIAS)
    return nodes


def evaluate_frozen_neural_surrogate(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    dense_signals: FastDenseSignals,
    paper_knowledge: PaperKnowledge | None = None,
    requirements_bundle: RequirementBundle | None = None,
    prototype_bank: Iterable[tuple[list[float], float]] | None = None,
) -> FrozenNeuralSignals:
    features = _feature_vector(spec, candidate, metrics, dense_signals, paper_knowledge, requirements_bundle)
    hidden = _input_projection(features)
    nodes = _graph_projection(features, hidden)
    backend = resolve_compute_backend()
    pooled = np.concatenate([hidden[:16], np.mean(nodes, axis=0), np.max(nodes, axis=0)], axis=0)
    constants = _torch_constants(backend.device) if backend.engine == "torch" and backend.gpu_enabled else None
    torch = get_torch_module() if constants is not None else None
    if torch is not None and constants is not None:
        pooled_tensor = torch.as_tensor(pooled, dtype=torch.float32, device=backend.device)
        logits = (pooled_tensor @ constants["OUT_W"] + constants["OUT_B"]).detach().cpu().numpy()
    else:
        logits = pooled @ _OUT_W + _OUT_B

    prototype_affinity = 0.5
    latent_head = pooled[:12]
    if prototype_bank:
        sims = []
        for latent, score in prototype_bank:
            proto = np.asarray(latent[:12], dtype=float)
            denom = np.linalg.norm(latent_head) * np.linalg.norm(proto)
            sim = float(np.dot(latent_head, proto) / max(denom, 1e-6))
            sims.append(0.5 + 0.5 * sim * max(min(score, 1.5), 0.0))
        if sims:
            prototype_affinity = max(0.0, min(1.25, float(np.mean(sims))))

    design_quality = max(0.0, min(1.25, 0.45 * _sigmoid(logits[0]) + 0.25 * _sigmoid(logits[1]) + 0.20 * prototype_affinity + 0.10 * (1.0 - dense_signals.effective_crosstalk_linear)))
    architecture_affinity = max(0.0, min(1.25, 0.55 * _sigmoid(logits[2]) + 0.25 * features[36] + 0.20 * features[44]))
    latency_risk = max(0.0, min(1.0, 0.65 * _sigmoid(logits[3]) + 0.35 * max(0.0, 1.0 - features[29])))
    crosstalk_risk = max(0.0, min(1.0, 0.55 * _sigmoid(logits[4]) + 0.45 * min(1.0, 8.0 * dense_signals.effective_crosstalk_linear)))
    yield_risk = max(0.0, min(1.0, 0.60 * _sigmoid(logits[5]) + 0.40 * max(0.0, 1.0 - features[32])))
    routing_risk = max(0.0, min(1.0, 0.60 * _sigmoid(logits[6]) + 0.40 * max(0.0, 0.8 - features[33])))
    confidence = max(0.1, min(1.2, 0.50 + 0.22 * prototype_affinity + 0.18 * features[26] + 0.10 * features[35]))

    pitch_delta = max(-2.5, min(3.0, 4.5 * crosstalk_risk + 1.5 * latency_risk - 1.8 * design_quality + 0.8 * routing_risk - 1.2))
    optical_delta = int(round(max(-2.0, min(3.0, 2.4 * routing_risk + 1.2 * crosstalk_risk - 1.0 * design_quality))))
    microwave_delta = int(round(max(-2.0, min(3.0, 2.0 * crosstalk_risk + 1.6 * latency_risk - 1.0 * design_quality))))
    shielding_delta = int(round(max(-1.0, min(2.0, 2.6 * crosstalk_risk + 0.9 * routing_risk - 1.2))))
    mux_delta = int(round(max(-2.0, min(1.0, 0.8 - 2.2 * latency_risk - 1.2 * crosstalk_risk + 0.6 * design_quality))))
    cavity_scale = max(0.90, min(1.15, 1.0 + 0.06 * architecture_affinity - 0.04 * latency_risk + 0.03 * design_quality))
    power_delta = max(-0.6, min(0.8, 0.5 * design_quality - 0.7 * crosstalk_risk - 0.3 * yield_risk))

    return FrozenNeuralSignals(
        design_quality=design_quality,
        architecture_affinity=architecture_affinity,
        latency_risk=latency_risk,
        crosstalk_risk=crosstalk_risk,
        yield_risk=yield_risk,
        routing_risk=routing_risk,
        confidence=confidence,
        prototype_affinity=prototype_affinity,
        pitch_delta_um=pitch_delta,
        optical_bus_delta=optical_delta,
        microwave_line_delta=microwave_delta,
        shielding_delta=shielding_delta,
        mux_delta=mux_delta,
        cavity_scale=cavity_scale,
        power_delta_dbm=power_delta,
        latent_vector=[float(value) for value in pooled[:16]],
    )


def calibrate_frozen_neural_surrogate(
    signals: FrozenNeuralSignals,
    metrics: SimulationMetrics,
    qec_plan: Any | None = None,
    robust_logical_pass_rate: float = 0.5,
) -> FrozenNeuralSignals:
    logical_pressure = 0.0
    if qec_plan is not None:
        logical_pressure = min(qec_plan.logical_error_rate / max(getattr(qec_plan, "threshold", 1e-6), 1e-6), 2.0)
    calibrated_quality = max(0.0, min(1.25, 0.55 * signals.design_quality + 0.20 * metrics.gate_fidelity + 0.15 * metrics.readout_fidelity + 0.10 * robust_logical_pass_rate))
    calibrated_latency = max(0.0, min(1.0, 0.60 * signals.latency_risk + 0.40 * max(metrics.latency_ns / 260.0 - 0.4, 0.0)))
    calibrated_crosstalk = max(0.0, min(1.0, 0.60 * signals.crosstalk_risk + 0.40 * min(metrics.crosstalk_linear * 8.0, 1.0)))
    calibrated_yield = max(0.0, min(1.0, 0.55 * signals.yield_risk + 0.45 * max(0.9 - metrics.yield_estimate, 0.0)))
    calibrated_confidence = max(0.2, min(1.2, 0.55 * signals.confidence + 0.20 * robust_logical_pass_rate + 0.15 * (1.0 - calibrated_latency) + 0.10 * (1.0 - logical_pressure / 2.0)))
    return replace(
        signals,
        design_quality=calibrated_quality,
        latency_risk=calibrated_latency,
        crosstalk_risk=calibrated_crosstalk,
        yield_risk=calibrated_yield,
        confidence=calibrated_confidence,
    )
