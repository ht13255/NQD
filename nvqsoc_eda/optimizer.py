from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .dense_placement import FastDenseSignals, analyze_candidate_dense_fast
from .neural_surrogate import FrozenNeuralSignals, calibrate_frozen_neural_surrogate, evaluate_frozen_neural_surrogate
from .papers import PaperKnowledge
from .qec import QECPlan, build_qec_plan, estimate_distance_resources, estimate_required_physical_qubits
from .requirements import RequirementBundle
from .simulator import CandidateDesign, MonteCarloSummary, SimulationMetrics, evaluate_candidate, run_monte_carlo
from .spec import DesignSpec
from .world_model import WorldModelSignals, calibrate_world_model_signals, simulate_world_model_rollout


@dataclass(slots=True)
class FastRobustSignals:
    scenario_pass_rate: float
    logical_pass_rate: float
    schedule_pass_rate: float
    gate_floor: float
    readout_floor: float
    worst_latency_ns: float
    worst_logical_error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_pass_rate": self.scenario_pass_rate,
            "logical_pass_rate": self.logical_pass_rate,
            "schedule_pass_rate": self.schedule_pass_rate,
            "gate_floor": self.gate_floor,
            "readout_floor": self.readout_floor,
            "worst_latency_ns": self.worst_latency_ns,
            "worst_logical_error_rate": self.worst_logical_error_rate,
        }


@dataclass(slots=True)
class RankedCandidate:
    candidate: CandidateDesign
    metrics: SimulationMetrics
    score: float
    robustness_score: float
    dense_signals: FastDenseSignals | None = None
    neural_signals: FrozenNeuralSignals | None = None
    qec_plan: QECPlan | None = None
    world_model_signals: WorldModelSignals | None = None
    robust_signals: FastRobustSignals | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "score": self.score,
            "robustness_score": self.robustness_score,
            "dense_signals": self.dense_signals.to_dict() if self.dense_signals else None,
            "neural_signals": self.neural_signals.to_dict() if self.neural_signals else None,
            "qec_plan": self.qec_plan.to_dict() if self.qec_plan else None,
            "world_model_signals": self.world_model_signals.to_dict() if self.world_model_signals else None,
            "robust_signals": self.robust_signals.to_dict() if self.robust_signals else None,
        }


@dataclass(slots=True)
class OptimizationResult:
    best_candidate: CandidateDesign
    best_metrics: SimulationMetrics
    monte_carlo: MonteCarloSummary
    pareto_like_frontier: list[RankedCandidate]
    search_log: list[dict[str, Any]]
    seed_provenance: list[dict[str, Any]] = field(default_factory=list)
    paper_bias_audit: dict[str, Any] = field(default_factory=dict)
    selected_ranked_candidate: RankedCandidate | None = None
    seed_feedback: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_candidate": self.best_candidate.to_dict(),
            "best_metrics": self.best_metrics.to_dict(),
            "monte_carlo": self.monte_carlo.to_dict(),
            "pareto_like_frontier": [item.to_dict() for item in self.pareto_like_frontier],
            "search_log": self.search_log,
            "seed_provenance": self.seed_provenance,
            "paper_bias_audit": self.paper_bias_audit,
            "selected_ranked_candidate": self.selected_ranked_candidate.to_dict() if self.selected_ranked_candidate else None,
            "seed_feedback": self.seed_feedback,
        }


@dataclass(slots=True)
class SeedFeedback:
    recommended_qubit_delta: int = 0
    recommended_factory_lane_delta: int = 0
    recommended_pitch_delta_um: float = 0.0
    confidence: float = 0.0
    source_candidate_keys: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return bool(
            self.recommended_qubit_delta
            or self.recommended_factory_lane_delta
            or abs(self.recommended_pitch_delta_um) >= 1e-9
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_qubit_delta": self.recommended_qubit_delta,
            "recommended_factory_lane_delta": self.recommended_factory_lane_delta,
            "recommended_pitch_delta_um": self.recommended_pitch_delta_um,
            "confidence": self.confidence,
            "source_candidate_keys": self.source_candidate_keys,
            "active": self.is_active(),
        }


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _architecture_space(spec: DesignSpec, paper_knowledge: PaperKnowledge | None = None) -> list[str]:
    biases = paper_knowledge.architecture_priors if paper_knowledge else spec.architecture_biases()
    return [name for name, _weight in sorted(biases.items(), key=lambda item: item[1], reverse=True)]


def _grid_shape(qubits: int) -> tuple[int, int]:
    rows = max(2, round(math.sqrt(qubits)))
    cols = math.ceil(qubits / rows)
    while rows * cols < qubits:
        cols += 1
    return rows, cols


def _candidate_signature(candidate: CandidateDesign) -> str:
    return (
        f"{candidate.architecture}_q{candidate.qubits}_p{candidate.cell_pitch_um:.2f}"
        f"_ob{candidate.optical_bus_count}_mw{candidate.microwave_line_count}"
    )


def _paper_recommendation_bindings(candidate: CandidateDesign, paper_knowledge: PaperKnowledge | None) -> list[dict[str, Any]]:
    if paper_knowledge is None:
        return []
    return [
        {
            "parameter": "cavity_q",
            "recommended_value": paper_knowledge.recommended_cavity_q,
            "candidate_value": candidate.cavity_q,
            "delta": candidate.cavity_q - paper_knowledge.recommended_cavity_q,
            "applied_in_seed": abs(math.log10(max(candidate.cavity_q, 1.0) / max(paper_knowledge.recommended_cavity_q, 1.0))) <= 0.30,
            "provenance": paper_knowledge.recommendation_provenance.get("cavity_q", []),
        },
        {
            "parameter": "waveguide_width_um",
            "recommended_value": paper_knowledge.recommended_waveguide_width_um,
            "candidate_value": candidate.waveguide_width_um,
            "delta": candidate.waveguide_width_um - paper_knowledge.recommended_waveguide_width_um,
            "applied_in_seed": abs(candidate.waveguide_width_um - paper_knowledge.recommended_waveguide_width_um) <= 0.08,
            "provenance": paper_knowledge.recommendation_provenance.get("waveguide_width_um", []),
        },
        {
            "parameter": "pitch_um",
            "recommended_value": paper_knowledge.recommended_pitch_um,
            "candidate_value": candidate.cell_pitch_um,
            "delta": candidate.cell_pitch_um - paper_knowledge.recommended_pitch_um,
            "applied_in_seed": abs(candidate.cell_pitch_um - paper_knowledge.recommended_pitch_um) <= 8.0,
            "provenance": paper_knowledge.recommendation_provenance.get("pitch_um", []),
        },
        {
            "parameter": "optical_bus_ratio",
            "recommended_value": paper_knowledge.recommended_optical_bus_ratio,
            "candidate_value": candidate.optical_bus_count / max(candidate.qubits, 1),
            "delta": candidate.optical_bus_count / max(candidate.qubits, 1) - paper_knowledge.recommended_optical_bus_ratio,
            "applied_in_seed": abs(candidate.optical_bus_count / max(candidate.qubits, 1) - paper_knowledge.recommended_optical_bus_ratio) <= 0.03,
            "provenance": paper_knowledge.recommendation_provenance.get("optical_bus_ratio", []),
        },
    ]


def _seed_provenance_entry(
    candidate: CandidateDesign,
    architecture_bias: float,
    paper_knowledge: PaperKnowledge | None = None,
    *,
    seed_origin: str = "generated",
    generation_context: dict[str, Any] | None = None,
    seed_feedback: SeedFeedback | None = None,
) -> dict[str, Any]:
    provenance = {
        "candidate_key": _candidate_signature(candidate),
        "architecture": candidate.architecture,
        "architecture_bias": architecture_bias,
        "seed_origin": seed_origin,
    }
    if generation_context:
        provenance.update(generation_context)
    if seed_feedback and seed_feedback.is_active():
        provenance["world_model_feedback_binding"] = {
            "recommended_qubit_delta": seed_feedback.recommended_qubit_delta,
            "recommended_factory_lane_delta": seed_feedback.recommended_factory_lane_delta,
            "recommended_pitch_delta_um": seed_feedback.recommended_pitch_delta_um,
            "confidence": seed_feedback.confidence,
            "source_candidate_keys": seed_feedback.source_candidate_keys,
            "applied_qubit_delta": candidate.qubits - int(generation_context.get("base_target_qubits", candidate.qubits)) if generation_context else 0,
        }
    if paper_knowledge is None:
        provenance["paper_guided"] = False
        return provenance
    provenance.update(
        {
            "paper_guided": True,
            "recommended_cavity_q": paper_knowledge.recommended_cavity_q,
            "candidate_cavity_q": candidate.cavity_q,
            "cavity_q_alignment": candidate.cavity_q / max(paper_knowledge.recommended_cavity_q, 1.0),
            "recommended_waveguide_width_um": paper_knowledge.recommended_waveguide_width_um,
            "candidate_waveguide_width_um": candidate.waveguide_width_um,
            "waveguide_alignment_delta_um": candidate.waveguide_width_um - paper_knowledge.recommended_waveguide_width_um,
            "recommended_pitch_um": paper_knowledge.recommended_pitch_um,
            "candidate_pitch_um": candidate.cell_pitch_um,
            "pitch_alignment_delta_um": candidate.cell_pitch_um - paper_knowledge.recommended_pitch_um,
            "recommended_optical_bus_ratio": paper_knowledge.recommended_optical_bus_ratio,
            "candidate_optical_bus_ratio": candidate.optical_bus_count / max(candidate.qubits, 1),
            "paper_recommendation_bindings": _paper_recommendation_bindings(candidate, paper_knowledge),
        }
    )
    return provenance


def generate_seed_candidates(
    spec: DesignSpec,
    seed: int,
    paper_knowledge: PaperKnowledge | None = None,
    seed_feedback: SeedFeedback | None = None,
    injected_candidates: list[CandidateDesign] | None = None,
) -> tuple[list[CandidateDesign], list[dict[str, Any]]]:
    rng = random.Random(seed)
    candidates: list[CandidateDesign] = []
    provenance: list[dict[str, Any]] = []
    architecture_bias = paper_knowledge.architecture_priors if paper_knowledge else spec.architecture_biases()
    active_feedback = seed_feedback if seed_feedback and seed_feedback.is_active() else None
    area_target_scale = max(0.72, min(1.28, math.sqrt(max(spec.max_die_area_mm2, 1.0) / 36.0)))
    logical_scale = max(0.85, min(1.35, 1.0 + 0.08 * spec.target_logical_qubits))
    _recommended_family, _recommended_distance, required_physical_qubits = estimate_required_physical_qubits(spec)
    feedback_qubit_delta = active_feedback.recommended_qubit_delta if active_feedback else 0
    expanded_target = max(spec.target_qubits + max(feedback_qubit_delta, 0), int(round(required_physical_qubits * 1.25)))
    physical_targets = sorted(
        {
            spec.target_qubits,
            max(spec.target_qubits + max(feedback_qubit_delta, 0), required_physical_qubits),
            expanded_target,
        }
    )
    for architecture in _architecture_space(spec, paper_knowledge):
        for base_target in physical_targets:
            for qubit_scale in (0.90, 1.00, 1.15, 1.30):
                qubits = max(8, int(round(base_target * qubit_scale)))
                rows, cols = _grid_shape(qubits)
                pitch_base = {
                    "sensor_dense": 30.0,
                    "hybrid_router": 36.0,
                    "network_node": 42.0,
                }[architecture]
                if paper_knowledge:
                    pitch_base = 0.55 * pitch_base + 0.45 * paper_knowledge.recommended_pitch_um
                if active_feedback:
                    pitch_base += 0.55 * active_feedback.recommended_pitch_delta_um
                qec_pressure = 1.06 if base_target > spec.target_qubits else 0.97
                pitch = pitch_base * area_target_scale * qec_pressure / logical_scale + rng.uniform(-4.0, 4.0)
                optical_ratio = paper_knowledge.recommended_optical_bus_ratio if paper_knowledge else 0.085
                if active_feedback:
                    optical_ratio *= 1.0 + 0.035 * active_feedback.recommended_factory_lane_delta
                if spec.routing_preference == "compact":
                    optical_ratio *= 0.90
                elif spec.routing_preference == "low_loss":
                    optical_ratio *= 1.12
                gain_range = (10.5, 17.0) if spec.max_latency_ns <= 220.0 else (12.0, 22.0)
                mux_choices = (1, 2, 3, 4) if spec.max_latency_ns <= 220.0 else (2, 3, 4, 5, 6)
                candidate = CandidateDesign(
                    architecture=architecture,
                    qubits=qubits,
                    rows=rows,
                    cols=cols,
                    cell_pitch_um=pitch,
                    optical_bus_count=max(2, int(round(qubits * optical_ratio * rng.uniform(0.9, 1.2)))),
                    microwave_line_count=max(4, int(round(qubits / (rng.choice((10, 12, 16, 20)) if spec.target_logical_qubits > 1 else rng.choice((12, 16, 20, 24)))))),
                    resonator_count=max(2, int(round(qubits / rng.choice((6, 8, 10))))),
                    metal_layers=rng.choice((4, 5, 6, 7)),
                    shielding_layers=rng.choice((1, 2, 3)),
                    waveguide_width_um=(paper_knowledge.recommended_waveguide_width_um if paper_knowledge else 0.50) + rng.uniform(-0.08, 0.08),
                    cavity_q=(paper_knowledge.recommended_cavity_q if paper_knowledge else 10 ** rng.uniform(4.8, 6.2)) * rng.uniform(0.75, 1.25),
                    implant_dose_scale=rng.uniform(0.75, 1.20),
                    anneal_quality=rng.uniform(0.78, 0.96),
                    microwave_power_dbm=rng.uniform(0.5, 5.0),
                    control_mux_factor=rng.choice(mux_choices),
                    amplifier_gain_db=rng.uniform(*gain_range),
                    amplifier_noise_temp_k=rng.uniform(2.0, 7.0),
                    detector_qe=rng.uniform(0.62, 0.88),
                )
                scaled_candidate = _scale_candidate_to_bias(candidate, architecture_bias.get(architecture, 1.0), paper_knowledge)
                candidates.append(scaled_candidate)
                provenance.append(
                    _seed_provenance_entry(
                        scaled_candidate,
                        architecture_bias.get(architecture, 1.0),
                        paper_knowledge,
                        seed_origin="generated",
                        generation_context={"base_target_qubits": base_target, "qubit_scale": qubit_scale},
                        seed_feedback=active_feedback,
                    )
                )
    for injected_candidate in injected_candidates or []:
        candidates.append(injected_candidate)
        provenance.append(
            _seed_provenance_entry(
                injected_candidate,
                architecture_bias.get(injected_candidate.architecture, 1.0),
                paper_knowledge,
                seed_origin="round_seed_injection",
                generation_context={"base_target_qubits": injected_candidate.qubits, "qubit_scale": 1.0},
                seed_feedback=active_feedback,
            )
        )
    return candidates, provenance


def _scale_candidate_to_bias(candidate: CandidateDesign, bias: float, paper_knowledge: PaperKnowledge | None = None) -> CandidateDesign:
    target_width = paper_knowledge.recommended_waveguide_width_um if paper_knowledge else candidate.waveguide_width_um
    target_q = paper_knowledge.recommended_cavity_q if paper_knowledge else candidate.cavity_q
    return CandidateDesign(
        architecture=candidate.architecture,
        qubits=candidate.qubits,
        rows=candidate.rows,
        cols=candidate.cols,
        cell_pitch_um=candidate.cell_pitch_um / max(min(bias, 1.3), 0.75),
        optical_bus_count=max(2, int(round(candidate.optical_bus_count * bias))),
        microwave_line_count=max(4, int(round(candidate.microwave_line_count * math.sqrt(bias)))),
        resonator_count=max(2, int(round(candidate.resonator_count * bias))),
        metal_layers=_clamp_int(candidate.metal_layers + (1 if bias > 1.1 else 0), 4, 8),
        shielding_layers=_clamp_int(candidate.shielding_layers + (1 if bias > 1.15 else 0), 1, 4),
        waveguide_width_um=0.55 * candidate.waveguide_width_um + 0.45 * target_width,
        cavity_q=(0.65 * candidate.cavity_q + 0.35 * target_q) * (0.95 + 0.08 * bias),
        implant_dose_scale=max(0.65, candidate.implant_dose_scale / max(bias, 0.85)),
        anneal_quality=min(0.99, candidate.anneal_quality * (0.98 + 0.03 * bias)),
        microwave_power_dbm=candidate.microwave_power_dbm,
        control_mux_factor=_clamp_int(candidate.control_mux_factor + (1 if bias > 1.1 else 0), 1, 8),
        amplifier_gain_db=candidate.amplifier_gain_db,
        amplifier_noise_temp_k=max(1.4, candidate.amplifier_noise_temp_k / max(bias, 0.9)),
        detector_qe=min(0.93, candidate.detector_qe * (0.97 + 0.05 * bias)),
    )


def _robust_logical_gate_threshold(spec: DesignSpec) -> float:
    return 0.78 if spec.qec_enabled else 0.60


ADAPTIVE_MUTATION_STREAK = 2
LATE_REHEAT_GENERATION = 60


def _generation_search_controls(
    generation: int,
    generations: int,
    beam_width: int,
    stagnation_count: int,
    convergence_detected: bool,
) -> tuple[int, float, bool, bool]:
    progress = (generation + 1) / max(generations, 1)
    widened_beam = beam_width
    if progress >= 0.60:
        widened_beam = max(widened_beam, int(math.ceil(beam_width * (1.18 + 0.22 * progress))))
    reheat_active = stagnation_count >= 2 or progress >= 0.72
    if reheat_active:
        widened_beam = max(widened_beam, int(math.ceil(beam_width * 1.55)))
    mutation_temperature = 1.0 + 0.12 * progress + (0.42 if reheat_active else 0.0)
    late_reheat_active = generation >= LATE_REHEAT_GENERATION and convergence_detected
    if late_reheat_active:
        widened_beam = max(widened_beam, int(math.ceil(beam_width * 2.05)))
        mutation_temperature += 0.55
        reheat_active = True
    return widened_beam, mutation_temperature, reheat_active, late_reheat_active


def _frontier_eligible(item: RankedCandidate) -> bool:
    return item.robust_signals is None or item.robust_signals.logical_pass_rate > 0.0


def _adaptive_mutation_targets(spec: DesignSpec, violation_streaks: dict[str, int]) -> set[str]:
    targets: set[str] = set()
    if violation_streaks.get("readout_fidelity", 0) >= ADAPTIVE_MUTATION_STREAK:
        targets.update({"detector_qe", "optical_loss"})
    if violation_streaks.get("logical_error_rate", 0) >= ADAPTIVE_MUTATION_STREAK:
        targets.update({"cavity_q", "optical_loss"})
    if spec.qec_enabled and violation_streaks.get("distance_escalation", 0) >= ADAPTIVE_MUTATION_STREAK:
        targets.add("cavity_q")
    return targets


def _aggregate_seed_feedback(ranked_frontier: list[RankedCandidate]) -> SeedFeedback:
    carriers = [item for item in ranked_frontier if item.world_model_signals is not None][:3]
    if not carriers:
        return SeedFeedback()
    total_weight = 0.0
    qubit_delta = 0.0
    factory_delta = 0.0
    pitch_delta = 0.0
    confidence = 0.0
    source_keys: list[str] = []
    for item in carriers:
        signals = item.world_model_signals
        if signals is None:
            continue
        weight = max(0.15, signals.confidence) * max(0.4, signals.terminal_quality)
        total_weight += weight
        qubit_delta += signals.recommended_qubit_delta * weight
        factory_delta += signals.recommended_factory_lane_delta * weight
        pitch_delta += signals.recommended_pitch_delta_um * weight
        confidence += signals.confidence * weight
        source_keys.append(_candidate_signature(item.candidate))
    if total_weight <= 0.0:
        return SeedFeedback()
    return SeedFeedback(
        recommended_qubit_delta=int(round(qubit_delta / total_weight)),
        recommended_factory_lane_delta=int(round(factory_delta / total_weight)),
        recommended_pitch_delta_um=float(pitch_delta / total_weight),
        confidence=float(confidence / total_weight),
        source_candidate_keys=source_keys,
    )


def mutate_candidate(
    candidate: CandidateDesign,
    spec: DesignSpec,
    rng: random.Random,
    paper_knowledge: PaperKnowledge | None = None,
    *,
    temperature: float = 1.0,
    force_readout_recovery: bool = False,
    forced_mutation_targets: set[str] | None = None,
) -> CandidateDesign:
    temp = max(0.65, float(temperature))
    forced_targets = set(forced_mutation_targets or ())
    qubits = max(8, int(round(candidate.qubits * rng.uniform(0.92, 1.10))))
    rows, cols = _grid_shape(qubits)
    routing_preference = spec.routing_preference.lower()
    pitch_shift = {
        "compact": -2.2,
        "balanced": 0.0,
        "low_loss": 2.5,
    }.get(routing_preference, 0.0)
    if paper_knowledge:
        pitch_shift += 0.12 * (paper_knowledge.recommended_pitch_um - candidate.cell_pitch_um)
    target_width = paper_knowledge.recommended_waveguide_width_um if paper_knowledge else candidate.waveguide_width_um
    target_q = paper_knowledge.recommended_cavity_q if paper_knowledge else candidate.cavity_q
    area_bias = max(0.75, min(1.20, math.sqrt(max(spec.max_die_area_mm2, 1.0) / 36.0)))
    gain_low, gain_high = ((10.0, 18.0) if spec.max_latency_ns <= 220.0 else (10.0, 28.0))
    pitch_scale_low = max(0.84, 1.0 - 0.06 * temp)
    pitch_scale_high = 1.0 + 0.08 * temp
    bus_scale_low = max(0.72, 1.0 - 0.14 * temp)
    bus_scale_high = 1.0 + 0.18 * temp
    readout_qe = candidate.detector_qe * rng.uniform(max(0.90, 1.0 - 0.04 * temp), 1.0 + 0.03 * temp)
    readout_cavity_q = (0.75 * candidate.cavity_q + 0.25 * target_q) * rng.uniform(max(0.82, 1.0 - 0.12 * temp), 1.0 + 0.12 * temp)
    readout_waveguide = 0.7 * candidate.waveguide_width_um * rng.uniform(max(0.92, 1.0 - 0.06 * temp), 1.0 + 0.06 * temp) + 0.3 * target_width
    readout_pitch = candidate.cell_pitch_um * area_bias * rng.uniform(pitch_scale_low, pitch_scale_high) + pitch_shift
    readout_buses = int(round(candidate.optical_bus_count * rng.uniform(bus_scale_low, bus_scale_high) * (1.08 if spec.target_logical_qubits > 1 else 1.0)))
    readout_metals = candidate.metal_layers + rng.choice((-1, 0, 1))
    readout_shielding = candidate.shielding_layers + rng.choice((-1, 0, 1))
    if "detector_qe" in forced_targets:
        readout_qe = max(readout_qe, candidate.detector_qe * rng.uniform(1.05, 1.16))
    if "optical_loss" in forced_targets:
        readout_waveguide = max(readout_waveguide, 0.55 * candidate.waveguide_width_um + 0.45 * target_width + rng.uniform(0.01, 0.06) * temp)
        readout_pitch = max(readout_pitch, candidate.cell_pitch_um + rng.uniform(0.8, 2.6) * temp)
        readout_buses = max(readout_buses, candidate.optical_bus_count + rng.choice((1, 1, 2)))
    if "cavity_q" in forced_targets:
        readout_cavity_q = max(readout_cavity_q, max(candidate.cavity_q, target_q) * rng.uniform(1.08, 1.26))
    if force_readout_recovery:
        qubits = max(qubits, candidate.qubits + max(4, int(round(0.02 * candidate.qubits))))
        readout_pitch = max(readout_pitch, candidate.cell_pitch_um + rng.uniform(0.6, 1.8) * temp)
        readout_buses = max(readout_buses + rng.choice((1, 1, 2)), candidate.optical_bus_count + 1)
        readout_metals = max(readout_metals, candidate.metal_layers + 1)
        readout_shielding = max(readout_shielding, candidate.shielding_layers)
        readout_waveguide = max(readout_waveguide, 0.55 * candidate.waveguide_width_um + 0.45 * target_width + rng.uniform(0.0, 0.03))
        readout_cavity_q = max(readout_cavity_q, candidate.cavity_q * rng.uniform(1.08, 1.22))
        readout_qe = max(readout_qe, candidate.detector_qe * rng.uniform(1.03, 1.09))
        pitch_shift += 0.8
    rows, cols = _grid_shape(qubits)
    return CandidateDesign(
        architecture=rng.choice((candidate.architecture, candidate.architecture, *_architecture_space(spec, paper_knowledge))),
        qubits=qubits,
        rows=rows,
        cols=cols,
        cell_pitch_um=max(22.0, readout_pitch),
        optical_bus_count=max(2, _clamp_int(readout_buses, 2, max(4, qubits // 3))),
        microwave_line_count=max(4, _clamp_int(int(round(candidate.microwave_line_count * rng.uniform(bus_scale_low, min(1.24, 1.0 + 0.18 * temp)) * (1.10 if spec.target_logical_qubits > 1 else 1.0))), 4, max(8, qubits // 2))),
        resonator_count=max(2, _clamp_int(int(round(candidate.resonator_count * rng.uniform(max(0.78, 1.0 - 0.16 * temp), 1.0 + 0.18 * temp))), 2, max(4, qubits))),
        metal_layers=_clamp_int(readout_metals, 4, 8),
        shielding_layers=_clamp_int(readout_shielding, 1, 4),
        waveguide_width_um=min(0.90, max(0.32, readout_waveguide)),
        cavity_q=readout_cavity_q,
        implant_dose_scale=min(1.35, max(0.60, candidate.implant_dose_scale * rng.uniform(max(0.86, 1.0 - 0.10 * temp), 1.0 + 0.10 * temp))),
        anneal_quality=min(0.99, max(0.70, candidate.anneal_quality * rng.uniform(max(0.94, 1.0 - 0.04 * temp), 1.0 + 0.05 * temp))),
        microwave_power_dbm=min(7.0, max(0.0, candidate.microwave_power_dbm + rng.uniform(-0.6 * temp, 0.6 * temp))),
        control_mux_factor=_clamp_int(candidate.control_mux_factor + rng.choice((-1, 0, 0, 1)), 1, 8),
        amplifier_gain_db=min(gain_high, max(gain_low, candidate.amplifier_gain_db + rng.uniform(-1.2 * temp, 1.2 * temp))),
        amplifier_noise_temp_k=min(10.0, max(1.5, candidate.amplifier_noise_temp_k * rng.uniform(max(0.86, 1.0 - 0.10 * temp), 1.0 + 0.08 * temp))),
        detector_qe=min(0.95, max(0.55, readout_qe)),
    )


def _normalized_ratio(goal: float, actual: float, larger_is_better: bool) -> float:
    if larger_is_better:
        return min(actual / max(goal, 1e-9), 1.35)
    return min(goal / max(actual, 1e-9), 1.35)


def _fast_candidate_key(candidate: CandidateDesign) -> tuple[Any, ...]:
    return (
        candidate.architecture,
        candidate.qubits,
        round(candidate.cell_pitch_um, 3),
        candidate.optical_bus_count,
        candidate.microwave_line_count,
        candidate.resonator_count,
        candidate.metal_layers,
        candidate.shielding_layers,
        round(candidate.waveguide_width_um, 3),
        round(candidate.cavity_q, -3),
    )


def _evaluate_fast_robustness(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    dense_signals: FastDenseSignals,
) -> FastRobustSignals:
    scenarios = [
        {},
        {"operating_temp_k": spec.operating_temp_k * 1.08, "magnetic_field_mT": spec.magnetic_field_mT * 0.97},
        {"operating_temp_k": spec.operating_temp_k * 0.94, "optical_power_budget_mw": spec.optical_power_budget_mw * 0.92},
        {"max_latency_ns": spec.max_latency_ns * 0.92, "optical_power_budget_mw": spec.optical_power_budget_mw * 1.05},
    ]
    passes = 0
    logical_passes = 0
    schedule_passes = 0
    gate_floor = metrics.gate_fidelity
    readout_floor = metrics.readout_fidelity
    worst_latency = metrics.latency_ns
    worst_logical_error = 0.0
    for overrides in scenarios:
        scenario_spec = DesignSpec.from_dict(spec.to_dict() | overrides)
        scenario_metrics = evaluate_candidate(scenario_spec, candidate)
        scenario_qec = build_qec_plan(scenario_spec, candidate, scenario_metrics, dense_signals=dense_signals)
        gate_floor = min(gate_floor, scenario_metrics.gate_fidelity)
        readout_floor = min(readout_floor, scenario_metrics.readout_fidelity)
        worst_latency = max(worst_latency, scenario_metrics.latency_ns)
        worst_logical_error = max(worst_logical_error, scenario_qec.logical_error_rate)
        if not scenario_metrics.constraint_violations:
            passes += 1
        if scenario_qec.achievable_logical_qubits >= scenario_spec.target_logical_qubits and scenario_qec.logical_error_rate <= scenario_spec.target_logical_error_rate * 1.5:
            logical_passes += 1
        if scenario_qec.schedule_makespan_ns <= max(scenario_qec.logical_cycle_ns * max(len(scenario_qec.patches), 1) * 8.0, 1.0):
            schedule_passes += 1
    total = max(len(scenarios), 1)
    return FastRobustSignals(
        scenario_pass_rate=passes / total,
        logical_pass_rate=logical_passes / total,
        schedule_pass_rate=schedule_passes / total,
        gate_floor=gate_floor,
        readout_floor=readout_floor,
        worst_latency_ns=worst_latency,
        worst_logical_error_rate=worst_logical_error,
    )


def _escape_candidates(
    parent: RankedCandidate,
    spec: DesignSpec,
    rng: random.Random,
) -> list[CandidateDesign]:
    candidate = parent.candidate
    qec_plan = parent.qec_plan
    dense_signals = parent.dense_signals
    escapes: list[CandidateDesign] = []
    logical_gap = 0
    if qec_plan is not None:
        logical_gap = max(spec.target_logical_qubits - qec_plan.achievable_logical_qubits, 0)
    for direction in (1, 2):
        qubits = candidate.qubits + direction * max(16, logical_gap * max(qec_plan.physical_qubits_per_logical if qec_plan else 24, 16))
        rows, cols = _grid_shape(qubits)
        pitch = candidate.cell_pitch_um + direction * (1.6 if dense_signals and dense_signals.effective_crosstalk_linear > 0.024 else 0.8)
        escapes.append(
            CandidateDesign(
                architecture=rng.choice((candidate.architecture, "hybrid_router", "network_node")),
                qubits=qubits,
                rows=rows,
                cols=cols,
                cell_pitch_um=max(22.0, pitch),
                optical_bus_count=max(candidate.optical_bus_count + direction, 2),
                microwave_line_count=max(candidate.microwave_line_count + 2 * direction, 4),
                resonator_count=max(candidate.resonator_count + direction, 2),
                metal_layers=_clamp_int(candidate.metal_layers + direction, 4, 8),
                shielding_layers=_clamp_int(candidate.shielding_layers + 1, 1, 5),
                waveguide_width_um=candidate.waveguide_width_um,
                cavity_q=candidate.cavity_q * (1.02 + 0.02 * direction),
                implant_dose_scale=max(0.60, candidate.implant_dose_scale * 0.98),
                anneal_quality=min(0.99, candidate.anneal_quality * 1.01),
                microwave_power_dbm=max(0.0, candidate.microwave_power_dbm - 0.2 + 0.1 * direction),
                control_mux_factor=_clamp_int(candidate.control_mux_factor - 1, 1, 8),
                amplifier_gain_db=min(28.0, candidate.amplifier_gain_db + 0.8 * direction),
                amplifier_noise_temp_k=max(1.5, candidate.amplifier_noise_temp_k * 0.98),
                detector_qe=min(0.95, candidate.detector_qe * 1.01),
            )
        )
    return escapes


def _world_model_seed_candidate(parent: RankedCandidate, spec: DesignSpec, rng: random.Random) -> CandidateDesign | None:
    signals = parent.world_model_signals
    if signals is None:
        return None
    candidate = parent.candidate
    guided_qubits = max(8, candidate.qubits + signals.recommended_qubit_delta)
    if abs(guided_qubits - candidate.qubits) < 2 and abs(signals.recommended_pitch_delta_um) < 0.2 and signals.recommended_factory_lane_delta == 0:
        return None
    rows, cols = _grid_shape(guided_qubits)
    guided_buses = _clamp_int(
        candidate.optical_bus_count + max(signals.recommended_factory_lane_delta, 0) + (1 if signals.logical_pressure > 0.46 else 0),
        2,
        max(4, guided_qubits // 2),
    )
    guided_mw = _clamp_int(
        candidate.microwave_line_count + (1 if signals.schedule_pressure > 0.42 or signals.recommended_schedule_bias > 0.08 else 0),
        4,
        max(8, guided_qubits // 2),
    )
    guided_mux = _clamp_int(candidate.control_mux_factor - (1 if signals.schedule_pressure > 0.44 else 0), 1, 8)
    return CandidateDesign(
        architecture=candidate.architecture,
        qubits=guided_qubits,
        rows=rows,
        cols=cols,
        cell_pitch_um=max(22.0, candidate.cell_pitch_um + signals.recommended_pitch_delta_um + rng.uniform(-0.4, 0.6)),
        optical_bus_count=guided_buses,
        microwave_line_count=guided_mw,
        resonator_count=max(candidate.resonator_count, int(round(candidate.resonator_count * (1.0 + 0.03 * max(signals.recommended_factory_lane_delta, 0))))),
        metal_layers=_clamp_int(candidate.metal_layers + (1 if signals.routing_pressure > 0.45 else 0), 4, 8),
        shielding_layers=_clamp_int(candidate.shielding_layers + (1 if signals.routing_pressure > 0.52 else 0), 1, 5),
        waveguide_width_um=candidate.waveguide_width_um,
        cavity_q=candidate.cavity_q,
        implant_dose_scale=candidate.implant_dose_scale,
        anneal_quality=candidate.anneal_quality,
        microwave_power_dbm=max(0.0, candidate.microwave_power_dbm - 0.2 * signals.recommended_schedule_bias),
        control_mux_factor=guided_mux,
        amplifier_gain_db=candidate.amplifier_gain_db,
        amplifier_noise_temp_k=candidate.amplifier_noise_temp_k,
        detector_qe=candidate.detector_qe,
    )


def _qec_distance_escalation_candidate(parent: RankedCandidate, spec: DesignSpec, rng: random.Random) -> CandidateDesign | None:
    qec_plan = parent.qec_plan
    if qec_plan is None or not qec_plan.escalation_rule_triggered:
        return None
    return _build_qec_distance_escalation_seed(parent.candidate, qec_plan, spec, rng)


def _build_qec_distance_escalation_seed(candidate: CandidateDesign, qec_plan: QECPlan, spec: DesignSpec, rng: random.Random) -> CandidateDesign | None:
    if qec_plan is None or not qec_plan.escalation_rule_triggered:
        return None
    target_distance = max(qec_plan.escalated_distance_candidate, 5)
    physical_per_logical, _factory_qubits, total_required = estimate_distance_resources(spec, qec_plan.code_family, target_distance)
    guided_qubits = max(candidate.qubits + max(8, physical_per_logical // 2), total_required + max(physical_per_logical // 3, 8))
    rows, cols = _grid_shape(guided_qubits)
    return CandidateDesign(
        architecture=candidate.architecture,
        qubits=guided_qubits,
        rows=rows,
        cols=cols,
        cell_pitch_um=max(22.0, candidate.cell_pitch_um + 1.0 + rng.uniform(0.0, 1.4)),
        optical_bus_count=_clamp_int(candidate.optical_bus_count + 1, 2, max(4, guided_qubits // 2)),
        microwave_line_count=_clamp_int(candidate.microwave_line_count + 1, 4, max(8, guided_qubits // 2)),
        resonator_count=max(candidate.resonator_count + 1, candidate.resonator_count),
        metal_layers=_clamp_int(candidate.metal_layers + 1, 4, 8),
        shielding_layers=_clamp_int(candidate.shielding_layers + 1, 1, 5),
        waveguide_width_um=min(0.90, candidate.waveguide_width_um + 0.02),
        cavity_q=candidate.cavity_q * 1.08,
        implant_dose_scale=max(0.60, candidate.implant_dose_scale * 0.98),
        anneal_quality=min(0.99, candidate.anneal_quality * 1.01),
        microwave_power_dbm=max(0.0, candidate.microwave_power_dbm - 0.15),
        control_mux_factor=_clamp_int(candidate.control_mux_factor - 1, 1, 8),
        amplifier_gain_db=min(28.0, candidate.amplifier_gain_db + 0.6),
        amplifier_noise_temp_k=max(1.5, candidate.amplifier_noise_temp_k * 0.98),
        detector_qe=min(0.95, candidate.detector_qe * 1.02),
    )


def build_qec_escalation_seed(candidate: CandidateDesign, qec_plan: QECPlan, spec: DesignSpec, seed: int) -> CandidateDesign | None:
    return _build_qec_distance_escalation_seed(candidate, qec_plan, spec, random.Random(seed))


def _build_paper_bias_audit(paper_knowledge: PaperKnowledge | None, ranked_frontier: list[RankedCandidate]) -> dict[str, Any]:
    if paper_knowledge is None:
        return {}
    total_frontier = max(len(ranked_frontier), 1)
    architecture_counts = {name: 0 for name in {item.candidate.architecture for item in ranked_frontier} | set(paper_knowledge.architecture_priors)}
    for item in ranked_frontier:
        architecture_counts[item.candidate.architecture] = architecture_counts.get(item.candidate.architecture, 0) + 1
    actual_ratios = {name: count / total_frontier for name, count in architecture_counts.items()}
    total_prior = sum(max(value, 0.0) for value in paper_knowledge.architecture_priors.values()) or 1.0
    expected_ratios = {name: max(value, 0.0) / total_prior for name, value in paper_knowledge.architecture_priors.items()}
    warnings: list[str] = []
    for architecture, expected in expected_ratios.items():
        actual = actual_ratios.get(architecture, 0.0)
        if abs(actual - expected) > 0.28:
            warnings.append(f"architecture `{architecture}` actual ratio {actual:.2f} diverges from paper prior ratio {expected:.2f}")
    dominant_topics = sorted(paper_knowledge.topic_counts.items(), key=lambda item: item[1], reverse=True)[:2]
    return {
        "topic_counts": paper_knowledge.topic_counts,
        "dominant_topics": dominant_topics,
        "architecture_counts": architecture_counts,
        "actual_architecture_ratios": actual_ratios,
        "expected_architecture_ratios": expected_ratios,
        "warnings": warnings,
    }


def _rank_key(item: RankedCandidate, spec: DesignSpec) -> tuple[float, float, float, float, float, float, float, float]:
    logical_feasible = 0.0
    physical_feasible = 0.0
    robust_logical_gate = 0.0
    robust_logical = 0.0
    robust_schedule = 0.0
    infeasibility = 0.0
    if item.qec_plan is not None:
        logical_feasible = 1.0 if not any(v in {"logical_qubits", "logical_error_rate"} for v in item.qec_plan.violations) else 0.0
        infeasibility += max(item.qec_plan.target_logical_qubits - item.qec_plan.achievable_logical_qubits, 0) * 1.2
        infeasibility += min(item.qec_plan.logical_error_rate / 1e-3, 50.0) * 0.08
    if not item.metrics.constraint_violations:
        physical_feasible = 1.0
    infeasibility += 2.0 * len(item.metrics.constraint_violations)
    infeasibility += max(item.metrics.latency_ns - 220.0, 0.0) / 120.0
    infeasibility += max(item.metrics.die_area_mm2 - 40.0, 0.0) / 20.0
    infeasibility += max(0.80 - item.metrics.yield_estimate, 0.0) * 5.0
    if item.robust_signals is not None:
        robust_logical = item.robust_signals.logical_pass_rate
        robust_schedule = item.robust_signals.schedule_pass_rate
        robust_logical_gate = 1.0 if item.robust_signals.logical_pass_rate >= _robust_logical_gate_threshold(spec) else 0.0
        infeasibility += max(0.60 - item.robust_signals.logical_pass_rate, 0.0) * 4.0
        infeasibility += max(0.60 - item.robust_signals.scenario_pass_rate, 0.0) * 3.0
    return (logical_feasible, physical_feasible, robust_logical_gate, robust_logical, robust_schedule, -infeasibility, item.score, item.metrics.gate_fidelity)


def _repair_candidate(
    candidate: CandidateDesign,
    spec: DesignSpec,
    dense_signals: FastDenseSignals,
    neural_signals: FrozenNeuralSignals | None = None,
    qec_plan: QECPlan | None = None,
    world_model_signals: WorldModelSignals | None = None,
) -> CandidateDesign:
    updated_pitch = candidate.cell_pitch_um
    updated_shield = candidate.shielding_layers
    updated_metal = candidate.metal_layers
    updated_mw = candidate.microwave_line_count
    updated_mux = candidate.control_mux_factor
    updated_buses = candidate.optical_bus_count
    updated_qubits = candidate.qubits
    if dense_signals.effective_crosstalk_linear > 0.030:
        updated_pitch += 1.6
        updated_shield = _clamp_int(candidate.shielding_layers + 1, 1, 5)
        updated_metal = _clamp_int(candidate.metal_layers + 1, 4, 8)
        updated_mw = _clamp_int(candidate.microwave_line_count + 1, 4, max(8, candidate.qubits // 2))
        updated_mux = _clamp_int(candidate.control_mux_factor - 1, 1, 8)
    if dense_signals.min_spacing_um < 0.86 * candidate.cell_pitch_um:
        updated_pitch += 1.2
        updated_buses = _clamp_int(candidate.optical_bus_count + 1, 2, max(4, candidate.qubits // 3))
    if dense_signals.optical_load_std > 0.70:
        updated_buses = _clamp_int(candidate.optical_bus_count + 1, 2, max(4, candidate.qubits // 3))
    if neural_signals is not None:
        updated_pitch = max(22.0, updated_pitch + neural_signals.pitch_delta_um)
        updated_buses = _clamp_int(updated_buses + neural_signals.optical_bus_delta, 2, max(4, candidate.qubits // 2))
        updated_mw = _clamp_int(updated_mw + neural_signals.microwave_line_delta, 4, max(8, candidate.qubits // 2))
        updated_shield = _clamp_int(updated_shield + neural_signals.shielding_delta, 1, 5)
        updated_mux = _clamp_int(updated_mux + neural_signals.mux_delta, 1, 8)
    if qec_plan is not None:
        if qec_plan.achievable_logical_qubits < spec.target_logical_qubits:
            logical_gap = spec.target_logical_qubits - qec_plan.achievable_logical_qubits
            updated_qubits += max(16, logical_gap * max(qec_plan.physical_qubits_per_logical, 16))
        if qec_plan.logical_error_rate > spec.target_logical_error_rate:
            updated_qubits += max(12, qec_plan.physical_qubits_per_logical)
            updated_pitch += 1.1
            updated_shield = _clamp_int(updated_shield + 1, 1, 5)
            updated_metal = _clamp_int(updated_metal + 1, 4, 8)
        if qec_plan.magic_state_rate_per_us < max(0.3 * spec.target_logical_qubits, 0.5):
            updated_mw = _clamp_int(updated_mw + 1, 4, max(8, updated_qubits // 2))
            updated_buses = _clamp_int(updated_buses + 1, 2, max(4, updated_qubits // 2))
        if qec_plan.schedule_makespan_ns > max(qec_plan.logical_cycle_ns * max(len(qec_plan.patches), 1) * 6.0, 1.0):
            updated_mux = _clamp_int(updated_mux - 1, 1, 8)
            updated_metal = _clamp_int(updated_metal + 1, 4, 8)
            updated_mw = _clamp_int(updated_mw + 1, 4, max(8, updated_qubits // 2))
    if spec.max_latency_ns < 210.0:
        updated_mw = _clamp_int(updated_mw + 1, 4, max(8, max(updated_qubits, candidate.qubits) // 2))
        updated_mux = _clamp_int(updated_mux - 1, 1, 8)
        updated_metal = _clamp_int(updated_metal + 1, 4, 8)
    if world_model_signals is not None:
        updated_qubits += max(0, world_model_signals.recommended_qubit_delta)
        updated_pitch = max(22.0, updated_pitch + world_model_signals.recommended_pitch_delta_um)
        updated_buses = _clamp_int(updated_buses + max(0, world_model_signals.recommended_factory_lane_delta // 2), 2, max(4, max(updated_qubits, candidate.qubits) // 2))
        if world_model_signals.schedule_pressure > 0.42:
            updated_mw = _clamp_int(updated_mw + 1, 4, max(8, max(updated_qubits, candidate.qubits) // 2))
            updated_mux = _clamp_int(updated_mux - 1, 1, 8)
    updated_qubits = min(max(updated_qubits, 8), max(spec.target_qubits * 2, candidate.qubits + 24))
    rows, cols = _grid_shape(updated_qubits)
    return CandidateDesign(
        architecture=candidate.architecture,
        qubits=updated_qubits,
        rows=rows,
        cols=cols,
        cell_pitch_um=max(22.0, updated_pitch),
        optical_bus_count=updated_buses,
        microwave_line_count=updated_mw,
        resonator_count=max(candidate.resonator_count, int(round(candidate.resonator_count * (1.0 + 0.06 * (updated_shield - candidate.shielding_layers))))),
        metal_layers=updated_metal,
        shielding_layers=updated_shield,
        waveguide_width_um=candidate.waveguide_width_um,
        cavity_q=candidate.cavity_q * (1.0 + 0.02 * (updated_metal - candidate.metal_layers)) * (neural_signals.cavity_scale if neural_signals else 1.0),
        implant_dose_scale=candidate.implant_dose_scale,
        anneal_quality=candidate.anneal_quality,
        microwave_power_dbm=min(7.0, max(0.0, candidate.microwave_power_dbm + (neural_signals.power_delta_dbm if neural_signals else 0.0))),
        control_mux_factor=updated_mux,
        amplifier_gain_db=max(10.0, candidate.amplifier_gain_db - (1.6 if spec.max_latency_ns < 210.0 else 0.0)),
        amplifier_noise_temp_k=candidate.amplifier_noise_temp_k,
        detector_qe=candidate.detector_qe,
    )


def score_candidate(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    dense_signals: FastDenseSignals | None = None,
    neural_signals: FrozenNeuralSignals | None = None,
    qec_plan: QECPlan | None = None,
    world_model_signals: WorldModelSignals | None = None,
    robust_signals: FastRobustSignals | None = None,
    mc: MonteCarloSummary | None = None,
    paper_knowledge: PaperKnowledge | None = None,
) -> tuple[float, float]:
    weights = spec.objective_weights
    placement_term = 0.0
    crosstalk_term = 0.0
    if dense_signals:
        placement_term = max(
            0.0,
            min(
                1.35,
                0.62 * dense_signals.placement_score
                + 0.42 * dense_signals.min_spacing_um / max(candidate.cell_pitch_um, 1e-6)
                + 0.18 / (1.0 + dense_signals.mean_offset_um / max(candidate.cell_pitch_um, 1e-6)),
            ),
        )
        crosstalk_term = max(
            0.0,
            min(
                1.35,
                1.12
                - 4.0 * dense_signals.effective_crosstalk_linear
                - 0.40 * dense_signals.spectral_radius
                - 0.12 * dense_signals.optical_load_std
                - 0.10 * dense_signals.microwave_load_std,
            ),
        )
    score_terms = {
        "fidelity": 0.55 * _normalized_ratio(spec.target_gate_fidelity, metrics.gate_fidelity, True)
        + 0.45 * _normalized_ratio(spec.target_readout_fidelity, metrics.readout_fidelity, True),
        "coherence": _normalized_ratio(spec.target_t2_us, metrics.t2_us, True),
        "yield": _normalized_ratio(spec.min_yield, metrics.yield_estimate, True),
        "power": _normalized_ratio(spec.max_power_mw, metrics.power_mw, False),
        "latency": _normalized_ratio(spec.max_latency_ns, metrics.latency_ns, False),
        "area": _normalized_ratio(spec.max_die_area_mm2, metrics.die_area_mm2, False),
        "scalability": metrics.scalability_score,
        "robustness": mc.robustness_mean if mc else metrics.robustness_margin,
        "placement": placement_term,
        "crosstalk": crosstalk_term,
        "logical": 0.0,
        "qec": 0.0,
        "schedule": 0.0,
        "world_model": 0.0,
    }
    if qec_plan is not None:
        logical_term = max(
            0.0,
            min(
                1.35,
                0.60 * qec_plan.achievable_logical_qubits / max(spec.target_logical_qubits, 1)
                + 0.40 * spec.target_logical_error_rate / max(qec_plan.logical_error_rate, 1e-12),
            ),
        )
        qec_term = max(
            0.0,
            min(
                1.35,
                0.22 * qec_plan.logical_success_probability
                + 0.18 * qec_plan.logical_yield
                + 0.20 * min(qec_plan.threshold / max(qec_plan.physical_error_rate, 1e-12), 1.35)
                + 0.16 / (1.0 + qec_plan.logical_cycle_ns / max(spec.max_latency_ns * 2.0, 1e-6))
                + 0.12 * qec_plan.decoder_locality_score
                + 0.12 * min(qec_plan.magic_state_rate_per_us / max(0.3 * spec.target_logical_qubits, 1e-6), 1.35),
            ),
        )
        score_terms["logical"] = logical_term
        score_terms["qec"] = qec_term
        schedule_baseline = max(qec_plan.logical_cycle_ns * max(len(qec_plan.patches), 1), 1e-6)
        schedule_term = max(
            0.0,
            min(
                1.35,
                0.24 / (1.0 + qec_plan.schedule_makespan_ns / (6.0 * schedule_baseline))
                + 0.18 / (1.0 + qec_plan.schedule_critical_path_ns / (5.0 * schedule_baseline))
                + 0.18 * qec_plan.decoder_locality_score
                + 0.16 * min(qec_plan.surgery_throughput_ops_per_us / max(0.6 * max(spec.target_logical_qubits, 1), 1e-6), 1.35)
                + 0.12 * (1.0 - abs(qec_plan.factory_utilization - 0.78))
                + 0.12 * (1.0 - abs(qec_plan.decoder_utilization - 0.42)),
            ),
        )
        score_terms["schedule"] = schedule_term
    if world_model_signals is not None:
        world_model_term = max(
            0.0,
            min(
                1.35,
                0.42 * world_model_signals.terminal_quality
                + 0.16 * world_model_signals.confidence
                + 0.16 * (1.0 - world_model_signals.schedule_pressure)
                + 0.14 * (1.0 - world_model_signals.logical_pressure)
                + 0.07 * (1.0 - world_model_signals.routing_pressure)
                + 0.05 * (1.0 - world_model_signals.thermal_pressure),
            ),
        )
        score_terms["world_model"] = world_model_term
    physical_keys = ["fidelity", "coherence", "yield", "power", "latency", "area", "scalability", "robustness", "placement", "crosstalk"]
    logical_keys = ["logical", "qec", "schedule", "world_model"]
    physical_weight = sum(weights.get(key, 0.0) for key in physical_keys) or 1.0
    logical_weight = sum(weights.get(key, 0.0) for key in logical_keys) or 1.0
    physical_weighted = sum(score_terms.get(key, 0.0) * weights.get(key, 0.0) for key in physical_keys) / physical_weight
    logical_weighted = sum(score_terms.get(key, 0.0) * weights.get(key, 0.0) for key in logical_keys) / logical_weight
    logical_priority = 0.58 if spec.qec_enabled else 0.22
    weighted = (1.0 - logical_priority) * physical_weighted + logical_priority * logical_weighted
    hard_penalty = 0.04 * metrics.route_drc_violations
    violation_penalties = {
        "die_area": 0.36,
        "power": 0.18,
        "latency": 0.34,
        "coherence": 0.18,
        "gate_fidelity": 0.20,
        "readout_fidelity": 0.22,
        "yield": 0.20,
    }
    hard_penalty += sum(violation_penalties.get(item, 0.12) for item in metrics.constraint_violations)
    hard_penalty += 0.08 * max(metrics.latency_ns / max(spec.max_latency_ns, 1e-6) - 1.0, 0.0)
    hard_penalty += 0.08 * max(metrics.die_area_mm2 / max(spec.max_die_area_mm2, 1e-6) - 1.0, 0.0)
    if dense_signals:
        hard_penalty += 0.06 * max(dense_signals.effective_crosstalk_linear - 0.035, 0.0) * 10.0
        hard_penalty += 0.04 * max(0.75 - dense_signals.min_spacing_um / max(candidate.cell_pitch_um, 1e-6), 0.0)
    if qec_plan is not None:
        violation_penalties = {
            "logical_qubits": 0.42,
            "logical_error_rate": 0.36,
            "qec_latency": 0.18,
            "magic_state_rate": 0.20,
        }
        hard_penalty += sum(violation_penalties.get(item, 0.08) for item in qec_plan.violations)
        hard_penalty += 0.04 * max(qec_plan.schedule_makespan_ns / max(qec_plan.logical_cycle_ns * max(len(qec_plan.patches), 1) * 8.0, 1e-6) - 1.0, 0.0)
    if world_model_signals is not None:
        hard_penalty += 0.03 * max(world_model_signals.schedule_pressure - 0.55, 0.0) * 4.0
        hard_penalty += 0.03 * max(world_model_signals.logical_pressure - 0.55, 0.0) * 4.0
    if robust_signals is not None:
        hard_penalty += 0.16 * max(0.72 - robust_signals.logical_pass_rate, 0.0)
        hard_penalty += 0.10 * max(0.68 - robust_signals.schedule_pass_rate, 0.0)
        hard_penalty += 0.08 * max(0.75 - robust_signals.scenario_pass_rate, 0.0)
    soft_penalty = max(0.0, 0.35 - metrics.routing_capacity) * 0.25
    robustness_score = mc.pass_rate if mc else metrics.robustness_margin
    paper_bonus = 0.0
    if paper_knowledge:
        q_alignment = 1.0 - min(abs(math.log10(max(candidate.cavity_q, 1.0) / max(paper_knowledge.recommended_cavity_q, 1.0))) / 1.2, 1.0)
        width_alignment = 1.0 - min(abs(candidate.waveguide_width_um - paper_knowledge.recommended_waveguide_width_um) / 0.28, 1.0)
        pitch_alignment = 1.0 - min(abs(candidate.cell_pitch_um - paper_knowledge.recommended_pitch_um) / 18.0, 1.0)
        paper_bonus = 0.035 * max(0.0, (q_alignment + width_alignment + pitch_alignment) / 3.0)
    neural_bonus = 0.0
    if neural_signals is not None:
        neural_bonus = (
            0.055 * neural_signals.design_quality
            + 0.020 * neural_signals.architecture_affinity
            + 0.018 * neural_signals.prototype_affinity
            + 0.012 * neural_signals.confidence
            - 0.022 * neural_signals.latency_risk
            - 0.028 * neural_signals.crosstalk_risk
            - 0.015 * neural_signals.routing_risk
            - 0.012 * neural_signals.yield_risk
        )
    robust_bonus = 0.0
    if robust_signals is not None:
        robust_bonus = (
            0.06 * robust_signals.scenario_pass_rate
            + 0.08 * robust_signals.logical_pass_rate
            + 0.06 * robust_signals.schedule_pass_rate
            + 0.03 * _normalized_ratio(spec.target_gate_fidelity, robust_signals.gate_floor, True)
            + 0.03 * _normalized_ratio(spec.target_readout_fidelity, robust_signals.readout_floor, True)
        )
    total_score = weighted + 0.08 * robustness_score + robust_bonus + paper_bonus + neural_bonus - hard_penalty - soft_penalty
    if robust_signals is not None:
        gate_threshold = _robust_logical_gate_threshold(spec)
        if robust_signals.logical_pass_rate < gate_threshold:
            gate_gap = gate_threshold - robust_signals.logical_pass_rate
            total_score = min(total_score, -1.0 - 12.0 * gate_gap)
    return total_score, robustness_score


def optimize_design(
    spec: DesignSpec,
    seed: int = 7,
    generations: int = 7,
    beam_width: int = 8,
    mutations_per_parent: int = 7,
    monte_carlo_trials: int = 256,
    paper_knowledge: PaperKnowledge | None = None,
    requirements_bundle: RequirementBundle | None = None,
    seed_feedback: SeedFeedback | None = None,
    injected_seed_candidates: list[CandidateDesign] | None = None,
) -> OptimizationResult:
    rng = random.Random(seed)
    population, seed_provenance = generate_seed_candidates(
        spec,
        seed,
        paper_knowledge=paper_knowledge,
        seed_feedback=seed_feedback,
        injected_candidates=injected_seed_candidates,
    )
    search_log: list[dict[str, Any]] = []
    best_ranked: list[RankedCandidate] = []
    propagation_ranked: list[RankedCandidate] = []
    dense_cache: dict[tuple[Any, ...], FastDenseSignals] = {}
    prototype_bank: list[tuple[list[float], float]] = []
    stagnation_count = 0
    best_score_seen = -1.0e18
    best_score_history: list[float] = []
    readout_violation_streak = 0
    violation_streaks = {"readout_fidelity": 0, "logical_error_rate": 0, "distance_escalation": 0}

    for generation in range(generations):
        recent_scores = best_score_history[-4:]
        convergence_detected = generation >= LATE_REHEAT_GENERATION and len(recent_scores) >= 4 and max(recent_scores) - min(recent_scores) <= 5.0e-4
        effective_beam_width, mutation_temperature, reheat_active, late_reheat_active = _generation_search_controls(
            generation,
            generations,
            beam_width,
            stagnation_count,
            convergence_detected,
        )
        ranked: list[RankedCandidate] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for candidate in population:
            key = (
                candidate.architecture,
                candidate.qubits,
                round(candidate.cell_pitch_um, 1),
                candidate.optical_bus_count,
                candidate.microwave_line_count,
                candidate.resonator_count,
                candidate.metal_layers,
                candidate.shielding_layers,
                round(candidate.waveguide_width_um, 2),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            metrics = evaluate_candidate(spec, candidate)
            dense_key = _fast_candidate_key(candidate)
            dense_signals = dense_cache.get(dense_key)
            if dense_signals is None:
                dense_signals = analyze_candidate_dense_fast(spec, candidate, samples=4, trials=20, seed_offset=generation)
                dense_cache[dense_key] = dense_signals
            qec_plan = build_qec_plan(spec, candidate, metrics, dense_signals=dense_signals)
            robust_signals = _evaluate_fast_robustness(spec, candidate, metrics, dense_signals)
            neural_signals = evaluate_frozen_neural_surrogate(
                spec,
                candidate,
                metrics,
                dense_signals,
                paper_knowledge=paper_knowledge,
                requirements_bundle=requirements_bundle,
                prototype_bank=prototype_bank,
            )
            neural_signals = calibrate_frozen_neural_surrogate(neural_signals, metrics, qec_plan, robust_signals.logical_pass_rate)
            world_model_signals = simulate_world_model_rollout(
                spec,
                candidate,
                metrics,
                dense_signals,
                qec_plan,
                neural_signals=neural_signals,
            )
            world_model_signals = calibrate_world_model_signals(world_model_signals, qec_plan, robust_signals.schedule_pass_rate)
            score, robustness = score_candidate(
                spec,
                candidate,
                metrics,
                dense_signals=dense_signals,
                neural_signals=neural_signals,
                qec_plan=qec_plan,
                world_model_signals=world_model_signals,
                robust_signals=robust_signals,
                paper_knowledge=paper_knowledge,
            )
            ranked.append(RankedCandidate(candidate, metrics, score, robustness, dense_signals, neural_signals, qec_plan, world_model_signals, robust_signals))

        ranked.sort(key=lambda item: _rank_key(item, spec), reverse=True)
        eligible_frontier = [item for item in ranked if _frontier_eligible(item)]
        best_ranked = eligible_frontier[:effective_beam_width]
        propagation_ranked = best_ranked or ranked[: max(1, min(effective_beam_width, len(ranked)))]
        prototype_bank.extend(
            (item.neural_signals.latent_vector, item.score)
            for item in propagation_ranked
            if item.neural_signals is not None
        )
        prototype_bank = prototype_bank[-24:]
        current_best_score = propagation_ranked[0].score if propagation_ranked else -1.0e18
        best_score_history.append(current_best_score)
        if current_best_score > best_score_seen + 1.0e-4:
            best_score_seen = current_best_score
            stagnation_count = 0
        else:
            stagnation_count += 1
        best_candidate = propagation_ranked[0] if propagation_ranked else None
        best_has_readout_violation = bool(
            best_candidate
            and (
                "readout_fidelity" in best_candidate.metrics.constraint_violations
                or (
                    best_candidate.robust_signals is not None
                    and best_candidate.robust_signals.readout_floor < spec.target_readout_fidelity
                )
            )
        )
        readout_violation_streak = readout_violation_streak + 1 if best_has_readout_violation else 0
        force_readout_mutation = readout_violation_streak >= 2
        logical_error_active = bool(
            best_candidate
            and best_candidate.qec_plan is not None
            and (
                "logical_error_rate" in best_candidate.qec_plan.violations
                or best_candidate.qec_plan.logical_error_rate > spec.target_logical_error_rate
            )
        )
        escalation_active = bool(best_candidate and best_candidate.qec_plan is not None and best_candidate.qec_plan.escalation_rule_triggered)
        violation_streaks["readout_fidelity"] = violation_streaks["readout_fidelity"] + 1 if best_has_readout_violation else 0
        violation_streaks["logical_error_rate"] = violation_streaks["logical_error_rate"] + 1 if logical_error_active else 0
        violation_streaks["distance_escalation"] = violation_streaks["distance_escalation"] + 1 if escalation_active else 0
        adaptive_targets = _adaptive_mutation_targets(spec, violation_streaks)
        mutation_budget = mutations_per_parent + (2 if late_reheat_active else 0)
        search_log.append(
            {
                "generation": generation,
                "candidate_count": len(ranked),
                "eligible_frontier_count": len(eligible_frontier),
                "frontier_candidate_count": len(best_ranked),
                "frontier_gate_filtered_count": max(len(ranked) - len(eligible_frontier), 0),
                "effective_beam_width": effective_beam_width,
                "mutation_temperature": mutation_temperature,
                "mutation_budget": mutation_budget,
                "reheat_active": reheat_active,
                "late_reheat_active": late_reheat_active,
                "convergence_detected": convergence_detected,
                "readout_violation_streak": readout_violation_streak,
                "forced_readout_mutation": force_readout_mutation,
                "adaptive_mutation_targets": sorted(adaptive_targets),
                "violation_streaks": dict(violation_streaks),
                "best_score": best_candidate.score if best_candidate else 0.0,
                "best_architecture": best_candidate.candidate.architecture if best_candidate else None,
                "best_constraint_violations": best_candidate.metrics.constraint_violations if best_candidate else [],
                "best_dense_crosstalk_db": best_candidate.dense_signals.effective_crosstalk_db if best_candidate and best_candidate.dense_signals else None,
                "best_neural_quality": best_candidate.neural_signals.design_quality if best_candidate and best_candidate.neural_signals else None,
                "best_logical_qubits": best_candidate.qec_plan.achievable_logical_qubits if best_candidate and best_candidate.qec_plan else None,
                "best_schedule_makespan_ns": best_candidate.qec_plan.schedule_makespan_ns if best_candidate and best_candidate.qec_plan else None,
                "best_world_quality": best_candidate.world_model_signals.terminal_quality if best_candidate and best_candidate.world_model_signals else None,
                "best_robust_logical_pass_rate": best_candidate.robust_signals.logical_pass_rate if best_candidate and best_candidate.robust_signals else None,
                "best_qec_escalation_triggered": best_candidate.qec_plan.escalation_rule_triggered if best_candidate and best_candidate.qec_plan else None,
                "best_qec_feasibility_warnings": best_candidate.qec_plan.feasibility_warnings if best_candidate and best_candidate.qec_plan else [],
            }
        )
        next_population = [item.candidate for item in propagation_ranked]
        world_model_guided_seed_count = 0
        qec_escalation_injected = 0
        for parent in propagation_ranked:
            if parent.dense_signals is not None:
                next_population.append(_repair_candidate(parent.candidate, spec, parent.dense_signals, parent.neural_signals, parent.qec_plan, parent.world_model_signals))
            world_model_seed = _world_model_seed_candidate(parent, spec, rng)
            if world_model_seed is not None:
                next_population.append(world_model_seed)
                world_model_guided_seed_count += 1
            for _ in range(mutation_budget):
                next_population.append(
                    mutate_candidate(
                        parent.candidate,
                        spec,
                        rng,
                        paper_knowledge=paper_knowledge,
                        temperature=mutation_temperature,
                        force_readout_recovery=force_readout_mutation,
                        forced_mutation_targets=adaptive_targets,
                    )
                )
            if force_readout_mutation or adaptive_targets:
                targeted_mutation_targets = set(adaptive_targets)
                if force_readout_mutation:
                    targeted_mutation_targets.update({"detector_qe", "optical_loss"})
                next_population.append(
                    mutate_candidate(
                        parent.candidate,
                        spec,
                        rng,
                        paper_knowledge=paper_knowledge,
                        temperature=max(mutation_temperature, 1.2 + (0.15 if late_reheat_active else 0.0)),
                        force_readout_recovery=force_readout_mutation,
                        forced_mutation_targets=targeted_mutation_targets,
                    )
                )
            if generation == 2 or late_reheat_active:
                escalation_candidate = _qec_distance_escalation_candidate(parent, spec, rng)
                if escalation_candidate is not None:
                    next_population.append(escalation_candidate)
                    qec_escalation_injected += 1
        if len(search_log) >= 2:
            recent_signatures = [tuple(entry.get("best_constraint_violations", [])) for entry in search_log[-2:]]
            if recent_signatures[0] == recent_signatures[1] and recent_signatures[0]:
                for parent in propagation_ranked[:2]:
                    next_population.extend(_escape_candidates(parent, spec, rng))
        search_log[-1]["world_model_guided_seed_count"] = world_model_guided_seed_count
        search_log[-1]["qec_escalation_injected"] = qec_escalation_injected
        population = next_population

    robust_ranked: list[RankedCandidate] = []
    for offset, ranked in enumerate(propagation_ranked):
        mc = run_monte_carlo(spec, ranked.candidate, trials=monte_carlo_trials, seed=seed + 100 + offset)
        dense_signals = ranked.dense_signals
        if dense_signals is None:
            dense_signals = analyze_candidate_dense_fast(spec, ranked.candidate, samples=6, trials=32, seed_offset=offset + generations)
        qec_plan = ranked.qec_plan or build_qec_plan(spec, ranked.candidate, ranked.metrics, dense_signals=dense_signals)
        robust_signals = ranked.robust_signals or _evaluate_fast_robustness(spec, ranked.candidate, ranked.metrics, dense_signals)
        neural_signals = evaluate_frozen_neural_surrogate(
            spec,
            ranked.candidate,
            ranked.metrics,
            dense_signals,
            paper_knowledge=paper_knowledge,
            requirements_bundle=requirements_bundle,
            prototype_bank=prototype_bank,
        )
        neural_signals = calibrate_frozen_neural_surrogate(neural_signals, ranked.metrics, qec_plan, robust_signals.logical_pass_rate)
        world_model_signals = simulate_world_model_rollout(
            spec,
            ranked.candidate,
            ranked.metrics,
            dense_signals,
            qec_plan,
            neural_signals=neural_signals,
        )
        world_model_signals = calibrate_world_model_signals(world_model_signals, qec_plan, robust_signals.schedule_pass_rate)
        score, robustness = score_candidate(
            spec,
            ranked.candidate,
            ranked.metrics,
            dense_signals=dense_signals,
            neural_signals=neural_signals,
            qec_plan=qec_plan,
            world_model_signals=world_model_signals,
            robust_signals=robust_signals,
            mc=mc,
            paper_knowledge=paper_knowledge,
        )
        robust_ranked.append(RankedCandidate(ranked.candidate, ranked.metrics, score, robustness, dense_signals, neural_signals, qec_plan, world_model_signals, robust_signals))
    robust_ranked.sort(key=lambda item: _rank_key(item, spec), reverse=True)
    robust_frontier = [item for item in robust_ranked if _frontier_eligible(item)]
    selected = robust_frontier[0] if robust_frontier else robust_ranked[0]
    selected_mc = run_monte_carlo(spec, selected.candidate, trials=monte_carlo_trials, seed=seed + 999)
    paper_bias_audit = _build_paper_bias_audit(paper_knowledge, robust_frontier or robust_ranked)
    aggregated_feedback = _aggregate_seed_feedback(robust_frontier or robust_ranked)

    return OptimizationResult(
        best_candidate=selected.candidate,
        best_metrics=selected.metrics,
        monte_carlo=selected_mc,
        pareto_like_frontier=robust_frontier,
        search_log=search_log,
        seed_provenance=seed_provenance,
        paper_bias_audit=paper_bias_audit,
        selected_ranked_candidate=selected,
        seed_feedback=aggregated_feedback.to_dict(),
    )
