from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from .dense_placement import FastDenseSignals
from .neural_surrogate import FrozenNeuralSignals
from .qec import QECPlan
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec


_RNG = np.random.default_rng(271828)
_STATE_DIM = 18
_A = _RNG.normal(0.0, 0.24, size=(_STATE_DIM, _STATE_DIM))
_B = _RNG.normal(0.0, 0.18, size=(7, _STATE_DIM))
_C = _RNG.normal(0.0, 0.10, size=(10, _STATE_DIM))
_HEAD = _RNG.normal(0.0, 0.20, size=(_STATE_DIM, 6))
_BIAS = _RNG.normal(0.0, 0.05, size=(6,))

_ACTION_NAMES = [
    "increase_pitch",
    "increase_shielding",
    "increase_optical_buses",
    "increase_microwave_lines",
    "reduce_mux",
    "expand_physical_qubits",
    "add_factory_parallelism",
]


@dataclass(slots=True)
class WorldModelStep:
    step: int
    action: str
    predicted_quality: float
    schedule_pressure: float
    logical_pressure: float
    routing_pressure: float
    thermal_pressure: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorldModelSignals:
    terminal_quality: float
    schedule_pressure: float
    logical_pressure: float
    routing_pressure: float
    thermal_pressure: float
    confidence: float
    recommended_qubit_delta: int
    recommended_factory_lane_delta: int
    recommended_pitch_delta_um: float
    recommended_schedule_bias: float
    rollout: list[WorldModelStep] = field(default_factory=list)
    latent_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "rollout": [step.to_dict() for step in self.rollout],
        }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


def _state_vector(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    dense_signals: FastDenseSignals,
    qec_plan: QECPlan,
    neural_signals: FrozenNeuralSignals | None,
) -> np.ndarray:
    neural = neural_signals or FrozenNeuralSignals(0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.8, 0.5, 0.0, 0, 0, 0, 0, 1.0, 0.0, [])
    values = np.asarray(
        [
            candidate.qubits / max(spec.target_qubits, 1),
            candidate.cell_pitch_um / 64.0,
            candidate.optical_bus_count / max(candidate.qubits, 1),
            candidate.microwave_line_count / max(candidate.qubits, 1),
            candidate.shielding_layers / 5.0,
            candidate.control_mux_factor / 8.0,
            metrics.gate_fidelity,
            metrics.readout_fidelity,
            metrics.routing_capacity,
            dense_signals.effective_crosstalk_linear,
            qec_plan.achievable_logical_qubits / max(spec.target_logical_qubits, 1),
            spec.target_logical_error_rate / max(qec_plan.logical_error_rate, 1e-12),
            qec_plan.decoder_locality_score,
            qec_plan.magic_state_rate_per_us / max(0.3 * spec.target_logical_qubits, 1e-6),
            qec_plan.schedule_makespan_ns / max(qec_plan.logical_cycle_ns * max(len(qec_plan.patches), 1), 1e-6),
            neural.design_quality,
            neural.latency_risk,
            neural.crosstalk_risk,
        ],
        dtype=float,
    )
    return values


def _choose_action(state: np.ndarray) -> int:
    action_scores = np.asarray(
        [
            1.2 * state[9] + 0.8 * state[17],
            1.0 * state[9] + 0.6 * state[8],
            0.8 * state[13] + 0.4 * state[8],
            0.8 * state[14] + 0.5 * state[16],
            0.6 * state[14] + 0.5 * state[16],
            1.1 * max(1.0 - state[10], 0.0) + 0.8 * max(1.0 - state[11], 0.0),
            1.0 * max(1.0 - state[13], 0.0) + 0.6 * state[14],
        ]
    )
    return int(np.argmax(action_scores))


def simulate_world_model_rollout(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    dense_signals: FastDenseSignals,
    qec_plan: QECPlan,
    neural_signals: FrozenNeuralSignals | None = None,
    steps: int = 4,
) -> WorldModelSignals:
    state = _state_vector(spec, candidate, metrics, dense_signals, qec_plan, neural_signals)
    context = np.asarray(
        [
            spec.target_logical_qubits / 8.0,
            spec.target_logical_error_rate * 1.0e4,
            spec.max_latency_ns / 400.0,
            spec.max_power_mw / 800.0,
            spec.target_qubits / 192.0,
            candidate.metal_layers / 8.0,
            candidate.detector_qe,
            metrics.scalability_score,
            qec_plan.factory_utilization,
            qec_plan.decoder_utilization,
        ],
        dtype=float,
    )
    rollout: list[WorldModelStep] = []
    latent = np.tanh(state)
    recommended_qubits = 0
    recommended_factory_lanes = 0
    recommended_pitch = 0.0
    schedule_bias = 0.0
    for step in range(steps):
        action_idx = _choose_action(latent)
        action_vec = _B[action_idx]
        latent = np.tanh(latent @ _A + action_vec + context @ _C)
        logits = latent @ _HEAD + _BIAS
        quality = max(0.0, min(1.25, 0.55 * _sigmoid(logits[0]) + 0.25 * _sigmoid(logits[1]) + 0.20 * latent[10]))
        schedule_pressure = max(0.0, min(1.0, 0.65 * _sigmoid(logits[2]) + 0.35 * max(latent[14], 0.0)))
        logical_pressure = max(0.0, min(1.0, 0.65 * _sigmoid(logits[3]) + 0.35 * max(1.0 - latent[10], 0.0)))
        routing_pressure = max(0.0, min(1.0, 0.60 * _sigmoid(logits[4]) + 0.40 * max(1.0 - latent[8], 0.0)))
        thermal_pressure = max(0.0, min(1.0, 0.55 * _sigmoid(logits[5]) + 0.45 * max(metrics.power_mw / max(spec.max_power_mw, 1e-6) - 0.4, 0.0)))
        rollout.append(
            WorldModelStep(
                step=step,
                action=_ACTION_NAMES[action_idx],
                predicted_quality=quality,
                schedule_pressure=schedule_pressure,
                logical_pressure=logical_pressure,
                routing_pressure=routing_pressure,
                thermal_pressure=thermal_pressure,
            )
        )
        if action_idx == 5:
            recommended_qubits += max(4, qec_plan.physical_qubits_per_logical // 3)
        elif action_idx == 6:
            recommended_factory_lanes += 1
        elif action_idx == 0:
            recommended_pitch += 0.8
        elif action_idx == 4:
            schedule_bias += 0.12

    final = rollout[-1] if rollout else WorldModelStep(0, "none", 0.5, 0.5, 0.5, 0.5, 0.5)
    confidence = max(0.2, min(1.2, 0.55 + 0.18 * final.predicted_quality + 0.10 * (1.0 - final.schedule_pressure) + 0.10 * (1.0 - final.logical_pressure)))
    return WorldModelSignals(
        terminal_quality=final.predicted_quality,
        schedule_pressure=final.schedule_pressure,
        logical_pressure=final.logical_pressure,
        routing_pressure=final.routing_pressure,
        thermal_pressure=final.thermal_pressure,
        confidence=confidence,
        recommended_qubit_delta=recommended_qubits,
        recommended_factory_lane_delta=recommended_factory_lanes,
        recommended_pitch_delta_um=recommended_pitch,
        recommended_schedule_bias=schedule_bias,
        rollout=rollout,
        latent_vector=[float(value) for value in latent[:12]],
    )


def calibrate_world_model_signals(
    signals: WorldModelSignals,
    qec_plan: QECPlan,
    robust_schedule_pass_rate: float,
) -> WorldModelSignals:
    calibrated_quality = max(0.0, min(1.25, 0.55 * signals.terminal_quality + 0.25 * qec_plan.decoder_locality_score + 0.20 * robust_schedule_pass_rate))
    calibrated_schedule = max(0.0, min(1.0, 0.60 * signals.schedule_pressure + 0.40 * max(qec_plan.schedule_makespan_ns / max(qec_plan.logical_cycle_ns * max(len(qec_plan.patches), 1) * 8.0, 1e-6) - 0.5, 0.0)))
    calibrated_logical = max(0.0, min(1.0, 0.60 * signals.logical_pressure + 0.40 * max(1.0 - qec_plan.achievable_logical_qubits / max(qec_plan.target_logical_qubits, 1), 0.0)))
    calibrated_confidence = max(0.2, min(1.2, 0.50 * signals.confidence + 0.25 * robust_schedule_pass_rate + 0.25 * qec_plan.decoder_locality_score))
    return replace(
        signals,
        terminal_quality=calibrated_quality,
        schedule_pressure=calibrated_schedule,
        logical_pressure=calibrated_logical,
        confidence=calibrated_confidence,
    )
