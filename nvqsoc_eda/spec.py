from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _normalized_weights(raw: dict[str, float] | None) -> dict[str, float]:
    defaults = {
        "fidelity": 0.19,
        "coherence": 0.14,
        "yield": 0.11,
        "power": 0.08,
        "latency": 0.08,
        "area": 0.07,
        "scalability": 0.06,
        "robustness": 0.05,
        "placement": 0.05,
        "crosstalk": 0.05,
        "logical": 0.06,
        "qec": 0.06,
        "schedule": 0.04,
        "world_model": 0.03,
    }
    merged = defaults | (raw or {})
    total = sum(max(value, 0.0) for value in merged.values()) or 1.0
    return {key: max(value, 0.0) / total for key, value in merged.items()}


@dataclass(slots=True)
class DesignSpec:
    design_name: str
    application: str
    target_qubits: int = 64
    target_logical_qubits: int = 1
    max_die_area_mm2: float = 36.0
    max_power_mw: float = 480.0
    operating_temp_k: float = 4.2
    magnetic_field_mT: float = 18.0
    optical_wavelength_nm: float = 637.0
    target_t2_us: float = 1200.0
    target_gate_fidelity: float = 0.995
    target_readout_fidelity: float = 0.965
    target_logical_error_rate: float = 1e-3
    max_latency_ns: float = 180.0
    min_yield: float = 0.72
    fabrication_node_nm: int = 90
    diamond_thickness_um: float = 25.0
    isotopic_purity_ppm: float = 30.0
    optical_power_budget_mw: float = 140.0
    cryo_stage_count: int = 3
    qec_code: str = "auto"
    qec_enabled: bool = True
    syndrome_cycle_ns: float = 120.0
    decoder_margin_ns: float = 60.0
    logical_connectivity: str = "grid"
    routing_preference: str = "balanced"
    objective_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignSpec":
        weights = _normalized_weights(data.get("objective_weights"))
        target_qubits = int(data.get("target_qubits", 64))
        target_logical_qubits = int(data.get("target_logical_qubits", max(1, target_qubits // 48)))
        derived_die_area = max(24.0, 20.0 + 0.14 * target_qubits + 6.0 * target_logical_qubits)
        values = {
            "design_name": data.get("design_name", "nv_qsoc_design"),
            "application": data.get("application", "general"),
            "target_qubits": target_qubits,
            "target_logical_qubits": target_logical_qubits,
            "max_die_area_mm2": float(data.get("max_die_area_mm2", derived_die_area)),
            "max_power_mw": float(data.get("max_power_mw", 480.0)),
            "operating_temp_k": float(data.get("operating_temp_k", 4.2)),
            "magnetic_field_mT": float(data.get("magnetic_field_mT", 18.0)),
            "optical_wavelength_nm": float(data.get("optical_wavelength_nm", 637.0)),
            "target_t2_us": float(data.get("target_t2_us", 1200.0)),
            "target_gate_fidelity": float(data.get("target_gate_fidelity", 0.995)),
            "target_readout_fidelity": float(data.get("target_readout_fidelity", 0.965)),
            "target_logical_error_rate": float(data.get("target_logical_error_rate", 1e-3)),
            "max_latency_ns": float(data.get("max_latency_ns", 180.0)),
            "min_yield": float(data.get("min_yield", 0.72)),
            "fabrication_node_nm": int(data.get("fabrication_node_nm", 90)),
            "diamond_thickness_um": float(data.get("diamond_thickness_um", 25.0)),
            "isotopic_purity_ppm": float(data.get("isotopic_purity_ppm", 30.0)),
            "optical_power_budget_mw": float(data.get("optical_power_budget_mw", 140.0)),
            "cryo_stage_count": int(data.get("cryo_stage_count", 3)),
            "qec_code": data.get("qec_code", "auto"),
            "qec_enabled": bool(data.get("qec_enabled", True)),
            "syndrome_cycle_ns": float(data.get("syndrome_cycle_ns", 120.0)),
            "decoder_margin_ns": float(data.get("decoder_margin_ns", 60.0)),
            "logical_connectivity": data.get("logical_connectivity", "grid"),
            "routing_preference": data.get("routing_preference", "balanced"),
            "objective_weights": weights,
        }
        spec = cls(**values)
        spec.validate()
        return spec

    @classmethod
    def from_json(cls, path: str | Path) -> "DesignSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.target_qubits <= 0:
            raise ValueError("target_qubits must be positive")
        if self.target_logical_qubits <= 0:
            raise ValueError("target_logical_qubits must be positive")
        if self.max_die_area_mm2 <= 0:
            raise ValueError("max_die_area_mm2 must be positive")
        if self.max_power_mw <= 0:
            raise ValueError("max_power_mw must be positive")
        if self.operating_temp_k <= 0:
            raise ValueError("operating_temp_k must be positive")
        if not 0.0 < self.target_gate_fidelity <= 1.0:
            raise ValueError("target_gate_fidelity must be in (0, 1]")
        if not 0.0 < self.target_readout_fidelity <= 1.0:
            raise ValueError("target_readout_fidelity must be in (0, 1]")
        if self.target_logical_error_rate <= 0.0:
            raise ValueError("target_logical_error_rate must be positive")
        if self.syndrome_cycle_ns <= 0.0:
            raise ValueError("syndrome_cycle_ns must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def architecture_biases(self) -> dict[str, float]:
        application = self.application.lower()
        mapping = {
            "quantum_repeater": {"network_node": 1.20, "hybrid_router": 1.10, "sensor_dense": 0.90},
            "magnetometer": {"sensor_dense": 1.25, "hybrid_router": 1.00, "network_node": 0.82},
            "imager": {"sensor_dense": 1.20, "hybrid_router": 1.08, "network_node": 0.85},
            "memory": {"network_node": 1.18, "hybrid_router": 1.06, "sensor_dense": 0.90},
            "processor": {"hybrid_router": 1.18, "network_node": 1.04, "sensor_dense": 0.92},
            "general": {"hybrid_router": 1.12, "sensor_dense": 1.00, "network_node": 1.00},
        }
        return mapping.get(application, mapping["general"])

    def normalized_targets(self) -> dict[str, float]:
        return {
            "qubits": float(self.target_qubits),
            "logical_qubits": float(self.target_logical_qubits),
            "die_area_mm2": self.max_die_area_mm2,
            "power_mw": self.max_power_mw,
            "t2_us": self.target_t2_us,
            "gate_fidelity": self.target_gate_fidelity,
            "readout_fidelity": self.target_readout_fidelity,
            "logical_error_rate": self.target_logical_error_rate,
            "latency_ns": self.max_latency_ns,
            "yield": self.min_yield,
        }
