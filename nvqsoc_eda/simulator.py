from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from .physics import (
    crosstalk_db,
    crosstalk_linear,
    effective_spin_photon_coupling_mhz,
    effective_t2_us,
    gate_fidelity,
    latency_ns,
    line_loss_db,
    magnetic_uniformity_ppm,
    manufacturing_yield,
    microwave_rabi_rate_mhz,
    nv_transition_frequency_ghz,
    optical_collection_efficiency,
    readout_fidelity,
    readout_photon_counts,
    readout_snr,
    resonator_q_loaded,
    robustness_margin,
    routing_capacity_score,
    scalability_score,
    single_qubit_gate_time_ns,
    strain_mhz,
    thermal_load_mw,
    two_qubit_gate_time_ns,
)
from .spec import DesignSpec


@dataclass(slots=True)
class CandidateDesign:
    architecture: str
    qubits: int
    rows: int
    cols: int
    cell_pitch_um: float
    optical_bus_count: int
    microwave_line_count: int
    resonator_count: int
    metal_layers: int
    shielding_layers: int
    waveguide_width_um: float
    cavity_q: float
    implant_dose_scale: float
    anneal_quality: float
    microwave_power_dbm: float
    control_mux_factor: int
    amplifier_gain_db: float
    amplifier_noise_temp_k: float
    detector_qe: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def core_width_mm(self) -> float:
        array_width_um = self.cols * self.cell_pitch_um
        edge_overhead_um = 1000.0 + 180.0 * self.metal_layers + 140.0 * self.shielding_layers
        return (array_width_um + edge_overhead_um) / 1000.0

    @property
    def core_height_mm(self) -> float:
        array_height_um = self.rows * self.cell_pitch_um
        edge_overhead_um = 1250.0 + 160.0 * self.optical_bus_count + 90.0 * self.microwave_line_count
        return (array_height_um + edge_overhead_um) / 1000.0

    @property
    def die_width_mm(self) -> float:
        pad_margin_um = 780.0 + 50.0 * self.metal_layers + 18.0 * math.sqrt(max(self.qubits, 1))
        array_width_um = self.cols * self.cell_pitch_um
        return max(self.core_width_mm * 1000.0 + 2.0 * pad_margin_um, array_width_um + 2.0 * (pad_margin_um + 600.0)) / 1000.0

    @property
    def die_height_mm(self) -> float:
        pad_margin_um = 780.0 + 50.0 * self.metal_layers + 18.0 * math.sqrt(max(self.qubits, 1))
        array_height_um = self.rows * self.cell_pitch_um
        return max(self.core_height_mm * 1000.0 + 2.0 * pad_margin_um, array_height_um + 2.0 * (pad_margin_um + 700.0)) / 1000.0

    @property
    def die_area_mm2(self) -> float:
        return self.die_width_mm * self.die_height_mm

    @property
    def route_length_um(self) -> float:
        array_span_um = (self.rows + self.cols) * self.cell_pitch_um
        fanout_um = 880.0 * self.microwave_line_count + 620.0 * self.optical_bus_count
        hierarchy_gain = 0.92 if self.metal_layers >= 6 else 1.0
        return hierarchy_gain * (1.65 * array_span_um + fanout_um)

    @property
    def via_count(self) -> int:
        return int(self.qubits * (0.9 + 0.25 * self.metal_layers + 0.2 * self.optical_bus_count))


@dataclass(slots=True)
class SimulationMetrics:
    nv_frequency_ghz: float
    loaded_q: float
    optical_efficiency: float
    crosstalk_linear: float
    crosstalk_db: float
    t2_us: float
    single_gate_ns: float
    two_qubit_gate_ns: float
    gate_fidelity: float
    readout_snr: float
    readout_fidelity: float
    power_mw: float
    latency_ns: float
    die_area_mm2: float
    routing_length_mm: float
    routing_capacity: float
    yield_estimate: float
    scalability_score: float
    robustness_margin: float
    route_drc_violations: int
    constraint_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MonteCarloSummary:
    trials: int
    pass_rate: float
    gate_fidelity_mean: float
    gate_fidelity_p05: float
    readout_fidelity_mean: float
    readout_fidelity_p05: float
    t2_mean_us: float
    t2_p05_us: float
    power_mean_mw: float
    yield_mean: float
    robustness_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_candidate(spec: DesignSpec, candidate: CandidateDesign) -> SimulationMetrics:
    route_length_um = candidate.route_length_um
    route_length_mm = route_length_um / 1000.0
    route_loss = line_loss_db(route_length_um, candidate.metal_layers, candidate.shielding_layers)
    local_strain_mhz = strain_mhz(candidate.implant_dose_scale, candidate.anneal_quality, candidate.cell_pitch_um)
    field_uniformity_ppm = magnetic_uniformity_ppm(
        spec.magnetic_field_mT,
        math.sqrt(candidate.die_width_mm**2 + candidate.die_height_mm**2),
        candidate.shielding_layers,
    )
    nv_frequency = nv_transition_frequency_ghz(spec.magnetic_field_mT, local_strain_mhz)
    optical_efficiency = optical_collection_efficiency(
        candidate.cavity_q,
        candidate.optical_bus_count,
        candidate.waveguide_width_um,
        route_loss * 0.65,
        candidate.detector_qe,
    )
    xtalk_lin = crosstalk_linear(
        candidate.cell_pitch_um,
        candidate.shielding_layers,
        candidate.microwave_line_count,
        candidate.qubits,
    )
    t2_us_value = effective_t2_us(
        base_t2_us=5400.0 * candidate.anneal_quality / (0.82 + 0.23 * candidate.implant_dose_scale),
        temperature_k=spec.operating_temp_k,
        isotopic_purity_ppm=spec.isotopic_purity_ppm,
        strain_mhz_value=local_strain_mhz,
        crosstalk_lin=xtalk_lin,
        magnetic_uniformity_ppm_value=field_uniformity_ppm,
    )
    rabi_rate_mhz = microwave_rabi_rate_mhz(candidate.microwave_power_dbm, route_loss, candidate.control_mux_factor)
    single_gate_ns_value = single_qubit_gate_time_ns(rabi_rate_mhz)
    loaded_q = resonator_q_loaded(candidate.cavity_q, route_loss, candidate.via_count / 1000.0)
    coupling_mhz = effective_spin_photon_coupling_mhz(
        candidate.resonator_count,
        candidate.qubits,
        loaded_q,
        optical_efficiency,
        candidate.cell_pitch_um,
    )
    two_qubit_gate_ns_value = two_qubit_gate_time_ns(coupling_mhz, candidate.control_mux_factor)
    single_gate_fidelity = gate_fidelity(single_gate_ns_value, t2_us_value, xtalk_lin, 0.8, 0.012)
    two_qubit_fidelity = gate_fidelity(two_qubit_gate_ns_value, t2_us_value, xtalk_lin * 1.35, 1.3, 0.018)
    aggregate_gate_fidelity = 0.45 * single_gate_fidelity + 0.55 * two_qubit_fidelity
    photon_counts = readout_photon_counts(
        optical_efficiency,
        spec.optical_power_budget_mw,
        candidate.qubits,
        candidate.optical_bus_count,
        pulse_ns=240.0,
    )
    readout_snr_value = readout_snr(photon_counts, contrast=0.28 + 0.03 * candidate.anneal_quality, dark_counts=4.0, amplifier_noise_temp_k=candidate.amplifier_noise_temp_k)
    routing_capacity = routing_capacity_score(candidate.metal_layers, candidate.die_area_mm2, route_length_mm)
    latency_ns_value = latency_ns(single_gate_ns_value, two_qubit_gate_ns_value, route_length_mm, candidate.amplifier_gain_db)
    readout_fidelity_value = readout_fidelity(readout_snr_value, latency_ns_value)
    power_mw_value = thermal_load_mw(
        candidate.microwave_power_dbm,
        candidate.microwave_line_count,
        spec.optical_power_budget_mw,
        candidate.amplifier_gain_db,
        spec.cryo_stage_count,
        duty_cycle=0.36 + 0.05 * min(candidate.control_mux_factor, 6),
    )
    route_drc_violations = 0
    if candidate.cell_pitch_um < 22.0:
        route_drc_violations += 1
    if routing_capacity < 0.30:
        route_drc_violations += 1
    if candidate.waveguide_width_um < 0.35 or candidate.waveguide_width_um > 0.85:
        route_drc_violations += 1
    active_area_mm2 = 0.62 * candidate.die_area_mm2 + 0.015 * route_length_mm
    yield_value = manufacturing_yield(
        active_area_mm2,
        candidate.via_count,
        route_drc_violations,
        candidate.anneal_quality,
        candidate.implant_dose_scale,
    )
    scalability_value = scalability_score(candidate.qubits, candidate.control_mux_factor, candidate.optical_bus_count, routing_capacity)
    robustness_value = robustness_margin(aggregate_gate_fidelity, readout_fidelity_value, yield_value, routing_capacity)

    violations: list[str] = []
    if candidate.die_area_mm2 > spec.max_die_area_mm2:
        violations.append("die_area")
    if power_mw_value > spec.max_power_mw:
        violations.append("power")
    if latency_ns_value > spec.max_latency_ns:
        violations.append("latency")
    if t2_us_value < spec.target_t2_us:
        violations.append("coherence")
    if aggregate_gate_fidelity < spec.target_gate_fidelity:
        violations.append("gate_fidelity")
    if readout_fidelity_value < spec.target_readout_fidelity:
        violations.append("readout_fidelity")
    if yield_value < spec.min_yield:
        violations.append("yield")

    return SimulationMetrics(
        nv_frequency_ghz=nv_frequency,
        loaded_q=loaded_q,
        optical_efficiency=optical_efficiency,
        crosstalk_linear=xtalk_lin,
        crosstalk_db=crosstalk_db(xtalk_lin),
        t2_us=t2_us_value,
        single_gate_ns=single_gate_ns_value,
        two_qubit_gate_ns=two_qubit_gate_ns_value,
        gate_fidelity=aggregate_gate_fidelity,
        readout_snr=readout_snr_value,
        readout_fidelity=readout_fidelity_value,
        power_mw=power_mw_value,
        latency_ns=latency_ns_value,
        die_area_mm2=candidate.die_area_mm2,
        routing_length_mm=route_length_mm,
        routing_capacity=routing_capacity,
        yield_estimate=yield_value,
        scalability_score=scalability_value,
        robustness_margin=robustness_value,
        route_drc_violations=route_drc_violations,
        constraint_violations=violations,
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def run_monte_carlo(spec: DesignSpec, candidate: CandidateDesign, trials: int, seed: int) -> MonteCarloSummary:
    rng = random.Random(seed)
    gate_values: list[float] = []
    readout_values: list[float] = []
    t2_values: list[float] = []
    power_values: list[float] = []
    yield_values: list[float] = []
    robustness_values: list[float] = []
    passes = 0

    for _ in range(trials):
        perturbed = CandidateDesign(
            architecture=candidate.architecture,
            qubits=candidate.qubits,
            rows=candidate.rows,
            cols=candidate.cols,
            cell_pitch_um=candidate.cell_pitch_um * rng.uniform(0.97, 1.03),
            optical_bus_count=max(1, int(round(candidate.optical_bus_count * rng.uniform(0.9, 1.1)))),
            microwave_line_count=max(1, int(round(candidate.microwave_line_count * rng.uniform(0.9, 1.1)))),
            resonator_count=max(1, int(round(candidate.resonator_count * rng.uniform(0.92, 1.08)))),
            metal_layers=candidate.metal_layers,
            shielding_layers=candidate.shielding_layers,
            waveguide_width_um=candidate.waveguide_width_um * rng.uniform(0.95, 1.05),
            cavity_q=candidate.cavity_q * rng.uniform(0.87, 1.10),
            implant_dose_scale=candidate.implant_dose_scale * rng.uniform(0.94, 1.07),
            anneal_quality=max(0.55, min(0.99, candidate.anneal_quality * rng.uniform(0.95, 1.03))),
            microwave_power_dbm=candidate.microwave_power_dbm + rng.uniform(-0.7, 0.7),
            control_mux_factor=max(1, candidate.control_mux_factor + rng.choice([-1, 0, 0, 1])),
            amplifier_gain_db=candidate.amplifier_gain_db + rng.uniform(-1.2, 1.2),
            amplifier_noise_temp_k=max(1.0, candidate.amplifier_noise_temp_k * rng.uniform(0.90, 1.12)),
            detector_qe=max(0.40, min(0.95, candidate.detector_qe * rng.uniform(0.95, 1.03))),
        )
        perturbed_spec = DesignSpec.from_dict(
            spec.to_dict()
            | {
                "operating_temp_k": spec.operating_temp_k * rng.uniform(0.95, 1.10),
                "magnetic_field_mT": spec.magnetic_field_mT * rng.uniform(0.97, 1.03),
                "optical_power_budget_mw": spec.optical_power_budget_mw * rng.uniform(0.92, 1.08),
            }
        )
        metrics = evaluate_candidate(perturbed_spec, perturbed)
        gate_values.append(metrics.gate_fidelity)
        readout_values.append(metrics.readout_fidelity)
        t2_values.append(metrics.t2_us)
        power_values.append(metrics.power_mw)
        yield_values.append(metrics.yield_estimate)
        robustness_values.append(metrics.robustness_margin)
        if not metrics.constraint_violations:
            passes += 1

    return MonteCarloSummary(
        trials=trials,
        pass_rate=passes / max(trials, 1),
        gate_fidelity_mean=statistics.fmean(gate_values) if gate_values else 0.0,
        gate_fidelity_p05=_percentile(gate_values, 0.05),
        readout_fidelity_mean=statistics.fmean(readout_values) if readout_values else 0.0,
        readout_fidelity_p05=_percentile(readout_values, 0.05),
        t2_mean_us=statistics.fmean(t2_values) if t2_values else 0.0,
        t2_p05_us=_percentile(t2_values, 0.05),
        power_mean_mw=statistics.fmean(power_values) if power_values else 0.0,
        yield_mean=statistics.fmean(yield_values) if yield_values else 0.0,
        robustness_mean=statistics.fmean(robustness_values) if robustness_values else 0.0,
    )
