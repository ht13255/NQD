from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.special import erf

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


def _percentile_array(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return 0.0
    try:
        return float(np.quantile(values, p, method="linear"))
    except TypeError:
        return float(np.quantile(values, p, interpolation="linear"))


def _run_monte_carlo_vectorized(spec: DesignSpec, candidate: CandidateDesign, trials: int, seed: int) -> MonteCarloSummary:
    if trials <= 0:
        return MonteCarloSummary(
            trials=0,
            pass_rate=0.0,
            gate_fidelity_mean=0.0,
            gate_fidelity_p05=0.0,
            readout_fidelity_mean=0.0,
            readout_fidelity_p05=0.0,
            t2_mean_us=0.0,
            t2_p05_us=0.0,
            power_mean_mw=0.0,
            yield_mean=0.0,
            robustness_mean=0.0,
        )

    rng = np.random.default_rng(seed)
    qubits = candidate.qubits
    rows = candidate.rows
    cols = candidate.cols
    metal_layers = candidate.metal_layers
    shielding_layers = candidate.shielding_layers
    pad_margin_um = 780.0 + 50.0 * metal_layers + 18.0 * math.sqrt(max(qubits, 1))
    hierarchy_gain = 0.92 if metal_layers >= 6 else 1.0

    cell_pitch_um = candidate.cell_pitch_um * rng.uniform(0.97, 1.03, size=trials)
    optical_bus_count = np.maximum(1, np.rint(candidate.optical_bus_count * rng.uniform(0.9, 1.1, size=trials)).astype(int))
    microwave_line_count = np.maximum(1, np.rint(candidate.microwave_line_count * rng.uniform(0.9, 1.1, size=trials)).astype(int))
    resonator_count = np.maximum(1, np.rint(candidate.resonator_count * rng.uniform(0.92, 1.08, size=trials)).astype(int))
    waveguide_width_um = candidate.waveguide_width_um * rng.uniform(0.95, 1.05, size=trials)
    cavity_q = candidate.cavity_q * rng.uniform(0.87, 1.10, size=trials)
    implant_dose_scale = candidate.implant_dose_scale * rng.uniform(0.94, 1.07, size=trials)
    anneal_quality = np.clip(candidate.anneal_quality * rng.uniform(0.95, 1.03, size=trials), 0.55, 0.99)
    microwave_power_dbm = candidate.microwave_power_dbm + rng.uniform(-0.7, 0.7, size=trials)
    control_mux_factor = np.maximum(1, candidate.control_mux_factor + rng.choice(np.array([-1, 0, 0, 1]), size=trials))
    amplifier_gain_db = candidate.amplifier_gain_db + rng.uniform(-1.2, 1.2, size=trials)
    amplifier_noise_temp_k = np.maximum(1.0, candidate.amplifier_noise_temp_k * rng.uniform(0.90, 1.12, size=trials))
    detector_qe = np.clip(candidate.detector_qe * rng.uniform(0.95, 1.03, size=trials), 0.40, 0.95)

    operating_temp_k = spec.operating_temp_k * rng.uniform(0.95, 1.10, size=trials)
    magnetic_field_mT = spec.magnetic_field_mT * rng.uniform(0.97, 1.03, size=trials)
    optical_power_budget_mw = spec.optical_power_budget_mw * rng.uniform(0.92, 1.08, size=trials)

    array_width_um = cols * cell_pitch_um
    array_height_um = rows * cell_pitch_um
    core_width_mm = (array_width_um + 1000.0 + 180.0 * metal_layers + 140.0 * shielding_layers) / 1000.0
    core_height_mm = (array_height_um + 1250.0 + 160.0 * optical_bus_count + 90.0 * microwave_line_count) / 1000.0
    die_width_mm = np.maximum(core_width_mm * 1000.0 + 2.0 * pad_margin_um, array_width_um + 2.0 * (pad_margin_um + 600.0)) / 1000.0
    die_height_mm = np.maximum(core_height_mm * 1000.0 + 2.0 * pad_margin_um, array_height_um + 2.0 * (pad_margin_um + 700.0)) / 1000.0
    die_area_mm2 = die_width_mm * die_height_mm

    route_length_um = hierarchy_gain * (1.65 * ((rows + cols) * cell_pitch_um) + 880.0 * microwave_line_count + 620.0 * optical_bus_count)
    route_length_mm = route_length_um / 1000.0
    route_loss_db = np.maximum(0.10, 0.22 * route_length_mm * (0.85 ** max(metal_layers - 2, 0)) * (1.0 + 0.04 * max(shielding_layers - 1, 0)))
    local_strain_mhz = 2.5 + 9.0 * implant_dose_scale + 12.0 * (1.0 - anneal_quality) + 20.0 / np.maximum(cell_pitch_um, 1.0)
    field_uniformity_ppm = 1e6 * (0.09 / (1.0 + 0.45 * shielding_layers)) * np.sqrt(die_width_mm**2 + die_height_mm**2) / np.maximum(magnetic_field_mT, 1e-6)
    nv_frequency_ghz = 2.87 + 28.024 * (magnetic_field_mT / 1000.0) + local_strain_mhz / 1000.0

    cavity_term = np.clip(0.28 + 0.09 * np.log10(np.maximum(cavity_q, 1e-12)) + 0.03 * optical_bus_count - 0.01 * np.abs(waveguide_width_um - 0.55), 0.10, 0.95)
    optical_efficiency = np.clip(cavity_term * np.power(10.0, -(0.65 * route_loss_db) / 10.0) * detector_qe, 0.01, 0.92)
    crosstalk_lin = np.clip(0.025 * (qubits / np.maximum(microwave_line_count, 1)) * np.exp(-cell_pitch_um / 36.0) * (1.0 / (1.0 + 1.15 * shielding_layers)), 1e-6, 0.40)

    base_t2_us = 5400.0 * anneal_quality / (0.82 + 0.23 * implant_dose_scale)
    inv_t2 = (
        1.0 / np.maximum(base_t2_us, 1e-6)
        + 8.0e-5 * np.maximum(operating_temp_k - 3.0, 0.0)
        + 3.0e-6 * spec.isotopic_purity_ppm
        + 5.0e-6 * local_strain_mhz
        + 3.0e-3 * crosstalk_lin
        + 2.0e-8 * field_uniformity_ppm
    )
    t2_us = 1.0 / inv_t2

    linear_power = np.power(10.0, microwave_power_dbm / 10.0)
    mux_penalty = 1.0 + 0.18 * np.maximum(control_mux_factor - 1, 0)
    rabi_rate_mhz = np.maximum(0.5, 22.0 * np.sqrt(linear_power) * np.power(10.0, -route_loss_db / 20.0) / mux_penalty)
    single_gate_ns = 500.0 / np.maximum(rabi_rate_mhz, 1e-6)

    via_count = np.floor(qubits * (0.9 + 0.25 * metal_layers + 0.2 * optical_bus_count)).astype(int)
    loaded_q = np.maximum(5.0e3, cavity_q / (1.0 + 0.18 * route_loss_db + 0.15 * (via_count / 1000.0)))
    coupling_mhz = np.maximum(
        0.01,
        32.0 * np.sqrt(resonator_count / max(qubits, 1)) * np.sqrt(loaded_q / 1.0e5) * optical_efficiency * np.exp(-cell_pitch_um / 220.0),
    )
    two_qubit_gate_ns = 1000.0 * (1.0 + 0.10 * np.maximum(control_mux_factor - 1, 0)) / np.maximum(coupling_mhz, 1e-6)

    single_gate_fidelity = np.clip(
        np.exp(-((single_gate_ns / np.maximum(t2_us * 1000.0, 1e-6)) ** 1.35 + 2.4 * crosstalk_lin**2 + (0.8 / np.maximum(single_gate_ns, 1e-6)) ** 2 + 0.012**2)),
        0.0,
        0.999999,
    )
    two_qubit_fidelity = np.clip(
        np.exp(-((two_qubit_gate_ns / np.maximum(t2_us * 1000.0, 1e-6)) ** 1.35 + 2.4 * (1.35 * crosstalk_lin) ** 2 + (1.3 / np.maximum(two_qubit_gate_ns, 1e-6)) ** 2 + 0.018**2)),
        0.0,
        0.999999,
    )
    gate_fidelity_value = 0.45 * single_gate_fidelity + 0.55 * two_qubit_fidelity

    bus_utilization = qubits / np.maximum(optical_bus_count, 1)
    power_per_bus = optical_power_budget_mw / np.maximum(optical_bus_count, 1)
    photon_counts = np.maximum(5.0, 300.0 * optical_efficiency * power_per_bus * 240.0 / (10.0 + bus_utilization))
    readout_snr_value = np.maximum(0.01, (0.28 + 0.03 * anneal_quality) * photon_counts / np.sqrt(np.maximum(photon_counts + 4.0 + 2.0 * amplifier_noise_temp_k, 1e-6)))
    routing_capacity = np.clip(1.0 - route_length_mm / ((8.0 + 2.2 * metal_layers) * np.maximum(die_area_mm2, 0.1)), 0.0, 1.0)
    latency_ns_value = single_gate_ns + 0.35 * two_qubit_gate_ns + 3.9 * route_length_mm + 2.0 * amplifier_gain_db
    readout_fidelity_value = np.clip(0.5 * (1.0 + erf(readout_snr_value / np.sqrt(2.0))) * np.exp(-latency_ns_value / 9000.0), 0.0, 0.999999)

    duty_cycle = 0.36 + 0.05 * np.minimum(control_mux_factor, 6)
    power_mw_value = microwave_line_count * linear_power * duty_cycle * 0.25 + 0.18 * optical_power_budget_mw + amplifier_gain_db * 1.6 / max(spec.cryo_stage_count, 1) + 24.0

    route_drc_violations = np.zeros(trials, dtype=int)
    route_drc_violations += (cell_pitch_um < 22.0).astype(int)
    route_drc_violations += (routing_capacity < 0.30).astype(int)
    route_drc_violations += ((waveguide_width_um < 0.35) | (waveguide_width_um > 0.85)).astype(int)

    active_area_mm2 = 0.62 * die_area_mm2 + 0.015 * route_length_mm
    defect_density = 0.0105 * implant_dose_scale / np.maximum(anneal_quality, 0.2)
    yield_estimate = np.clip(np.exp(-defect_density * active_area_mm2) * np.exp(-1.4e-4 * via_count) * np.exp(-0.25 * route_drc_violations), 0.0, 0.999999)
    scalability = np.clip((0.35 * np.minimum(control_mux_factor / 6.0, 1.2) + 0.30 * np.minimum(optical_bus_count / max(qubits / 16.0, 1.0), 1.2) + 0.35 * min(qubits / 128.0, 1.2)) * routing_capacity, 0.0, 1.0)
    robustness = np.clip(0.35 * gate_fidelity_value + 0.25 * readout_fidelity_value + 0.25 * yield_estimate + 0.15 * routing_capacity, 0.0, 1.0)

    pass_mask = (
        (die_area_mm2 <= spec.max_die_area_mm2)
        & (power_mw_value <= spec.max_power_mw)
        & (latency_ns_value <= spec.max_latency_ns)
        & (t2_us >= spec.target_t2_us)
        & (gate_fidelity_value >= spec.target_gate_fidelity)
        & (readout_fidelity_value >= spec.target_readout_fidelity)
        & (yield_estimate >= spec.min_yield)
    )

    return MonteCarloSummary(
        trials=trials,
        pass_rate=float(np.mean(pass_mask.astype(float))),
        gate_fidelity_mean=float(np.mean(gate_fidelity_value)),
        gate_fidelity_p05=_percentile_array(gate_fidelity_value, 0.05),
        readout_fidelity_mean=float(np.mean(readout_fidelity_value)),
        readout_fidelity_p05=_percentile_array(readout_fidelity_value, 0.05),
        t2_mean_us=float(np.mean(t2_us)),
        t2_p05_us=_percentile_array(t2_us, 0.05),
        power_mean_mw=float(np.mean(power_mw_value)),
        yield_mean=float(np.mean(yield_estimate)),
        robustness_mean=float(np.mean(robustness)),
    )


def _run_monte_carlo_scalar(spec: DesignSpec, candidate: CandidateDesign, trials: int, seed: int) -> MonteCarloSummary:
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


def run_monte_carlo(spec: DesignSpec, candidate: CandidateDesign, trials: int, seed: int) -> MonteCarloSummary:
    try:
        return _run_monte_carlo_vectorized(spec, candidate, trials, seed)
    except Exception:
        return _run_monte_carlo_scalar(spec, candidate, trials, seed)
