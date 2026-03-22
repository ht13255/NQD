from __future__ import annotations

import math


D_ZFS_GHZ = 2.87
GAMMA_E_GHZ_PER_T = 28.024
PLANCK = 6.62607015e-34
HBAR = 1.054571817e-34
KB = 1.380649e-23


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_log10(value: float, floor: float = 1e-12) -> float:
    return math.log10(max(value, floor))


def nv_transition_frequency_ghz(magnetic_field_mT: float, strain_mhz: float) -> float:
    return D_ZFS_GHZ + GAMMA_E_GHZ_PER_T * (magnetic_field_mT / 1000.0) + strain_mhz / 1000.0


def magnetic_uniformity_ppm(magnetic_field_mT: float, die_diag_mm: float, shielding_layers: int) -> float:
    gradient_mT_per_mm = 0.09 / (1.0 + 0.45 * shielding_layers)
    return 1e6 * gradient_mT_per_mm * die_diag_mm / max(magnetic_field_mT, 1e-6)


def strain_mhz(implant_dose_scale: float, anneal_quality: float, cell_pitch_um: float) -> float:
    return 2.5 + 9.0 * implant_dose_scale + 12.0 * (1.0 - anneal_quality) + 20.0 / max(cell_pitch_um, 1.0)


def cavity_enhancement(cavity_q: float, optical_bus_count: int, waveguide_width_um: float) -> float:
    return clamp(0.28 + 0.09 * safe_log10(cavity_q) + 0.03 * optical_bus_count - 0.01 * abs(waveguide_width_um - 0.55), 0.10, 0.95)


def optical_collection_efficiency(
    cavity_q: float,
    optical_bus_count: int,
    waveguide_width_um: float,
    routing_loss_db: float,
    detector_qe: float,
) -> float:
    cavity_term = cavity_enhancement(cavity_q, optical_bus_count, waveguide_width_um)
    routing_term = 10.0 ** (-routing_loss_db / 10.0)
    return clamp(cavity_term * routing_term * detector_qe, 0.01, 0.92)


def line_loss_db(route_length_um: float, metal_layers: int, shielding_layers: int) -> float:
    length_mm = route_length_um / 1000.0
    base = 0.22 * length_mm
    metal_gain = 0.85 ** max(metal_layers - 2, 0)
    shielding_penalty = 1.0 + 0.04 * max(shielding_layers - 1, 0)
    return max(0.10, base * metal_gain * shielding_penalty)


def microwave_rabi_rate_mhz(power_dbm: float, route_loss_db: float, control_mux_factor: int) -> float:
    linear_power = 10.0 ** (power_dbm / 10.0)
    mux_penalty = 1.0 + 0.18 * max(control_mux_factor - 1, 0)
    return max(0.5, 22.0 * math.sqrt(linear_power) * 10.0 ** (-route_loss_db / 20.0) / mux_penalty)


def crosstalk_linear(cell_pitch_um: float, shielding_layers: int, microwave_line_count: int, qubits: int) -> float:
    aggressor_density = qubits / max(microwave_line_count, 1)
    pitch_term = math.exp(-cell_pitch_um / 36.0)
    shield_term = 1.0 / (1.0 + 1.15 * shielding_layers)
    return clamp(0.025 * aggressor_density * pitch_term * shield_term, 1e-6, 0.40)


def crosstalk_db(crosstalk_lin: float) -> float:
    return 20.0 * safe_log10(crosstalk_lin)


def effective_t2_us(
    base_t2_us: float,
    temperature_k: float,
    isotopic_purity_ppm: float,
    strain_mhz_value: float,
    crosstalk_lin: float,
    magnetic_uniformity_ppm_value: float,
) -> float:
    inv_t2 = (
        1.0 / max(base_t2_us, 1e-6)
        + 8.0e-5 * max(temperature_k - 3.0, 0.0)
        + 3.0e-6 * isotopic_purity_ppm
        + 5.0e-6 * strain_mhz_value
        + 3.0e-3 * crosstalk_lin
        + 2.0e-8 * magnetic_uniformity_ppm_value
    )
    return 1.0 / inv_t2


def resonator_q_loaded(cavity_q: float, route_loss_db: float, via_factor: float) -> float:
    return max(5.0e3, cavity_q / (1.0 + 0.18 * route_loss_db + 0.15 * via_factor))


def effective_spin_photon_coupling_mhz(
    resonator_count: int,
    qubits: int,
    loaded_q: float,
    optical_efficiency: float,
    cell_pitch_um: float,
) -> float:
    density_term = math.sqrt(resonator_count / max(qubits, 1))
    q_term = math.sqrt(loaded_q / 1.0e5)
    pitch_term = math.exp(-cell_pitch_um / 220.0)
    return max(0.01, 32.0 * density_term * q_term * optical_efficiency * pitch_term)


def single_qubit_gate_time_ns(rabi_rate_mhz: float) -> float:
    return 500.0 / max(rabi_rate_mhz, 1e-6)


def two_qubit_gate_time_ns(coupling_mhz: float, control_mux_factor: int) -> float:
    return 1000.0 * (1.0 + 0.10 * max(control_mux_factor - 1, 0)) / max(coupling_mhz, 1e-6)


def gate_fidelity(
    gate_time_ns: float,
    t2_us: float,
    crosstalk_lin: float,
    jitter_ns: float,
    calibration_error: float,
) -> float:
    decoherence_term = (gate_time_ns / max(t2_us * 1000.0, 1e-6)) ** 1.35
    crosstalk_term = 2.4 * crosstalk_lin ** 2
    jitter_term = (jitter_ns / max(gate_time_ns, 1e-6)) ** 2
    calibration_term = calibration_error ** 2
    return clamp(math.exp(-(decoherence_term + crosstalk_term + jitter_term + calibration_term)), 0.0, 0.999999)


def readout_photon_counts(
    optical_efficiency: float,
    optical_power_budget_mw: float,
    target_qubits: int,
    optical_bus_count: int,
    pulse_ns: float,
) -> float:
    bus_utilization = target_qubits / max(optical_bus_count, 1)
    power_per_bus = optical_power_budget_mw / max(optical_bus_count, 1)
    return max(5.0, 300.0 * optical_efficiency * power_per_bus * pulse_ns / (10.0 + bus_utilization))


def readout_snr(photon_counts: float, contrast: float, dark_counts: float, amplifier_noise_temp_k: float) -> float:
    noise = math.sqrt(max(photon_counts + dark_counts + 2.0 * amplifier_noise_temp_k, 1e-6))
    return max(0.01, contrast * photon_counts / noise)


def readout_fidelity(snr: float, latency_ns: float) -> float:
    classifier_margin = 0.5 * (1.0 + math.erf(snr / math.sqrt(2.0)))
    latency_penalty = math.exp(-latency_ns / 9000.0)
    return clamp(classifier_margin * latency_penalty, 0.0, 0.999999)


def thermal_load_mw(
    power_dbm: float,
    microwave_line_count: int,
    optical_power_budget_mw: float,
    amplifier_gain_db: float,
    cryo_stage_count: int,
    duty_cycle: float,
) -> float:
    microwave_static = microwave_line_count * (10.0 ** (power_dbm / 10.0)) * duty_cycle * 0.25
    optical_absorption = optical_power_budget_mw * 0.18
    amplifier_chain = amplifier_gain_db * 1.6 / max(cryo_stage_count, 1)
    return microwave_static + optical_absorption + amplifier_chain + 24.0


def routing_capacity_score(metal_layers: int, die_area_mm2: float, routing_length_mm: float) -> float:
    capacity = (8.0 + 2.2 * metal_layers) * max(die_area_mm2, 0.1)
    congestion = routing_length_mm / capacity
    return clamp(1.0 - congestion, 0.0, 1.0)


def manufacturing_yield(
    active_area_mm2: float,
    via_count: int,
    route_drc_violations: int,
    anneal_quality: float,
    implant_dose_scale: float,
) -> float:
    defect_density = 0.0105 * implant_dose_scale / max(anneal_quality, 0.2)
    area_term = math.exp(-defect_density * active_area_mm2)
    via_term = math.exp(-1.4e-4 * via_count)
    drc_term = math.exp(-0.25 * route_drc_violations)
    return clamp(area_term * via_term * drc_term, 0.0, 0.999999)


def scalability_score(target_qubits: int, control_mux_factor: int, optical_bus_count: int, routing_capacity: float) -> float:
    mux_term = min(control_mux_factor / 6.0, 1.2)
    bus_term = min(optical_bus_count / max(target_qubits / 16.0, 1.0), 1.2)
    qubit_term = min(target_qubits / 128.0, 1.2)
    return clamp((0.35 * mux_term + 0.30 * bus_term + 0.35 * qubit_term) * routing_capacity, 0.0, 1.0)


def latency_ns(single_gate_ns: float, two_qubit_gate_ns_value: float, routing_length_mm: float, amplifier_gain_db: float) -> float:
    route_delay = 3.9 * routing_length_mm
    electronics_delay = 2.0 * amplifier_gain_db
    return single_gate_ns + 0.35 * two_qubit_gate_ns_value + route_delay + electronics_delay


def robustness_margin(
    gate_fidelity_value: float,
    readout_fidelity_value: float,
    yield_value: float,
    routing_capacity: float,
) -> float:
    return clamp(
        0.35 * gate_fidelity_value + 0.25 * readout_fidelity_value + 0.25 * yield_value + 0.15 * routing_capacity,
        0.0,
        1.0,
    )
