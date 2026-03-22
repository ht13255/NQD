from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import networkx as nx
import numpy as np
import qutip
import skrf as rf
from scipy import signal, sparse
from scipy.sparse import linalg as spla
from shapely.geometry import LineString, Point, box

from .dense_placement import (
    DenseCrosstalkSummary,
    FastDenseSignals,
    PlacementPlan,
    build_probabilistic_placement,
    simulate_dense_crosstalk,
    write_dense_placement_artifacts,
)
from .layout import LayoutBundle, generate_layout
from .physics import readout_photon_counts
from .qec import QECPlan, build_qec_plan
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec
from .topology import TopologyPlan, build_topology_plan


matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(slots=True)
class QutipGateSummary:
    single_gate_state_fidelity: float
    two_qubit_state_fidelity: float
    simulated_rabi_rate_mhz: float
    simulated_exchange_rate_mhz: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MicrowaveSweepSummary:
    center_frequency_ghz: float
    center_s21_db: float
    peak_s21_db: float
    bandwidth_mhz: float
    group_delay_ns: float
    return_loss_db: float
    vswr: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThermalMapSummary:
    peak_temp_rise_k: float
    mean_temp_rise_k: float
    hotspot_x_norm: float
    hotspot_y_norm: float
    max_gradient_k_per_mm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PhotonStatisticsSummary:
    empirical_readout_fidelity: float
    optimal_threshold: int
    bright_mean_counts: float
    dark_mean_counts: float
    shot_noise_margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ControlLoopSummary:
    closed_loop_bandwidth_mhz: float
    phase_margin_deg: float
    overshoot_pct: float
    settling_time_ns: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NoiseSpectrumSummary:
    rms_jitter_ps: float
    integrated_phase_noise_rad: float
    noise_floor_db: float
    dephasing_proxy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PowerGridSummary:
    worst_drop_mV: float
    mean_drop_mV: float
    hotspot_x_norm: float
    hotspot_y_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EntanglementSummary:
    bell_fidelity: float
    heralding_success_probability: float
    entanglement_rate_hz: float
    repeater_link_margin: float
    error_mitigation_gain: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ErrorChannelSummary:
    depolarizing_error: float
    measurement_error: float
    erasure_error: float
    correlated_error: float
    decoder_failure_proxy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlacementSummary:
    placement_score: float
    min_spacing_um: float
    mean_offset_um: float
    optical_load_std: float
    microwave_load_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RoutingGraphSummary:
    average_path_um: float
    worst_path_um: float
    congestion_ratio: float
    max_edge_load: float
    connected_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OpticalSpectrumSummary:
    peak_drop_db: float
    extinction_db: float
    fsr_nm: float
    linewidth_pm: float
    net_collection_efficiency: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeometryDRCSummary:
    min_spacing_um: float
    density_min: float
    density_max: float
    overlap_violations: int
    via_density_per_mm2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LayoutParasiticSummary:
    total_route_length_um: float
    route_density: float
    max_coupling_risk: float
    optical_route_length_um: float
    microwave_route_length_um: float
    min_feature_spacing_um: float
    metal_density_min: float
    metal_density_max: float
    overlap_violations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdvancedSimulationBundle:
    candidate_key: str
    base_score: float
    composite_score: float
    placement: PlacementSummary
    dense_crosstalk: DenseCrosstalkSummary
    qutip: QutipGateSummary
    microwave: MicrowaveSweepSummary
    thermal: ThermalMapSummary
    photon: PhotonStatisticsSummary
    control: ControlLoopSummary
    noise: NoiseSpectrumSummary
    power_grid: PowerGridSummary
    entanglement: EntanglementSummary
    error_channels: ErrorChannelSummary
    routing: RoutingGraphSummary
    optical: OpticalSpectrumSummary
    geometry: GeometryDRCSummary
    topology: dict[str, Any]
    plots: dict[str, str]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "base_score": self.base_score,
            "composite_score": self.composite_score,
            "placement": self.placement.to_dict(),
            "dense_crosstalk": self.dense_crosstalk.to_dict(),
            "qutip": self.qutip.to_dict(),
            "microwave": self.microwave.to_dict(),
            "thermal": self.thermal.to_dict(),
            "photon": self.photon.to_dict(),
            "control": self.control.to_dict(),
            "noise": self.noise.to_dict(),
            "power_grid": self.power_grid.to_dict(),
            "entanglement": self.entanglement.to_dict(),
            "error_channels": self.error_channels.to_dict(),
            "routing": self.routing.to_dict(),
            "optical": self.optical.to_dict(),
            "geometry": self.geometry.to_dict(),
            "topology": self.topology,
            "plots": self.plots,
            "artifacts": self.artifacts,
        }


@dataclass(slots=True)
class AdvancedRefinementResult:
    selected_candidate_key: str
    bundles: list[AdvancedSimulationBundle]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_key": self.selected_candidate_key,
            "bundles": [bundle.to_dict() for bundle in self.bundles],
        }


def _candidate_key(candidate: CandidateDesign) -> str:
    return (
        f"{candidate.architecture}_q{candidate.qubits}_p{candidate.cell_pitch_um:.2f}"
        f"_ob{candidate.optical_bus_count}_mw{candidate.microwave_line_count}"
    )


def _qubit_dephasing(metrics: SimulationMetrics) -> QutipGateSummary:
    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    sm = qutip.sigmam()
    ket0 = qutip.basis(2, 0)
    rho0 = ket0.proj()
    duration_1q_us = max(metrics.single_gate_ns / 1000.0, 1e-4)
    rabi_rate_mhz = 500.0 / max(metrics.single_gate_ns, 1e-6)
    detuning_1q_mhz = max(0.02, 0.015 * metrics.nv_frequency_ghz * 1000.0 * metrics.crosstalk_linear)
    omega_1q = 2.0 * math.pi * rabi_rate_mhz
    delta_1q = 2.0 * math.pi * detuning_1q_mhz
    t1_us = max(1.8 * metrics.t2_us, metrics.t2_us + 1.0)
    tphi_rate = max(0.0, 1.0 / max(metrics.t2_us, 1e-6) - 1.0 / (2.0 * t1_us))
    collapse_ops = []
    if t1_us > 0:
        collapse_ops.append(math.sqrt(1.0 / t1_us) * sm)
    if tphi_rate > 0:
        collapse_ops.append(math.sqrt(tphi_rate) * sz)
    h_ideal_1q = 0.5 * omega_1q * sx
    h_noisy_1q = h_ideal_1q + 0.5 * delta_1q * sz
    times_1q = np.linspace(0.0, duration_1q_us, 128)
    noisy_1q = qutip.mesolve(h_noisy_1q, rho0, times_1q, c_ops=collapse_ops).states[-1]
    target_1q = (-(1j) * h_ideal_1q * duration_1q_us).expm() * ket0
    single_fidelity = float(qutip.fidelity(noisy_1q, target_1q))

    si = qutip.qeye(2)
    xx = qutip.tensor(qutip.sigmax(), qutip.sigmax())
    yy = qutip.tensor(qutip.sigmay(), qutip.sigmay())
    sz1 = qutip.tensor(qutip.sigmaz(), si)
    sz2 = qutip.tensor(si, qutip.sigmaz())
    sm1 = qutip.tensor(qutip.sigmam(), si)
    sm2 = qutip.tensor(si, qutip.sigmam())
    rho_in = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 1)).proj()
    duration_2q_us = max(metrics.two_qubit_gate_ns / 1000.0, 1e-4)
    exchange_rate_mhz = 1000.0 / max(metrics.two_qubit_gate_ns, 1e-6)
    omega_2q = 2.0 * math.pi * exchange_rate_mhz
    detuning_2q = 2.0 * math.pi * max(0.01, 0.025 * metrics.nv_frequency_ghz * 1000.0 * metrics.crosstalk_linear)
    h_ideal_2q = 0.25 * omega_2q * (xx + yy)
    h_noisy_2q = h_ideal_2q + 0.5 * detuning_2q * (sz1 - sz2)
    collapse_2q = []
    if t1_us > 0:
        collapse_2q.extend([math.sqrt(1.0 / t1_us) * sm1, math.sqrt(1.0 / t1_us) * sm2])
    if tphi_rate > 0:
        collapse_2q.extend([math.sqrt(tphi_rate) * sz1, math.sqrt(tphi_rate) * sz2])
    times_2q = np.linspace(0.0, duration_2q_us, 160)
    noisy_2q = qutip.mesolve(h_noisy_2q, rho_in, times_2q, c_ops=collapse_2q).states[-1]
    target_2q = (-(1j) * h_ideal_2q * duration_2q_us).expm() * qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 1))
    two_fidelity = float(qutip.fidelity(noisy_2q, target_2q))
    return QutipGateSummary(
        single_gate_state_fidelity=single_fidelity,
        two_qubit_state_fidelity=two_fidelity,
        simulated_rabi_rate_mhz=rabi_rate_mhz,
        simulated_exchange_rate_mhz=exchange_rate_mhz,
    )


def _microwave_sweep(candidate: CandidateDesign, metrics: SimulationMetrics, plot_dir: Path, label: str) -> tuple[MicrowaveSweepSummary, str]:
    f0_hz = metrics.nv_frequency_ghz * 1.0e9
    frequencies_hz = np.linspace(f0_hz * 0.86, f0_hz * 1.14, 420)
    omega = 2.0 * np.pi * frequencies_hz
    z0 = 50.0
    sections = max(4, candidate.microwave_line_count + candidate.shielding_layers)
    c_total = 0.14e-12 * (1.0 + 0.06 * candidate.optical_bus_count)
    l_total = 1.0 / ((2.0 * np.pi * f0_hz) ** 2 * c_total)
    r_total = 1.5 + 0.48 * metrics.routing_length_mm + 14.0 * metrics.crosstalk_linear
    g_total = 2.0e-4 * (1.0 + 0.08 * candidate.shielding_layers)
    c_section = c_total / sections
    l_section = l_total / sections
    r_section = r_total / sections
    g_section = g_total / sections
    s_mats = []
    for w in omega:
        zs = r_section + 1j * w * l_section
        yp = g_section + 1j * w * c_section
        t_section = np.array([[1.0 + zs * yp, zs], [yp, 1.0]], dtype=complex)
        t_total = np.eye(2, dtype=complex)
        for _ in range(sections):
            t_total = t_total @ t_section
        a, b = t_total[0, 0], t_total[0, 1]
        c, d = t_total[1, 0], t_total[1, 1]
        denom = a + b / z0 + c * z0 + d
        s11 = (a + b / z0 - c * z0 - d) / denom
        s21 = 2.0 / denom
        s_mats.append([[s11, s21], [s21, s11]])
    s_arr = np.asarray(s_mats, dtype=complex)
    network = rf.Network(frequency=rf.Frequency.from_f(frequencies_hz, unit="hz"), s=s_arr, z0=z0)
    s21_db = network.s_db[:, 1, 0]
    s11_db = network.s_db[:, 0, 0]
    center_idx = int(np.argmin(np.abs(frequencies_hz - f0_hz)))
    center_db = float(s21_db[center_idx])
    peak_db = float(np.max(s21_db))
    return_loss_db = float(-s11_db[center_idx])
    s11_mag = abs(network.s[center_idx, 0, 0])
    vswr = float((1.0 + s11_mag) / max(1.0 - s11_mag, 1e-6))
    phase = np.unwrap(np.angle(network.s[:, 1, 0]))
    group_delay_ns = float(np.mean(-np.gradient(phase, omega)) * 1.0e9)
    mask = s21_db >= peak_db - 3.0
    bandwidth_mhz = float((frequencies_hz[mask][-1] - frequencies_hz[mask][0]) / 1.0e6) if np.any(mask) else 0.0
    plot_path = plot_dir / f"{label}_microwave_response.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.plot(frequencies_hz / 1.0e9, s21_db, color="#1d3557", linewidth=2.1, label="S21")
    ax.plot(frequencies_hz / 1.0e9, s11_db, color="#e76f51", linewidth=1.4, label="S11")
    ax.axvline(metrics.nv_frequency_ghz, color="#2a9d8f", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("scikit-rf Microwave Network")
    ax.grid(alpha=0.24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return (
        MicrowaveSweepSummary(
            center_frequency_ghz=metrics.nv_frequency_ghz,
            center_s21_db=center_db,
            peak_s21_db=peak_db,
            bandwidth_mhz=bandwidth_mhz,
            group_delay_ns=group_delay_ns,
            return_loss_db=return_loss_db,
            vswr=vswr,
        ),
        str(plot_path),
    )


def _thermal_map(candidate: CandidateDesign, metrics: SimulationMetrics, plot_dir: Path, label: str) -> tuple[ThermalMapSummary, str]:
    grid = 46
    total_nodes = grid * grid
    source = np.zeros(total_nodes, dtype=float)
    rows = []
    cols = []
    data = []

    def idx(r: int, c: int) -> int:
        return r * grid + c

    def deposit(x0: int, x1: int, y0: int, y1: int, power: float) -> None:
        for r in range(y0, y1):
            for c in range(x0, x1):
                source[idx(r, c)] += power / max((x1 - x0) * (y1 - y0), 1)

    deposit(2, 8, 10, grid - 10, 0.20 * metrics.power_mw)
    deposit(grid - 8, grid - 2, 10, grid - 10, 0.20 * metrics.power_mw)
    deposit(8, grid - 8, 2, 7, 0.18 * metrics.power_mw)
    deposit(8, grid - 8, grid - 8, grid - 3, 0.22 * metrics.power_mw)
    deposit(14, grid - 14, 14, grid - 14, 0.16 * metrics.power_mw)
    conductivity = 650.0
    boundary = 0.0
    for r in range(grid):
        for c in range(grid):
            node = idx(r, c)
            if r in {0, grid - 1} or c in {0, grid - 1}:
                rows.append(node)
                cols.append(node)
                data.append(1.0)
                source[node] = boundary
                continue
            rows.append(node)
            cols.append(node)
            data.append(4.0)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                rows.append(node)
                cols.append(idx(nr, nc))
                data.append(-1.0)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
    rhs = source / conductivity
    solution = spla.spsolve(matrix, rhs).reshape((grid, grid))
    solution -= solution.min()
    peak = float(np.max(solution))
    mean = float(np.mean(solution))
    grad_y, grad_x = np.gradient(solution)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    max_gradient = float(np.max(gradient) * 1000.0)
    hotspot = np.unravel_index(int(np.argmax(solution)), solution.shape)
    plot_path = plot_dir / f"{label}_thermal_map.png"
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(solution, cmap="inferno", origin="lower")
    ax.set_title("Sparse Thermal Solve")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta T (K)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return (
        ThermalMapSummary(
            peak_temp_rise_k=peak,
            mean_temp_rise_k=mean,
            hotspot_x_norm=float(hotspot[1] / (grid - 1)),
            hotspot_y_norm=float(hotspot[0] / (grid - 1)),
            max_gradient_k_per_mm=max_gradient,
        ),
        str(plot_path),
    )


def _photon_statistics(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, plot_dir: Path, label: str, seed: int) -> tuple[PhotonStatisticsSummary, str]:
    rng = np.random.default_rng(seed)
    bright_mean = readout_photon_counts(
        metrics.optical_efficiency,
        spec.optical_power_budget_mw,
        candidate.qubits,
        candidate.optical_bus_count,
        pulse_ns=260.0,
    )
    contrast = 0.28 + 0.03 * candidate.anneal_quality
    dark_mean = max(1.5, bright_mean * (1.0 - contrast) + 2.0 + candidate.amplifier_noise_temp_k)
    bright = rng.poisson(bright_mean, size=6000)
    dark = rng.poisson(dark_mean, size=6000)
    thresholds = range(int(min(bright.min(), dark.min())), int(max(bright.max(), dark.max())) + 1)
    best_threshold = 0
    best_fidelity = 0.0
    for threshold in thresholds:
        bright_correct = np.mean(bright >= threshold)
        dark_correct = np.mean(dark < threshold)
        fidelity = 0.5 * (bright_correct + dark_correct)
        if fidelity > best_fidelity:
            best_fidelity = float(fidelity)
            best_threshold = int(threshold)
    shot_margin = float((np.mean(bright) - np.mean(dark)) / max(np.std(bright) + np.std(dark), 1e-6))
    plot_path = plot_dir / f"{label}_photon_hist.png"
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bins = np.arange(int(min(dark.min(), bright.min())), int(max(dark.max(), bright.max())) + 2)
    ax.hist(dark, bins=bins, alpha=0.55, label="dark", color="#457b9d", density=True)
    ax.hist(bright, bins=bins, alpha=0.55, label="bright", color="#f4a261", density=True)
    ax.axvline(best_threshold, color="#e63946", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Detected photons")
    ax.set_ylabel("Density")
    ax.set_title("Readout Photon Statistics")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return (
        PhotonStatisticsSummary(
            empirical_readout_fidelity=best_fidelity,
            optimal_threshold=best_threshold,
            bright_mean_counts=float(np.mean(bright)),
            dark_mean_counts=float(np.mean(dark)),
            shot_noise_margin=shot_margin,
        ),
        str(plot_path),
    )


def _control_loop_response(candidate: CandidateDesign, metrics: SimulationMetrics, plot_dir: Path, label: str) -> tuple[ControlLoopSummary, str]:
    wn = 2.0 * np.pi * max(20.0, 550.0 / max(metrics.latency_ns, 1e-6)) * 1.0e6
    zeta = max(0.25, min(0.95, 0.82 - 0.16 * metrics.crosstalk_linear + 0.03 * candidate.shielding_layers))
    loop_gain = 1.0 + 0.08 * candidate.amplifier_gain_db + 0.04 * candidate.control_mux_factor
    num = [loop_gain * wn**2]
    den = [1.0, 2.0 * zeta * wn, wn**2]
    sys = signal.TransferFunction(num, den)
    time = np.linspace(0.0, max(8.0 / wn, 20e-9), 5000)
    tout, yout = signal.step(sys, T=time)
    final = float(yout[-1]) if len(yout) else 1.0
    overshoot = max(0.0, (float(np.max(yout)) - final) / max(final, 1e-9) * 100.0)
    within = np.where(np.abs(yout - final) <= 0.02 * max(final, 1e-9))[0]
    settling_idx = int(within[0]) if len(within) else len(tout) - 1
    settling_ns = float(tout[settling_idx] * 1.0e9)
    freqs = np.logspace(4, 9, 500)
    _, h = signal.freqresp(sys, w=2.0 * np.pi * freqs)
    mag = np.abs(h)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    phase_deg = np.unwrap(np.angle(h)) * 180.0 / np.pi
    crossing = np.argmin(np.abs(mag_db))
    bandwidth_mhz = float(freqs[crossing] / 1.0e6)
    phase_margin = float(180.0 + phase_deg[crossing])
    plot_path = plot_dir / f"{label}_control_loop.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(tout * 1.0e9, yout, color="#264653", linewidth=1.8)
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Step response")
    axes[0].set_title("Control Loop Step")
    axes[0].grid(alpha=0.24)
    axes[1].semilogx(freqs / 1.0e6, mag_db, color="#e76f51", linewidth=1.8)
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title("Closed-Loop Bode")
    axes[1].grid(alpha=0.24)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return ControlLoopSummary(bandwidth_mhz, phase_margin, overshoot, settling_ns), str(plot_path)


def _phase_noise_spectrum(candidate: CandidateDesign, metrics: SimulationMetrics, plot_dir: Path, label: str, seed: int) -> tuple[NoiseSpectrumSummary, str]:
    rng = np.random.default_rng(seed + 2024)
    fs = 2.0e9
    samples = 8192
    white = rng.normal(0.0, 1.0, size=samples)
    pink = np.cumsum(rng.normal(0.0, 1.0, size=samples))
    pink = (pink - np.mean(pink)) / max(np.std(pink), 1e-6)
    noise = 0.65 * white + 0.35 * pink
    loop_filter = signal.butter(3, min(0.25, max(metrics.gate_fidelity, 0.6) * 0.18))
    shaped = signal.filtfilt(*loop_filter, noise)
    freqs, psd = signal.welch(shaped, fs=fs, nperseg=1024, scaling="density")
    phase_noise = psd * (1.0 + 8.0 * metrics.crosstalk_linear + 0.3 * candidate.control_mux_factor)
    integrated = float(np.trapezoid(phase_noise, freqs))
    rms_jitter_ps = float(math.sqrt(max(integrated, 1e-18)) * 1.0e12 / (2.0 * np.pi * max(metrics.nv_frequency_ghz * 1.0e9, 1.0)))
    noise_floor_db = float(10.0 * np.log10(np.median(phase_noise[1:]) + 1e-18))
    dephasing_proxy = float(min(1.0, math.sqrt(max(integrated, 0.0)) * 1.0e3))
    plot_path = plot_dir / f"{label}_phase_noise.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.semilogx(freqs[1:], 10.0 * np.log10(np.maximum(phase_noise[1:], 1e-18)), color="#6a4c93", linewidth=1.8)
    ax.set_xlabel("Offset frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Phase Noise Proxy")
    ax.grid(alpha=0.24)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return NoiseSpectrumSummary(rms_jitter_ps, integrated, noise_floor_db, dephasing_proxy), str(plot_path)


def _power_grid_ir_drop(candidate: CandidateDesign, metrics: SimulationMetrics, topology_plan: TopologyPlan, plot_dir: Path, label: str) -> tuple[PowerGridSummary, str]:
    grid = 34
    total = grid * grid
    rows = []
    cols = []
    data = []
    rhs = np.zeros(total, dtype=float)

    def idx(r: int, c: int) -> int:
        return r * grid + c

    source_regions = [
        (2, 7, 8, grid - 8, 0.18 * metrics.power_mw),
        (grid - 7, grid - 2, 8, grid - 8, 0.18 * metrics.power_mw),
        (10, grid - 10, 10, grid - 10, 0.24 * metrics.power_mw),
    ]
    for x0, x1, y0, y1, value in source_regions:
        for r in range(y0, y1):
            for c in range(x0, x1):
                rhs[idx(r, c)] += value / max((x1 - x0) * (y1 - y0), 1)

    conductance = 1.0 + 0.10 * candidate.metal_layers + 0.08 * topology_plan.route_width_scale
    for r in range(grid):
        for c in range(grid):
            node = idx(r, c)
            if r in {0, grid - 1} or c in {0, grid - 1}:
                rows.append(node)
                cols.append(node)
                data.append(1.0)
                rhs[node] = 0.0
                continue
            rows.append(node)
            cols.append(node)
            data.append(4.0 * conductance)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                rows.append(node)
                cols.append(idx(nr, nc))
                data.append(-conductance)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(total, total))
    voltage_drop = spla.spsolve(matrix, rhs).reshape((grid, grid))
    voltage_drop -= voltage_drop.min()
    voltage_drop *= (0.18 + 0.04 * candidate.control_mux_factor)
    worst_drop = float(np.max(voltage_drop) * 1000.0)
    mean_drop = float(np.mean(voltage_drop) * 1000.0)
    hotspot = np.unravel_index(int(np.argmax(voltage_drop)), voltage_drop.shape)
    plot_path = plot_dir / f"{label}_ir_drop.png"
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(voltage_drop * 1000.0, cmap="magma", origin="lower")
    ax.set_title("Power Grid IR Drop")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mV")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return PowerGridSummary(worst_drop, mean_drop, float(hotspot[1] / (grid - 1)), float(hotspot[0] / (grid - 1))), str(plot_path)


def _entanglement_link(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, qec_plan: QECPlan, plot_dir: Path, label: str) -> tuple[EntanglementSummary, str]:
    photon_capture = metrics.optical_efficiency
    spin_preservation = math.exp(-metrics.latency_ns / max(metrics.t2_us * 1000.0, 1e-6))
    two_photon_interference = math.exp(-4.2 * metrics.crosstalk_linear) * min(metrics.loaded_q / 1.2e6, 1.2)
    bell_fidelity = max(0.0, min(0.999999, 0.54 * metrics.readout_fidelity + 0.22 * spin_preservation + 0.24 * photon_capture * two_photon_interference))
    heralding = max(1e-6, min(0.95, 0.36 * photon_capture**2 * math.exp(-candidate.cell_pitch_um / 160.0) * (1.0 - 0.5 * metrics.crosstalk_linear)))
    ent_rate = 1.0e6 * heralding / max(metrics.latency_ns + spec.decoder_margin_ns + spec.syndrome_cycle_ns, 1.0)
    link_margin = max(0.0, min(1.5, bell_fidelity / max(0.85, 1e-6) * (1.0 + 0.05 * qec_plan.decoder_locality_score)))
    mitigation = max(0.0, min(1.5, 1.08 - qec_plan.logical_error_rate / max(spec.target_logical_error_rate, 1e-12)))
    distances = np.linspace(0.0, 1.0, 160)
    fidelity_curve = bell_fidelity * np.exp(-0.55 * distances) + 0.02 * np.cos(8.0 * distances)
    plot_path = plot_dir / f"{label}_entanglement_link.png"
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(distances, fidelity_curve, color="#2a9d8f", linewidth=1.8)
    ax.set_xlabel("Normalized link distance")
    ax.set_ylabel("Bell-state fidelity")
    ax.set_title("Entanglement Link Fidelity")
    ax.grid(alpha=0.24)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return EntanglementSummary(bell_fidelity, heralding, ent_rate, link_margin, mitigation), str(plot_path)


def _error_channel_bundle(spec: DesignSpec, metrics: SimulationMetrics, dense_crosstalk: DenseCrosstalkSummary, qec_plan: QECPlan, plot_dir: Path, label: str) -> tuple[ErrorChannelSummary, str]:
    depolarizing = max(1e-6, min(0.20, 1.0 - metrics.gate_fidelity))
    measurement = max(1e-6, min(0.20, 1.0 - metrics.readout_fidelity))
    erasure = max(1e-6, min(0.20, 0.015 + 0.08 * (1.0 - metrics.optical_efficiency)))
    correlated = max(1e-6, min(0.30, 0.55 * dense_crosstalk.effective_crosstalk_linear + 0.35 * dense_crosstalk.spectral_radius + 0.10 * qec_plan.decoder_utilization))
    decoder_failure = max(1e-6, min(0.50, depolarizing + 0.6 * measurement + 0.4 * correlated + qec_plan.logical_error_rate))
    categories = ["depol", "meas", "erase", "corr", "decode"]
    values = [depolarizing, measurement, erasure, correlated, decoder_failure]
    plot_path = plot_dir / f"{label}_error_channels.png"
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(categories, values, color=["#457b9d", "#f4a261", "#e9c46a", "#e76f51", "#6a4c93"])
    ax.set_ylabel("Error probability")
    ax.set_title("Error Channel Bundle")
    ax.grid(axis="y", alpha=0.24)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return ErrorChannelSummary(depolarizing, measurement, erasure, correlated, decoder_failure), str(plot_path)


def _routing_graph(candidate: CandidateDesign, metrics: SimulationMetrics, plan: PlacementPlan, plot_dir: Path, label: str) -> tuple[RoutingGraphSummary, str]:
    rows = candidate.rows + 2
    cols = candidate.cols + 2
    graph = nx.grid_2d_graph(rows, cols)
    base_capacity = 1.0 + 0.35 * candidate.metal_layers + 0.2 * candidate.shielding_layers
    loads: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for edge in graph.edges:
        graph.edges[edge]["capacity"] = base_capacity
        graph.edges[edge]["weight"] = 1.0 / base_capacity
        loads[tuple(sorted(edge))] = 0.0

    lengths = []
    connected = 0
    tile_map = {(tile.row, tile.col): tile for tile in plan.tiles}
    for row in range(candidate.rows):
        for col in range(candidate.cols):
            tile = tile_map.get((row, col))
            if tile is None:
                continue
            target = (row + 1, col + 1)
            control_source = (min(rows - 1, max(0, tile.microwave_domain + 1)), 0 if col < candidate.cols / 2 else cols - 1)
            optical_source = (0, min(cols - 1, tile.optical_bus + 1))
            for source in (control_source, optical_source):
                path = nx.shortest_path(graph, source=source, target=target, weight="weight")
                connected += 1
                lengths.append((len(path) - 1) * candidate.cell_pitch_um)
                for u, v in zip(path, path[1:]):
                    edge = tuple(sorted((u, v)))
                    loads[edge] += 1.0

    max_edge_load = max(loads.values()) if loads else 0.0
    congestion_ratio = max_edge_load / max(base_capacity, 1e-6)
    avg_path = float(np.mean(lengths)) if lengths else 0.0
    worst_path = float(np.max(lengths)) if lengths else 0.0
    connected_ratio = connected / max(2 * candidate.qubits, 1)

    heat = np.zeros((rows, cols), dtype=float)
    for (u, v), value in loads.items():
        heat[u] += value
        heat[v] += value
    plot_path = plot_dir / f"{label}_routing_heatmap.png"
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(heat[1:-1, 1:-1], cmap="viridis", origin="lower")
    ax.set_title("Routing Congestion Graph")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Edge load")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return (
        RoutingGraphSummary(
            average_path_um=avg_path,
            worst_path_um=worst_path,
            congestion_ratio=congestion_ratio,
            max_edge_load=max_edge_load,
            connected_ratio=connected_ratio,
        ),
        str(plot_path),
    )


def _optical_spectrum(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, plot_dir: Path, label: str) -> tuple[OpticalSpectrumSummary, str]:
    lambda0 = spec.optical_wavelength_nm
    wavelengths = np.linspace(lambda0 - 1.8, lambda0 + 1.8, 600)
    linewidth_nm = max(lambda0 / max(metrics.loaded_q, 1e-6), 0.0003)
    fsr_nm = max(0.08, 18.0 / max(math.sqrt(candidate.resonator_count), 1.0) + candidate.cell_pitch_um / 95.0)
    peak_drop = min(0.995, metrics.optical_efficiency * (0.82 + 0.08 * candidate.detector_qe))
    detuning = (wavelengths - lambda0) / linewidth_nm
    drop = peak_drop / (1.0 + 4.0 * detuning**2)
    periodic = 0.18 * np.cos(2.0 * np.pi * (wavelengths - lambda0) / fsr_nm)
    through = np.clip(1.0 - drop + periodic * 0.05, 1e-4, 1.0)
    drop = np.clip(drop * (1.0 + periodic), 1e-6, 1.0)
    peak_drop_db = float(10.0 * np.log10(np.max(drop)))
    extinction_db = float(10.0 * np.log10(np.max(through) / max(np.min(through), 1e-6)))
    linewidth_pm = float(linewidth_nm * 1000.0)
    plot_path = plot_dir / f"{label}_optical_spectrum.png"
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.plot(wavelengths, 10.0 * np.log10(through), color="#1d3557", linewidth=1.8, label="through")
    ax.plot(wavelengths, 10.0 * np.log10(drop), color="#ffb703", linewidth=1.8, label="drop")
    ax.axvline(lambda0, color="#e63946", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (dB)")
    ax.set_title("Optical Resonator Transfer")
    ax.grid(alpha=0.24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return (
        OpticalSpectrumSummary(
            peak_drop_db=peak_drop_db,
            extinction_db=extinction_db,
            fsr_nm=fsr_nm,
            linewidth_pm=linewidth_pm,
            net_collection_efficiency=float(np.max(drop)),
        ),
        str(plot_path),
    )


def _geometry_from_layout(layout: LayoutBundle) -> GeometryDRCSummary:
    metal_layers = {"GROUND", "M1", "M2", "M3", "M4", "M5", "M6", "RESONATOR", "OPTICAL", "READOUT", "DETECTOR"}
    shapes: list[tuple[str, Any]] = []
    for rect in layout.rects:
        if rect.layer in {"DIE", "SCRIBE", "PASSIVATION", "FILL"}:
            continue
        shapes.append((rect.layer, box(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)))
    for route in layout.routes:
        geom = LineString(route.points).buffer(route.width / 2.0, cap_style=1, join_style=1)
        shapes.append((route.layer, geom))
    for circle in layout.circles:
        shapes.append((circle.layer, Point(circle.cx, circle.cy).buffer(circle.radius)))

    densities = []
    windows_x = 5
    windows_y = 5
    win_w = layout.die_width_um / windows_x
    win_h = layout.die_height_um / windows_y
    metal_geoms = [geom for layer, geom in shapes if layer in metal_layers]
    for wx in range(windows_x):
        for wy in range(windows_y):
            window = box(wx * win_w, wy * win_h, (wx + 1) * win_w, (wy + 1) * win_h)
            metal_area = 0.0
            for geom in metal_geoms:
                inter = geom.intersection(window)
                if not inter.is_empty:
                    metal_area += inter.area
            densities.append(metal_area / max(window.area, 1e-6))

    via_count = sum(count for layer, count in layout.stats.get("rect_count_by_layer", {}).items() if layer.startswith("VIA"))
    via_density = via_count / max((layout.die_width_um * layout.die_height_um) / 1.0e6, 1e-6)
    return GeometryDRCSummary(
        min_spacing_um=float(layout.drc.get("min_spacing_um", 0.0)),
        density_min=float(min(densities) if densities else 0.0),
        density_max=float(max(densities) if densities else 0.0),
        overlap_violations=int(layout.drc.get("route_drc_violations", 0)),
        via_density_per_mm2=float(via_density),
    )


def evaluate_candidate_with_open_source_sims(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    output_dir: Path,
    base_score: float,
    seed: int,
    dense_signals: FastDenseSignals | None = None,
    qec_plan: QECPlan | None = None,
    neural_signals: Any | None = None,
    paper_knowledge: Any | None = None,
    requirements_bundle: Any | None = None,
) -> AdvancedSimulationBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = _candidate_key(candidate)
    plan = build_probabilistic_placement(spec, candidate, samples=28, seed_offset=seed)
    dense_crosstalk = simulate_dense_crosstalk(spec, candidate, plan, trials=192, seed_offset=seed)
    placement_artifacts = write_dense_placement_artifacts(output_dir, plan, dense_crosstalk, label)
    qec_plan = qec_plan or build_qec_plan(spec, candidate, metrics, dense_signals=dense_signals)
    topology_plan = build_topology_plan(
        spec,
        candidate,
        metrics,
        dense_signals=dense_signals,
        placement=plan,
        dense_crosstalk=dense_crosstalk,
        neural_signals=neural_signals,
        paper_knowledge=paper_knowledge,
        requirements_bundle=requirements_bundle,
    )
    qutip_summary = _qubit_dephasing(metrics)
    microwave_summary, microwave_plot = _microwave_sweep(candidate, metrics, output_dir, label)
    thermal_summary, thermal_plot = _thermal_map(candidate, metrics, output_dir, label)
    photon_summary, photon_plot = _photon_statistics(spec, candidate, metrics, output_dir, label, seed)
    control_summary, control_plot = _control_loop_response(candidate, metrics, output_dir, label)
    noise_summary, noise_plot = _phase_noise_spectrum(candidate, metrics, output_dir, label, seed)
    power_grid_summary, power_grid_plot = _power_grid_ir_drop(candidate, metrics, topology_plan, output_dir, label)
    entanglement_summary, entanglement_plot = _entanglement_link(spec, candidate, metrics, qec_plan, output_dir, label)
    error_channel_summary, error_plot = _error_channel_bundle(spec, metrics, dense_crosstalk, qec_plan, output_dir, label)
    routing_summary, routing_plot = _routing_graph(candidate, metrics, plan, output_dir, label)
    optical_summary, optical_plot = _optical_spectrum(spec, candidate, metrics, output_dir, label)
    layout = generate_layout(spec, candidate, metrics, topology_plan=topology_plan, placement=plan, qec_plan=qec_plan)
    geometry_summary = _geometry_from_layout(layout)
    placement_summary = PlacementSummary(
        placement_score=plan.placement_score,
        min_spacing_um=plan.min_spacing_um,
        mean_offset_um=plan.mean_offset_um,
        optical_load_std=float(np.std(plan.optical_bus_loads) / max(np.mean(plan.optical_bus_loads), 1e-6)) if plan.optical_bus_loads else 0.0,
        microwave_load_std=float(np.std(plan.microwave_domain_loads) / max(np.mean(plan.microwave_domain_loads), 1e-6)) if plan.microwave_domain_loads else 0.0,
    )

    qutip_term = min(1.25, (0.45 * qutip_summary.single_gate_state_fidelity + 0.55 * qutip_summary.two_qubit_state_fidelity) / max(spec.target_gate_fidelity, 1e-6))
    photon_term = min(1.25, photon_summary.empirical_readout_fidelity / max(spec.target_readout_fidelity, 1e-6))
    thermal_term = max(0.0, min(1.2, 1.15 - 2.2 * thermal_summary.peak_temp_rise_k - 0.0002 * thermal_summary.max_gradient_k_per_mm))
    microwave_term = max(0.0, min(1.2, 1.12 - abs(microwave_summary.center_s21_db) / 16.0 - max(microwave_summary.vswr - 1.0, 0.0) / 8.0))
    control_term = max(0.0, min(1.2, 1.10 - control_summary.overshoot_pct / 80.0 - max(0.0, 45.0 - control_summary.phase_margin_deg) / 120.0))
    noise_term = max(0.0, min(1.2, 1.08 - 0.30 * noise_summary.dephasing_proxy - noise_summary.rms_jitter_ps / 200.0))
    power_term = max(0.0, min(1.2, 1.10 - power_grid_summary.worst_drop_mV / 180.0))
    entanglement_term = max(0.0, min(1.2, 0.70 * entanglement_summary.repeater_link_margin + 0.30 * entanglement_summary.error_mitigation_gain))
    error_term = max(0.0, min(1.2, 1.10 - 1.6 * error_channel_summary.decoder_failure_proxy - 0.6 * error_channel_summary.correlated_error))
    routing_term = max(0.0, min(1.2, 1.10 - 0.18 * max(routing_summary.congestion_ratio - 1.0, 0.0)))
    optical_term = max(0.0, min(1.2, optical_summary.net_collection_efficiency / max(metrics.optical_efficiency, 1e-6)))
    geometry_term = max(0.0, min(1.2, geometry_summary.density_min / 0.12 + geometry_summary.min_spacing_um / 15.0 - 0.04 * geometry_summary.overlap_violations))
    placement_term = max(0.0, min(1.2, 0.85 * placement_summary.min_spacing_um / max(candidate.cell_pitch_um, 1e-6) + 0.35 * placement_summary.placement_score))
    crosstalk_term = max(0.0, min(1.2, 1.05 - 2.8 * dense_crosstalk.effective_crosstalk_linear - 0.18 * dense_crosstalk.spectral_radius))
    constraint_penalty = 0.18 * len(metrics.constraint_violations)
    composite_score = (
        base_score
        + 0.11 * qutip_term
        + 0.06 * photon_term
        + 0.05 * thermal_term
        + 0.05 * microwave_term
        + 0.06 * control_term
        + 0.05 * noise_term
        + 0.05 * power_term
        + 0.07 * entanglement_term
        + 0.07 * error_term
        + 0.06 * routing_term
        + 0.05 * optical_term
        + 0.04 * geometry_term
        + 0.05 * placement_term
        + 0.08 * crosstalk_term
        - constraint_penalty
    )
    bundle = AdvancedSimulationBundle(
        candidate_key=label,
        base_score=base_score,
        composite_score=composite_score,
        placement=placement_summary,
        dense_crosstalk=dense_crosstalk,
        qutip=qutip_summary,
        microwave=microwave_summary,
        thermal=thermal_summary,
        photon=photon_summary,
        control=control_summary,
        noise=noise_summary,
        power_grid=power_grid_summary,
        entanglement=entanglement_summary,
        error_channels=error_channel_summary,
        routing=routing_summary,
        optical=optical_summary,
        geometry=geometry_summary,
        topology=topology_plan.to_dict(),
        plots={
            "microwave_response": microwave_plot,
            "thermal_map": thermal_plot,
            "photon_histogram": photon_plot,
            "control_loop": control_plot,
            "phase_noise": noise_plot,
            "ir_drop": power_grid_plot,
            "entanglement_link": entanglement_plot,
            "error_channels": error_plot,
            "routing_heatmap": routing_plot,
            "optical_spectrum": optical_plot,
            "placement_plot": placement_artifacts["placement_plot"],
        },
        artifacts=placement_artifacts | {"topology_plan_inline": f"complexity={topology_plan.layout_complexity_score:.3f}"},
    )
    (output_dir / f"{label}_advanced.json").write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    return bundle


def refine_frontier_with_open_source_sims(
    spec: DesignSpec,
    frontier: Sequence[Any],
    output_dir: Path,
    seed: int,
    top_k: int = 3,
    paper_knowledge: Any | None = None,
    requirements_bundle: Any | None = None,
) -> AdvancedRefinementResult:
    bundles: list[AdvancedSimulationBundle] = []
    for index, ranked in enumerate(frontier[:top_k]):
        bundles.append(
            evaluate_candidate_with_open_source_sims(
                spec,
                ranked.candidate,
                ranked.metrics,
                output_dir=output_dir,
                base_score=ranked.score,
                seed=seed + 37 * (index + 1),
                dense_signals=getattr(ranked, "dense_signals", None),
                qec_plan=getattr(ranked, "qec_plan", None),
                neural_signals=getattr(ranked, "neural_signals", None),
                paper_knowledge=paper_knowledge,
                requirements_bundle=requirements_bundle,
            )
        )
    bundles.sort(key=lambda item: item.composite_score, reverse=True)
    return AdvancedRefinementResult(selected_candidate_key=bundles[0].candidate_key if bundles else "", bundles=bundles)


def _route_length(route_points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(route_points, route_points[1:]))


def analyze_layout_parasitics(layout: LayoutBundle) -> LayoutParasiticSummary:
    total_route_length = sum(_route_length(route.points) for route in layout.routes)
    optical_length = sum(_route_length(route.points) for route in layout.routes if route.layer == "OPTICAL")
    microwave_length = sum(_route_length(route.points) for route in layout.routes if route.layer in {"GROUND", "M1", "M2", "M3", "M4", "M5", "M6"})
    die_area = max(layout.die_width_um * layout.die_height_um, 1e-6)
    route_density = total_route_length / math.sqrt(die_area)

    route_centers = [
        (route.layer, sum(point[0] for point in route.points) / len(route.points), sum(point[1] for point in route.points) / len(route.points), route.width)
        for route in layout.routes
    ]
    max_coupling = 0.0
    capped_routes = route_centers[: min(len(route_centers), 220)]
    for index, (_layer_a, ax, ay, aw) in enumerate(capped_routes):
        for _layer_b, bx, by, bw in capped_routes[index + 1 :]:
            dist = math.hypot(ax - bx, ay - by)
            coupling = (aw * bw) / max(dist, 10.0)
            max_coupling = max(max_coupling, coupling)

    geometry = _geometry_from_layout(layout)
    return LayoutParasiticSummary(
        total_route_length_um=total_route_length,
        route_density=route_density,
        max_coupling_risk=max_coupling,
        optical_route_length_um=optical_length,
        microwave_route_length_um=microwave_length,
        min_feature_spacing_um=geometry.min_spacing_um,
        metal_density_min=geometry.density_min,
        metal_density_max=geometry.density_max,
        overlap_violations=geometry.overlap_violations,
    )
