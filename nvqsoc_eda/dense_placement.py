from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from .simulator import CandidateDesign
from .spec import DesignSpec


matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(slots=True)
class TilePlacement:
    qubit_id: int
    row: int
    col: int
    x_um: float
    y_um: float
    optical_bus: int
    microwave_domain: int
    shield_strength: float
    resonator_span_um: float
    local_density: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlacementPlan:
    die_width_um: float
    die_height_um: float
    core_x_um: float
    core_y_um: float
    array_width_um: float
    array_height_um: float
    pad_margin_um: float
    optical_bus_x_um: list[float]
    microwave_lane_y_um: list[float]
    tiles: list[TilePlacement]
    placement_score: float
    min_spacing_um: float
    mean_offset_um: float
    optical_bus_loads: list[int] = field(default_factory=list)
    microwave_domain_loads: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "die_width_um": self.die_width_um,
            "die_height_um": self.die_height_um,
            "core_x_um": self.core_x_um,
            "core_y_um": self.core_y_um,
            "array_width_um": self.array_width_um,
            "array_height_um": self.array_height_um,
            "pad_margin_um": self.pad_margin_um,
            "optical_bus_x_um": self.optical_bus_x_um,
            "microwave_lane_y_um": self.microwave_lane_y_um,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "placement_score": self.placement_score,
            "min_spacing_um": self.min_spacing_um,
            "mean_offset_um": self.mean_offset_um,
            "optical_bus_loads": self.optical_bus_loads,
            "microwave_domain_loads": self.microwave_domain_loads,
        }


@dataclass(slots=True)
class DenseCrosstalkSummary:
    effective_crosstalk_linear: float
    effective_crosstalk_db: float
    mean_crosstalk_linear: float
    worst_case_linear: float
    spectral_radius: float
    p95_active_interference: float
    hotspot_qubit: int
    hotspot_x_um: float
    hotspot_y_um: float
    active_probability: float
    monte_carlo_trials: int
    interference_histogram: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FastDenseSignals:
    placement_score: float
    min_spacing_um: float
    mean_offset_um: float
    optical_load_std: float
    microwave_load_std: float
    effective_crosstalk_linear: float
    effective_crosstalk_db: float
    spectral_radius: float
    p95_active_interference: float
    hotspot_qubit: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_seed(candidate: CandidateDesign, salt: int = 0) -> int:
    base = (
        candidate.qubits * 31
        + int(candidate.cell_pitch_um * 100) * 17
        + candidate.optical_bus_count * 19
        + candidate.microwave_line_count * 23
        + candidate.resonator_count * 29
        + candidate.metal_layers * 37
        + candidate.shielding_layers * 41
        + int(candidate.cavity_q) % 1000003
    )
    return (base + salt * 9973) % (2**32 - 1)


def base_layout_frame(candidate: CandidateDesign) -> tuple[float, float, float, float, float, float, float]:
    pad_margin = 780.0 + 50.0 * candidate.metal_layers + 18.0 * math.sqrt(max(candidate.qubits, 1))
    array_width_um = candidate.cols * candidate.cell_pitch_um
    array_height_um = candidate.rows * candidate.cell_pitch_um
    die_width_um = candidate.die_width_mm * 1000.0
    die_height_um = candidate.die_height_mm * 1000.0
    core_x = (die_width_um - array_width_um) / 2.0
    core_y = (die_height_um - array_height_um) / 2.0
    return die_width_um, die_height_um, core_x, core_y, array_width_um, array_height_um, pad_margin


def _smooth_field(rng: np.random.Generator, length: int, scale: float) -> np.ndarray:
    if length <= 1:
        return np.zeros(length)
    raw = rng.normal(0.0, scale, size=length + 4)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    smoothed = np.convolve(raw, kernel / kernel.sum(), mode="valid")[:length]
    return smoothed - np.mean(smoothed)


def _pairwise_min_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=2))
    distances += np.eye(len(points)) * 1.0e9
    return float(np.min(distances))


def _route_alignment_score(x: float, y: float, bus_x: float, lane_y: float, pitch: float) -> float:
    distance = abs(x - bus_x) + abs(y - lane_y)
    return math.exp(-distance / max(2.6 * pitch, 1.0))


def build_probabilistic_placement(spec: DesignSpec, candidate: CandidateDesign, samples: int = 24, seed_offset: int = 0) -> PlacementPlan:
    die_width_um, die_height_um, core_x, core_y, array_width_um, array_height_um, pad_margin = base_layout_frame(candidate)
    bus_x = np.linspace(core_x + 0.5 * candidate.cell_pitch_um, core_x + array_width_um - 0.5 * candidate.cell_pitch_um, max(candidate.optical_bus_count, 1))
    lane_y = np.linspace(core_y + 0.5 * candidate.cell_pitch_um, core_y + array_height_um - 0.5 * candidate.cell_pitch_um, max(candidate.microwave_line_count, 1))
    rng = np.random.default_rng(_candidate_seed(candidate, salt=seed_offset))
    best_plan: PlacementPlan | None = None

    base_positions = []
    for row in range(candidate.rows):
        for col in range(candidate.cols):
            qubit_id = row * candidate.cols + col
            if qubit_id >= candidate.qubits:
                continue
            base_positions.append((qubit_id, row, col, core_x + (col + 0.5) * candidate.cell_pitch_um, core_y + (row + 0.5) * candidate.cell_pitch_um))

    for sample_idx in range(samples):
        row_field = _smooth_field(rng, candidate.rows, candidate.cell_pitch_um * 0.085)
        col_field = _smooth_field(rng, candidate.cols, candidate.cell_pitch_um * 0.085)
        tiles: list[TilePlacement] = []
        offsets = []
        bus_loads = [0 for _ in range(max(candidate.optical_bus_count, 1))]
        domain_loads = [0 for _ in range(max(candidate.microwave_line_count, 1))]
        points = []
        alignment_accumulator = 0.0

        for qubit_id, row, col, x0, y0 in base_positions:
            local_jitter = rng.normal(0.0, candidate.cell_pitch_um * 0.028, size=2)
            bus_index = int(np.argmin(np.abs(bus_x - x0 + rng.normal(0.0, candidate.cell_pitch_um * 0.02))))
            domain_index = int(np.argmin(np.abs(lane_y - y0 + rng.normal(0.0, candidate.cell_pitch_um * 0.02))))
            attraction_x = 0.18 * (bus_x[bus_index] - x0)
            attraction_y = 0.16 * (lane_y[domain_index] - y0)
            architecture_bias = {"sensor_dense": 0.75, "hybrid_router": 0.55, "network_node": 0.45}.get(candidate.architecture, 0.55)
            x = x0 + 0.58 * col_field[col] + 0.22 * row_field[row] + architecture_bias * local_jitter[0] + 0.18 * attraction_x
            y = y0 + 0.58 * row_field[row] + 0.22 * col_field[col] + architecture_bias * local_jitter[1] + 0.18 * attraction_y
            x = float(np.clip(x, core_x + 0.24 * candidate.cell_pitch_um, core_x + array_width_um - 0.24 * candidate.cell_pitch_um))
            y = float(np.clip(y, core_y + 0.24 * candidate.cell_pitch_um, core_y + array_height_um - 0.24 * candidate.cell_pitch_um))
            local_density = 1.0 + 0.12 * abs(row_field[row]) / max(candidate.cell_pitch_um, 1e-6) + 0.12 * abs(col_field[col]) / max(candidate.cell_pitch_um, 1e-6)
            shield_strength = min(1.0, 0.35 + 0.12 * candidate.shielding_layers + 0.05 * candidate.metal_layers + 0.08 * local_density)
            resonator_span = candidate.cell_pitch_um * (0.70 + 0.10 * rng.uniform(0.0, 1.0))
            tiles.append(
                TilePlacement(
                    qubit_id=qubit_id,
                    row=row,
                    col=col,
                    x_um=x,
                    y_um=y,
                    optical_bus=bus_index,
                    microwave_domain=domain_index,
                    shield_strength=shield_strength,
                    resonator_span_um=resonator_span,
                    local_density=local_density,
                )
            )
            bus_loads[bus_index] += 1
            domain_loads[domain_index] += 1
            offsets.append(math.hypot(x - x0, y - y0))
            points.append((x, y))
            alignment_accumulator += _route_alignment_score(x, y, bus_x[bus_index], lane_y[domain_index], candidate.cell_pitch_um)

        point_array = np.asarray(points, dtype=float)
        min_spacing = _pairwise_min_spacing(point_array)
        spacing_score = min_spacing / max(candidate.cell_pitch_um, 1e-6)
        bus_balance = np.std(bus_loads) / max(np.mean(bus_loads), 1e-6)
        domain_balance = np.std(domain_loads) / max(np.mean(domain_loads), 1e-6)
        score = 0.46 * spacing_score + 0.26 * (alignment_accumulator / max(len(tiles), 1)) - 0.14 * bus_balance - 0.14 * domain_balance
        plan = PlacementPlan(
            die_width_um=die_width_um,
            die_height_um=die_height_um,
            core_x_um=core_x,
            core_y_um=core_y,
            array_width_um=array_width_um,
            array_height_um=array_height_um,
            pad_margin_um=pad_margin,
            optical_bus_x_um=[float(value) for value in bus_x],
            microwave_lane_y_um=[float(value) for value in lane_y],
            tiles=tiles,
            placement_score=score,
            min_spacing_um=min_spacing,
            mean_offset_um=float(np.mean(offsets)) if offsets else 0.0,
            optical_bus_loads=bus_loads,
            microwave_domain_loads=domain_loads,
        )
        if best_plan is None or plan.placement_score > best_plan.placement_score:
            best_plan = plan

    if best_plan is None:
        raise RuntimeError("failed to build placement plan")
    return best_plan


def simulate_dense_crosstalk(spec: DesignSpec, candidate: CandidateDesign, plan: PlacementPlan, trials: int = 160, seed_offset: int = 0) -> DenseCrosstalkSummary:
    if not plan.tiles:
        return DenseCrosstalkSummary(0.0, -120.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, trials, [])

    rng = np.random.default_rng(_candidate_seed(candidate, salt=4096 + seed_offset))
    positions = np.asarray([(tile.x_um, tile.y_um) for tile in plan.tiles], dtype=float)
    bus_ids = np.asarray([tile.optical_bus for tile in plan.tiles], dtype=int)
    domain_ids = np.asarray([tile.microwave_domain for tile in plan.tiles], dtype=int)
    shield = np.asarray([tile.shield_strength for tile in plan.tiles], dtype=float)

    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=2)) + np.eye(len(plan.tiles))
    same_bus = (bus_ids[:, None] == bus_ids[None, :]).astype(float)
    same_domain = (domain_ids[:, None] == domain_ids[None, :]).astype(float)
    nearest_neighbor = (distances < 1.35 * candidate.cell_pitch_um).astype(float)
    distance_term = np.exp(-distances / max(0.92 * candidate.cell_pitch_um, 1.0))
    bus_term = same_bus * np.exp(-distances / max(1.8 * candidate.cell_pitch_um, 1.0))
    domain_term = same_domain * np.exp(-distances / max(1.25 * candidate.cell_pitch_um, 1.0))
    shield_term = np.exp(-0.18 * candidate.shielding_layers) / (1.0 + 0.22 * (shield[:, None] + shield[None, :]))
    metal_term = 1.0 / (1.0 + 0.05 * candidate.metal_layers)
    coupling = (0.010 * distance_term + 0.008 * domain_term + 0.006 * bus_term + 0.006 * nearest_neighbor * distance_term) * shield_term * metal_term
    np.fill_diagonal(coupling, 0.0)

    active_probability = min(0.60, 0.16 + 0.045 * candidate.control_mux_factor + 0.01 * candidate.optical_bus_count / max(candidate.qubits, 1))
    max_interference = []
    mean_interference = np.zeros(len(plan.tiles), dtype=float)
    for _ in range(trials):
        domain_activity = rng.random(max(candidate.microwave_line_count, 1)) < active_probability
        bus_activity = rng.random(max(candidate.optical_bus_count, 1)) < min(0.55, active_probability + 0.08)
        base_activity = rng.random(len(plan.tiles)) < 0.28
        tile_activity = base_activity | domain_activity[domain_ids] | bus_activity[bus_ids]
        tile_drive = tile_activity.astype(float) * (0.65 + 0.35 * rng.random(len(plan.tiles)))
        interference = coupling @ tile_drive
        mean_interference += interference
        max_interference.append(float(np.max(interference)))

    mean_interference /= max(trials, 1)
    eigvals = np.linalg.eigvalsh((coupling + coupling.T) / 2.0)
    spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    p95 = float(np.quantile(max_interference, 0.95)) if max_interference else 0.0
    mean_linear = float(np.mean(mean_interference))
    worst_linear = float(np.max(mean_interference))
    effective_linear = float(np.clip(0.70 * p95 + 0.20 * worst_linear + 0.10 * spectral_radius, 1e-9, 0.85))
    hotspot_index = int(np.argmax(mean_interference))
    hotspot_tile = plan.tiles[hotspot_index]
    return DenseCrosstalkSummary(
        effective_crosstalk_linear=effective_linear,
        effective_crosstalk_db=20.0 * math.log10(max(effective_linear, 1e-12)),
        mean_crosstalk_linear=mean_linear,
        worst_case_linear=worst_linear,
        spectral_radius=spectral_radius,
        p95_active_interference=p95,
        hotspot_qubit=hotspot_tile.qubit_id,
        hotspot_x_um=hotspot_tile.x_um,
        hotspot_y_um=hotspot_tile.y_um,
        active_probability=active_probability,
        monte_carlo_trials=trials,
        interference_histogram=max_interference[:80],
    )


def write_dense_placement_artifacts(output_dir: Path, plan: PlacementPlan, crosstalk: DenseCrosstalkSummary, label: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / f"{label}_placement.json"
    crosstalk_path = output_dir / f"{label}_crosstalk.json"
    plot_path = output_dir / f"{label}_placement_heatmap.png"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    crosstalk_path.write_text(json.dumps(crosstalk.to_dict(), indent=2), encoding="utf-8")

    xs = [tile.x_um for tile in plan.tiles]
    ys = [tile.y_um for tile in plan.tiles]
    colors = [tile.optical_bus for tile in plan.tiles]
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=38, edgecolors="black", linewidths=0.25)
    ax.scatter([crosstalk.hotspot_x_um], [crosstalk.hotspot_y_um], color="#e63946", s=120, marker="x", linewidths=2.0)
    ax.set_title("Probabilistic Dense Placement")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Optical bus")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "placement_json": str(plan_path),
        "crosstalk_json": str(crosstalk_path),
        "placement_plot": str(plot_path),
    }


def analyze_candidate_dense_fast(
    spec: DesignSpec,
    candidate: CandidateDesign,
    samples: int = 5,
    trials: int = 24,
    seed_offset: int = 0,
) -> FastDenseSignals:
    plan = build_probabilistic_placement(spec, candidate, samples=max(samples, 2), seed_offset=seed_offset)
    crosstalk = simulate_dense_crosstalk(spec, candidate, plan, trials=max(trials, 8), seed_offset=seed_offset)
    optical_std = float(np.std(plan.optical_bus_loads) / max(np.mean(plan.optical_bus_loads), 1e-6)) if plan.optical_bus_loads else 0.0
    microwave_std = float(np.std(plan.microwave_domain_loads) / max(np.mean(plan.microwave_domain_loads), 1e-6)) if plan.microwave_domain_loads else 0.0
    return FastDenseSignals(
        placement_score=plan.placement_score,
        min_spacing_um=plan.min_spacing_um,
        mean_offset_um=plan.mean_offset_um,
        optical_load_std=optical_std,
        microwave_load_std=microwave_std,
        effective_crosstalk_linear=crosstalk.effective_crosstalk_linear,
        effective_crosstalk_db=crosstalk.effective_crosstalk_db,
        spectral_radius=crosstalk.spectral_radius,
        p95_active_interference=crosstalk.p95_active_interference,
        hotspot_qubit=crosstalk.hotspot_qubit,
    )
