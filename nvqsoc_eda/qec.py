from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dense_placement import FastDenseSignals
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec


QEC_LIBRARY = {
    "surface_code": {
        "threshold": 1.0e-2,
        "alpha": 0.010,
        "qubits_per_distance": lambda d: 2 * d * d + 2 * d,
        "syndrome_factor": 1.00,
        "decoder_factor": 1.00,
        "patch_aspect": (1.0, 1.0),
        "factory_scale": 0.36,
        "factory_protocol": "15to1",
    },
    "color_code": {
        "threshold": 8.0e-3,
        "alpha": 0.014,
        "qubits_per_distance": lambda d: int(round(1.55 * d * d + 1.5 * d)),
        "syndrome_factor": 0.88,
        "decoder_factor": 1.10,
        "patch_aspect": (1.2, 1.0),
        "factory_scale": 0.78,
        "factory_protocol": "20to4",
    },
    "bacon_shor": {
        "threshold": 6.0e-3,
        "alpha": 0.018,
        "qubits_per_distance": lambda d: int(round(1.30 * d * d + 2.0 * d)),
        "syndrome_factor": 0.82,
        "decoder_factor": 0.92,
        "patch_aspect": (1.35, 0.85),
        "factory_scale": 0.56,
        "factory_protocol": "8to3",
    },
}


def _select_code_family(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, dense_signals: FastDenseSignals | None) -> str:
    explicit = spec.qec_code.lower()
    if explicit in QEC_LIBRARY:
        return explicit
    physical_error = _effective_physical_error_rate(metrics, dense_signals)
    target_logicals = spec.target_logical_qubits
    latency_pressure = spec.max_latency_ns / max(metrics.latency_ns, 1e-6)
    area_pressure = spec.max_die_area_mm2 / max(metrics.die_area_mm2, 1e-6)
    candidates: list[tuple[float, str]] = []
    for family, cfg in QEC_LIBRARY.items():
        threshold_score = min(cfg["threshold"] / max(physical_error, 1e-12), 1.5)
        distance = _required_distance(spec.target_logical_error_rate, physical_error, cfg["threshold"], cfg["alpha"])
        phys_per_logical = max(9, int(cfg["qubits_per_distance"](distance)))
        factory_qubits = _factory_count(spec, target_logicals) * _factory_qubits(family, distance)
        achievable = max(0, (candidate.qubits - factory_qubits) // max(phys_per_logical, 1))
        logical_fit = min(achievable / max(target_logicals, 1), 1.3)
        latency_fit = min(latency_pressure * (1.10 if family == "bacon_shor" else 0.95 if family == "color_code" else 1.0), 1.3)
        area_fit = min(area_pressure * (1.08 if family == "bacon_shor" else 0.96 if family == "color_code" else 1.0), 1.3)
        family_score = 0.38 * threshold_score + 0.28 * logical_fit + 0.18 * latency_fit + 0.16 * area_fit - 0.015 * distance
        candidates.append((family_score, family))
    candidates.sort(reverse=True)
    return candidates[0][1]


def estimate_required_physical_qubits(spec: DesignSpec) -> tuple[str, int, int]:
    gate_error = max(1.0 - spec.target_gate_fidelity, 1e-6)
    readout_error = max(1.0 - spec.target_readout_fidelity, 1e-6)
    p_phys_est = max(1e-6, min(0.18, 0.64 * gate_error + 0.10 * readout_error + 0.002 + 0.0003 * spec.target_logical_qubits))
    best_family = "surface_code"
    best_total = 10**9
    best_distance = 3
    family_names = list(QEC_LIBRARY) if spec.qec_code == "auto" or spec.qec_code not in QEC_LIBRARY else [spec.qec_code]
    for family in family_names:
        cfg = QEC_LIBRARY[family]
        distance = _required_distance(spec.target_logical_error_rate, p_phys_est, cfg["threshold"], cfg["alpha"])
        phys_per_logical = max(9, int(cfg["qubits_per_distance"](distance)))
        factory_qubits = _factory_count(spec, spec.target_logical_qubits) * _factory_qubits(family, distance)
        total = spec.target_logical_qubits * phys_per_logical + factory_qubits
        if total < best_total:
            best_total = total
            best_family = family
            best_distance = distance
    return best_family, best_distance, best_total


@dataclass(slots=True)
class LogicalPatch:
    logical_id: int
    qubit_indices: list[int]
    ancilla_qubits: int
    patch_rows: int
    patch_cols: int
    patch_width_um: float
    patch_height_um: float
    syndrome_bus_count: int
    grid_row: int
    grid_col: int
    center_x_norm: float
    center_y_norm: float
    decoder_cluster: int
    neighboring_patches: list[int] = field(default_factory=list)
    nearest_factory_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LatticeSurgeryChannel:
    source_patch_id: int
    target_patch_id: int
    surgery_type: str
    shared_boundary_qubits: int
    estimated_length_um: float
    cycle_ns: float
    round_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MagicStateFactory:
    factory_id: int
    protocol_name: str
    physical_qubits: int
    code_distance: int
    output_magic_states_per_us: float
    support_logical_qubits: int
    center_x_norm: float
    center_y_norm: float
    region: str
    parallel_lanes: int = 1
    client_patch_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LogicalInterconnect:
    source_kind: str
    source_id: int
    target_kind: str
    target_id: int
    route_class: str
    estimated_length_um: float
    estimated_latency_ns: float
    decoder_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LogicalOperation:
    operation_id: str
    op_type: str
    logical_patch_ids: list[int]
    start_ns: float
    end_ns: float
    duration_ns: float
    decoder_cluster: int
    factory_id: int | None = None
    dependency_ids: list[str] = field(default_factory=list)
    lane_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FactoryTimelineEntry:
    factory_id: int
    batch_id: int
    protocol_name: str
    start_ns: float
    end_ns: float
    produced_magic_states: int
    lane_index: int
    reserved_patch_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QECPlan:
    enabled: bool
    code_family: str
    code_distance: int
    target_logical_qubits: int
    achievable_logical_qubits: int
    physical_qubits_per_logical: int
    ancilla_qubits_total: int
    data_qubits_total: int
    total_physical_qubits_required: int
    factory_qubits_total: int
    physical_error_rate: float
    logical_error_rate: float
    threshold: float
    syndrome_cycle_ns: float
    decoder_latency_ns: float
    logical_cycle_ns: float
    logical_success_probability: float
    logical_yield: float
    qec_overhead_ratio: float
    logical_area_efficiency: float
    decoder_graph_nodes: int
    decoder_graph_edges: int
    decoder_locality_score: float
    surgery_throughput_ops_per_us: float
    magic_state_rate_per_us: float
    schedule_makespan_ns: float
    schedule_critical_path_ns: float
    average_patch_idle_ns: float
    factory_utilization: float
    decoder_utilization: float
    patches: list[LogicalPatch] = field(default_factory=list)
    surgery_channels: list[LatticeSurgeryChannel] = field(default_factory=list)
    magic_state_factories: list[MagicStateFactory] = field(default_factory=list)
    logical_interconnects: list[LogicalInterconnect] = field(default_factory=list)
    logical_schedule: list[LogicalOperation] = field(default_factory=list)
    factory_timelines: list[FactoryTimelineEntry] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            **asdict(self),
            "patches": [patch.to_dict() for patch in self.patches],
            "surgery_channels": [channel.to_dict() for channel in self.surgery_channels],
            "magic_state_factories": [factory.to_dict() for factory in self.magic_state_factories],
            "logical_interconnects": [interconnect.to_dict() for interconnect in self.logical_interconnects],
            "logical_schedule": [item.to_dict() for item in self.logical_schedule],
            "factory_timelines": [item.to_dict() for item in self.factory_timelines],
        })


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _effective_physical_error_rate(metrics: SimulationMetrics, dense_signals: FastDenseSignals | None) -> float:
    dense_xtalk = dense_signals.effective_crosstalk_linear if dense_signals else metrics.crosstalk_linear
    gate_component = 1.0 - metrics.gate_fidelity
    readout_component = 1.0 - metrics.readout_fidelity
    decoherence_component = metrics.latency_ns / max(metrics.t2_us * 1000.0, 1e-6)
    mitigation = max(0.58, min(1.0, 0.92 - 1.6 * dense_xtalk + 0.04 * metrics.routing_capacity))
    effective = (0.48 * gate_component + 0.03 * readout_component + 0.08 * dense_xtalk + 0.03 * decoherence_component) * mitigation
    return max(5e-5, min(0.20, effective))


def _required_distance(target_logical_error_rate: float, physical_error_rate: float, threshold: float, alpha: float) -> int:
    if physical_error_rate >= threshold:
        return 9
    ratio = max(physical_error_rate / max(threshold, 1e-9), 1e-9)
    for distance in range(3, 26, 2):
        logical = alpha * ratio ** ((distance + 1) / 2.0)
        if logical <= target_logical_error_rate:
            return distance
    return 25


def _logical_error_rate(alpha: float, physical_error_rate: float, threshold: float, distance: int) -> float:
    ratio = max(physical_error_rate / max(threshold, 1e-9), 1e-9)
    if physical_error_rate >= threshold:
        return min(0.49, alpha * (1.0 + ratio) * ratio)
    return alpha * ratio ** ((distance + 1) / 2.0)


def _grid_shape(count: int, connectivity: str) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    if connectivity == "line":
        return 1, count
    rows = max(1, round(math.sqrt(count)))
    cols = math.ceil(count / rows)
    while rows * cols < count:
        cols += 1
    return rows, cols


def _factory_count(spec: DesignSpec, target_logicals: int) -> int:
    base = 1 if target_logicals > 0 else 0
    if spec.application in {"processor", "memory"}:
        base += 1
    if target_logicals >= 4:
        base += 1
    return max(0, min(base, 4))


def _factory_qubits(code_family: str, distance: int) -> int:
    lib = QEC_LIBRARY[code_family]
    return max(12, int(round(lib["factory_scale"] * lib["qubits_per_distance"](distance))))


def _estimate_factory_resources(spec: DesignSpec, code_family: str, distance: int, target_logicals: int) -> tuple[int, int]:
    count = _factory_count(spec, target_logicals)
    base = _factory_qubits(code_family, distance)
    total = 0
    for factory_id in range(count):
        parallel_lanes = 1 + (1 if (target_logicals >= 2 and distance >= 5 and factory_id % 2 == 0) or (target_logicals >= 3 and factory_id % 2 == 0) else 0)
        total += int(round(base * (1.0 + 0.35 * max(parallel_lanes - 1, 0))))
    return count, total


def _patch_partition(
    candidate: CandidateDesign,
    achievable_logical_qubits: int,
    physical_qubits_per_logical: int,
    code_distance: int,
    patch_aspect: tuple[float, float],
    control_clusters: int,
    logical_connectivity: str,
) -> list[LogicalPatch]:
    patches: list[LogicalPatch] = []
    assigned = 0
    all_indices = list(range(candidate.qubits))
    rows, cols = _grid_shape(achievable_logical_qubits, logical_connectivity)
    x_grid = [0.20] if cols <= 1 else [0.18 + 0.62 * idx / (cols - 1) for idx in range(cols)]
    y_grid = [0.58] if rows <= 1 else [0.32 + 0.46 * idx / (rows - 1) for idx in range(rows)]
    for logical_id in range(achievable_logical_qubits):
        chunk = all_indices[assigned : assigned + physical_qubits_per_logical]
        if not chunk:
            break
        assigned += len(chunk)
        grid_row = logical_id // cols
        grid_col = logical_id % cols
        ancilla = max(4, len(chunk) // 3)
        patch_rows = max(3, int(round(code_distance * patch_aspect[1])))
        patch_cols = max(3, int(round(code_distance * patch_aspect[0])))
        patches.append(
            LogicalPatch(
                logical_id=logical_id,
                qubit_indices=chunk,
                ancilla_qubits=ancilla,
                patch_rows=patch_rows,
                patch_cols=patch_cols,
                patch_width_um=patch_cols * candidate.cell_pitch_um,
                patch_height_um=patch_rows * candidate.cell_pitch_um,
                syndrome_bus_count=max(2, code_distance // 2),
                grid_row=grid_row,
                grid_col=grid_col,
                center_x_norm=x_grid[grid_col],
                center_y_norm=y_grid[grid_row],
                decoder_cluster=logical_id % max(control_clusters, 1),
            )
        )
    patch_lookup = {(patch.grid_row, patch.grid_col): patch for patch in patches}
    for patch in patches:
        neighbors = []
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = patch_lookup.get((patch.grid_row + delta_row, patch.grid_col + delta_col))
            if other is not None:
                neighbors.append(other.logical_id)
        patch.neighboring_patches = sorted(set(neighbors))
    return patches


def _build_magic_state_factories(
    spec: DesignSpec,
    code_family: str,
    code_distance: int,
    target_logicals: int,
    logical_cycle_ns: float,
) -> list[MagicStateFactory]:
    count = _factory_count(spec, target_logicals)
    base_physical_qubits = _factory_qubits(code_family, code_distance)
    rate = max(0.01, 1000.0 / max((6.0 + 0.4 * code_distance) * logical_cycle_ns, 1e-6))
    factories: list[MagicStateFactory] = []
    if count == 0:
        return factories
    x_positions = [0.20] if count <= 1 else [0.18 + 0.64 * idx / (count - 1) for idx in range(count)]
    for factory_id, x_pos in enumerate(x_positions):
        parallel_lanes = 1 + (1 if (target_logicals >= 2 and code_distance >= 5 and factory_id % 2 == 0) or (target_logicals >= 3 and factory_id % 2 == 0) else 0)
        physical_qubits = int(round(base_physical_qubits * (1.0 + 0.35 * max(parallel_lanes - 1, 0))))
        factories.append(
            MagicStateFactory(
                factory_id=factory_id,
                protocol_name=QEC_LIBRARY[code_family]["factory_protocol"],
                physical_qubits=physical_qubits,
                code_distance=max(3, code_distance),
                output_magic_states_per_us=rate * parallel_lanes,
                support_logical_qubits=max(1, math.ceil(target_logicals / max(count, 1))),
                center_x_norm=x_pos,
                center_y_norm=0.12,
                region="north_edge",
                parallel_lanes=parallel_lanes,
            )
        )
    return factories


def _build_surgery_channels(patches: list[LogicalPatch], code_distance: int, syndrome_cycle_ns: float, candidate: CandidateDesign) -> list[LatticeSurgeryChannel]:
    channels: list[LatticeSurgeryChannel] = []
    seen: set[tuple[int, int]] = set()
    for patch in patches:
        for neighbor_id in patch.neighboring_patches:
            edge = tuple(sorted((patch.logical_id, neighbor_id)))
            if edge in seen:
                continue
            seen.add(edge)
            neighbor = next((item for item in patches if item.logical_id == neighbor_id), None)
            if neighbor is None:
                continue
            x_dist = abs(patch.center_x_norm - neighbor.center_x_norm)
            y_dist = abs(patch.center_y_norm - neighbor.center_y_norm)
            length = (x_dist + y_dist) * max(candidate.rows, candidate.cols) * candidate.cell_pitch_um
            channels.append(
                LatticeSurgeryChannel(
                    source_patch_id=patch.logical_id,
                    target_patch_id=neighbor_id,
                    surgery_type="merge_split" if patch.grid_row == neighbor.grid_row or patch.grid_col == neighbor.grid_col else "bridge",
                    shared_boundary_qubits=max(2, code_distance - 1),
                    estimated_length_um=length,
                    cycle_ns=syndrome_cycle_ns * (1.0 + 0.02 * code_distance + length / max(candidate.cell_pitch_um * 200.0, 1.0)),
                )
            )
    return channels


def _attach_factories_to_patches(factories: list[MagicStateFactory], patches: list[LogicalPatch]) -> None:
    for factory in factories:
        ranked = sorted(
            patches,
            key=lambda patch: abs(factory.center_x_norm - patch.center_x_norm) + abs(factory.center_y_norm - patch.center_y_norm),
        )
        factory.client_patch_ids = [patch.logical_id for patch in ranked[: factory.support_logical_qubits]]
        for patch in ranked[: factory.support_logical_qubits]:
            patch.nearest_factory_ids.append(factory.factory_id)


def _build_logical_interconnects(
    patches: list[LogicalPatch],
    channels: list[LatticeSurgeryChannel],
    factories: list[MagicStateFactory],
    candidate: CandidateDesign,
    decoder_locality_score: float,
) -> list[LogicalInterconnect]:
    interconnects: list[LogicalInterconnect] = []
    for channel in channels:
        interconnects.append(
            LogicalInterconnect(
                source_kind="patch",
                source_id=channel.source_patch_id,
                target_kind="patch",
                target_id=channel.target_patch_id,
                route_class="lattice_surgery",
                estimated_length_um=channel.estimated_length_um,
                estimated_latency_ns=channel.cycle_ns,
                decoder_aware=decoder_locality_score > 0.45,
            )
        )
    patch_lookup = {patch.logical_id: patch for patch in patches}
    for factory in factories:
        for patch_id in factory.client_patch_ids:
            patch = patch_lookup.get(patch_id)
            if patch is None:
                continue
            distance = (abs(factory.center_x_norm - patch.center_x_norm) + abs(factory.center_y_norm - patch.center_y_norm)) * max(candidate.rows, candidate.cols) * candidate.cell_pitch_um
            interconnects.append(
                LogicalInterconnect(
                    source_kind="factory",
                    source_id=factory.factory_id,
                    target_kind="patch",
                    target_id=patch_id,
                    route_class="magic_state_feed",
                    estimated_length_um=distance,
                    estimated_latency_ns=distance / max(candidate.cell_pitch_um * 2.2, 1.0),
                    decoder_aware=patch.decoder_cluster == factory.factory_id % max(len(factories), 1),
                )
            )
    return interconnects


def _magic_demand_per_patch(spec: DesignSpec, patch: LogicalPatch, surgery_channels: list[LatticeSurgeryChannel]) -> int:
    surgery_degree = sum(1 for channel in surgery_channels if patch.logical_id in {channel.source_patch_id, channel.target_patch_id})
    base = 1 if spec.application in {"processor", "memory", "quantum_repeater"} else 0
    return max(0, base + math.ceil(surgery_degree / 2.0))


def _group_surgery_rounds(channels: list[LatticeSurgeryChannel]) -> list[list[LatticeSurgeryChannel]]:
    remaining = list(sorted(channels, key=lambda item: (item.estimated_length_um, item.source_patch_id, item.target_patch_id)))
    rounds: list[list[LatticeSurgeryChannel]] = []
    while remaining:
        used_patches: set[int] = set()
        current_round: list[LatticeSurgeryChannel] = []
        for channel in list(remaining):
            if channel.source_patch_id in used_patches or channel.target_patch_id in used_patches:
                continue
            current_round.append(channel)
            used_patches.add(channel.source_patch_id)
            used_patches.add(channel.target_patch_id)
            remaining.remove(channel)
        for channel in current_round:
            channel.round_index = len(rounds)
        rounds.append(current_round)
    return rounds


def _build_logical_schedule(
    spec: DesignSpec,
    candidate: CandidateDesign,
    logical_cycle_ns: float,
    syndrome_cycle_ns: float,
    decoder_latency_ns: float,
    patches: list[LogicalPatch],
    surgery_channels: list[LatticeSurgeryChannel],
    factories: list[MagicStateFactory],
) -> tuple[list[LogicalOperation], list[FactoryTimelineEntry], float, float, float, float, float, float]:
    if not patches:
        return [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    operations: list[LogicalOperation] = []
    timelines: list[FactoryTimelineEntry] = []
    patch_ready = {patch.logical_id: 0.0 for patch in patches}
    patch_busy = {patch.logical_id: 0.0 for patch in patches}
    patch_last_op: dict[int, str] = {patch.logical_id: "" for patch in patches}
    decoder_ready = {cluster: 0.0 for cluster in {patch.decoder_cluster for patch in patches}}
    decoder_busy = {cluster: 0.0 for cluster in decoder_ready}
    op_counter = 0
    factory_lane_ready = {factory.factory_id: [0.0 for _ in range(max(factory.parallel_lanes, 1))] for factory in factories}

    def add_op(
        op_type: str,
        patch_ids: list[int],
        duration_ns: float,
        decoder_cluster: int,
        factory_id: int | None = None,
        dependency_ids: list[str] | None = None,
        lane_id: str = "",
        earliest_start_ns: float = 0.0,
    ) -> str:
        nonlocal op_counter
        start = max([patch_ready[patch_id] for patch_id in patch_ids] + [decoder_ready.get(decoder_cluster, 0.0), earliest_start_ns])
        end = start + duration_ns
        op_id = f"op_{op_counter}_{op_type}"
        op_counter += 1
        operations.append(
            LogicalOperation(
                operation_id=op_id,
                op_type=op_type,
                logical_patch_ids=patch_ids,
                start_ns=start,
                end_ns=end,
                duration_ns=duration_ns,
                decoder_cluster=decoder_cluster,
                factory_id=factory_id,
                dependency_ids=[dep for dep in (dependency_ids or []) if dep],
                lane_id=lane_id,
            )
        )
        decoder_ready[decoder_cluster] = end
        decoder_busy[decoder_cluster] += duration_ns
        for patch_id in patch_ids:
            patch_ready[patch_id] = end
            patch_busy[patch_id] += duration_ns
            patch_last_op[patch_id] = op_id
        return op_id

    for patch in patches:
        prep = add_op(
            "patch_prepare",
            [patch.logical_id],
            duration_ns=0.70 * syndrome_cycle_ns + 0.08 * patch.ancilla_qubits,
            decoder_cluster=patch.decoder_cluster,
        )
        add_op(
            "stabilizer_calibration",
            [patch.logical_id],
            duration_ns=syndrome_cycle_ns,
            decoder_cluster=patch.decoder_cluster,
            dependency_ids=[prep],
        )

    surgery_rounds = _group_surgery_rounds(surgery_channels)
    for surgery_round in surgery_rounds:
        for channel in surgery_round:
            source_cluster = next(patch.decoder_cluster for patch in patches if patch.logical_id == channel.source_patch_id)
            target_cluster = next(patch.decoder_cluster for patch in patches if patch.logical_id == channel.target_patch_id)
            chosen_cluster = source_cluster if decoder_ready.get(source_cluster, 0.0) <= decoder_ready.get(target_cluster, 0.0) else target_cluster
            add_op(
                "lattice_surgery",
                [channel.source_patch_id, channel.target_patch_id],
                duration_ns=channel.cycle_ns,
                decoder_cluster=chosen_cluster,
                dependency_ids=[patch_last_op[channel.source_patch_id], patch_last_op[channel.target_patch_id]],
                lane_id=f"surgery_{channel.source_patch_id}_{channel.target_patch_id}",
            )

    patch_lookup = {patch.logical_id: patch for patch in patches}
    next_factory_batch = {factory.factory_id: 0 for factory in factories}
    for patch in patches:
        demand = _magic_demand_per_patch(spec, patch, surgery_channels)
        candidate_factories = patch.nearest_factory_ids or [factory.factory_id for factory in factories]
        for demand_idx in range(demand):
            if not candidate_factories:
                break
            factory_id = min(
                candidate_factories,
                key=lambda item: min(factory_lane_ready.get(item, [0.0]))
                + 180.0 * abs(next(factory for factory in factories if factory.factory_id == item).center_x_norm - patch.center_x_norm),
            )
            factory = next(item for item in factories if item.factory_id == factory_id)
            lane_index = min(range(len(factory_lane_ready[factory_id])), key=lambda idx: factory_lane_ready[factory_id][idx])
            batch_id = next_factory_batch[factory_id]
            next_factory_batch[factory_id] += 1
            batch_duration_ns = 1000.0 / max(factory.output_magic_states_per_us, 1e-6)
            start = factory_lane_ready[factory_id][lane_index]
            end = start + batch_duration_ns
            timelines.append(
                FactoryTimelineEntry(
                    factory_id=factory_id,
                    batch_id=batch_id,
                    protocol_name=factory.protocol_name,
                    start_ns=start,
                    end_ns=end,
                    produced_magic_states=1,
                    lane_index=lane_index,
                    reserved_patch_ids=[patch.logical_id],
                )
            )
            factory_lane_ready[factory_id][lane_index] = end
            inject_cluster = patch.decoder_cluster
            add_op(
                "magic_state_injection",
                [patch.logical_id],
                duration_ns=0.45 * logical_cycle_ns,
                decoder_cluster=inject_cluster,
                factory_id=factory_id,
                dependency_ids=[patch_last_op[patch.logical_id]],
                lane_id=f"factory_{factory_id}_lane_{lane_index}_to_patch_{patch.logical_id}_{demand_idx}",
                earliest_start_ns=end,
            )

    for patch in patches:
        add_op(
            "logical_measure",
            [patch.logical_id],
            duration_ns=0.40 * syndrome_cycle_ns + 0.35 * decoder_latency_ns,
            decoder_cluster=patch.decoder_cluster,
            dependency_ids=[patch_last_op[patch.logical_id]],
        )

    makespan = max((op.end_ns for op in operations), default=0.0)
    critical_path = max(patch_ready.values(), default=0.0)
    avg_idle = float(
        sum(max(makespan - patch_busy[patch.logical_id], 0.0) for patch in patches) / max(len(patches), 1)
    )
    total_factory_lanes = sum(max(factory.parallel_lanes, 1) for factory in factories)
    factory_utilization = float(
        sum(entry.end_ns - entry.start_ns for entry in timelines) / max(makespan * max(total_factory_lanes, 1), 1e-6)
    ) if factories else 0.0
    decoder_utilization = float(
        sum(decoder_busy.values()) / max(makespan * max(len(decoder_busy), 1), 1e-6)
    ) if decoder_busy else 0.0
    logical_ops_per_us = len(operations) / max(makespan / 1000.0, 1e-6) if makespan > 0 else 0.0
    return operations, timelines, makespan, critical_path, avg_idle, factory_utilization, min(decoder_utilization, 1.5), logical_ops_per_us


def _decoder_locality_score(patches: list[LogicalPatch], channels: list[LatticeSurgeryChannel], control_clusters: int) -> float:
    if not patches:
        return 0.0
    cluster_penalty = sum(abs(patch.decoder_cluster - patch.grid_col % max(control_clusters, 1)) for patch in patches) / max(len(patches), 1)
    channel_penalty = sum(channel.estimated_length_um for channel in channels) / max(len(channels), 1) if channels else 0.0
    return max(0.0, min(1.0, 1.12 - 0.09 * cluster_penalty - channel_penalty / 1500.0))


def build_qec_plan(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, dense_signals: FastDenseSignals | None = None) -> QECPlan:
    if not spec.qec_enabled:
        return QECPlan(
            enabled=False,
            code_family=spec.qec_code,
            code_distance=1,
            target_logical_qubits=spec.target_logical_qubits,
            achievable_logical_qubits=max(1, spec.target_logical_qubits),
            physical_qubits_per_logical=1,
            ancilla_qubits_total=0,
            data_qubits_total=min(candidate.qubits, spec.target_logical_qubits),
            total_physical_qubits_required=spec.target_logical_qubits,
            factory_qubits_total=0,
            physical_error_rate=1.0 - metrics.gate_fidelity,
            logical_error_rate=1.0 - metrics.gate_fidelity,
            threshold=1.0,
            syndrome_cycle_ns=0.0,
            decoder_latency_ns=0.0,
            logical_cycle_ns=0.0,
            logical_success_probability=metrics.gate_fidelity,
            logical_yield=metrics.yield_estimate,
            qec_overhead_ratio=1.0,
            logical_area_efficiency=spec.target_logical_qubits / max(candidate.qubits, 1),
            decoder_graph_nodes=0,
            decoder_graph_edges=0,
            decoder_locality_score=1.0,
            surgery_throughput_ops_per_us=0.0,
            magic_state_rate_per_us=0.0,
            schedule_makespan_ns=0.0,
            schedule_critical_path_ns=0.0,
            average_patch_idle_ns=0.0,
            factory_utilization=0.0,
            decoder_utilization=0.0,
            patches=[
                LogicalPatch(
                    logical_id=i,
                    qubit_indices=[i],
                    ancilla_qubits=0,
                    patch_rows=1,
                    patch_cols=1,
                    patch_width_um=candidate.cell_pitch_um,
                    patch_height_um=candidate.cell_pitch_um,
                    syndrome_bus_count=0,
                    grid_row=0,
                    grid_col=i,
                    center_x_norm=0.20 + 0.10 * i,
                    center_y_norm=0.50,
                    decoder_cluster=0,
                )
                for i in range(spec.target_logical_qubits)
            ],
            surgery_channels=[],
            magic_state_factories=[],
            logical_interconnects=[],
            logical_schedule=[],
            factory_timelines=[],
            violations=[],
        )

    code_family = _select_code_family(spec, candidate, metrics, dense_signals)
    library = QEC_LIBRARY[code_family]
    p_phys = _effective_physical_error_rate(metrics, dense_signals)
    required_distance = _required_distance(spec.target_logical_error_rate, p_phys, library["threshold"], library["alpha"])
    best_distance = 3
    best_score = -1.0e9
    best_factory_count = 0
    best_factory_qubits = 0
    for distance_candidate in range(3, max(required_distance, 9) + 2, 2):
        physical_per_logical_candidate = max(9, int(library["qubits_per_distance"](distance_candidate)))
        factory_count_candidate, factory_qubits_candidate = _estimate_factory_resources(spec, code_family, distance_candidate, spec.target_logical_qubits)
        residual_qubits = max(candidate.qubits - factory_qubits_candidate, 0)
        achievable_candidate = max(0, residual_qubits // max(physical_per_logical_candidate, 1))
        logical_error_candidate = _logical_error_rate(library["alpha"], p_phys, library["threshold"], distance_candidate)
        success_ratio = min(achievable_candidate / max(spec.target_logical_qubits, 1), 1.2)
        error_ratio = min(spec.target_logical_error_rate / max(logical_error_candidate, 1e-12), 1.2)
        resource_penalty = 0.018 * distance_candidate + 0.0010 * factory_qubits_candidate
        if achievable_candidate >= spec.target_logical_qubits:
            resource_penalty *= 0.65
        score = 0.45 * success_ratio + 0.45 * error_ratio + 0.10 * min(candidate.qubits / max(physical_per_logical_candidate, 1), 1.2) - resource_penalty
        if achievable_candidate == 0:
            score -= 1.2
        if score > best_score:
            best_score = score
            best_distance = distance_candidate
            best_factory_count = factory_count_candidate
            best_factory_qubits = factory_qubits_candidate

    distance = best_distance
    physical_per_logical = max(9, int(library["qubits_per_distance"](distance)))
    factory_qubits_total = best_factory_qubits
    residual_qubits = max(candidate.qubits - factory_qubits_total, 0)
    achievable_logical = max(0, residual_qubits // max(physical_per_logical, 1))
    data_total = achievable_logical * max(1, physical_per_logical * 2 // 3)
    ancilla_total = achievable_logical * max(1, physical_per_logical - physical_per_logical * 2 // 3)
    total_required = spec.target_logical_qubits * physical_per_logical + factory_qubits_total
    logical_error = _logical_error_rate(library["alpha"], p_phys, library["threshold"], distance)
    syndrome_cycle_ns = spec.syndrome_cycle_ns * library["syndrome_factor"] * (1.0 + 0.06 * max(distance - 3, 0))
    decoder_latency_ns = spec.decoder_margin_ns * library["decoder_factor"] * (1.0 + 0.06 * distance + 0.015 * candidate.qubits)
    logical_cycle_ns = syndrome_cycle_ns + decoder_latency_ns
    logical_success_probability = max(0.0, min(0.999999, math.exp(-logical_error * max(spec.target_logical_qubits, 1))))
    logical_yield = max(0.0, min(0.999999, metrics.yield_estimate * logical_success_probability * min(achievable_logical / max(spec.target_logical_qubits, 1), 1.0)))
    qec_overhead_ratio = total_required / max(spec.target_logical_qubits, 1)
    logical_area_efficiency = achievable_logical / max(metrics.die_area_mm2, 1e-6)
    patches = _patch_partition(
        candidate,
        achievable_logical,
        physical_per_logical,
        distance,
        library["patch_aspect"],
        control_clusters=max(candidate.microwave_line_count, 1),
        logical_connectivity=spec.logical_connectivity,
    )
    factories = _build_magic_state_factories(spec, code_family, distance, achievable_logical, logical_cycle_ns)
    _attach_factories_to_patches(factories, patches)
    surgery_channels = _build_surgery_channels(patches, distance, syndrome_cycle_ns, candidate)
    decoder_locality = _decoder_locality_score(patches, surgery_channels, max(candidate.microwave_line_count, 1))
    logical_interconnects = _build_logical_interconnects(patches, surgery_channels, factories, candidate, decoder_locality)
    logical_schedule, factory_timelines, makespan_ns, critical_path_ns, average_patch_idle_ns, factory_utilization, decoder_utilization, logical_ops_per_us = _build_logical_schedule(
        spec,
        candidate,
        logical_cycle_ns,
        syndrome_cycle_ns,
        decoder_latency_ns,
        patches,
        surgery_channels,
        factories,
    )
    decoder_nodes = achievable_logical * max(4, distance * distance // 2) + len(factories) * max(6, distance)
    decoder_edges = int(round(decoder_nodes * (2.2 + 0.12 * candidate.microwave_line_count) + len(surgery_channels) * 3))
    surgery_throughput = max(len(surgery_channels) / max(logical_cycle_ns / 1000.0, 1e-6), logical_ops_per_us * 0.15)
    magic_state_rate = sum(factory.output_magic_states_per_us for factory in factories)

    violations: list[str] = []
    if achievable_logical < spec.target_logical_qubits:
        violations.append("logical_qubits")
    if logical_error > spec.target_logical_error_rate:
        violations.append("logical_error_rate")
    if logical_cycle_ns > spec.max_latency_ns * 2.0:
        violations.append("qec_latency")
    if magic_state_rate < max(0.5, 0.3 * spec.target_logical_qubits):
        violations.append("magic_state_rate")

    return QECPlan(
        enabled=True,
        code_family=code_family,
        code_distance=distance,
        target_logical_qubits=spec.target_logical_qubits,
        achievable_logical_qubits=achievable_logical,
        physical_qubits_per_logical=physical_per_logical,
        ancilla_qubits_total=ancilla_total,
        data_qubits_total=data_total,
        total_physical_qubits_required=total_required,
        factory_qubits_total=factory_qubits_total,
        physical_error_rate=p_phys,
        logical_error_rate=logical_error,
        threshold=library["threshold"],
        syndrome_cycle_ns=syndrome_cycle_ns,
        decoder_latency_ns=decoder_latency_ns,
        logical_cycle_ns=logical_cycle_ns,
        logical_success_probability=logical_success_probability,
        logical_yield=logical_yield,
        qec_overhead_ratio=qec_overhead_ratio,
        logical_area_efficiency=logical_area_efficiency,
        decoder_graph_nodes=decoder_nodes,
        decoder_graph_edges=decoder_edges,
        decoder_locality_score=decoder_locality,
        surgery_throughput_ops_per_us=surgery_throughput,
        magic_state_rate_per_us=magic_state_rate,
        schedule_makespan_ns=makespan_ns,
        schedule_critical_path_ns=critical_path_ns,
        average_patch_idle_ns=average_patch_idle_ns,
        factory_utilization=factory_utilization,
        decoder_utilization=decoder_utilization,
        patches=patches,
        surgery_channels=surgery_channels,
        magic_state_factories=factories,
        logical_interconnects=logical_interconnects,
        logical_schedule=logical_schedule,
        factory_timelines=factory_timelines,
        violations=violations,
    )


def write_qec_artifacts(output_dir: Path, qec_plan: QECPlan) -> dict[str, str]:
    path = output_dir / "qec_plan.json"
    path.write_text(json.dumps(qec_plan.to_dict(), indent=2), encoding="utf-8")
    return {"qec_plan": str(path)}
