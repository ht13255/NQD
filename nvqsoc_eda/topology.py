from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dense_placement import DenseCrosstalkSummary, PlacementPlan, base_layout_frame
from .dense_placement import FastDenseSignals
from .neural_surrogate import FrozenNeuralSignals
from .papers import PaperKnowledge
from .requirements import RequirementBundle
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec


@dataclass(slots=True)
class TopologyPlan:
    optical_topology: str
    control_topology: str
    readout_topology: str
    power_mesh_pitch_um: float
    shield_pitch_um: float
    route_width_scale: float
    optical_route_width_scale: float
    resonator_trace_scale: float
    tile_guard_multiplier: float
    keepout_margin_um: float
    optical_redundancy: int
    microwave_redundancy: int
    detector_clusters: int
    macro_segment_count: int
    control_cluster_count: int
    add_ground_moat: bool
    bus_escape_offset_um: float
    left_bank_x_um: float
    right_bank_x_um: float
    photonics_header_y_um: float
    readout_macro_y_um: float
    detector_cluster_x_um: list[float]
    control_spine_x_um: list[float]
    hotspot_quadrant: str
    layout_complexity_score: float = 1.0
    hotspot_keepouts: list[dict[str, float]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_topology_plan(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    dense_signals: FastDenseSignals | None = None,
    placement: PlacementPlan | None = None,
    dense_crosstalk: DenseCrosstalkSummary | None = None,
    neural_signals: FrozenNeuralSignals | None = None,
    paper_knowledge: PaperKnowledge | None = None,
    requirements_bundle: RequirementBundle | None = None,
) -> TopologyPlan:
    dense_signals = dense_signals or FastDenseSignals(0.5, candidate.cell_pitch_um * 0.85, 0.0, 0.3, 0.3, 0.03, -30.0, 0.03, 0.03, 0)
    neural_signals = neural_signals or FrozenNeuralSignals(0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.8, 0.5, 0.0, 0, 0, 0, 0, 1.0, 0.0, [])
    routing_pref = spec.routing_preference.lower()
    optical_topology = "striped"
    control_topology = "split_domain"
    readout_topology = "balanced_tree"
    reasoning: list[str] = []

    if paper_knowledge and paper_knowledge.topic_counts.get("network", 0) > paper_knowledge.topic_counts.get("sensing", 0):
        optical_topology = "redundant_spine"
        reasoning.append("paper priors favor network-style photonic redundancy")
    if neural_signals.crosstalk_risk > 0.40 or dense_signals.effective_crosstalk_linear > 0.026:
        optical_topology = "serpentine_redundant"
        control_topology = "shielded_clusters"
        readout_topology = "segmented_tree"
        reasoning.append("high crosstalk risk triggers separated redundant routing")
    elif neural_signals.latency_risk > 0.32:
        control_topology = "direct_fanout"
        readout_topology = "short_tree"
        reasoning.append("latency risk pushes direct fanout control topology")
    elif routing_pref == "compact":
        optical_topology = "compact_spine"
        readout_topology = "compressed_tree"
        reasoning.append("compact routing preference compresses the topology")
    elif routing_pref == "low_loss":
        optical_topology = "wide_redundant"
        control_topology = "shielded_clusters"
        reasoning.append("low-loss preference widens and duplicates trunks")
    if spec.target_logical_qubits > 1:
        control_topology = "shielded_clusters"
        readout_topology = "segmented_tree"
        reasoning.append("logical-qubit target increases segmentation and control isolation")

    area_pressure = max(spec.max_die_area_mm2 / max(metrics.die_area_mm2, 1e-6), 0.3)
    route_width_scale = 1.0 + 0.08 * candidate.shielding_layers + 0.04 * neural_signals.architecture_affinity - 0.03 * neural_signals.latency_risk + 0.04 * min(area_pressure, 1.2)
    optical_route_scale = 1.0 + 0.10 * neural_signals.architecture_affinity + 0.08 * neural_signals.prototype_affinity - 0.06 * neural_signals.crosstalk_risk
    resonator_trace_scale = 1.0 + 0.06 * neural_signals.design_quality + 0.04 * (paper_knowledge.recommended_cavity_q / max(candidate.cavity_q, 1.0) - 1.0 if paper_knowledge else 0.0)
    tile_guard_multiplier = 1.0 + 0.22 * neural_signals.crosstalk_risk + 0.10 * dense_signals.optical_load_std
    power_mesh_pitch_um = max(72.0, candidate.cell_pitch_um * (2.1 - 0.35 * neural_signals.crosstalk_risk + 0.10 * neural_signals.latency_risk))
    shield_pitch_um = max(8.0, candidate.cell_pitch_um * (0.55 - 0.18 * neural_signals.crosstalk_risk + 0.06 * dense_signals.mean_offset_um / max(candidate.cell_pitch_um, 1.0)))
    keepout_margin_um = max(22.0, 0.55 * candidate.cell_pitch_um + 9.0 * neural_signals.crosstalk_risk + 5.0 * dense_signals.microwave_load_std)
    optical_redundancy = max(0, min(3, int(round(1.8 * neural_signals.crosstalk_risk + 0.9 * neural_signals.architecture_affinity))))
    microwave_redundancy = max(0, min(2, int(round(1.5 * neural_signals.routing_risk + 0.9 * neural_signals.latency_risk))))
    detector_clusters = max(2, min(8, 2 + int(round(candidate.optical_bus_count / 6.0 + neural_signals.architecture_affinity + 0.6 * spec.target_logical_qubits))))
    macro_segment_count = max(3, min(8, 3 + int(round(candidate.qubits / 40.0 + 2.0 * neural_signals.design_quality))))
    control_cluster_count = max(2, min(6, 2 + int(round(candidate.microwave_line_count / 4.0 + neural_signals.routing_risk + neural_signals.crosstalk_risk))))
    add_ground_moat = bool(neural_signals.crosstalk_risk > 0.32 or dense_signals.effective_crosstalk_linear > 0.024)
    bus_escape_offset_um = max(26.0, candidate.cell_pitch_um * (0.75 + 0.18 * neural_signals.crosstalk_risk + 0.10 * neural_signals.latency_risk))
    die_width_um, die_height_um, core_x, core_y, array_width_um, array_height_um, _ = base_layout_frame(candidate)
    left_bank_x_um = 250.0
    right_bank_x_um = die_width_um - 680.0
    photonics_header_y_um = 260.0
    readout_macro_y_um = die_height_um - 520.0
    detector_cluster_x_um = [
        core_x - 290.0 + (index + 0.5) * (array_width_um + 580.0) / max(detector_clusters, 1)
        for index in range(detector_clusters)
    ]
    control_spine_x_um = [
        core_x - 180.0 + index * (array_width_um + 360.0) / max(control_cluster_count, 1)
        for index in range(control_cluster_count)
    ]
    hotspot_quadrant = "center"
    hotspot_keepouts: list[dict[str, float]] = []
    iterations: list[dict[str, Any]] = []

    hotspot_tile = None
    if placement is not None and dense_crosstalk is not None:
        hotspot_tile = next((tile for tile in placement.tiles if tile.qubit_id == dense_crosstalk.hotspot_qubit), None)
    if hotspot_tile is not None:
        hot_x = hotspot_tile.x_um
        hot_y = hotspot_tile.y_um
        if hot_x < core_x + 0.33 * array_width_um:
            hotspot_quadrant = "left"
        elif hot_x > core_x + 0.67 * array_width_um:
            hotspot_quadrant = "right"
        elif hot_y < core_y + 0.35 * array_height_um:
            hotspot_quadrant = "top"
        elif hot_y > core_y + 0.65 * array_height_um:
            hotspot_quadrant = "bottom"
        pressure = max(dense_signals.effective_crosstalk_linear * 22.0, neural_signals.crosstalk_risk)
        keepout_w = candidate.cell_pitch_um * (2.2 + 2.8 * pressure)
        keepout_h = candidate.cell_pitch_um * (2.2 + 2.8 * pressure)
        hotspot_keepouts.append(
            {
                "x": hot_x - keepout_w / 2.0,
                "y": hot_y - keepout_h / 2.0,
                "width": keepout_w,
                "height": keepout_h,
            }
        )
        for iteration in range(3):
            step = 1.0 + 0.35 * iteration
            if hotspot_quadrant == "left":
                left_bank_x_um = max(120.0, left_bank_x_um - 26.0 * step)
                control_spine_x_um = [min(x, core_x - keepout_margin_um - 60.0) if x < hot_x else x + 18.0 * step for x in control_spine_x_um]
                detector_cluster_x_um = [x + 12.0 * step for x in detector_cluster_x_um]
            elif hotspot_quadrant == "right":
                right_bank_x_um = min(die_width_um - 560.0, right_bank_x_um + 26.0 * step)
                control_spine_x_um = [max(x, core_x + array_width_um + keepout_margin_um + 60.0) if x > hot_x else x - 18.0 * step for x in control_spine_x_um]
                detector_cluster_x_um = [x - 12.0 * step for x in detector_cluster_x_um]
            elif hotspot_quadrant == "top":
                photonics_header_y_um = max(160.0, photonics_header_y_um - 18.0 * step)
                readout_macro_y_um += 8.0 * step
            elif hotspot_quadrant == "bottom":
                readout_macro_y_um = min(die_height_um - 480.0, readout_macro_y_um + 18.0 * step)
                photonics_header_y_um += 8.0 * step
            else:
                left_bank_x_um = max(140.0, left_bank_x_um - 12.0 * step)
                right_bank_x_um = min(die_width_um - 560.0, right_bank_x_um + 12.0 * step)
                detector_cluster_x_um = [x + ((-1.0) ** idx) * 8.0 * step for idx, x in enumerate(detector_cluster_x_um)]
            iterations.append(
                {
                    "iteration": iteration,
                    "hotspot_quadrant": hotspot_quadrant,
                    "left_bank_x_um": left_bank_x_um,
                    "right_bank_x_um": right_bank_x_um,
                    "photonics_header_y_um": photonics_header_y_um,
                    "readout_macro_y_um": readout_macro_y_um,
                }
            )
        reasoning.append("iterative hotspot-aware topology refinement applied")

    layout_complexity_score = max(
        0.0,
        min(
            1.5,
            0.35 * neural_signals.design_quality
            + 0.18 * neural_signals.prototype_affinity
            + 0.15 * (1.0 + optical_redundancy)
            + 0.12 * (1.0 + microwave_redundancy)
            + 0.10 * tile_guard_multiplier
            + 0.10 * control_cluster_count / 4.0,
        ),
    )

    if requirements_bundle and requirements_bundle.layout_focus:
        reasoning.append("requirements bundle influences layout focus and complexity")
    return TopologyPlan(
        optical_topology=optical_topology,
        control_topology=control_topology,
        readout_topology=readout_topology,
        power_mesh_pitch_um=power_mesh_pitch_um,
        shield_pitch_um=shield_pitch_um,
        route_width_scale=route_width_scale,
        optical_route_width_scale=optical_route_scale,
        resonator_trace_scale=resonator_trace_scale,
        tile_guard_multiplier=tile_guard_multiplier,
        keepout_margin_um=keepout_margin_um,
        optical_redundancy=optical_redundancy,
        microwave_redundancy=microwave_redundancy,
        detector_clusters=detector_clusters,
        macro_segment_count=macro_segment_count,
        control_cluster_count=control_cluster_count,
        add_ground_moat=add_ground_moat,
        bus_escape_offset_um=bus_escape_offset_um,
        left_bank_x_um=left_bank_x_um,
        right_bank_x_um=right_bank_x_um,
        photonics_header_y_um=photonics_header_y_um,
        readout_macro_y_um=readout_macro_y_um,
        detector_cluster_x_um=detector_cluster_x_um,
        control_spine_x_um=control_spine_x_um,
        hotspot_quadrant=hotspot_quadrant,
        hotspot_keepouts=hotspot_keepouts,
        layout_complexity_score=layout_complexity_score,
        iterations=iterations,
        reasoning=reasoning,
    )


def write_topology_artifacts(output_dir: Path, plan: TopologyPlan) -> dict[str, str]:
    path = output_dir / "topology_plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"topology_plan": str(path)}
