from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gdstk
import matplotlib
import numpy as np

from .dense_placement import PlacementPlan, base_layout_frame, build_probabilistic_placement
from .qec import QECPlan
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec
from .topology import TopologyPlan


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Rectangle


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


LAYER_MAP = {
    "DIE": (1, 0),
    "SCRIBE": (2, 0),
    "SEAL": (3, 0),
    "DIAMOND": (4, 0),
    "PAD": (5, 0),
    "PASSIVATION": (6, 0),
    "GROUND": (7, 0),
    "M1": (10, 0),
    "VIA1": (11, 0),
    "M2": (12, 0),
    "VIA2": (13, 0),
    "M3": (14, 0),
    "VIA3": (15, 0),
    "M4": (16, 0),
    "VIA4": (17, 0),
    "M5": (18, 0),
    "VIA5": (19, 0),
    "M6": (20, 0),
    "RESONATOR": (30, 0),
    "OPTICAL": (40, 0),
    "READOUT": (50, 0),
    "DETECTOR": (51, 0),
    "FILL": (60, 0),
    "MARKER": (61, 0),
    "NV": (70, 0),
    "LOGICAL": (71, 0),
    "FACTORY": (72, 0),
    "SURGERY": (73, 0),
    "TEXT": (80, 0),
}

LAYER_COLORS = {
    "DIE": "#f5efe1",
    "SCRIBE": "#cbd5e1",
    "SEAL": "#475569",
    "DIAMOND": "#d7f0f6",
    "PAD": "#ad8b3a",
    "PASSIVATION": "#e2e8f0",
    "GROUND": "#2a6f97",
    "M1": "#4c6ef5",
    "VIA1": "#748ffc",
    "M2": "#0ca678",
    "VIA2": "#63e6be",
    "M3": "#1971c2",
    "VIA3": "#74c0fc",
    "M4": "#7048e8",
    "VIA4": "#b197fc",
    "M5": "#e67700",
    "VIA5": "#ffc078",
    "M6": "#c92a2a",
    "RESONATOR": "#118ab2",
    "OPTICAL": "#ffb703",
    "READOUT": "#ef476f",
    "DETECTOR": "#f28482",
    "FILL": "#ced4da",
    "MARKER": "#1f2937",
    "NV": "#073b4c",
    "LOGICAL": "#2b2d42",
    "FACTORY": "#8d99ae",
    "SURGERY": "#e76f51",
    "TEXT": "#111827",
}


@dataclass(slots=True)
class Rect:
    name: str
    x: float
    y: float
    width: float
    height: float
    layer: str
    color: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Route:
    name: str
    points: list[tuple[float, float]]
    width: float
    layer: str
    color: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": [{"x": x, "y": y} for x, y in self.points],
            "width": self.width,
            "layer": self.layer,
            "color": self.color,
        }


@dataclass(slots=True)
class Circle:
    name: str
    cx: float
    cy: float
    radius: float
    layer: str
    color: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Label:
    text: str
    x: float
    y: float
    size: float
    layer: str
    color: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LayoutBundle:
    die_width_um: float
    die_height_um: float
    rects: list[Rect] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    drc: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "die_width_um": self.die_width_um,
            "die_height_um": self.die_height_um,
            "rects": [item.to_dict() for item in self.rects],
            "routes": [item.to_dict() for item in self.routes],
            "circles": [item.to_dict() for item in self.circles],
            "labels": [item.to_dict() for item in self.labels],
            "stats": _json_safe(self.stats),
            "drc": _json_safe(self.drc),
        }


def _route(points: list[tuple[float, float]], width: float, name: str, layer: str) -> Route:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or (abs(cleaned[-1][0] - point[0]) > 1e-9 or abs(cleaned[-1][1] - point[1]) > 1e-9):
            cleaned.append(point)
    if len(cleaned) == 1:
        cleaned.append((cleaned[0][0] + max(width, 1.0), cleaned[0][1]))
    return Route(name=name, points=cleaned, width=width, layer=layer, color=LAYER_COLORS[layer])


def _rect(name: str, x: float, y: float, width: float, height: float, layer: str) -> Rect:
    return Rect(name=name, x=x, y=y, width=width, height=height, layer=layer, color=LAYER_COLORS[layer])


def _circle(name: str, cx: float, cy: float, radius: float, layer: str) -> Circle:
    return Circle(name=name, cx=cx, cy=cy, radius=radius, layer=layer, color=LAYER_COLORS[layer])


def _label(text: str, x: float, y: float, size: float, layer: str = "TEXT") -> Label:
    return Label(text=text, x=x, y=y, size=size, layer=layer, color=LAYER_COLORS[layer])


def _add_cross_marker(bundle: LayoutBundle, name: str, cx: float, cy: float, span: float, arm: float) -> None:
    bundle.rects.append(_rect(f"{name}_h", cx - span / 2.0, cy - arm / 2.0, span, arm, "MARKER"))
    bundle.rects.append(_rect(f"{name}_v", cx - arm / 2.0, cy - span / 2.0, arm, span, "MARKER"))


def _add_via_array(bundle: LayoutBundle, prefix: str, x: float, y: float, cols: int, rows: int, pitch: float, size: float, layer: str) -> int:
    count = 0
    for row in range(rows):
        for col in range(cols):
            bundle.rects.append(_rect(f"{prefix}_{row}_{col}", x + col * pitch, y + row * pitch, size, size, layer))
            count += 1
    return count


def _add_grating_coupler(bundle: LayoutBundle, name: str, x: float, y: float, width: float, height: float, teeth: int) -> None:
    tooth_pitch = height / max(teeth, 1)
    tooth_height = tooth_pitch * 0.55
    for index in range(teeth):
        bundle.rects.append(_rect(f"{name}_tooth_{index}", x, y + index * tooth_pitch, width, tooth_height, "OPTICAL"))
    bundle.rects.append(_rect(f"{name}_backplane", x - 10.0, y - 10.0, width + 20.0, 10.0, "M3"))


def _add_guard_ring(bundle: LayoutBundle, name: str, x: float, y: float, width: float, height: float, ring: float, layer: str) -> None:
    bundle.rects.extend(
        [
            _rect(f"{name}_top", x + ring, y + height - ring, max(width - 2.0 * ring, ring), ring, layer),
            _rect(f"{name}_bottom", x + ring, y, max(width - 2.0 * ring, ring), ring, layer),
            _rect(f"{name}_left", x, y, ring, height, layer),
            _rect(f"{name}_right", x + width - ring, y, ring, height, layer),
        ]
    )


def _add_shield_fence(bundle: LayoutBundle, name: str, x0: float, y0: float, x1: float, y1: float, pitch: float = 24.0) -> int:
    count = 0
    if abs(y1 - y0) < 1e-6:
        x = min(x0, x1)
        while x <= max(x0, x1):
            bundle.rects.append(_rect(f"{name}_{count}", x - 1.6, y0 - 1.6, 3.2, 3.2, "GROUND"))
            count += 1
            x += pitch
    elif abs(x1 - x0) < 1e-6:
        y = min(y0, y1)
        while y <= max(y0, y1):
            bundle.rects.append(_rect(f"{name}_{count}", x0 - 1.6, y - 1.6, 3.2, 3.2, "GROUND"))
            count += 1
            y += pitch
    return count


def _add_square_resonator(bundle: LayoutBundle, name: str, cx: float, cy: float, size: float, trace: float) -> None:
    x0 = cx - size / 2.0
    y0 = cy - size / 2.0
    bundle.rects.extend(
        [
            _rect(f"{name}_top", x0 + trace, y0 + size - trace, max(size - 2.0 * trace, trace), trace, "RESONATOR"),
            _rect(f"{name}_bottom", x0 + trace, y0, max(size - 2.0 * trace, trace), trace, "RESONATOR"),
            _rect(f"{name}_left", x0, y0, trace, size, "RESONATOR"),
            _rect(f"{name}_right", x0 + size - trace, y0, trace, size, "RESONATOR"),
        ]
    )


def _add_fill_region(bundle: LayoutBundle, name: str, x: float, y: float, width: float, height: float, pitch: float = 120.0) -> int:
    if width <= 0.0 or height <= 0.0:
        return 0
    fill_size = pitch * 0.42
    count = 0
    rows = max(1, int(height // pitch))
    cols = max(1, int(width // pitch))
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                rx = x + col * pitch + 0.5 * (pitch - fill_size)
                ry = y + row * pitch + 0.5 * (pitch - fill_size)
                bundle.rects.append(_rect(f"{name}_{row}_{col}", rx, ry, fill_size, fill_size, "FILL"))
                count += 1
    return count


def _route_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(points, points[1:]))


def _default_topology_plan(candidate: CandidateDesign) -> TopologyPlan:
    die_width_um, die_height_um, core_x, _core_y, array_width_um, _array_height_um, _ = base_layout_frame(candidate)
    detector_clusters = max(2, min(4, candidate.optical_bus_count // 3 + 1))
    control_cluster_count = max(2, min(4, candidate.microwave_line_count // 2 + 1))
    return TopologyPlan(
        optical_topology="striped",
        control_topology="split_domain",
        readout_topology="balanced_tree",
        power_mesh_pitch_um=max(candidate.cell_pitch_um * 2.4, 100.0),
        shield_pitch_um=max(candidate.cell_pitch_um * 0.45, 10.0),
        route_width_scale=1.0,
        optical_route_width_scale=1.0,
        resonator_trace_scale=1.0,
        tile_guard_multiplier=1.0,
        keepout_margin_um=max(candidate.cell_pitch_um * 0.5, 20.0),
        optical_redundancy=0,
        microwave_redundancy=0,
        detector_clusters=detector_clusters,
        macro_segment_count=4,
        control_cluster_count=control_cluster_count,
        add_ground_moat=False,
        bus_escape_offset_um=max(candidate.cell_pitch_um * 0.7, 24.0),
        left_bank_x_um=250.0,
        right_bank_x_um=die_width_um - 680.0,
        photonics_header_y_um=260.0,
        readout_macro_y_um=die_height_um - 520.0,
        detector_cluster_x_um=[core_x - 290.0 + (index + 0.5) * (array_width_um + 580.0) / max(detector_clusters, 1) for index in range(detector_clusters)],
        control_spine_x_um=[core_x - 180.0 + index * (array_width_um + 360.0) / max(control_cluster_count, 1) for index in range(control_cluster_count)],
        hotspot_quadrant="center",
        hotspot_keepouts=[],
        layout_complexity_score=1.0,
        iterations=[],
        reasoning=["default topology plan"],
    )


def generate_layout(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    topology_plan: TopologyPlan | None = None,
    placement: PlacementPlan | None = None,
    qec_plan: QECPlan | None = None,
) -> LayoutBundle:
    die_width_um, die_height_um, core_x, core_y, array_width_um, array_height_um, pad_margin = base_layout_frame(candidate)
    topology_plan = topology_plan or _default_topology_plan(candidate)
    placement = placement or build_probabilistic_placement(spec, candidate, samples=28)
    ring_margin = 180.0
    tile_size = candidate.cell_pitch_um * 0.72
    qubit_radius = max(3.8, min(candidate.cell_pitch_um * 0.12, 7.5))
    route_width_scale = max(0.75, topology_plan.route_width_scale)
    optical_route_scale = max(0.75, topology_plan.optical_route_width_scale)
    resonator_trace_scale = max(0.70, topology_plan.resonator_trace_scale)
    keepout = topology_plan.keepout_margin_um
    via_count = 0
    fill_count = 0
    fence_count = 0
    bundle = LayoutBundle(die_width_um=die_width_um, die_height_um=die_height_um)

    bundle.rects.extend(
        [
            _rect("die", 0.0, 0.0, die_width_um, die_height_um, "DIE"),
            _rect("scribe_top", 0.0, die_height_um - 80.0, die_width_um, 80.0, "SCRIBE"),
            _rect("scribe_bottom", 0.0, 0.0, die_width_um, 80.0, "SCRIBE"),
            _rect("scribe_left", 0.0, 0.0, 80.0, die_height_um, "SCRIBE"),
            _rect("scribe_right", die_width_um - 80.0, 0.0, 80.0, die_height_um, "SCRIBE"),
            _rect("diamond_core", core_x - 120.0, core_y - 120.0, array_width_um + 240.0, array_height_um + 240.0, "DIAMOND"),
            _rect("photonics_header", core_x - 260.0, topology_plan.photonics_header_y_um, array_width_um + 520.0, 220.0, "OPTICAL"),
            _rect("readout_macro", core_x - 340.0, topology_plan.readout_macro_y_um, array_width_um + 680.0, 240.0, "READOUT"),
            _rect("left_ctrl_bank", topology_plan.left_bank_x_um, core_y - 260.0, 430.0, array_height_um + 520.0, "M4"),
            _rect("right_ctrl_bank", topology_plan.right_bank_x_um, core_y - 260.0, 430.0, array_height_um + 520.0, "M4"),
            _rect("top_left_cryo_macro", 240.0, 240.0, 460.0, 300.0, "M5"),
            _rect("top_right_cryo_macro", die_width_um - 700.0, 240.0, 460.0, 300.0, "M5"),
        ]
    )

    _add_guard_ring(bundle, "seal_ring", ring_margin, ring_margin, die_width_um - 2.0 * ring_margin, die_height_um - 2.0 * ring_margin, 36.0, "SEAL")
    _add_guard_ring(bundle, "ground_guard", core_x - 260.0, core_y - 260.0, array_width_um + 520.0, array_height_um + 520.0, 24.0, "GROUND")
    if topology_plan.add_ground_moat:
        _add_guard_ring(bundle, "ground_moat", core_x - 320.0, core_y - 320.0, array_width_um + 640.0, array_height_um + 640.0, 38.0, "GROUND")
    for index, hotspot_keepout in enumerate(topology_plan.hotspot_keepouts):
        bundle.rects.append(
            _rect(
                f"hotspot_keepout_{index}",
                hotspot_keepout["x"],
                hotspot_keepout["y"],
                hotspot_keepout["width"],
                hotspot_keepout["height"],
                "PASSIVATION",
            )
        )

    macro_span = array_height_um + 520.0
    segment_height = macro_span / max(topology_plan.macro_segment_count, 1)
    for segment in range(topology_plan.macro_segment_count):
        y0 = core_y - 260.0 + segment * segment_height + 12.0
        seg_h = max(24.0, segment_height - 24.0)
        bundle.rects.append(_rect(f"left_ctrl_seg_{segment}", 270.0, y0, 180.0, seg_h, "M3"))
        bundle.rects.append(_rect(f"right_ctrl_seg_{segment}", die_width_um - 450.0, y0, 180.0, seg_h, "M3"))
    for segment in range(topology_plan.detector_clusters):
        width = (array_width_um + 580.0) / max(topology_plan.detector_clusters, 1)
        x0 = topology_plan.detector_cluster_x_um[segment] - width / 2.0 if segment < len(topology_plan.detector_cluster_x_um) else core_x - 290.0 + segment * width
        bundle.rects.append(_rect(f"readout_seg_{segment}", x0 + 10.0, topology_plan.readout_macro_y_um + 20.0, max(40.0, width - 20.0), 70.0, "M4"))

    for marker_name, mx, my in (
        ("marker_tl", 190.0, 190.0),
        ("marker_tr", die_width_um - 190.0, 190.0),
        ("marker_bl", 190.0, die_height_um - 190.0),
        ("marker_br", die_width_um - 190.0, die_height_um - 190.0),
    ):
        _add_cross_marker(bundle, marker_name, mx, my, 110.0, 18.0)

    pad_w = 110.0
    pad_h = 110.0
    pad_pitch = 165.0
    top_pad_count = max(18, candidate.optical_bus_count + candidate.resonator_count // 2 + 8)
    side_pad_count = max(16, candidate.microwave_line_count * 2 + candidate.control_mux_factor + 6)

    start_top = (die_width_um - (top_pad_count - 1) * pad_pitch - pad_w) / 2.0
    for idx in range(top_pad_count):
        x = start_top + idx * pad_pitch
        bundle.rects.append(_rect(f"pad_top_{idx}", x, 120.0, pad_w, pad_h, "PAD"))
        bundle.rects.append(_rect(f"pass_top_{idx}", x - 10.0, 110.0, pad_w + 20.0, pad_h + 20.0, "PASSIVATION"))
    for idx in range(top_pad_count):
        x = start_top + idx * pad_pitch
        bundle.rects.append(_rect(f"pad_bottom_{idx}", x, die_height_um - 230.0, pad_w, pad_h, "PAD"))
        bundle.rects.append(_rect(f"pass_bottom_{idx}", x - 10.0, die_height_um - 240.0, pad_w + 20.0, pad_h + 20.0, "PASSIVATION"))

    start_side = (die_height_um - (side_pad_count - 1) * pad_pitch - pad_h) / 2.0
    for idx in range(side_pad_count):
        y = start_side + idx * pad_pitch
        bundle.rects.append(_rect(f"pad_left_{idx}", 120.0, y, pad_h, pad_w, "PAD"))
        bundle.rects.append(_rect(f"pad_right_{idx}", die_width_um - 230.0, y, pad_h, pad_w, "PAD"))

    power_grid_pitch_x = max(topology_plan.power_mesh_pitch_um, 92.0)
    power_grid_pitch_y = max(topology_plan.power_mesh_pitch_um * 0.92, 92.0)
    x = core_x - 240.0
    while x <= core_x + array_width_um + 240.0:
        bundle.routes.append(_route([(x, core_y - 280.0), (x, core_y + array_height_um + 280.0)], 12.0, f"m6_grid_{x:.0f}", "M6"))
        x += power_grid_pitch_x
    y = core_y - 280.0
    while y <= core_y + array_height_um + 280.0:
        bundle.routes.append(_route([(core_x - 280.0, y), (core_x + array_width_um + 280.0, y)], 10.0, f"m5_grid_{y:.0f}", "M5"))
        y += power_grid_pitch_y

    optical_bus_xs = placement.optical_bus_x_um
    optical_entry_y = topology_plan.photonics_header_y_um + 110.0
    for bus, x in enumerate(optical_bus_xs):
        main_points = [(x, optical_entry_y), (x, core_y - 40.0), (x, core_y + array_height_um + 40.0)]
        if topology_plan.optical_topology in {"serpentine_redundant", "wide_redundant"}:
            detour = topology_plan.bus_escape_offset_um * (1.0 if bus % 2 == 0 else -1.0)
            mid_y = core_y + array_height_um * (0.22 + 0.10 * (bus % 3))
            main_points = [(x, optical_entry_y), (x + detour, mid_y), (x, core_y + array_height_um + 40.0)]
        bundle.routes.append(_route(main_points, 14.0 * optical_route_scale, f"opt_bus_{bus}", "OPTICAL"))
        _add_grating_coupler(bundle, f"grating_{bus}", x - 18.0, 260.0, 36.0, 92.0, 9)
        bundle.routes.append(_route([(x, 352.0), (x, 260.0), (die_width_um / 2.0, 260.0)], 8.0 * route_width_scale, f"opt_backhaul_{bus}", "M3"))
        fence_count += _add_shield_fence(bundle, f"opt_fence_left_{bus}", x - 22.0, core_y - 50.0, x - 22.0, core_y + array_height_um + 50.0, pitch=topology_plan.shield_pitch_um)
        fence_count += _add_shield_fence(bundle, f"opt_fence_right_{bus}", x + 22.0, core_y - 50.0, x + 22.0, core_y + array_height_um + 50.0, pitch=topology_plan.shield_pitch_um)
        for redundancy in range(topology_plan.optical_redundancy):
            offset = (redundancy + 1) * topology_plan.bus_escape_offset_um * 0.35
            bundle.routes.append(
                _route(
                    [(x + offset, optical_entry_y + 20.0), (x + offset, core_y - 15.0), (x + offset, core_y + array_height_um + 15.0)],
                    5.0 * optical_route_scale,
                    f"opt_redundant_{bus}_{redundancy}",
                    "M6",
                )
            )

    lane_ys = placement.microwave_lane_y_um
    for line, y_line in enumerate(lane_ys):
        left = line % 2 == 0
        x0 = 360.0 if left else die_width_um - 360.0
        x1 = core_x - 50.0 if left else core_x + array_width_um + 50.0
        target_cluster_x = topology_plan.control_spine_x_um[line % max(len(topology_plan.control_spine_x_um), 1)] if topology_plan.control_spine_x_um else core_x + array_width_um / 2.0
        center_x = target_cluster_x
        sig_points = [(x0, y_line), (x1, y_line), (center_x, y_line)]
        if topology_plan.control_topology in {"shielded_clusters", "direct_fanout"}:
            knee = core_x + array_width_um * (0.32 if left else 0.68)
            sig_points = [(x0, y_line), (x1, y_line), (knee, y_line), (center_x, y_line)]
        bundle.routes.append(_route(sig_points, 10.0 * route_width_scale, f"mw_line_{line}", "M2"))
        bundle.routes.append(_route([(x0, y_line - 18.0), (x1, y_line - 18.0), (center_x, y_line - 18.0)], 6.0 * route_width_scale, f"mw_gnd_low_{line}", "GROUND"))
        bundle.routes.append(_route([(x0, y_line + 18.0), (x1, y_line + 18.0), (center_x, y_line + 18.0)], 6.0 * route_width_scale, f"mw_gnd_high_{line}", "GROUND"))
        fence_count += _add_shield_fence(bundle, f"mw_sig_fence_low_{line}", x0, y_line - 28.0, center_x, y_line - 28.0, pitch=max(10.0, topology_plan.shield_pitch_um * 1.2))
        fence_count += _add_shield_fence(bundle, f"mw_sig_fence_high_{line}", x0, y_line + 28.0, center_x, y_line + 28.0, pitch=max(10.0, topology_plan.shield_pitch_um * 1.2))
        for redundancy in range(topology_plan.microwave_redundancy):
            offset = (redundancy + 1) * 11.0
            bundle.routes.append(_route([(x0, y_line + offset), (center_x, y_line + offset)], 3.2 * route_width_scale, f"mw_return_{line}_{redundancy}", "M3"))

    readout_trunk_y = topology_plan.readout_macro_y_um + 120.0
    for lane, x in enumerate(optical_bus_xs):
        trunk_y = readout_trunk_y - 28.0 * (lane % max(topology_plan.detector_clusters, 1))
        cluster_x = topology_plan.detector_cluster_x_um[lane % max(len(topology_plan.detector_cluster_x_um), 1)] if topology_plan.detector_cluster_x_um else die_width_um / 2.0
        lane_points = [(x, core_y + array_height_um + 35.0), (x, trunk_y), (cluster_x, trunk_y)]
        if topology_plan.readout_topology in {"segmented_tree", "compressed_tree"}:
            merge_x = cluster_x
            lane_points = [(x, core_y + array_height_um + 35.0), (x, trunk_y), (merge_x, trunk_y), (merge_x, topology_plan.readout_macro_y_um + 155.0)]
        bundle.routes.append(_route(lane_points, 12.0 * route_width_scale, f"readout_lane_{lane}", "READOUT"))
        bundle.rects.append(_rect(f"detector_macro_{lane}", x - 28.0, die_height_um - 465.0, 56.0, 40.0, "DETECTOR"))
        bundle.rects.append(_rect(f"detector_cap_{lane}", x - 34.0, die_height_um - 500.0, 68.0, 18.0, "M4"))

    row_to_y: dict[int, list[float]] = {}
    for tile in placement.tiles:
        row_to_y.setdefault(tile.row, []).append(tile.y_um)
    for row in range(candidate.rows):
        y_center = float(np.mean(row_to_y.get(row, [core_y + (row + 0.5) * candidate.cell_pitch_um])))
        bundle.routes.append(_route([(core_x - 120.0, y_center), (core_x + array_width_um + 120.0, y_center)], 6.5 * resonator_trace_scale, f"resonator_spine_{row}", "RESONATOR"))
        bundle.routes.append(_route([(core_x - 120.0, y_center - 12.0), (core_x + array_width_um + 120.0, y_center - 12.0)], 3.2 * route_width_scale, f"resonator_aux_low_{row}", "M3"))
        bundle.routes.append(_route([(core_x - 120.0, y_center + 12.0), (core_x + array_width_um + 120.0, y_center + 12.0)], 3.2 * route_width_scale, f"resonator_aux_high_{row}", "M3"))

    for cluster in range(topology_plan.control_cluster_count):
        x_cluster = topology_plan.control_spine_x_um[cluster] if cluster < len(topology_plan.control_spine_x_um) else core_x - 180.0 + cluster * (array_width_um + 360.0) / max(topology_plan.control_cluster_count, 1)
        bundle.routes.append(_route([(x_cluster, core_y - 150.0), (x_cluster, core_y + array_height_um + 150.0)], 4.6 * route_width_scale, f"control_cluster_spine_{cluster}", "M5"))

    for tile in placement.tiles:
        tile_x = tile.x_um - tile_size / 2.0
        tile_y = tile.y_um - tile_size / 2.0
        cx = tile.x_um
        cy = tile.y_um
        bus_x = optical_bus_xs[tile.optical_bus]
        lane_y = lane_ys[tile.microwave_domain]
        resonator_size = min(candidate.cell_pitch_um * 0.94, max(candidate.cell_pitch_um * 0.60, tile.resonator_span_um))
        bundle.rects.append(_rect(f"tile_{tile.qubit_id}", tile_x, tile_y, tile_size, tile_size, "DIAMOND"))
        _add_guard_ring(bundle, f"tile_guard_{tile.qubit_id}", tile_x - 2.8, tile_y - 2.8, tile_size + 5.6, tile_size + 5.6, 3.2 * topology_plan.tile_guard_multiplier, "GROUND")
        _add_guard_ring(bundle, f"tile_shield_{tile.qubit_id}", tile_x - keepout / 2.0, tile_y - keepout / 2.0, tile_size + keepout, tile_size + keepout, 2.4 * topology_plan.tile_guard_multiplier, "M4")
        _add_square_resonator(bundle, f"tile_res_{tile.qubit_id}", cx, cy, resonator_size, max(2.6, candidate.cell_pitch_um * 0.045 * resonator_trace_scale))
        bundle.rects.append(_rect(f"control_cap_{tile.qubit_id}", tile_x - 8.0, cy - 5.0, 12.0, 10.0, "M1"))
        bundle.rects.append(_rect(f"readout_stub_{tile.qubit_id}", cx - 7.0, tile_y + tile_size - 4.0, 14.0, 12.0, "READOUT"))
        bundle.rects.append(_rect(f"opt_stub_{tile.qubit_id}", cx - 6.0, tile_y - 8.0, 12.0, 8.0, "OPTICAL"))
        bundle.rects.append(_rect(f"decap_{tile.qubit_id}", tile_x + tile_size - 10.0, tile_y - 3.0, 8.0, 14.0, "M3"))
        bundle.rects.append(_rect(f"spreader_h_{tile.qubit_id}", cx - 10.0, cy - 1.4, 20.0, 2.8, "M5"))
        bundle.rects.append(_rect(f"spreader_v_{tile.qubit_id}", cx - 1.4, cy - 10.0, 2.8, 20.0, "M5"))
        bundle.circles.append(_circle(f"q_{tile.qubit_id}", cx, cy, qubit_radius, "NV"))
        opt_points = [(bus_x, cy), (cx, cy), (cx, tile_y - 2.0)]
        if topology_plan.optical_topology == "serpentine_redundant":
            detour = topology_plan.bus_escape_offset_um * (1.0 if tile.qubit_id % 2 == 0 else -1.0)
            opt_points = [(bus_x, cy), (bus_x + detour, cy - 0.25 * candidate.cell_pitch_um), (cx, cy), (cx, tile_y - 2.0)]
        bundle.routes.append(_route(opt_points, 2.8 * optical_route_scale, f"opt_coupler_{tile.qubit_id}", "OPTICAL"))
        bundle.routes.append(_route([(cx, cy), (cx, tile_y + tile_size + 18.0)], 3.0 * route_width_scale, f"local_readout_{tile.qubit_id}", "M1"))
        bundle.routes.append(_route([(cx, cy), (cx, lane_y)], 2.4 * route_width_scale, f"mw_drop_{tile.qubit_id}", "M1"))
        bundle.routes.append(_route([(cx - 6.0, lane_y), (cx + 6.0, lane_y)], 1.8 * route_width_scale, f"mw_stub_{tile.qubit_id}", "M2"))
        bridge_layer = "M3" if tile.qubit_id % 2 == 0 else "M5"
        bundle.routes.append(_route([(cx, cy), (bus_x, cy)], 1.4 * route_width_scale, f"sense_bridge_{tile.qubit_id}", bridge_layer))
        fence_pitch = max(8.0, topology_plan.shield_pitch_um)
        fence_count += _add_shield_fence(bundle, f"tile_fence_l_{tile.qubit_id}", tile_x - keepout / 2.0 - 3.0, tile_y - keepout / 2.0, tile_x - keepout / 2.0 - 3.0, tile_y + tile_size + keepout / 2.0, pitch=fence_pitch)
        fence_count += _add_shield_fence(bundle, f"tile_fence_r_{tile.qubit_id}", tile_x + tile_size + keepout / 2.0 + 3.0, tile_y - keepout / 2.0, tile_x + tile_size + keepout / 2.0 + 3.0, tile_y + tile_size + keepout / 2.0, pitch=fence_pitch)
        fence_count += _add_shield_fence(bundle, f"tile_fence_t_{tile.qubit_id}", tile_x - keepout / 2.0, tile_y - keepout / 2.0 - 3.0, tile_x + tile_size + keepout / 2.0, tile_y - keepout / 2.0 - 3.0, pitch=fence_pitch)
        fence_count += _add_shield_fence(bundle, f"tile_fence_b_{tile.qubit_id}", tile_x - keepout / 2.0, tile_y + tile_size + keepout / 2.0 + 3.0, tile_x + tile_size + keepout / 2.0, tile_y + tile_size + keepout / 2.0 + 3.0, pitch=fence_pitch)
        via_cols = 2 + (1 if tile.local_density > 1.08 else 0)
        via_rows = 2 + (1 if tile.shield_strength > 0.82 else 0)
        via_count += _add_via_array(bundle, f"via1_ctrl_{tile.qubit_id}", tile_x + 4.0, tile_y + 4.0, via_cols, via_rows, 5.0, 2.0, "VIA1")
        via_count += _add_via_array(bundle, f"via2_res_{tile.qubit_id}", tile_x + tile_size - 14.0, tile_y + 4.0, via_cols, via_rows, 5.0, 2.0, "VIA2")
        via_count += _add_via_array(bundle, f"via3_rd_{tile.qubit_id}", tile_x + 4.0, tile_y + tile_size - 14.0, via_cols, via_rows, 5.0, 2.0, "VIA3")
        via_count += _add_via_array(bundle, f"via4_bus_{tile.qubit_id}", cx - 3.5, cy - 3.5, 2, 2, 5.0, 2.0, "VIA4")

    if qec_plan and qec_plan.enabled:
        patch_centers: dict[int, tuple[float, float]] = {}
        op_lookup = {op.lane_id: op for op in qec_plan.logical_schedule if op.lane_id}
        for patch in qec_plan.patches:
            center_x = core_x + patch.center_x_norm * array_width_um
            center_y = core_y + patch.center_y_norm * array_height_um
            patch_centers[patch.logical_id] = (center_x, center_y)
            patch_width = max(patch.patch_width_um * 1.10, candidate.cell_pitch_um * 2.4)
            patch_height = max(patch.patch_height_um * 1.10, candidate.cell_pitch_um * 2.4)
            min_x = center_x - patch_width / 2.0
            min_y = center_y - patch_height / 2.0
            bundle.rects.append(_rect(f"logical_patch_{patch.logical_id}", min_x, min_y, patch_width, patch_height, "LOGICAL"))
            _add_guard_ring(bundle, f"logical_patch_guard_{patch.logical_id}", min_x - 8.0, min_y - 8.0, patch_width + 16.0, patch_height + 16.0, 5.0, "LOGICAL")
            bundle.labels.append(_label(f"LQ{patch.logical_id} d={qec_plan.code_distance} dc={patch.decoder_cluster}", center_x - 30.0, center_y, 14.0, "LOGICAL"))
            for syndrome_lane in range(patch.syndrome_bus_count):
                lane_offset = (syndrome_lane - 0.5 * (patch.syndrome_bus_count - 1)) * 6.0
                bundle.routes.append(
                    _route(
                        [(min_x - 14.0, center_y + lane_offset), (center_x, center_y + lane_offset), (min_x + patch_width + 14.0, center_y + lane_offset)],
                        1.4 * route_width_scale,
                        f"syndrome_lane_{patch.logical_id}_{syndrome_lane}",
                        "SURGERY",
                    )
                )
            ancilla_cols = max(1, int(round(math.sqrt(max(patch.ancilla_qubits, 1)))))
            ancilla_rows = max(1, int(math.ceil(max(patch.ancilla_qubits, 1) / ancilla_cols)))
            via_count += _add_via_array(
                bundle,
                f"logical_vias_{patch.logical_id}",
                min_x + 6.0,
                min_y + 6.0,
                ancilla_cols,
                ancilla_rows,
                7.0,
                2.2,
                "VIA5",
            )
        for channel in qec_plan.surgery_channels:
            source = patch_centers.get(channel.source_patch_id)
            target = patch_centers.get(channel.target_patch_id)
            if source is None or target is None:
                continue
            mid_x = 0.5 * (source[0] + target[0])
            lane_id = f"surgery_{channel.source_patch_id}_{channel.target_patch_id}"
            bundle.routes.append(
                _route(
                    [source, (mid_x, source[1]), (mid_x, target[1]), target],
                    2.4 * route_width_scale,
                    lane_id,
                    "SURGERY",
                )
            )
            op = op_lookup.get(lane_id)
            if op is not None:
                bundle.labels.append(_label(f"LS {op.start_ns:.0f}-{op.end_ns:.0f}ns", mid_x - 22.0, 0.5 * (source[1] + target[1]), 10.0, "SURGERY"))
        factory_centers: dict[int, tuple[float, float]] = {}
        for factory in qec_plan.magic_state_factories:
            center_x = core_x + factory.center_x_norm * array_width_um
            center_y = core_y + factory.center_y_norm * array_height_um
            factory_centers[factory.factory_id] = (center_x, center_y)
            width = max(candidate.cell_pitch_um * 2.2, 0.24 * factory.physical_qubits)
            height = max(candidate.cell_pitch_um * 1.6, 0.18 * factory.physical_qubits)
            bundle.rects.append(_rect(f"factory_{factory.factory_id}", center_x - width / 2.0, center_y - height / 2.0, width, height, "FACTORY"))
            _add_guard_ring(bundle, f"factory_guard_{factory.factory_id}", center_x - width / 2.0 - 6.0, center_y - height / 2.0 - 6.0, width + 12.0, height + 12.0, 3.0, "FACTORY")
            bundle.labels.append(_label(f"MSF{factory.factory_id} {factory.protocol_name}", center_x - 24.0, center_y, 12.0, "FACTORY"))
        for interconnect in qec_plan.logical_interconnects:
            if interconnect.source_kind == "factory":
                source = factory_centers.get(interconnect.source_id)
            else:
                source = patch_centers.get(interconnect.source_id)
            if interconnect.target_kind == "factory":
                target = factory_centers.get(interconnect.target_id)
            else:
                target = patch_centers.get(interconnect.target_id)
            if source is None or target is None:
                continue
            if interconnect.route_class == "magic_state_feed":
                mid_y = min(source[1], target[1]) - 18.0
                lane_id = f"factory_{interconnect.source_id}_to_patch_{interconnect.target_id}_0"
                bundle.routes.append(_route([source, (source[0], mid_y), (target[0], mid_y), target], 1.6 * route_width_scale, f"logical_ic_{interconnect.source_kind}_{interconnect.source_id}_{interconnect.target_kind}_{interconnect.target_id}", "FACTORY"))
                op = next((item for item in qec_plan.logical_schedule if item.op_type == "magic_state_injection" and item.factory_id == interconnect.source_id and interconnect.target_id in item.logical_patch_ids), None)
                if op is not None:
                    bundle.labels.append(_label(f"MS {op.start_ns:.0f}-{op.end_ns:.0f}ns", target[0] - 22.0, mid_y - 6.0, 10.0, "FACTORY"))

    fill_count += _add_fill_region(bundle, "fill_top_left", 250.0, 600.0, core_x - 340.0, max(240.0, core_y - 760.0))
    fill_count += _add_fill_region(bundle, "fill_top_right", core_x + array_width_um + 80.0, 600.0, die_width_um - (core_x + array_width_um + 330.0), max(240.0, core_y - 760.0))
    fill_count += _add_fill_region(bundle, "fill_bottom_left", 250.0, core_y + array_height_um + 300.0, core_x - 340.0, die_height_um - (core_y + array_height_um + 610.0))
    fill_count += _add_fill_region(bundle, "fill_bottom_right", core_x + array_width_um + 80.0, core_y + array_height_um + 300.0, die_width_um - (core_x + array_width_um + 330.0), die_height_um - (core_y + array_height_um + 610.0))

    bundle.labels.extend(
        [
            _label(spec.design_name, 260.0, 260.0, 92.0),
            _label(f"{candidate.architecture} | qubits={candidate.qubits}", 260.0, 360.0, 42.0),
            _label(f"gate={metrics.gate_fidelity:.5f} | readout={metrics.readout_fidelity:.5f} | T2={metrics.t2_us:.1f} us", 260.0, 430.0, 30.0),
        ]
    )

    min_spacing_um = placement.min_spacing_um - 2.0 * qubit_radius
    rect_count_by_layer: dict[str, int] = {}
    for rect in bundle.rects:
        rect_count_by_layer[rect.layer] = rect_count_by_layer.get(rect.layer, 0) + 1
    bundle.stats = {
        "rect_count": len(bundle.rects),
        "route_count": len(bundle.routes),
        "circle_count": len(bundle.circles),
        "label_count": len(bundle.labels),
        "pad_count": 2 * top_pad_count + 2 * side_pad_count,
        "via_count": via_count,
        "shield_fence_count": fence_count,
        "fill_count": fill_count,
        "placement_score": placement.placement_score,
        "placement_mean_offset_um": placement.mean_offset_um,
        "placement_bus_loads": placement.optical_bus_loads,
        "placement_domain_loads": placement.microwave_domain_loads,
        "topology": topology_plan.to_dict(),
        "qec": qec_plan.to_dict() if qec_plan else None,
        "magic_state_factories": len(qec_plan.magic_state_factories) if qec_plan else 0,
        "lattice_surgery_channels": len(qec_plan.surgery_channels) if qec_plan else 0,
        "logical_schedule_ops": len(qec_plan.logical_schedule) if qec_plan else 0,
        "factory_batches": len(qec_plan.factory_timelines) if qec_plan else 0,
        "rect_count_by_layer": rect_count_by_layer,
    }
    bundle.drc = {
        "min_spacing_um": min_spacing_um,
        "required_spacing_um": 8.0,
        "spacing_pass": min_spacing_um >= 8.0,
        "die_area_mm2": metrics.die_area_mm2,
        "area_pass": metrics.die_area_mm2 <= spec.max_die_area_mm2,
        "route_drc_violations": metrics.route_drc_violations,
        "routing_capacity": metrics.routing_capacity,
        "routing_pass": metrics.routing_capacity >= 0.30,
        "pad_count": 2 * top_pad_count + 2 * side_pad_count,
        "via_count": via_count,
        "shield_fence_count": fence_count,
        "fill_count": fill_count,
        "placement_score": placement.placement_score,
        "layout_complexity_score": topology_plan.layout_complexity_score,
        "logical_patches": len(qec_plan.patches) if qec_plan else 0,
        "magic_state_factories": len(qec_plan.magic_state_factories) if qec_plan else 0,
        "logical_schedule_ops": len(qec_plan.logical_schedule) if qec_plan else 0,
    }
    return bundle


def layout_to_svg(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, layout: LayoutBundle) -> str:
    scale = 0.16
    width = layout.die_width_um * scale
    height = layout.die_height_um * scale
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" height="{height:.1f}" viewBox="0 0 {layout.die_width_um:.1f} {layout.die_height_um:.1f}">',
        '<defs>',
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#fffaf0"/>',
        '<stop offset="100%" stop-color="#edf2f7"/>',
        '</linearGradient>',
        '</defs>',
        f'<rect x="0" y="0" width="{layout.die_width_um:.1f}" height="{layout.die_height_um:.1f}" fill="url(#bg)"/>',
    ]
    for rect in layout.rects:
        opacity = 0.38 if rect.layer == "FILL" else 0.84
        parts.append(
            f'<rect x="{rect.x:.1f}" y="{rect.y:.1f}" width="{rect.width:.1f}" height="{rect.height:.1f}" fill="{rect.color}" opacity="{opacity:.2f}" stroke="#111827" stroke-width="1"/>'
        )
    for route in layout.routes:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in route.points)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{route.color}" stroke-width="{route.width:.1f}" stroke-linecap="round" stroke-linejoin="round" opacity="0.90"/>'
        )
    for circle in layout.circles:
        parts.append(f'<circle cx="{circle.cx:.1f}" cy="{circle.cy:.1f}" r="{circle.radius:.1f}" fill="{circle.color}" opacity="0.96"/>')
    for label in layout.labels:
        parts.append(
            f'<text x="{label.x:.1f}" y="{label.y:.1f}" font-size="{label.size:.1f}" font-family="Georgia, serif" fill="{label.color}">{label.text}</text>'
        )
    parts.append('</svg>')
    return "".join(parts)


def _render_preview_png(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, layout: LayoutBundle, output_path: Path) -> str:
    fig_w = max(8.0, layout.die_width_um / 1800.0)
    fig_h = max(6.0, layout.die_height_um / 1800.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#fcfbf7")
    for rect in layout.rects:
        alpha = 0.28 if rect.layer == "FILL" else 0.80
        ax.add_patch(Rectangle((rect.x, rect.y), rect.width, rect.height, facecolor=rect.color, edgecolor="#243b53", linewidth=0.15, alpha=alpha))
    for route in layout.routes:
        xs = [point[0] for point in route.points]
        ys = [point[1] for point in route.points]
        ax.plot(xs, ys, color=route.color, linewidth=max(0.5, route.width / 18.0), solid_capstyle="round", alpha=0.88)
    for circle in layout.circles:
        ax.add_patch(MplCircle((circle.cx, circle.cy), circle.radius, facecolor=circle.color, edgecolor="none", alpha=0.95))
    ax.text(260.0, 230.0, spec.design_name, fontsize=18, color="#0f172a", family="serif")
    ax.text(260.0, 330.0, f"{candidate.architecture} | qubits={candidate.qubits} | gate={metrics.gate_fidelity:.5f}", fontsize=9.5, color="#334155")
    ax.set_xlim(0.0, layout.die_width_um)
    ax.set_ylim(layout.die_height_um, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(output_path)


def layout_to_klayout_script(layout: LayoutBundle, output_gds_name: str) -> str:
    layer_map_lines = [f'    "{name}": layout.layer({layer}, {datatype}),' for name, (layer, datatype) in LAYER_MAP.items()]
    rect_lines = [
        f'insert_box(top, {rect.x:.3f}, {rect.y:.3f}, {rect.x + rect.width:.3f}, {rect.y + rect.height:.3f}, layer_map["{rect.layer}"])'
        for rect in layout.rects
    ]
    route_lines = []
    for route in layout.routes:
        points = ", ".join(f'pya.Point(int({x:.3f} * dbu_inv), int({y:.3f} * dbu_inv))' for x, y in route.points)
        route_lines.append(f'insert_path(top, [{points}], {route.width:.3f}, layer_map["{route.layer}"])')
    circle_lines = [
        f'insert_disc(top, {circle.cx:.3f}, {circle.cy:.3f}, {circle.radius:.3f}, layer_map["{circle.layer}"])'
        for circle in layout.circles
    ]
    label_lines = [
        f'top.shapes(layer_map["{label.layer}"]).insert(pya.Text({label.text!r}, pya.Trans(int({label.x:.3f} * dbu_inv), int({label.y:.3f} * dbu_inv))))'
        for label in layout.labels
    ]
    return f'''import math\nimport pya\n\nlayout = pya.Layout()\nlayout.dbu = 0.001\ndbu_inv = 1.0 / layout.dbu\ntop = layout.create_cell("TOP")\nlayer_map = {{\n{chr(10).join(layer_map_lines)}\n}}\n\ndef insert_box(cell, x1, y1, x2, y2, layer):\n    cell.shapes(layer).insert(pya.Box(int(x1 * dbu_inv), int(y1 * dbu_inv), int(x2 * dbu_inv), int(y2 * dbu_inv)))\n\ndef insert_path(cell, pts, width, layer):\n    cell.shapes(layer).insert(pya.Path(pts, int(width * dbu_inv)))\n\ndef insert_disc(cell, cx, cy, radius, layer, segments=24):\n    pts = []\n    for i in range(segments):\n        angle = 2.0 * math.pi * i / segments\n        pts.append(pya.Point(int((cx + radius * math.cos(angle)) * dbu_inv), int((cy + radius * math.sin(angle)) * dbu_inv)))\n    cell.shapes(layer).insert(pya.Polygon(pts))\n\n{chr(10).join(rect_lines)}\n{chr(10).join(route_lines)}\n{chr(10).join(circle_lines)}\n{chr(10).join(label_lines)}\nlayout.write(r"{output_gds_name}")\nprint("Wrote", r"{output_gds_name}")\n'''


def write_gds_artifacts(output_dir: Path, layout: LayoutBundle) -> dict[str, str]:
    gds_path = output_dir / "layout.gds"
    oas_path = output_dir / "layout.oas"
    layer_map_path = output_dir / "layer_map.json"
    hierarchy_path = output_dir / "gds_hierarchy.json"
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("TOP")
    rect_cache: dict[tuple[str, int, int], gdstk.Cell] = {}
    circle_cache: dict[tuple[str, int], gdstk.Cell] = {}
    ref_counts: dict[str, int] = {}

    def _rect_cell(rect: Rect) -> gdstk.Cell:
        key = (rect.layer, int(round(rect.width * 1000.0)), int(round(rect.height * 1000.0)))
        cached = rect_cache.get(key)
        if cached is not None:
            return cached
        layer, datatype = LAYER_MAP[rect.layer]
        new_cell = library.new_cell(f"RECT_{rect.layer}_{key[1]}_{key[2]}")
        new_cell.add(gdstk.rectangle((0.0, 0.0), (rect.width, rect.height), layer=layer, datatype=datatype))
        rect_cache[key] = new_cell
        return new_cell

    def _circle_cell(circle: Circle) -> gdstk.Cell:
        key = (circle.layer, int(round(circle.radius * 1000.0)))
        cached = circle_cache.get(key)
        if cached is not None:
            return cached
        layer, datatype = LAYER_MAP[circle.layer]
        new_cell = library.new_cell(f"DISC_{circle.layer}_{key[1]}")
        new_cell.add(gdstk.ellipse((0.0, 0.0), circle.radius, layer=layer, datatype=datatype))
        circle_cache[key] = new_cell
        return new_cell

    for rect in layout.rects:
        primitive_cell = _rect_cell(rect)
        cell.add(gdstk.Reference(primitive_cell, (rect.x, rect.y)))
        ref_counts[primitive_cell.name] = ref_counts.get(primitive_cell.name, 0) + 1
    for route in layout.routes:
        if _route_length(route.points) <= 0.1:
            continue
        layer, datatype = LAYER_MAP[route.layer]
        cell.add(gdstk.FlexPath(route.points, route.width, layer=layer, datatype=datatype, ends="round"))
    for circle in layout.circles:
        primitive_cell = _circle_cell(circle)
        cell.add(gdstk.Reference(primitive_cell, (circle.cx, circle.cy)))
        ref_counts[primitive_cell.name] = ref_counts.get(primitive_cell.name, 0) + 1
    for label in layout.labels:
        layer, texttype = LAYER_MAP[label.layer]
        cell.add(gdstk.Label(label.text, (label.x, label.y), layer=layer, texttype=texttype))

    library.write_gds(str(gds_path))
    try:
        library.write_oas(str(oas_path))
        oas_value = str(oas_path)
    except Exception:
        oas_value = ""
    layer_map_path.write_text(json.dumps({name: {"layer": layer, "datatype": datatype} for name, (layer, datatype) in LAYER_MAP.items()}, indent=2), encoding="utf-8")
    hierarchy_path.write_text(
        json.dumps(
            {
                "cell_count": len(library.cells),
                "rect_cell_count": len(rect_cache),
                "circle_cell_count": len(circle_cache),
                "reference_counts": ref_counts,
                "top_references": sum(ref_counts.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"gds": str(gds_path), "oas": oas_value, "layer_map": str(layer_map_path), "gds_hierarchy": str(hierarchy_path)}


def write_layout_artifacts(output_dir: Path, spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, layout: LayoutBundle) -> dict[str, str]:
    layout_json = output_dir / "layout.json"
    layout_svg = output_dir / "layout.svg"
    preview_png = output_dir / "layout_preview.png"
    klayout_py = output_dir / "layout_klayout.py"
    layout_json.write_text(json.dumps(layout.to_dict(), indent=2), encoding="utf-8")
    layout_svg.write_text(layout_to_svg(spec, candidate, metrics, layout), encoding="utf-8")
    _render_preview_png(spec, candidate, metrics, layout, preview_png)
    klayout_py.write_text(layout_to_klayout_script(layout, str(output_dir / "layout.gds")), encoding="utf-8")
    gds_artifacts = write_gds_artifacts(output_dir, layout)
    return {
        "layout_json": str(layout_json),
        "layout_svg": str(layout_svg),
        "layout_preview_png": str(preview_png),
        "klayout_py": str(klayout_py),
        **gds_artifacts,
    }
