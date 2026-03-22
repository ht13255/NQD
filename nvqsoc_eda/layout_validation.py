from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .layout import LayoutBundle
from .qec import QECPlan
from .topology import TopologyPlan


@dataclass(slots=True)
class LayoutConsistencySummary:
    closure_pass: bool
    patch_count_match: bool
    surgery_route_coverage: bool
    factory_route_coverage: bool
    pad_ring_complete: bool
    power_mesh_present: bool
    hotspot_keepout_present: bool
    logical_schedule_labeled: bool
    hierarchy_present: bool
    route_count: int
    rect_count: int
    missing_items: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_layout_consistency(layout: LayoutBundle, topology_plan: TopologyPlan, qec_plan: QECPlan | None) -> LayoutConsistencySummary:
    route_names = [route.name for route in layout.routes]
    rect_names = [rect.name for rect in layout.rects]
    pad_count = int(layout.stats.get("pad_count", 0))
    power_mesh_present = any(name.startswith("m6_grid_") or name.startswith("m5_grid_") for name in route_names)
    hotspot_present = bool(topology_plan.hotspot_keepouts) == any(name.startswith("hotspot_keepout_") for name in rect_names)
    hierarchy_present = int(layout.stats.get("rect_count", 0)) > 200
    patch_count_match = True
    surgery_route_coverage = True
    factory_route_coverage = True
    logical_schedule_labeled = True
    missing_items: list[str] = []
    recommendations: list[str] = []
    if qec_plan is not None:
        logical_patch_rects = [name for name in rect_names if name.startswith("logical_patch_") and not name.startswith("logical_patch_guard_")]
        patch_count_match = len(logical_patch_rects) == len(qec_plan.patches)
        if not patch_count_match:
            missing_items.append("logical_patch_overlay")
            recommendations.append("regenerate logical patch overlays from QEC plan")
        surgery_route_coverage = all(f"surgery_{channel.source_patch_id}_{channel.target_patch_id}" in route_names for channel in qec_plan.surgery_channels)
        if not surgery_route_coverage:
            missing_items.append("lattice_surgery_routes")
            recommendations.append("ensure all lattice surgery channels are emitted into layout routes")
        factory_rects = [name for name in rect_names if name.startswith("factory_") and not name.startswith("factory_guard_")]
        factory_route_coverage = len(factory_rects) >= len(qec_plan.magic_state_factories)
        if not factory_route_coverage:
            missing_items.append("magic_state_factories")
            recommendations.append("emit one physical macro per planned magic-state factory")
        logical_schedule_labeled = True
        for op in qec_plan.logical_schedule:
            if op.op_type == "lattice_surgery" and not any(label.text.startswith("LS ") for label in layout.labels):
                logical_schedule_labeled = False
                break
        if not logical_schedule_labeled:
            missing_items.append("schedule_labels")
            recommendations.append("add schedule labels for logical operations")
    pad_ring_complete = pad_count >= 40
    if not pad_ring_complete:
        missing_items.append("pad_ring")
        recommendations.append("increase pad-ring density for signoff-style packaging realism")
    if not power_mesh_present:
        missing_items.append("power_mesh")
        recommendations.append("emit global power mesh and verify M5/M6 grid closure")
    if not hotspot_present:
        missing_items.append("hotspot_keepout")
        recommendations.append("propagate hotspot keepout windows into layout")
    closure_pass = patch_count_match and surgery_route_coverage and factory_route_coverage and pad_ring_complete and power_mesh_present and hotspot_present and logical_schedule_labeled
    return LayoutConsistencySummary(
        closure_pass=closure_pass,
        patch_count_match=patch_count_match,
        surgery_route_coverage=surgery_route_coverage,
        factory_route_coverage=factory_route_coverage,
        pad_ring_complete=pad_ring_complete,
        power_mesh_present=power_mesh_present,
        hotspot_keepout_present=hotspot_present,
        logical_schedule_labeled=logical_schedule_labeled,
        hierarchy_present=hierarchy_present,
        route_count=len(layout.routes),
        rect_count=len(layout.rects),
        missing_items=missing_items,
        recommendations=recommendations,
    )


def write_layout_consistency_artifacts(output_dir: Path, summary: LayoutConsistencySummary) -> dict[str, str]:
    path = output_dir / "layout_consistency.json"
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return {"layout_consistency": str(path)}
