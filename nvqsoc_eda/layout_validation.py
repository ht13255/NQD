from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .layout import LayoutBundle
from .qec import QECPlan
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec
from .topology import TopologyPlan


# ---------------------------------------------------------------------------
# DRC Rule Checking
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DRCRuleResult:
    """Result of a single DRC rule check."""
    rule_name: str
    passed: bool
    actual_value: float
    limit_value: float
    severity: str  # "critical", "major", "minor"
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DRCSummary:
    """Aggregate result of all DRC rule checks."""
    all_pass: bool
    critical_pass: bool
    rules: list[DRCRuleResult] = field(default_factory=list)
    total_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_pass": self.all_pass,
            "critical_pass": self.critical_pass,
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "rules": [r.to_dict() for r in self.rules],
        }


def run_drc_checks(layout: LayoutBundle, spec: DesignSpec) -> DRCSummary:
    """Run formal DRC checks against layout bundle and spec."""
    rules: list[DRCRuleResult] = []
    drc = layout.drc
    stats = layout.stats

    # 1. Minimum feature spacing
    min_spacing = float(drc.get("min_spacing_um", 0.0))
    required_spacing = float(drc.get("required_spacing_um", 8.0))
    rules.append(DRCRuleResult(
        rule_name="min_feature_spacing",
        passed=min_spacing >= required_spacing,
        actual_value=min_spacing,
        limit_value=required_spacing,
        severity="critical",
        message=f"Minimum spacing {min_spacing:.2f} um {'≥' if min_spacing >= required_spacing else '<'} required {required_spacing:.2f} um",
    ))

    # 2. Die area within spec limit
    die_area = float(drc.get("die_area_mm2", 0.0))
    max_area = spec.max_die_area_mm2
    rules.append(DRCRuleResult(
        rule_name="die_area_limit",
        passed=die_area <= max_area,
        actual_value=die_area,
        limit_value=max_area,
        severity="critical",
        message=f"Die area {die_area:.2f} mm² {'≤' if die_area <= max_area else '>'} limit {max_area:.2f} mm²",
    ))

    # 3. Route DRC violations
    route_violations = int(drc.get("route_drc_violations", 0))
    rules.append(DRCRuleResult(
        rule_name="route_drc_violations",
        passed=route_violations == 0,
        actual_value=float(route_violations),
        limit_value=0.0,
        severity="critical",
        message=f"Route DRC violations: {route_violations} (must be 0)",
    ))

    # 4. Routing capacity
    routing_capacity = float(drc.get("routing_capacity", 0.0))
    min_routing = 0.30
    rules.append(DRCRuleResult(
        rule_name="routing_capacity",
        passed=routing_capacity >= min_routing,
        actual_value=routing_capacity,
        limit_value=min_routing,
        severity="critical",
        message=f"Routing capacity {routing_capacity:.4f} {'≥' if routing_capacity >= min_routing else '<'} minimum {min_routing}",
    ))

    # 5. Pad count minimum
    pad_count = int(drc.get("pad_count", 0))
    min_pads = 40
    rules.append(DRCRuleResult(
        rule_name="pad_count_minimum",
        passed=pad_count >= min_pads,
        actual_value=float(pad_count),
        limit_value=float(min_pads),
        severity="major",
        message=f"Pad count {pad_count} {'≥' if pad_count >= min_pads else '<'} minimum {min_pads}",
    ))

    # 6. Via count consistency (non-zero)
    via_count = int(drc.get("via_count", 0))
    rules.append(DRCRuleResult(
        rule_name="via_count_nonzero",
        passed=via_count > 0,
        actual_value=float(via_count),
        limit_value=1.0,
        severity="major",
        message=f"Via count {via_count} {'> 0' if via_count > 0 else '= 0 (missing vias)'}",
    ))

    # 7. Shield fence density
    fence_count = int(drc.get("shield_fence_count", 0))
    min_fences = 10
    rules.append(DRCRuleResult(
        rule_name="shield_fence_density",
        passed=fence_count >= min_fences,
        actual_value=float(fence_count),
        limit_value=float(min_fences),
        severity="minor",
        message=f"Shield fences {fence_count} {'≥' if fence_count >= min_fences else '<'} minimum {min_fences}",
    ))

    # 8. Fill count nonzero
    fill_count = int(drc.get("fill_count", 0))
    rules.append(DRCRuleResult(
        rule_name="fill_region_present",
        passed=fill_count > 0,
        actual_value=float(fill_count),
        limit_value=1.0,
        severity="minor",
        message=f"Fill shapes {fill_count} {'present' if fill_count > 0 else 'missing'}",
    ))

    # 9. Power-grid IR drop limit
    ir_drop_mV = float(drc.get("worst_ir_drop_mV", 0.0))
    ir_drop_limit_mV = float(drc.get("ir_drop_drc_limit_mV", 200.0))
    rules.append(DRCRuleResult(
        rule_name="power_grid_ir_drop",
        passed=ir_drop_mV <= ir_drop_limit_mV,
        actual_value=ir_drop_mV,
        limit_value=ir_drop_limit_mV,
        severity="major",
        message=f"Power-grid IR drop {ir_drop_mV:.2f} mV {'<=' if ir_drop_mV <= ir_drop_limit_mV else '>'} limit {ir_drop_limit_mV:.2f} mV",
    ))

    # 10. Explicit metal density ceiling
    metal_density_max = float(drc.get("metal_density_max", drc.get("density_max_after_fill", 0.0)))
    metal_density_limit = float(drc.get("metal_density_limit", 1.0))
    rules.append(DRCRuleResult(
        rule_name="metal_density_limit",
        passed=metal_density_max <= metal_density_limit,
        actual_value=metal_density_max,
        limit_value=metal_density_limit,
        severity="critical",
        message=f"Metal density {metal_density_max:.4f} {'<=' if metal_density_max <= metal_density_limit else '>'} limit {metal_density_limit:.4f}",
    ))

    # 11. Placement quality
    placement_score = float(drc.get("placement_score", 0.0))
    min_placement = 0.10
    rules.append(DRCRuleResult(
        rule_name="placement_quality",
        passed=placement_score >= min_placement,
        actual_value=placement_score,
        limit_value=min_placement,
        severity="major",
        message=f"Placement score {placement_score:.4f} {'≥' if placement_score >= min_placement else '<'} minimum {min_placement}",
    ))

    # 12. Rect count sanity (non-trivial layout)
    rect_count = int(stats.get("rect_count", 0))
    min_rects = 100
    rules.append(DRCRuleResult(
        rule_name="layout_complexity",
        passed=rect_count >= min_rects,
        actual_value=float(rect_count),
        limit_value=float(min_rects),
        severity="major",
        message=f"Rect count {rect_count} {'≥' if rect_count >= min_rects else '<'} minimum {min_rects} for signoff-grade layout",
    ))

    passed_rules = sum(1 for r in rules if r.passed)
    failed_rules = len(rules) - passed_rules
    critical_pass = all(r.passed for r in rules if r.severity == "critical")

    return DRCSummary(
        all_pass=failed_rules == 0,
        critical_pass=critical_pass,
        rules=rules,
        total_rules=len(rules),
        passed_rules=passed_rules,
        failed_rules=failed_rules,
    )


# ---------------------------------------------------------------------------
# Design-vs-Layout Cross-validation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DesignLayoutMismatch:
    """A single mismatch between design intent and generated layout."""
    check_name: str
    expected: float
    actual: float
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DesignVsLayoutResult:
    """Aggregate result of design-vs-layout cross-validation."""
    all_match: bool
    mismatches: list[DesignLayoutMismatch] = field(default_factory=list)
    checks_performed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_match": self.all_match,
            "checks_performed": self.checks_performed,
            "mismatches": [m.to_dict() for m in self.mismatches],
        }


def _simulation_constraint_diffs(spec: DesignSpec, metrics: SimulationMetrics) -> list[dict[str, Any]]:
    diffs = [
        ("die_area", spec.max_die_area_mm2, metrics.die_area_mm2, "<="),
        ("power", spec.max_power_mw, metrics.power_mw, "<="),
        ("latency", spec.max_latency_ns, metrics.latency_ns, "<="),
        ("coherence", spec.target_t2_us, metrics.t2_us, ">="),
        ("gate_fidelity", spec.target_gate_fidelity, metrics.gate_fidelity, ">="),
        ("readout_fidelity", spec.target_readout_fidelity, metrics.readout_fidelity, ">="),
        ("yield", spec.min_yield, metrics.yield_estimate, ">="),
    ]
    violated = set(metrics.constraint_violations)
    rows: list[dict[str, Any]] = []
    for name, target, actual, comparison in diffs:
        rows.append(
            {
                "name": name,
                "target": float(target),
                "actual": float(actual),
                "comparison": comparison,
                "delta": float(actual - target),
                "violated": name in violated,
            }
        )
    return rows


def validate_design_vs_layout(
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    layout: LayoutBundle,
) -> DesignVsLayoutResult:
    """Cross-validate design spec and candidate against generated layout."""
    mismatches: list[DesignLayoutMismatch] = []
    checks = 0

    # 1. Qubit count vs NV circles
    checks += 1
    nv_circles = sum(1 for c in layout.circles if c.layer == "NV")
    if nv_circles != candidate.qubits:
        mismatches.append(DesignLayoutMismatch(
            check_name="qubit_count",
            expected=float(candidate.qubits),
            actual=float(nv_circles),
            severity="critical",
            message=f"Design specifies {candidate.qubits} qubits but layout has {nv_circles} NV circles",
        ))

    # 2. Optical bus count vs optical bus routes
    checks += 1
    opt_bus_routes = sum(1 for r in layout.routes if r.name.startswith("opt_bus_"))
    if opt_bus_routes != candidate.optical_bus_count:
        mismatches.append(DesignLayoutMismatch(
            check_name="optical_bus_count",
            expected=float(candidate.optical_bus_count),
            actual=float(opt_bus_routes),
            severity="major",
            message=f"Design specifies {candidate.optical_bus_count} optical buses but layout has {opt_bus_routes}",
        ))

    # 3. Microwave line count vs microwave routes
    checks += 1
    mw_routes = sum(1 for r in layout.routes if r.name.startswith("mw_line_"))
    if mw_routes != candidate.microwave_line_count:
        mismatches.append(DesignLayoutMismatch(
            check_name="microwave_line_count",
            expected=float(candidate.microwave_line_count),
            actual=float(mw_routes),
            severity="major",
            message=f"Design specifies {candidate.microwave_line_count} MW lines but layout has {mw_routes}",
        ))

    # 4. Die area agreement (layout vs metrics, ≤5% tolerance)
    checks += 1
    layout_area_mm2 = float(layout.die_width_um * layout.die_height_um / 1.0e6)
    metrics_area_mm2 = float(metrics.die_area_mm2)
    area_delta_pct = abs(layout_area_mm2 - metrics_area_mm2) / max(metrics_area_mm2, 1e-9) * 100.0
    if area_delta_pct > 5.0:
        mismatches.append(DesignLayoutMismatch(
            check_name="die_area_agreement",
            expected=metrics_area_mm2,
            actual=layout_area_mm2,
            severity="critical",
            message=f"Layout die area {layout_area_mm2:.3f} mm² differs from metrics {metrics_area_mm2:.3f} mm² by {area_delta_pct:.1f}%",
        ))

    # 5. Die area within spec
    checks += 1
    if layout_area_mm2 > spec.max_die_area_mm2:
        mismatches.append(DesignLayoutMismatch(
            check_name="die_area_spec_limit",
            expected=spec.max_die_area_mm2,
            actual=layout_area_mm2,
            severity="critical",
            message=f"Layout die area {layout_area_mm2:.3f} mm² exceeds spec limit {spec.max_die_area_mm2:.3f} mm²",
        ))

    # 6. Constraint violations in simulation
    checks += 1
    violation_count = len(metrics.constraint_violations)
    if violation_count > 0:
        mismatches.append(DesignLayoutMismatch(
            check_name="simulation_constraints",
            expected=0.0,
            actual=float(violation_count),
            severity="major",
            message=f"Simulation has {violation_count} constraint violations: {metrics.constraint_violations}",
            details={"constraint_diffs": _simulation_constraint_diffs(spec, metrics)},
        ))

    # 7. Routing capacity adequate
    checks += 1
    if metrics.routing_capacity < 0.30:
        mismatches.append(DesignLayoutMismatch(
            check_name="routing_capacity",
            expected=0.30,
            actual=metrics.routing_capacity,
            severity="major",
            message=f"Routing capacity {metrics.routing_capacity:.4f} below minimum 0.30",
        ))

    return DesignVsLayoutResult(
        all_match=len(mismatches) == 0,
        mismatches=mismatches,
        checks_performed=checks,
    )


# ---------------------------------------------------------------------------
# Original Layout Consistency Validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Artifact Writers
# ---------------------------------------------------------------------------

def write_layout_consistency_artifacts(output_dir: Path, summary: LayoutConsistencySummary) -> dict[str, str]:
    path = output_dir / "layout_consistency.json"
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return {"layout_consistency": str(path)}


def write_drc_report(output_dir: Path, drc_summary: DRCSummary, dvl_result: DesignVsLayoutResult) -> dict[str, str]:
    """Write DRC and design-vs-layout reports."""
    payload = {
        "drc": drc_summary.to_dict(),
        "design_vs_layout": dvl_result.to_dict(),
    }
    path = output_dir / "signoff_drc_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"signoff_drc_report": str(path)}



def write_layout_consistency_artifacts(output_dir: Path, summary: LayoutConsistencySummary) -> dict[str, str]:
    path = output_dir / "layout_consistency.json"
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return {"layout_consistency": str(path)}
