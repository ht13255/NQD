from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .layout import LayoutBundle
from .layout_validation import (
    DRCSummary,
    DesignVsLayoutResult,
    LayoutConsistencySummary,
)
from .qec import QECPlan
from .simulator import CandidateDesign, SimulationMetrics
from .signoff_schema import SIGNOFF_VERSION
from .spec import DesignSpec


# ---------------------------------------------------------------------------
# Signoff Gates - Individual check stages
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SignoffGate:
    """A single signoff gate representing one validation stage."""
    gate_id: str
    gate_name: str
    category: str  # "drc", "design_vs_layout", "consistency", "integrity", "artifact"
    passed: bool
    severity: str  # "critical", "major", "minor"
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_drc_gate(drc_summary: DRCSummary) -> SignoffGate:
    """Build a signoff gate from DRC check results."""
    failed = [r.rule_name for r in drc_summary.rules if not r.passed]
    return SignoffGate(
        gate_id="gate_drc",
        gate_name="Design Rule Check",
        category="drc",
        passed=drc_summary.critical_pass,
        severity="critical",
        timestamp=_now_iso(),
        details={
            "all_pass": drc_summary.all_pass,
            "critical_pass": drc_summary.critical_pass,
            "total_rules": drc_summary.total_rules,
            "passed_rules": drc_summary.passed_rules,
            "failed_rules": drc_summary.failed_rules,
            "failed_rule_names": failed,
        },
        message=f"DRC: {drc_summary.passed_rules}/{drc_summary.total_rules} rules passed" + (f" (failures: {', '.join(failed)})" if failed else ""),
    )


def build_design_vs_layout_gate(dvl_result: DesignVsLayoutResult) -> SignoffGate:
    """Build a signoff gate from design-vs-layout cross-validation."""
    mismatch_names = [m.check_name for m in dvl_result.mismatches]
    critical_mismatches = [m.check_name for m in dvl_result.mismatches if m.severity == "critical"]
    return SignoffGate(
        gate_id="gate_design_vs_layout",
        gate_name="Design vs Layout Cross-validation",
        category="design_vs_layout",
        passed=len(critical_mismatches) == 0,
        severity="critical",
        timestamp=_now_iso(),
        details={
            "all_match": dvl_result.all_match,
            "checks_performed": dvl_result.checks_performed,
            "mismatch_count": len(dvl_result.mismatches),
            "mismatch_names": mismatch_names,
            "critical_mismatches": critical_mismatches,
        },
        message=f"Design↔Layout: {dvl_result.checks_performed - len(dvl_result.mismatches)}/{dvl_result.checks_performed} checks passed" + (f" (mismatches: {', '.join(mismatch_names)})" if mismatch_names else ""),
    )


def build_consistency_gate(layout_consistency: LayoutConsistencySummary) -> SignoffGate:
    """Build a signoff gate from layout consistency validation."""
    return SignoffGate(
        gate_id="gate_consistency",
        gate_name="Layout Consistency Closure",
        category="consistency",
        passed=layout_consistency.closure_pass,
        severity="critical",
        timestamp=_now_iso(),
        details={
            "closure_pass": layout_consistency.closure_pass,
            "patch_count_match": layout_consistency.patch_count_match,
            "surgery_route_coverage": layout_consistency.surgery_route_coverage,
            "factory_route_coverage": layout_consistency.factory_route_coverage,
            "pad_ring_complete": layout_consistency.pad_ring_complete,
            "power_mesh_present": layout_consistency.power_mesh_present,
            "hotspot_keepout_present": layout_consistency.hotspot_keepout_present,
            "logical_schedule_labeled": layout_consistency.logical_schedule_labeled,
            "missing_items": layout_consistency.missing_items,
        },
        message=f"Consistency closure: {'PASS' if layout_consistency.closure_pass else 'FAIL'}" + (f" (missing: {', '.join(layout_consistency.missing_items)})" if layout_consistency.missing_items else ""),
    )


def build_area_agreement_gate(
    metrics: SimulationMetrics,
    layout: LayoutBundle,
    tolerance_mm2: float = 0.05,
) -> SignoffGate:
    """Build a gate checking metrics die area vs layout die area agreement."""
    layout_die_area = float(layout.die_width_um * layout.die_height_um / 1.0e6)
    metrics_die_area = float(metrics.die_area_mm2)
    delta = float(abs(layout_die_area - metrics_die_area))
    passed = delta <= tolerance_mm2
    return SignoffGate(
        gate_id="gate_area_agreement",
        gate_name="Die Area Agreement",
        category="design_vs_layout",
        passed=passed,
        severity="critical",
        timestamp=_now_iso(),
        details={
            "metrics_die_area_mm2": metrics_die_area,
            "layout_die_area_mm2": layout_die_area,
            "area_delta_mm2": delta,
            "tolerance_mm2": tolerance_mm2,
        },
        message=f"Area delta {delta:.6f} mm² {'≤' if passed else '>'} tolerance {tolerance_mm2} mm²",
    )


def build_lvs_gate(layout: LayoutBundle) -> SignoffGate:
    """Layout-vs-Schematic light check: verify expected cell hierarchy complexity."""
    rect_count = int(layout.stats.get("rect_count", 0))
    route_count = int(layout.stats.get("route_count", 0))
    circle_count = int(layout.stats.get("circle_count", 0))
    total_elements = rect_count + route_count + circle_count
    # A non-trivial signoff-grade layout should have significant elements
    min_elements = 200
    passed = total_elements >= min_elements
    return SignoffGate(
        gate_id="gate_lvs",
        gate_name="Layout vs Schematic (Light)",
        category="integrity",
        passed=passed,
        severity="major",
        timestamp=_now_iso(),
        details={
            "rect_count": rect_count,
            "route_count": route_count,
            "circle_count": circle_count,
            "total_elements": total_elements,
            "min_elements": min_elements,
        },
        message=f"LVS: {total_elements} layout elements {'≥' if passed else '<'} minimum {min_elements}",
    )


def build_qec_gate(qec_plan: QECPlan | None, spec: DesignSpec) -> SignoffGate:
    """Build a gate checking QEC plan compliance with spec."""
    if qec_plan is None or not qec_plan.enabled:
        return SignoffGate(
            gate_id="gate_qec",
            gate_name="QEC Plan Compliance",
            category="consistency",
            passed=not spec.qec_enabled,
            severity="major" if spec.qec_enabled else "minor",
            timestamp=_now_iso(),
            details={"qec_enabled_in_spec": spec.qec_enabled, "qec_plan_present": qec_plan is not None},
            message="QEC plan absent" + (" (spec requires QEC)" if spec.qec_enabled else " (QEC not required)"),
        )
    logical_ok = qec_plan.achievable_logical_qubits >= spec.target_logical_qubits
    error_ok = qec_plan.logical_error_rate <= spec.target_logical_error_rate
    passed = logical_ok and error_ok
    return SignoffGate(
        gate_id="gate_qec",
        gate_name="QEC Plan Compliance",
        category="consistency",
        passed=passed,
        severity="critical",
        timestamp=_now_iso(),
        details={
            "achievable_logical_qubits": qec_plan.achievable_logical_qubits,
            "target_logical_qubits": spec.target_logical_qubits,
            "logical_ok": logical_ok,
            "logical_error_rate": qec_plan.logical_error_rate,
            "target_logical_error_rate": spec.target_logical_error_rate,
            "error_ok": error_ok,
        },
        message=f"QEC: logical qubits {'OK' if logical_ok else 'FAIL'}, error rate {'OK' if error_ok else 'FAIL'}",
    )


# ---------------------------------------------------------------------------
# Signoff Report - Aggregate of all gates
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SignoffReport:
    """Comprehensive signoff report aggregating all gates."""
    run_id: str
    signoff_version: str
    timestamp: str
    gates: list[SignoffGate] = field(default_factory=list)
    all_gates_pass: bool = False
    critical_gates_pass: bool = False
    handoff_authorized: bool = False
    gate_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "signoff_version": self.signoff_version,
            "timestamp": self.timestamp,
            "all_gates_pass": self.all_gates_pass,
            "critical_gates_pass": self.critical_gates_pass,
            "handoff_authorized": self.handoff_authorized,
            "gate_summary": self.gate_summary,
            "gates": [g.to_dict() for g in self.gates],
        }


def build_signoff_report(
    run_id: str,
    gates: list[SignoffGate],
) -> SignoffReport:
    """Build the aggregate signoff report from individual gates."""
    all_pass = all(g.passed for g in gates)
    critical_pass = all(g.passed for g in gates if g.severity == "critical")
    # Handoff is authorized only when all critical gates pass
    handoff_authorized = critical_pass

    summary: dict[str, int] = {"total": len(gates), "passed": 0, "failed": 0}
    for gate in gates:
        if gate.passed:
            summary["passed"] += 1
        else:
            summary["failed"] += 1

    return SignoffReport(
        run_id=run_id,
        signoff_version=SIGNOFF_VERSION,
        timestamp=_now_iso(),
        gates=gates,
        all_gates_pass=all_pass,
        critical_gates_pass=critical_pass,
        handoff_authorized=handoff_authorized,
        gate_summary=summary,
    )


def write_signoff_report(output_dir: Path, report: SignoffReport) -> dict[str, str]:
    """Write the signoff report to disk."""
    path = output_dir / "signoff_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=_json_default), encoding="utf-8")
    return {"signoff_report": str(path)}


# ---------------------------------------------------------------------------
# Original SignoffManifest — Enhanced
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SignoffManifest:
    run_id: str
    candidate_key: str
    report_path: str
    gds_path: str
    oas_path: str
    layout_json_path: str
    qec_plan_path: str | None
    hierarchy_path: str | None
    report_sha256: str
    gds_sha256: str
    layout_json_sha256: str
    report_references_run_id: bool
    report_references_gds: bool
    report_references_candidate_key: bool
    report_references_die_area: bool
    hierarchy_cell_count: int
    hierarchy_reference_count: int
    metrics_die_area_mm2: float
    layout_die_area_mm2: float
    area_delta_mm2: float
    logical_patch_count: int
    closure_pass: bool
    handoff_ready: bool
    artifact_registry_path: str | None = None
    handoff_bundle_path: str | None = None
    signoff_version: str = SIGNOFF_VERSION
    timestamp: str = ""
    gate_count: int = 0
    gates_passed: int = 0
    gates_failed: int = 0
    critical_gates_pass: bool = False
    handoff_authorized: bool = False
    drc_all_pass: bool = False
    design_vs_layout_match: bool = False
    signoff_report_path: str | None = None
    handoff_quarantined: bool = False
    quarantine_flag: str = ""
    signoff_schema_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_id(spec: DesignSpec, candidate: CandidateDesign, metrics: SimulationMetrics, seed: int) -> str:
    payload = json.dumps(
        {
            "design_name": spec.design_name,
            "application": spec.application,
            "target_qubits": spec.target_qubits,
            "target_logical_qubits": spec.target_logical_qubits,
            "seed": seed,
            "candidate": candidate.to_dict(),
            "gate_fidelity": metrics.gate_fidelity,
            "latency_ns": metrics.latency_ns,
        },
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_signoff_manifest(
    output_dir: Path,
    run_id: str,
    spec: DesignSpec,
    candidate: CandidateDesign,
    metrics: SimulationMetrics,
    layout: LayoutBundle,
    qec_plan: QECPlan | None,
    layout_consistency: LayoutConsistencySummary,
    report_path: Path,
    layout_paths: dict[str, str],
    artifact_registry_path: str | None = None,
    handoff_bundle_path: str | None = None,
    signoff_report: SignoffReport | None = None,
    drc_summary: DRCSummary | None = None,
    dvl_result: DesignVsLayoutResult | None = None,
    handoff_quarantined: bool = False,
    quarantine_flag: str = "",
    signoff_schema_path: str | None = None,
) -> SignoffManifest:
    gds_path = Path(layout_paths["gds"])
    oas_path = Path(layout_paths.get("oas", "")) if layout_paths.get("oas") else None
    layout_json_path = Path(layout_paths["layout_json"])
    hierarchy_path = Path(layout_paths["gds_hierarchy"]) if layout_paths.get("gds_hierarchy") else None
    report_text = report_path.read_text(encoding="utf-8")
    hierarchy_payload = json.loads(hierarchy_path.read_text(encoding="utf-8")) if hierarchy_path and hierarchy_path.exists() else {}
    candidate_key = f"{candidate.architecture}_q{candidate.qubits}_p{candidate.cell_pitch_um:.2f}_ob{candidate.optical_bus_count}_mw{candidate.microwave_line_count}"
    layout_die_area = float(layout.die_width_um * layout.die_height_um / 1.0e6)
    metrics_die_area = float(metrics.die_area_mm2)
    area_delta = float(abs(layout_die_area - metrics_die_area))
    report_references_run_id = bool(run_id in report_text)
    report_references_gds = bool(layout_paths["gds"] in report_text)
    report_references_candidate_key = bool(candidate_key in report_text)
    report_references_die_area = bool(f"{metrics_die_area:.2f}" in report_text or f"{layout_die_area:.2f}" in report_text)
    handoff_ready = bool(
        layout_consistency.closure_pass
        and report_references_run_id
        and report_references_gds
        and report_references_candidate_key
        and report_references_die_area
        and area_delta <= 0.05
    )

    # Enhanced fields from signoff report and DRC
    sr_gate_count = 0
    sr_gates_passed = 0
    sr_gates_failed = 0
    sr_critical_pass = False
    sr_handoff_authorized = False
    sr_path: str | None = None
    if signoff_report is not None:
        sr_gate_count = signoff_report.gate_summary.get("total", 0)
        sr_gates_passed = signoff_report.gate_summary.get("passed", 0)
        sr_gates_failed = signoff_report.gate_summary.get("failed", 0)
        sr_critical_pass = signoff_report.critical_gates_pass
        sr_handoff_authorized = signoff_report.handoff_authorized
        # If signoff report was written, reference it
        sr_path_candidate = output_dir / "signoff_report.json"
        if sr_path_candidate.exists():
            sr_path = str(sr_path_candidate)

    drc_pass = drc_summary.all_pass if drc_summary else False
    dvl_match = dvl_result.all_match if dvl_result else False

    # Final handoff_authorized: both legacy handoff_ready and new gate system must agree
    final_handoff_authorized = handoff_ready and sr_handoff_authorized if signoff_report else handoff_ready

    return SignoffManifest(
        run_id=run_id,
        candidate_key=candidate_key,
        report_path=str(report_path),
        gds_path=str(gds_path),
        oas_path=str(oas_path) if oas_path else "",
        layout_json_path=str(layout_json_path),
        qec_plan_path=str(output_dir / "qec_plan.json") if qec_plan else None,
        hierarchy_path=str(hierarchy_path) if hierarchy_path else None,
        report_sha256=_sha256(report_path),
        gds_sha256=_sha256(gds_path),
        layout_json_sha256=_sha256(layout_json_path),
        report_references_run_id=report_references_run_id,
        report_references_gds=report_references_gds,
        report_references_candidate_key=report_references_candidate_key,
        report_references_die_area=report_references_die_area,
        hierarchy_cell_count=int(hierarchy_payload.get("cell_count", 0)),
        hierarchy_reference_count=int(hierarchy_payload.get("top_references", 0)),
        artifact_registry_path=artifact_registry_path,
        handoff_bundle_path=handoff_bundle_path,
        metrics_die_area_mm2=metrics_die_area,
        layout_die_area_mm2=layout_die_area,
        area_delta_mm2=area_delta,
        logical_patch_count=len(qec_plan.patches) if qec_plan else 0,
        closure_pass=bool(layout_consistency.closure_pass),
        handoff_ready=handoff_ready,
        signoff_version=SIGNOFF_VERSION,
        timestamp=_now_iso(),
        gate_count=sr_gate_count,
        gates_passed=sr_gates_passed,
        gates_failed=sr_gates_failed,
        critical_gates_pass=sr_critical_pass,
        handoff_authorized=final_handoff_authorized,
        drc_all_pass=drc_pass,
        design_vs_layout_match=dvl_match,
        signoff_report_path=sr_path,
        handoff_quarantined=handoff_quarantined,
        quarantine_flag=quarantine_flag,
        signoff_schema_path=signoff_schema_path,
    )


def write_signoff_manifest(output_dir: Path, manifest: SignoffManifest) -> dict[str, str]:
    json_path = output_dir / "signoff_manifest.json"
    json_path.write_text(json.dumps(manifest.to_dict(), indent=2, default=_json_default), encoding="utf-8")
    return {"signoff_manifest": str(json_path)}
