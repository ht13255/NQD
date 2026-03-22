from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .layout import LayoutBundle
from .layout_validation import LayoutConsistencySummary
from .qec import QECPlan
from .simulator import CandidateDesign, SimulationMetrics
from .spec import DesignSpec


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
    )


def write_signoff_manifest(output_dir: Path, manifest: SignoffManifest) -> dict[str, str]:
    json_path = output_dir / "signoff_manifest.json"
    json_path.write_text(json.dumps(manifest.to_dict(), indent=2, default=_json_default), encoding="utf-8")
    return {"signoff_manifest": str(json_path)}
