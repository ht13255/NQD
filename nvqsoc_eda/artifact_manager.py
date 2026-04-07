from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    run_id: str
    role: str
    stage: str
    path: str
    file_name: str
    extension: str
    sha256: str
    size_bytes: int
    immutable_copy: str | None = None
    producer: str = "nvqsoc_eda"
    related_artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArtifactRegistry:
    run_id: str
    output_dir: str
    records: list[ArtifactRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(slots=True)
class HandoffBundle:
    run_id: str
    package_dir: str
    package_manifest: str
    immutable_artifacts: dict[str, str]
    package_sha256: str
    quarantined: bool = False
    quarantine_flag: str = ""
    quarantine_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_for_name(name: str) -> tuple[str, str]:
    lowered = name.lower()
    mapping = [
        ("design_report.md", ("report", "reporting")),
        ("layout.gds", ("gds", "layout_export")),
        ("layout.oas", ("oas", "layout_export")),
        ("layout.json", ("layout_json", "layout_export")),
        ("layout.svg", ("layout_svg", "layout_export")),
        ("layout_preview.png", ("layout_preview", "layout_export")),
        ("layout_klayout.py", ("klayout_script", "layout_export")),
        ("layer_map.json", ("layer_map", "layout_export")),
        ("gds_hierarchy.json", ("gds_hierarchy", "layout_export")),
        ("topology_plan.json", ("topology_plan", "topology")),
        ("qec_plan.json", ("qec_plan", "qec")),
        ("layout_consistency.json", ("layout_consistency", "signoff")),
        ("layout_parasitics.json", ("layout_parasitics", "signoff")),
        ("signoff_schema.json", ("signoff_schema", "signoff")),
        ("signoff_manifest.json", ("signoff_manifest", "signoff")),
        ("optimization.json", ("optimization", "optimization")),
        ("normalized_spec.json", ("normalized_spec", "input")),
        ("summary.json", ("summary", "signoff")),
        ("requirements_analysis.json", ("requirements", "input")),
        ("paper_knowledge.json", ("paper_knowledge", "research")),
    ]
    for key, result in mapping:
        if lowered.endswith(key):
            return result
    if "advanced" in lowered:
        return "advanced_sim", "advanced"
    return "artifact", "misc"


def build_artifact_registry(run_id: str, output_dir: Path, artifact_paths: dict[str, str]) -> ArtifactRegistry:
    records: list[ArtifactRecord] = []
    for alias, value in sorted(artifact_paths.items()):
        if not value:
            continue
        path = Path(value)
        if not path.exists() or path.is_dir():
            continue
        role, stage = _role_for_name(path.name)
        record = ArtifactRecord(
            artifact_id=alias,
            run_id=run_id,
            role=role,
            stage=stage,
            path=str(path),
            file_name=path.name,
            extension=path.suffix.lower(),
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
        )
        records.append(record)

    id_to_record = {record.artifact_id: record for record in records}
    relations = {
        "report": ["gds", "layout_json", "qec_plan", "topology_plan", "signoff_manifest"],
        "gds": ["layout_json", "layer_map", "gds_hierarchy"],
        "optimization": ["requirements_json", "paper_knowledge_json", "qec_plan"],
    }
    for alias, linked in relations.items():
        record = id_to_record.get(alias)
        if record is None:
            continue
        record.related_artifacts = [item for item in linked if item in id_to_record]
    return ArtifactRegistry(run_id=run_id, output_dir=str(output_dir), records=records)


def write_artifact_registry(output_dir: Path, registry: ArtifactRegistry) -> dict[str, str]:
    path = output_dir / "artifact_registry.json"
    path.write_text(json.dumps(registry.to_dict(), indent=2), encoding="utf-8")
    return {"artifact_registry": str(path)}


def create_handoff_bundle(
    run_id: str,
    output_dir: Path,
    registry: ArtifactRegistry,
    *,
    quarantined: bool = False,
    quarantine_reason: str = "",
) -> HandoffBundle:
    package_dir = output_dir / ("QUARANTINE" if quarantined else "handoff")
    package_dir.mkdir(parents=True, exist_ok=True)
    immutable_artifacts: dict[str, str] = {}
    selected_roles = {"report", "gds", "oas", "layout_json", "qec_plan", "topology_plan", "signoff_manifest", "artifact_registry"}
    for record in registry.records:
        if record.role not in selected_roles:
            continue
        source = Path(record.path)
        blocked_suffix = "__BLOCKED" if quarantined else ""
        immutable_name = f"{run_id}__{source.stem}{blocked_suffix}{source.suffix}"
        destination = package_dir / immutable_name
        shutil.copy2(source, destination)
        immutable_artifacts[record.artifact_id] = str(destination)
        record.immutable_copy = str(destination)

    manifest_suffix = "__BLOCKED" if quarantined else ""
    package_manifest_path = package_dir / f"{run_id}__handoff_manifest{manifest_suffix}.json"
    package_payload = {
        "run_id": run_id,
        "quarantined": quarantined,
        "quarantine_flag": "QUARANTINE" if quarantined else "",
        "quarantine_reason": quarantine_reason,
        "blocked_suffix": "BLOCKED" if quarantined else "",
        "immutable_artifacts": immutable_artifacts,
        "registry": registry.to_dict(),
    }
    package_manifest_path.write_text(json.dumps(package_payload, indent=2), encoding="utf-8")
    package_sha = _sha256(package_manifest_path)
    return HandoffBundle(
        run_id=run_id,
        package_dir=str(package_dir),
        package_manifest=str(package_manifest_path),
        immutable_artifacts=immutable_artifacts,
        package_sha256=package_sha,
        quarantined=quarantined,
        quarantine_flag="QUARANTINE" if quarantined else "",
        quarantine_reason=quarantine_reason,
    )


def write_handoff_bundle(output_dir: Path, bundle: HandoffBundle) -> dict[str, str]:
    path = output_dir / "handoff_bundle.json"
    path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    return {"handoff_bundle": str(path)}


# ---------------------------------------------------------------------------
# Bundle Integrity & Completeness Verification
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BundleAuditResult:
    """Result of integrity and completeness verification of a handoff bundle."""
    integrity_pass: bool
    completeness_pass: bool
    overall_pass: bool
    timestamp: str
    sha_mismatches: list[dict[str, str]] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    present_artifacts: list[str] = field(default_factory=list)
    file_not_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_bundle_integrity(bundle: HandoffBundle, registry: ArtifactRegistry) -> BundleAuditResult:
    """Re-compute SHA256 for all immutable artifacts and verify against registry."""
    from datetime import datetime, timezone

    sha_mismatches: list[dict[str, str]] = []
    file_not_found: list[str] = []

    # Build lookup from registry
    registry_sha: dict[str, str] = {}
    for record in registry.records:
        if record.immutable_copy:
            registry_sha[record.immutable_copy] = record.sha256
        registry_sha[record.path] = record.sha256

    for alias, immutable_path_str in bundle.immutable_artifacts.items():
        immutable_path = Path(immutable_path_str)
        if not immutable_path.exists():
            file_not_found.append(alias)
            continue
        actual_sha = _sha256(immutable_path)
        expected_sha = registry_sha.get(immutable_path_str)
        if expected_sha is not None and actual_sha != expected_sha:
            sha_mismatches.append({
                "artifact_id": alias,
                "path": immutable_path_str,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            })

    integrity_pass = len(sha_mismatches) == 0 and len(file_not_found) == 0

    return BundleAuditResult(
        integrity_pass=integrity_pass,
        completeness_pass=True,  # Will be filled by check_bundle_completeness
        overall_pass=integrity_pass,
        timestamp=datetime.now(timezone.utc).isoformat(),
        sha_mismatches=sha_mismatches,
        file_not_found=file_not_found,
    )


def check_bundle_completeness(bundle: HandoffBundle, registry: ArtifactRegistry) -> BundleAuditResult:
    """Verify that all required roles are present in the handoff bundle."""
    from datetime import datetime, timezone

    required_roles = {"report", "gds", "layout_json", "signoff_manifest"}
    optional_roles = {"oas", "qec_plan", "topology_plan", "artifact_registry"}

    present_roles: set[str] = set()
    for record in registry.records:
        if record.role in required_roles or record.role in optional_roles:
            # Check if this artifact is in the immutable bundle
            if record.artifact_id in bundle.immutable_artifacts:
                present_roles.add(record.role)

    missing = [role for role in sorted(required_roles) if role not in present_roles]
    present = sorted(present_roles)
    completeness_pass = len(missing) == 0

    return BundleAuditResult(
        integrity_pass=True,  # Not checked here
        completeness_pass=completeness_pass,
        overall_pass=completeness_pass,
        timestamp=datetime.now(timezone.utc).isoformat(),
        missing_artifacts=missing,
        present_artifacts=present,
    )


def audit_handoff_bundle(bundle: HandoffBundle, registry: ArtifactRegistry) -> BundleAuditResult:
    """Full audit: integrity + completeness combined."""
    integrity = verify_bundle_integrity(bundle, registry)
    completeness = check_bundle_completeness(bundle, registry)

    overall = integrity.integrity_pass and completeness.completeness_pass

    return BundleAuditResult(
        integrity_pass=integrity.integrity_pass,
        completeness_pass=completeness.completeness_pass,
        overall_pass=overall,
        timestamp=integrity.timestamp,
        sha_mismatches=integrity.sha_mismatches,
        missing_artifacts=completeness.missing_artifacts,
        present_artifacts=completeness.present_artifacts,
        file_not_found=integrity.file_not_found,
    )


def write_bundle_audit(output_dir: Path, audit: BundleAuditResult) -> dict[str, str]:
    """Write the bundle audit result to disk."""
    path = output_dir / "bundle_audit.json"
    path.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    return {"bundle_audit": str(path)}
