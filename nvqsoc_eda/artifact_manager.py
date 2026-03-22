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


def create_handoff_bundle(run_id: str, output_dir: Path, registry: ArtifactRegistry) -> HandoffBundle:
    package_dir = output_dir / "handoff"
    package_dir.mkdir(parents=True, exist_ok=True)
    immutable_artifacts: dict[str, str] = {}
    selected_roles = {"report", "gds", "oas", "layout_json", "qec_plan", "topology_plan", "signoff_manifest", "artifact_registry"}
    for record in registry.records:
        if record.role not in selected_roles:
            continue
        source = Path(record.path)
        immutable_name = f"{run_id}__{source.name}"
        destination = package_dir / immutable_name
        shutil.copy2(source, destination)
        immutable_artifacts[record.artifact_id] = str(destination)
        record.immutable_copy = str(destination)

    package_manifest_path = package_dir / f"{run_id}__handoff_manifest.json"
    package_payload = {
        "run_id": run_id,
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
    )


def write_handoff_bundle(output_dir: Path, bundle: HandoffBundle) -> dict[str, str]:
    path = output_dir / "handoff_bundle.json"
    path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    return {"handoff_bundle": str(path)}
