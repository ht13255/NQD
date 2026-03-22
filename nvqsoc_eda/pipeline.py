from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .advanced_sim import analyze_layout_parasitics, refine_frontier_with_open_source_sims
from .artifact_manager import build_artifact_registry, create_handoff_bundle, write_artifact_registry, write_handoff_bundle
from .dense_placement import build_probabilistic_placement, simulate_dense_crosstalk
from .layout import generate_layout, write_layout_artifacts
from .layout_validation import validate_layout_consistency, write_layout_consistency_artifacts
from .optimizer import optimize_design
from .papers import build_paper_knowledge, write_paper_artifacts
from .qec import write_qec_artifacts
from .requirements import RequirementBundle, write_requirement_artifacts
from .reporting import build_report, write_report
from .simulator import run_monte_carlo
from .signoff import build_signoff_manifest, make_run_id, write_signoff_manifest
from .spec import DesignSpec
from .topology import build_topology_plan, write_topology_artifacts


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dump(payload: Any, *, ensure_ascii: bool = True) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=ensure_ascii, default=_json_default)


def _json_safe(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=_json_default))


def run_pipeline(
    spec_path: str | Path | None,
    output_dir: str | Path,
    spec_data: dict[str, Any] | None = None,
    seed: int = 7,
    monte_carlo_trials: int = 256,
    generations: int = 7,
    beam_width: int = 8,
    mutations_per_parent: int = 7,
    crawl_papers: bool = False,
    paper_limit: int = 6,
    advanced_top_k: int = 3,
    requirements_bundle: RequirementBundle | None = None,
) -> dict[str, Any]:
    def _candidate_key(candidate: Any) -> str:
        return (
            f"{candidate.architecture}_q{candidate.qubits}_p{candidate.cell_pitch_um:.2f}"
            f"_ob{candidate.optical_bus_count}_mw{candidate.microwave_line_count}"
        )

    if spec_data is not None:
        spec = DesignSpec.from_dict(spec_data)
    elif spec_path is not None:
        spec = DesignSpec.from_json(spec_path)
    else:
        raise ValueError("spec_path or spec_data must be provided")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "normalized_spec.json").write_text(_json_dump(spec.to_dict()), encoding="utf-8")
    requirement_artifacts = write_requirement_artifacts(out, requirements_bundle) if requirements_bundle else {}
    paper_knowledge = build_paper_knowledge(spec, max_papers=paper_limit) if crawl_papers else None
    paper_artifacts = write_paper_artifacts(out, paper_knowledge) if paper_knowledge else {}

    result = optimize_design(
        spec,
        seed=seed,
        generations=generations,
        beam_width=beam_width,
        mutations_per_parent=mutations_per_parent,
        monte_carlo_trials=monte_carlo_trials,
        paper_knowledge=paper_knowledge,
        requirements_bundle=requirements_bundle,
    )
    advanced_dir = out / "advanced"
    advanced = refine_frontier_with_open_source_sims(
        spec,
        result.pareto_like_frontier,
        advanced_dir,
        seed=seed,
        top_k=advanced_top_k,
        paper_knowledge=paper_knowledge,
        requirements_bundle=requirements_bundle,
    )

    selected_ranked = next(
        (item for item in result.pareto_like_frontier if _candidate_key(item.candidate) == advanced.selected_candidate_key),
        result.pareto_like_frontier[0],
    )
    final_candidate = selected_ranked.candidate
    final_metrics = selected_ranked.metrics
    final_qec_plan = selected_ranked.qec_plan
    final_monte_carlo = run_monte_carlo(spec, final_candidate, trials=monte_carlo_trials, seed=seed + 4096)
    final_placement = build_probabilistic_placement(spec, final_candidate, samples=32, seed_offset=seed + 5000)
    final_dense_crosstalk = simulate_dense_crosstalk(spec, final_candidate, final_placement, trials=192, seed_offset=seed + 5000)
    topology_plan = build_topology_plan(
        spec,
        final_candidate,
        final_metrics,
        dense_signals=selected_ranked.dense_signals,
        placement=final_placement,
        dense_crosstalk=final_dense_crosstalk,
        neural_signals=selected_ranked.neural_signals,
        paper_knowledge=paper_knowledge,
        requirements_bundle=requirements_bundle,
    )

    layout = generate_layout(spec, final_candidate, final_metrics, topology_plan=topology_plan, placement=final_placement, qec_plan=final_qec_plan)
    layout_paths = write_layout_artifacts(out, spec, final_candidate, final_metrics, layout)
    topology_artifacts = write_topology_artifacts(out, topology_plan)
    qec_artifacts = write_qec_artifacts(out, final_qec_plan) if final_qec_plan else {}
    layout_parasitics = analyze_layout_parasitics(layout)
    layout_consistency = validate_layout_consistency(layout, topology_plan, final_qec_plan)
    consistency_artifacts = write_layout_consistency_artifacts(out, layout_consistency)
    (out / "layout_parasitics.json").write_text(_json_dump(layout_parasitics.to_dict()), encoding="utf-8")
    run_id = make_run_id(spec, final_candidate, final_metrics, seed)
    report_path = write_report(
        out,
        build_report(
            spec,
            result,
            layout_paths,
            run_id=run_id,
            selected_candidate=final_candidate,
            selected_metrics=final_metrics,
            selected_monte_carlo=final_monte_carlo,
            selected_qec_plan=final_qec_plan,
            requirements_bundle=requirements_bundle,
            paper_knowledge=paper_knowledge,
            advanced_refinement=advanced,
            layout_parasitics=layout_parasitics,
            layout_consistency=layout_consistency,
        ),
    )
    base_artifact_paths = {
        **layout_paths,
        **topology_artifacts,
        **qec_artifacts,
        **consistency_artifacts,
        **requirement_artifacts,
        **paper_artifacts,
        "report": report_path,
        "normalized_spec": str(out / "normalized_spec.json"),
        "layout_parasitics": str(out / "layout_parasitics.json"),
    }
    artifact_registry = build_artifact_registry(run_id, out, base_artifact_paths)
    registry_artifacts = write_artifact_registry(out, artifact_registry)
    handoff_bundle = create_handoff_bundle(run_id, out, artifact_registry)
    handoff_artifacts = write_handoff_bundle(out, handoff_bundle)
    signoff_manifest = build_signoff_manifest(
        out,
        run_id,
        spec,
        final_candidate,
        final_metrics,
        layout,
        final_qec_plan,
        layout_consistency,
        Path(report_path),
        layout_paths,
        artifact_registry_path=registry_artifacts["artifact_registry"],
        handoff_bundle_path=handoff_artifacts["handoff_bundle"],
    )
    report_path = write_report(
        out,
        build_report(
            spec,
            result,
            layout_paths,
            run_id=run_id,
            selected_candidate=final_candidate,
            selected_metrics=final_metrics,
            selected_monte_carlo=final_monte_carlo,
            selected_qec_plan=final_qec_plan,
            requirements_bundle=requirements_bundle,
            paper_knowledge=paper_knowledge,
            advanced_refinement=advanced,
            layout_parasitics=layout_parasitics,
            layout_consistency=layout_consistency,
            signoff_manifest=signoff_manifest,
            handoff_bundle=handoff_bundle.to_dict(),
        ),
    )
    base_artifact_paths["report"] = report_path
    artifact_registry = build_artifact_registry(run_id, out, base_artifact_paths | registry_artifacts | handoff_artifacts)
    handoff_bundle = create_handoff_bundle(run_id, out, artifact_registry)
    registry_artifacts = write_artifact_registry(out, artifact_registry)
    handoff_artifacts = write_handoff_bundle(out, handoff_bundle)
    signoff_manifest = build_signoff_manifest(
        out,
        run_id,
        spec,
        final_candidate,
        final_metrics,
        layout,
        final_qec_plan,
        layout_consistency,
        Path(report_path),
        layout_paths,
        artifact_registry_path=registry_artifacts["artifact_registry"],
        handoff_bundle_path=handoff_artifacts["handoff_bundle"],
    )
    signoff_artifacts = write_signoff_manifest(out, signoff_manifest)
    optimization_path = out / "optimization.json"
    optimization_payload = _json_safe(
        {
            "optimization": result.to_dict(),
            "paper_knowledge": paper_knowledge.to_dict() if paper_knowledge else None,
            "requirements_bundle": requirements_bundle.to_dict() if requirements_bundle else None,
            "advanced_refinement": advanced.to_dict(),
            "selected_candidate_key": advanced.selected_candidate_key,
            "topology_plan": topology_plan.to_dict(),
            "qec_plan": final_qec_plan.to_dict() if final_qec_plan else None,
            "world_model_signals": selected_ranked.world_model_signals.to_dict() if selected_ranked.world_model_signals else None,
            "robust_signals": selected_ranked.robust_signals.to_dict() if selected_ranked.robust_signals else None,
            "layout_parasitics": layout_parasitics.to_dict(),
            "layout_consistency": layout_consistency.to_dict(),
            "run_id": run_id,
            "signoff_manifest": signoff_manifest.to_dict(),
            "artifact_registry": artifact_registry.to_dict(),
            "handoff_bundle": handoff_bundle.to_dict(),
        }
    )
    optimization_path.write_text(_json_dump(optimization_payload), encoding="utf-8")

    summary = _json_safe({
        "design_name": spec.design_name,
        "run_id": run_id,
        "output_dir": str(out),
        "best_candidate": final_candidate.to_dict(),
        "best_metrics": final_metrics.to_dict(),
        "best_dense_signals": selected_ranked.dense_signals.to_dict() if selected_ranked.dense_signals else None,
        "best_neural_signals": selected_ranked.neural_signals.to_dict() if selected_ranked.neural_signals else None,
        "best_qec_plan": final_qec_plan.to_dict() if final_qec_plan else None,
        "best_world_model_signals": selected_ranked.world_model_signals.to_dict() if selected_ranked.world_model_signals else None,
        "best_robust_signals": selected_ranked.robust_signals.to_dict() if selected_ranked.robust_signals else None,
        "monte_carlo": final_monte_carlo.to_dict(),
        "requirements_bundle": requirements_bundle.to_dict() if requirements_bundle else None,
        "paper_knowledge": paper_knowledge.to_dict() if paper_knowledge else None,
        "advanced_refinement": advanced.to_dict(),
        "topology_plan": topology_plan.to_dict(),
        "layout_parasitics": layout_parasitics.to_dict(),
        "layout_consistency": layout_consistency.to_dict(),
        "signoff_manifest": signoff_manifest.to_dict(),
        "artifacts": {
            **layout_paths,
            **topology_artifacts,
            **qec_artifacts,
            **consistency_artifacts,
            **registry_artifacts,
            **handoff_artifacts,
            **signoff_artifacts,
            **requirement_artifacts,
            **paper_artifacts,
            "report": report_path,
            "optimization": str(optimization_path),
            "normalized_spec": str(out / "normalized_spec.json"),
            "layout_parasitics": str(out / "layout_parasitics.json"),
        },
    })
    (out / "summary.json").write_text(_json_dump(summary), encoding="utf-8")
    return summary
