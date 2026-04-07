from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .advanced_sim import analyze_layout_parasitics, refine_frontier_with_open_source_sims
from .artifact_manager import (
    audit_handoff_bundle,
    build_artifact_registry,
    create_handoff_bundle,
    write_artifact_registry,
    write_bundle_audit,
    write_handoff_bundle,
)
from .dense_placement import build_probabilistic_placement, simulate_dense_crosstalk
from .layout import generate_layout, write_layout_artifacts
from .layout_validation import (
    run_drc_checks,
    validate_design_vs_layout,
    validate_layout_consistency,
    write_drc_report,
    write_layout_consistency_artifacts,
)
from .optimizer import OptimizationResult, RankedCandidate, SeedFeedback, build_qec_escalation_seed, optimize_design
from .compute_backend import resolve_compute_backend
from .papers import build_paper_knowledge, write_paper_artifacts
from .qec import estimate_qec_feasibility_precheck, write_qec_artifacts
from .requirements import RequirementBundle, write_requirement_artifacts
from .reporting import build_report, write_report
from .simulator import run_monte_carlo
from .signoff import (
    build_area_agreement_gate,
    build_consistency_gate,
    build_design_vs_layout_gate,
    build_drc_gate,
    build_lvs_gate,
    build_qec_gate,
    build_signoff_manifest,
    build_signoff_report,
    make_run_id,
    write_signoff_manifest,
    write_signoff_report,
)
from .signoff_schema import write_signoff_schema
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


def _candidate_key(candidate: Any) -> str:
    return (
        f"{candidate.architecture}_q{candidate.qubits}_p{candidate.cell_pitch_um:.2f}"
        f"_ob{candidate.optical_bus_count}_mw{candidate.microwave_line_count}"
    )


@dataclass(slots=True)
class QualityProfile:
    mode: str
    search_rounds: int
    frontier_limit: int
    layout_candidate_count: int
    topology_variant_count: int
    placement_samples: int
    dense_trials: int
    effective_monte_carlo_trials: int
    effective_generations: int
    effective_beam_width: int
    effective_mutations_per_parent: int
    effective_advanced_top_k: int
    effective_paper_limit: int
    auto_crawl_papers: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "search_rounds": self.search_rounds,
            "frontier_limit": self.frontier_limit,
            "layout_candidate_count": self.layout_candidate_count,
            "topology_variant_count": self.topology_variant_count,
            "placement_samples": self.placement_samples,
            "dense_trials": self.dense_trials,
            "effective_monte_carlo_trials": self.effective_monte_carlo_trials,
            "effective_generations": self.effective_generations,
            "effective_beam_width": self.effective_beam_width,
            "effective_mutations_per_parent": self.effective_mutations_per_parent,
            "effective_advanced_top_k": self.effective_advanced_top_k,
            "effective_paper_limit": self.effective_paper_limit,
            "auto_crawl_papers": self.auto_crawl_papers,
        }


def _canonical_quality_mode(quality_mode: str) -> str:
    lowered = (quality_mode or "standard").strip().lower()
    if lowered == "max":
        return "extreme"
    if lowered in {"standard", "high", "extreme"}:
        return lowered
    return "standard"


def _build_quality_profile(
    quality_mode: str,
    *,
    monte_carlo_trials: int,
    generations: int,
    beam_width: int,
    mutations_per_parent: int,
    paper_limit: int,
    advanced_top_k: int,
) -> QualityProfile:
    mode = _canonical_quality_mode(quality_mode)
    if mode == "extreme":
        effective_advanced_top_k = max(advanced_top_k, 6)
        effective_beam_width = max(beam_width, 14)
        return QualityProfile(
            mode=mode,
            search_rounds=4,
            frontier_limit=max(24, effective_advanced_top_k * 4, effective_beam_width * 2),
            layout_candidate_count=5,
            topology_variant_count=4,
            placement_samples=72,
            dense_trials=384,
            effective_monte_carlo_trials=max(monte_carlo_trials, 1536),
            effective_generations=max(generations, 16),
            effective_beam_width=effective_beam_width,
            effective_mutations_per_parent=max(mutations_per_parent, 14),
            effective_advanced_top_k=effective_advanced_top_k,
            effective_paper_limit=max(paper_limit, 10),
            auto_crawl_papers=True,
        )
    if mode == "high":
        effective_advanced_top_k = max(advanced_top_k, 4)
        effective_beam_width = max(beam_width, 10)
        return QualityProfile(
            mode=mode,
            search_rounds=2,
            frontier_limit=max(14, effective_advanced_top_k * 3, effective_beam_width * 2),
            layout_candidate_count=3,
            topology_variant_count=3,
            placement_samples=48,
            dense_trials=256,
            effective_monte_carlo_trials=max(monte_carlo_trials, 512),
            effective_generations=max(generations, 10),
            effective_beam_width=effective_beam_width,
            effective_mutations_per_parent=max(mutations_per_parent, 10),
            effective_advanced_top_k=effective_advanced_top_k,
            effective_paper_limit=max(paper_limit, 8),
            auto_crawl_papers=True,
        )
    return QualityProfile(
        mode="standard",
        search_rounds=1,
        frontier_limit=max(beam_width, advanced_top_k, 8),
        layout_candidate_count=1,
        topology_variant_count=1,
        placement_samples=32,
        dense_trials=192,
        effective_monte_carlo_trials=monte_carlo_trials,
        effective_generations=generations,
        effective_beam_width=beam_width,
        effective_mutations_per_parent=mutations_per_parent,
        effective_advanced_top_k=advanced_top_k,
        effective_paper_limit=paper_limit,
        auto_crawl_papers=False,
    )


def _merge_frontiers(results: list[OptimizationResult], frontier_limit: int) -> tuple[list[RankedCandidate], list[dict[str, Any]]]:
    merged: dict[str, RankedCandidate] = {}
    search_log: list[dict[str, Any]] = []
    for round_index, result in enumerate(results):
        for entry in result.search_log:
            search_log.append({"search_round": round_index, **entry})
        for ranked in result.pareto_like_frontier:
            key = _candidate_key(ranked.candidate)
            current = merged.get(key)
            if current is None or ranked.score > current.score:
                merged[key] = ranked
    combined = sorted(
        merged.values(),
        key=lambda item: (item.score, item.robustness_score, -len(item.metrics.constraint_violations), item.metrics.gate_fidelity),
        reverse=True,
    )
    return combined[:frontier_limit], search_log


def _fallback_ranked_candidate(results: list[OptimizationResult]) -> RankedCandidate | None:
    for result in reversed(results):
        if result.selected_ranked_candidate is not None:
            return result.selected_ranked_candidate
    return None


def _seed_feedback_from_payload(payload: dict[str, Any]) -> SeedFeedback | None:
    if not payload or not payload.get("active"):
        return None
    return SeedFeedback(
        recommended_qubit_delta=int(payload.get("recommended_qubit_delta", 0)),
        recommended_factory_lane_delta=int(payload.get("recommended_factory_lane_delta", 0)),
        recommended_pitch_delta_um=float(payload.get("recommended_pitch_delta_um", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        source_candidate_keys=list(payload.get("source_candidate_keys", [])),
    )


def _build_next_round_injected_seeds(result: OptimizationResult, spec: DesignSpec, seed: int) -> list[Any]:
    seeds: list[Any] = []
    candidate_pool = result.pareto_like_frontier[:2]
    if result.selected_ranked_candidate is not None:
        candidate_pool = [result.selected_ranked_candidate, *candidate_pool]
    seen: set[str] = set()
    for offset, ranked in enumerate(candidate_pool):
        if ranked.qec_plan is None:
            continue
        escalation_seed = build_qec_escalation_seed(ranked.candidate, ranked.qec_plan, spec, seed + 97 * (offset + 1))
        if escalation_seed is None:
            continue
        candidate_key = _candidate_key(escalation_seed)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        seeds.append(escalation_seed)
    return seeds


def _topology_variants(plan: Any, spec: DesignSpec, profile: QualityProfile) -> list[tuple[str, Any]]:
    variants = [("base", plan)]
    if profile.topology_variant_count >= 2:
        variants.append(
            (
                "guarded_ft",
                replace(
                    plan,
                    route_width_scale=plan.route_width_scale * 1.05,
                    optical_route_width_scale=plan.optical_route_width_scale * 1.08,
                    tile_guard_multiplier=plan.tile_guard_multiplier * 1.16,
                    keepout_margin_um=plan.keepout_margin_um * 1.14,
                    optical_redundancy=min(plan.optical_redundancy + 1, 4),
                    microwave_redundancy=min(plan.microwave_redundancy + 1, 3),
                    macro_segment_count=min(plan.macro_segment_count + 1, 10),
                    control_cluster_count=min(plan.control_cluster_count + 1, 8),
                    add_ground_moat=True,
                    reasoning=[*plan.reasoning, "quality variant: guarded fault-tolerant topology"],
                ),
            )
        )
    if profile.topology_variant_count >= 3:
        variants.append(
            (
                "latency_biased",
                replace(
                    plan,
                    control_topology="direct_fanout" if spec.max_latency_ns <= 220.0 else plan.control_topology,
                    readout_topology="short_tree",
                    route_width_scale=plan.route_width_scale * 1.10,
                    optical_route_width_scale=plan.optical_route_width_scale * 1.05,
                    power_mesh_pitch_um=max(60.0, plan.power_mesh_pitch_um * 0.92),
                    bus_escape_offset_um=max(20.0, plan.bus_escape_offset_um * 0.90),
                    reasoning=[*plan.reasoning, "quality variant: latency-biased topology"],
                ),
            )
        )
    if profile.topology_variant_count >= 4:
        variants.append(
            (
                "segmented_qec",
                replace(
                    plan,
                    optical_topology="wide_redundant",
                    control_topology="shielded_clusters",
                    readout_topology="segmented_tree",
                    optical_redundancy=min(plan.optical_redundancy + 1, 4),
                    microwave_redundancy=min(plan.microwave_redundancy + 1, 3),
                    detector_clusters=min(plan.detector_clusters + 1, 10),
                    macro_segment_count=min(plan.macro_segment_count + 2, 10),
                    control_cluster_count=min(plan.control_cluster_count + 1, 8),
                    add_ground_moat=True,
                    reasoning=[*plan.reasoning, "quality variant: segmented QEC-heavy topology"],
                ),
            )
        )
    return variants


def _layout_exploration_score(
    *,
    base_score: float,
    composite_score: float,
    layout_parasitics: Any,
    layout_consistency: Any,
    drc_summary: Any,
    dvl_result: Any,
    qec_plan: Any | None,
) -> float:
    qec_violation_count = len(qec_plan.violations) if qec_plan is not None else 0
    score = 0.55 * composite_score + 0.20 * base_score
    score += 0.16 * (drc_summary.passed_rules / max(drc_summary.total_rules, 1))
    score += 0.12 * (1.0 if dvl_result.all_match else max(0.0, 1.0 - len(dvl_result.mismatches) / max(dvl_result.checks_performed, 1)))
    score += 0.14 * (1.0 if layout_consistency.closure_pass else max(0.0, 1.0 - len(layout_consistency.missing_items) / 8.0))
    score += 0.08 / (1.0 + layout_parasitics.max_coupling_risk)
    score += 0.06 * min(layout_parasitics.min_feature_spacing_um / 12.0, 1.5)
    score += 0.05 / (1.0 + layout_parasitics.overlap_violations)
    if qec_plan is not None and not qec_plan.violations:
        score += 0.12
    penalties = 0.14 * drc_summary.failed_rules
    penalties += 0.12 * len(dvl_result.mismatches)
    penalties += 0.10 * len(layout_consistency.missing_items)
    penalties += 0.10 * layout_parasitics.overlap_violations
    penalties += 0.12 * qec_violation_count
    return score - penalties


def _explore_layout_variants(
    spec: DesignSpec,
    *,
    frontier_by_key: dict[str, RankedCandidate],
    advanced: Any,
    profile: QualityProfile,
    seed: int,
    paper_knowledge: Any | None,
    requirements_bundle: RequirementBundle | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    best_stage: dict[str, Any] | None = None
    for bundle_index, bundle in enumerate(advanced.bundles[: profile.layout_candidate_count]):
        ranked = frontier_by_key.get(bundle.candidate_key)
        if ranked is None:
            continue
        placement = build_probabilistic_placement(
            spec,
            ranked.candidate,
            samples=profile.placement_samples,
            seed_offset=seed + 7000 + 97 * bundle_index,
        )
        dense_crosstalk = simulate_dense_crosstalk(
            spec,
            ranked.candidate,
            placement,
            trials=profile.dense_trials,
            seed_offset=seed + 8000 + 131 * bundle_index,
        )
        base_topology = build_topology_plan(
            spec,
            ranked.candidate,
            ranked.metrics,
            dense_signals=ranked.dense_signals,
            placement=placement,
            dense_crosstalk=dense_crosstalk,
            neural_signals=ranked.neural_signals,
            paper_knowledge=paper_knowledge,
            requirements_bundle=requirements_bundle,
        )
        for variant_index, (variant_name, topology_plan) in enumerate(_topology_variants(base_topology, spec, profile)):
            layout = generate_layout(spec, ranked.candidate, ranked.metrics, topology_plan=topology_plan, placement=placement, qec_plan=ranked.qec_plan)
            layout_parasitics = analyze_layout_parasitics(layout)
            layout_consistency = validate_layout_consistency(layout, topology_plan, ranked.qec_plan)
            drc_summary = run_drc_checks(layout, spec)
            dvl_result = validate_design_vs_layout(spec, ranked.candidate, ranked.metrics, layout)
            variant_score = _layout_exploration_score(
                base_score=bundle.base_score,
                composite_score=bundle.composite_score,
                layout_parasitics=layout_parasitics,
                layout_consistency=layout_consistency,
                drc_summary=drc_summary,
                dvl_result=dvl_result,
                qec_plan=ranked.qec_plan,
            )
            record = {
                "candidate_key": bundle.candidate_key,
                "variant": variant_name,
                "variant_index": variant_index,
                "layout_score": variant_score,
                "advanced_composite_score": bundle.composite_score,
                "drc_failed_rules": drc_summary.failed_rules,
                "design_layout_mismatches": len(dvl_result.mismatches),
                "layout_closure_pass": layout_consistency.closure_pass,
                "layout_missing_items": list(layout_consistency.missing_items),
                "max_coupling_risk": layout_parasitics.max_coupling_risk,
                "min_feature_spacing_um": layout_parasitics.min_feature_spacing_um,
                "overlap_violations": layout_parasitics.overlap_violations,
                "qec_violations": list(ranked.qec_plan.violations) if ranked.qec_plan is not None else [],
            }
            records.append(record)
            if best_stage is None or variant_score > best_stage["layout_score"]:
                best_stage = {
                    "candidate_key": bundle.candidate_key,
                    "variant": variant_name,
                    "layout_score": variant_score,
                    "placement": placement,
                    "dense_crosstalk": dense_crosstalk,
                    "topology_plan": topology_plan,
                    "layout": layout,
                    "layout_parasitics": layout_parasitics,
                    "layout_consistency": layout_consistency,
                }
    records.sort(key=lambda item: item["layout_score"], reverse=True)
    return best_stage, {
        "selected_candidate_key": best_stage["candidate_key"] if best_stage else "",
        "selected_variant": best_stage["variant"] if best_stage else "",
        "stages": records,
    }


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
    quality_mode: str = "standard",
    requirements_bundle: RequirementBundle | None = None,
) -> dict[str, Any]:
    if spec_data is not None:
        spec = DesignSpec.from_dict(spec_data)
    elif spec_path is not None:
        spec = DesignSpec.from_json(spec_path)
    else:
        raise ValueError("spec_path or spec_data must be provided")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compute_backend = resolve_compute_backend().to_dict()
    quality_profile = _build_quality_profile(
        quality_mode,
        monte_carlo_trials=monte_carlo_trials,
        generations=generations,
        beam_width=beam_width,
        mutations_per_parent=mutations_per_parent,
        paper_limit=paper_limit,
        advanced_top_k=advanced_top_k,
    )
    feasibility_precheck = estimate_qec_feasibility_precheck(spec)
    (out / "normalized_spec.json").write_text(_json_dump(spec.to_dict()), encoding="utf-8")
    requirement_artifacts = write_requirement_artifacts(out, requirements_bundle) if requirements_bundle else {}
    signoff_schema_artifacts = write_signoff_schema(out)
    effective_crawl_papers = crawl_papers or quality_profile.auto_crawl_papers
    paper_knowledge = build_paper_knowledge(spec, max_papers=quality_profile.effective_paper_limit) if effective_crawl_papers else None
    paper_artifacts = write_paper_artifacts(out, paper_knowledge) if paper_knowledge else {}

    search_results: list[OptimizationResult] = []
    round_feedback: SeedFeedback | None = None
    injected_round_seeds: list[Any] = []
    for round_index in range(quality_profile.search_rounds):
        round_result = optimize_design(
            spec,
            seed=seed + 7919 * round_index,
            generations=quality_profile.effective_generations,
            beam_width=quality_profile.effective_beam_width,
            mutations_per_parent=quality_profile.effective_mutations_per_parent,
            monte_carlo_trials=quality_profile.effective_monte_carlo_trials,
            paper_knowledge=paper_knowledge,
            requirements_bundle=requirements_bundle,
            seed_feedback=round_feedback,
            injected_seed_candidates=injected_round_seeds,
        )
        search_results.append(round_result)
        round_feedback = _seed_feedback_from_payload(round_result.seed_feedback)
        injected_round_seeds = _build_next_round_injected_seeds(round_result, spec, seed + 12000 + 257 * round_index)
    seed_provenance = [item for result_item in search_results for item in result_item.seed_provenance]
    paper_bias_audit = next((result_item.paper_bias_audit for result_item in reversed(search_results) if result_item.paper_bias_audit), {})
    combined_frontier, merged_search_log = _merge_frontiers(search_results, quality_profile.frontier_limit)
    fallback_ranked = _fallback_ranked_candidate(search_results)
    advanced_dir = out / "advanced"
    advanced = refine_frontier_with_open_source_sims(
        spec,
        combined_frontier,
        advanced_dir,
        seed=seed,
        top_k=quality_profile.effective_advanced_top_k,
        paper_knowledge=paper_knowledge,
        requirements_bundle=requirements_bundle,
    )
    frontier_by_key = {_candidate_key(item.candidate): item for item in combined_frontier}
    selected_layout_stage, layout_exploration = _explore_layout_variants(
        spec,
        frontier_by_key=frontier_by_key,
        advanced=advanced,
        profile=quality_profile,
        seed=seed,
        paper_knowledge=paper_knowledge,
        requirements_bundle=requirements_bundle,
    )
    selected_candidate_key = selected_layout_stage["candidate_key"] if selected_layout_stage is not None else advanced.selected_candidate_key
    if not selected_candidate_key and combined_frontier:
        selected_candidate_key = _candidate_key(combined_frontier[0].candidate)
    if not selected_candidate_key and fallback_ranked is not None:
        selected_candidate_key = _candidate_key(fallback_ranked.candidate)
    advanced.selected_candidate_key = selected_candidate_key
    if advanced.bundles:
        advanced.bundles.sort(key=lambda item: (0 if item.candidate_key == selected_candidate_key else 1, -item.composite_score))

    selected_ranked = frontier_by_key.get(selected_candidate_key) or fallback_ranked or (combined_frontier[0] if combined_frontier else None)
    if selected_ranked is None:
        raise RuntimeError("optimizer returned no ranked candidates")
    final_candidate = selected_ranked.candidate
    final_metrics = selected_ranked.metrics
    final_qec_plan = selected_ranked.qec_plan
    final_monte_carlo = run_monte_carlo(spec, final_candidate, trials=quality_profile.effective_monte_carlo_trials, seed=seed + 4096)
    if selected_layout_stage is not None:
        final_placement = selected_layout_stage["placement"]
        final_dense_crosstalk = selected_layout_stage["dense_crosstalk"]
        topology_plan = selected_layout_stage["topology_plan"]
        layout = selected_layout_stage["layout"]
        layout_parasitics = selected_layout_stage["layout_parasitics"]
        layout_consistency = selected_layout_stage["layout_consistency"]
    else:
        final_placement = build_probabilistic_placement(spec, final_candidate, samples=quality_profile.placement_samples, seed_offset=seed + 5000)
        final_dense_crosstalk = simulate_dense_crosstalk(spec, final_candidate, final_placement, trials=quality_profile.dense_trials, seed_offset=seed + 5000)
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
        layout_parasitics = analyze_layout_parasitics(layout)
        layout_consistency = validate_layout_consistency(layout, topology_plan, final_qec_plan)

    result = OptimizationResult(
        best_candidate=final_candidate,
        best_metrics=final_metrics,
        monte_carlo=final_monte_carlo,
        pareto_like_frontier=combined_frontier,
        search_log=merged_search_log,
        seed_provenance=seed_provenance,
        paper_bias_audit=paper_bias_audit,
        selected_ranked_candidate=selected_ranked,
        seed_feedback=search_results[-1].seed_feedback if search_results else {},
    )

    layout_paths = write_layout_artifacts(out, spec, final_candidate, final_metrics, layout)
    topology_artifacts = write_topology_artifacts(out, topology_plan)
    qec_artifacts = write_qec_artifacts(out, final_qec_plan) if final_qec_plan else {}
    consistency_artifacts = write_layout_consistency_artifacts(out, layout_consistency)
    (out / "layout_parasitics.json").write_text(_json_dump(layout_parasitics.to_dict()), encoding="utf-8")
    quality_profile_path = out / "quality_profile.json"
    quality_profile_path.write_text(_json_dump(quality_profile.to_dict()), encoding="utf-8")
    layout_exploration_path = out / "layout_exploration.json"
    layout_exploration_path.write_text(_json_dump(layout_exploration), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Enhanced Signoff Flow: DRC + Design-vs-Layout + Multi-stage Gates
    # -----------------------------------------------------------------------
    drc_summary = run_drc_checks(layout, spec)
    dvl_result = validate_design_vs_layout(spec, final_candidate, final_metrics, layout)
    drc_report_artifacts = write_drc_report(out, drc_summary, dvl_result)

    # Build signoff gates
    signoff_gates = [
        build_drc_gate(drc_summary),
        build_design_vs_layout_gate(dvl_result),
        build_consistency_gate(layout_consistency),
        build_area_agreement_gate(final_metrics, layout),
        build_lvs_gate(layout),
        build_qec_gate(final_qec_plan, spec),
    ]

    run_id = make_run_id(spec, final_candidate, final_metrics, seed)

    # Build signoff report (first pass — before report generation)
    signoff_report = build_signoff_report(run_id, signoff_gates)
    signoff_report_artifacts = write_signoff_report(out, signoff_report)

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
            design_vs_layout=dvl_result,
            signoff_report=signoff_report,
            quality_profile=quality_profile.to_dict(),
            layout_exploration=layout_exploration,
            feasibility_precheck=feasibility_precheck,
        ),
    )
    base_artifact_paths = {
        **layout_paths,
        **topology_artifacts,
        **qec_artifacts,
        **consistency_artifacts,
        **requirement_artifacts,
        **paper_artifacts,
        **signoff_schema_artifacts,
        **drc_report_artifacts,
        **signoff_report_artifacts,
        "report": report_path,
        "normalized_spec": str(out / "normalized_spec.json"),
        "layout_parasitics": str(out / "layout_parasitics.json"),
        "quality_profile": str(quality_profile_path),
        "layout_exploration": str(layout_exploration_path),
    }
    artifact_registry = build_artifact_registry(run_id, out, base_artifact_paths)
    registry_artifacts = write_artifact_registry(out, artifact_registry)
    handoff_bundle = create_handoff_bundle(
        run_id,
        out,
        artifact_registry,
        quarantined=not signoff_report.critical_gates_pass,
        quarantine_reason="critical_gate_failure" if not signoff_report.critical_gates_pass else "",
    )
    handoff_artifacts = write_handoff_bundle(out, handoff_bundle)

    # Build signoff manifest with full gate information
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
        signoff_report=signoff_report,
        drc_summary=drc_summary,
        dvl_result=dvl_result,
        handoff_quarantined=handoff_bundle.quarantined,
        quarantine_flag=handoff_bundle.quarantine_flag,
        signoff_schema_path=signoff_schema_artifacts.get("signoff_schema"),
    )

    # Second pass: rebuild report with signoff manifest + handoff bundle
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
            design_vs_layout=dvl_result,
            signoff_manifest=signoff_manifest,
            handoff_bundle=handoff_bundle.to_dict(),
            signoff_report=signoff_report,
            quality_profile=quality_profile.to_dict(),
            layout_exploration=layout_exploration,
            feasibility_precheck=feasibility_precheck,
        ),
    )
    base_artifact_paths["report"] = report_path
    artifact_registry = build_artifact_registry(run_id, out, base_artifact_paths | registry_artifacts | handoff_artifacts)
    handoff_bundle = create_handoff_bundle(
        run_id,
        out,
        artifact_registry,
        quarantined=not signoff_report.critical_gates_pass,
        quarantine_reason="critical_gate_failure" if not signoff_report.critical_gates_pass else "",
    )
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
        signoff_report=signoff_report,
        drc_summary=drc_summary,
        dvl_result=dvl_result,
        handoff_quarantined=handoff_bundle.quarantined,
        quarantine_flag=handoff_bundle.quarantine_flag,
        signoff_schema_path=signoff_schema_artifacts.get("signoff_schema"),
    )
    signoff_artifacts = write_signoff_manifest(out, signoff_manifest)

    # -----------------------------------------------------------------------
    # Bundle Audit: verify integrity & completeness
    # -----------------------------------------------------------------------
    bundle_audit = audit_handoff_bundle(handoff_bundle, artifact_registry)
    audit_artifacts = write_bundle_audit(out, bundle_audit)

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
            "quality_profile": quality_profile.to_dict(),
            "layout_exploration": layout_exploration,
            "feasibility_precheck": feasibility_precheck,
            "seed_provenance": result.seed_provenance,
            "paper_bias_audit": result.paper_bias_audit,
            "run_id": run_id,
            "signoff_manifest": signoff_manifest.to_dict(),
            "signoff_report": signoff_report.to_dict(),
            "drc_summary": drc_summary.to_dict(),
            "design_vs_layout": dvl_result.to_dict(),
            "bundle_audit": bundle_audit.to_dict(),
            "artifact_registry": artifact_registry.to_dict(),
            "handoff_bundle": handoff_bundle.to_dict(),
            "compute_backend": compute_backend,
            "signoff_schema": signoff_schema_artifacts,
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
        "quality_profile": quality_profile.to_dict(),
        "layout_exploration": layout_exploration,
        "feasibility_precheck": feasibility_precheck,
        "seed_provenance": result.seed_provenance,
        "paper_bias_audit": result.paper_bias_audit,
        "compute_backend": compute_backend,
        "signoff_manifest": signoff_manifest.to_dict(),
        "signoff_report": signoff_report.to_dict(),
        "drc_summary": drc_summary.to_dict(),
        "design_vs_layout": dvl_result.to_dict(),
        "bundle_audit": bundle_audit.to_dict(),
        "artifacts": {
            **layout_paths,
            **topology_artifacts,
            **qec_artifacts,
            **consistency_artifacts,
            **registry_artifacts,
            **handoff_artifacts,
            **signoff_artifacts,
            **drc_report_artifacts,
            **signoff_report_artifacts,
            **audit_artifacts,
            **requirement_artifacts,
            **paper_artifacts,
            **signoff_schema_artifacts,
            "report": report_path,
            "optimization": str(optimization_path),
            "normalized_spec": str(out / "normalized_spec.json"),
            "layout_parasitics": str(out / "layout_parasitics.json"),
            "quality_profile": str(quality_profile_path),
            "layout_exploration": str(layout_exploration_path),
        },
    })
    (out / "summary.json").write_text(_json_dump(summary), encoding="utf-8")
    return summary
