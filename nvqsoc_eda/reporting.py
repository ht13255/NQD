from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .advanced_sim import AdvancedRefinementResult, LayoutParasiticSummary
from .compute_backend import resolve_compute_backend
from .layout_validation import DesignVsLayoutResult, LayoutConsistencySummary
from .optimizer import OptimizationResult
from .papers import PaperKnowledge
from .qec import QECPlan
from .requirements import RequirementBundle
from .signoff import SignoffManifest, SignoffReport
from .simulator import CandidateDesign, SimulationMetrics
from .simulator import MonteCarloSummary
from .spec import DesignSpec


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(payload: Any, *, ensure_ascii: bool = True, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, default=_json_default)


def _candidate_table(candidate: CandidateDesign) -> str:
    rows = []
    for key, value in candidate.to_dict().items():
        rows.append(f"| {key} | {value} |")
    return "\n".join(rows)


def _metrics_table(metrics: SimulationMetrics) -> str:
    rows = []
    for key, value in metrics.to_dict().items():
        rows.append(f"| {key} | {value} |")
    return "\n".join(rows)


def build_report(
    spec: DesignSpec,
    result: OptimizationResult,
    layout_paths: dict[str, str],
    run_id: str,
    selected_candidate: CandidateDesign,
    selected_metrics: SimulationMetrics,
    selected_monte_carlo: MonteCarloSummary,
    selected_qec_plan: QECPlan | None,
    requirements_bundle: RequirementBundle | None,
    paper_knowledge: PaperKnowledge | None,
    advanced_refinement: AdvancedRefinementResult,
    layout_parasitics: LayoutParasiticSummary,
    layout_consistency: LayoutConsistencySummary,
    design_vs_layout: DesignVsLayoutResult | None = None,
    signoff_manifest: SignoffManifest | None = None,
    handoff_bundle: dict[str, Any] | None = None,
    signoff_report: SignoffReport | None = None,
    quality_profile: dict[str, Any] | None = None,
    layout_exploration: dict[str, Any] | None = None,
    feasibility_precheck: dict[str, Any] | None = None,
) -> str:
    best = selected_candidate
    metrics = selected_metrics
    mc = selected_monte_carlo
    selected_ranked = next(
        (
            item
            for item in result.pareto_like_frontier
            if item.candidate.architecture == best.architecture
            and item.candidate.qubits == best.qubits
            and round(item.candidate.cell_pitch_um, 3) == round(best.cell_pitch_um, 3)
            and item.candidate.optical_bus_count == best.optical_bus_count
            and item.candidate.microwave_line_count == best.microwave_line_count
        ),
        None,
    )
    formula_block = """
1. NV transition frequency: `f_nv = D_zfs + gamma_e * B + strain / 1000`
2. Effective coherence: `1/T2_eff = 1/T2_base + a_T * (T - 3) + a_iso * purity + a_strain * strain + a_xtalk * X + a_B * dB`
3. Optical efficiency: `eta_opt = eta_cavity(Q, buses, width) * 10^(-loss_db/10) * eta_detector`
4. Rabi rate: `Omega_R = 22 * sqrt(P_lin) * 10^(-loss_db/20) / mux_penalty`
5. Single-qubit gate time: `t_1q = 500 / Omega_R`
6. Loaded resonator Q: `Q_L = Q_0 / (1 + 0.18 * loss_db + 0.15 * via_factor)`
7. Spin-photon coupling: `g_eff = 32 * sqrt(N_res/N_q) * sqrt(Q_L/1e5) * eta_opt * exp(-pitch/220)`
8. Two-qubit gate time: `t_2q = 1000 * (1 + 0.1 * (mux - 1)) / g_eff`
9. Gate fidelity: `F_gate = exp(-(t_gate/(T2_eff*1000))^1.35 - 10*X^2 - (jitter/t_gate)^2 - eps_cal^2)`
10. Readout photons: `N_ph = 300 * eta_opt * P_opt_per_bus * t_pulse / (10 + bus_utilization)`
11. Readout SNR: `SNR = contrast * N_ph / sqrt(N_ph + dark + 2*T_noise)`
12. Readout fidelity: `F_ro = 0.5 * (1 + erf(SNR/sqrt(2))) * exp(-latency/2200)`
13. Thermal load: `P_total = N_mw * P_mw * duty * 0.25 + 0.18 * P_opt + 1.6 * G_amp / stages + 24`
14. Routing capacity: `C_route = 1 - route_length / ((8 + 2.2*M_layers) * die_area)`
15. Yield: `Y = exp(-rho_defect * area) * exp(-k_via * N_via) * exp(-0.25 * DRC)`
16. Control loop: `G_cl(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)` with extracted bandwidth and phase margin proxies
17. Phase noise proxy: `sigma_t = sqrt(integral S_phi(f) df) / (2*pi*f0)`
18. IR-drop solve: solve `G * V = I` on the extracted sparse power-grid topology and measure worst-case `Delta V`
19. Logical error rate: `p_L = alpha * (p_phys / p_th)^((d+1)/2)` for sub-threshold QEC operation
20. Physical qubits per logical: `N_phys/logical ~= k_code(d)` with code-family dependent scaling
21. Decoder locality proxy: `S_loc ~ 1 - cluster_penalty - avg_surgery_length / scale`
22. Magic-state throughput: `R_ms ~ N_factory / ((a + b*d) * t_logical)`
23. Bell-pair fidelity proxy: `F_bell ~ 0.54*F_ro + 0.22*exp(-latency/T2) + 0.24*eta_opt*I_2photon`
24. Decoder failure proxy: `p_dec ~ p_dep + 0.6*p_meas + 0.4*p_corr + p_logical`
25. Global objective: `Score = sum(w_i * score_i) + 0.08 * robustness - penalties - hard(IR_drop > threshold)`
""".strip()
    requirements_section = "- No natural-language requirement bundle attached."
    if requirements_bundle:
        requirements_section = "\n".join(
            [
                f"- Source: `{requirements_bundle.source}`",
                f"- Model: `{requirements_bundle.model}`",
                f"- Parse confidence: `{requirements_bundle.parse_confidence:.2f}`",
                f"- Ollama endpoint: `{requirements_bundle.ollama_endpoint}`",
                f"- Ollama runtime ready / model ready / auto-started / auto-pulled: `{requirements_bundle.ollama_runtime_ready}` / `{requirements_bundle.ollama_model_available}` / `{requirements_bundle.ollama_started_local_service}` / `{requirements_bundle.ollama_model_pulled}`",
                f"- Ollama model loaded / auth mode: `{requirements_bundle.ollama_model_loaded}` / `{requirements_bundle.ollama_auth_mode}`",
                f"- Goals: `{json.dumps(requirements_bundle.goals, ensure_ascii=False)}`",
                f"- Constraints: `{json.dumps(requirements_bundle.constraints, ensure_ascii=False)}`",
                f"- Requirement layers: `{json.dumps(requirements_bundle.layers.to_dict(), ensure_ascii=False)}`",
                f"- Layout focus: `{json.dumps(requirements_bundle.layout_focus, ensure_ascii=False)}`",
            ]
        )

    compute_backend = resolve_compute_backend()
    compute_section = "\n".join(
        [
            f"- Engine: `{compute_backend.engine}`",
            f"- Device: `{compute_backend.device}`",
            f"- GPU enabled: `{compute_backend.gpu_enabled}`",
            f"- Reason: `{compute_backend.reason}`",
        ]
    )

    quality_section = "- Standard search depth used."
    if quality_profile is not None:
        quality_section = "\n".join(
            [
                f"- Mode: `{quality_profile.get('mode')}`",
                f"- Search rounds: `{quality_profile.get('search_rounds')}`",
                f"- Effective generations / beam / mutations: `{quality_profile.get('effective_generations')}` / `{quality_profile.get('effective_beam_width')}` / `{quality_profile.get('effective_mutations_per_parent')}`",
                f"- Effective MC trials / advanced top-k: `{quality_profile.get('effective_monte_carlo_trials')}` / `{quality_profile.get('effective_advanced_top_k')}`",
                f"- Placement samples / dense trials: `{quality_profile.get('placement_samples')}` / `{quality_profile.get('dense_trials')}`",
                f"- Auto paper crawl: `{quality_profile.get('auto_crawl_papers')}`",
            ]
        )

    feasibility_section = "- Early feasibility pre-check unavailable."
    if feasibility_precheck is not None:
        family_lines = [
            f"- `{item['code_family']}`: d=`{item['recommended_distance']}` | required qubits=`{item['estimated_total_required_qubits']}` | projected logical error=`{item['projected_logical_error_rate']:.3e}` | qubit/error feasible=`{item['meets_qubit_budget']}`/`{item['meets_error_budget']}`"
            for item in feasibility_precheck.get("families", [])[:3]
        ]
        feasibility_section = "\n".join(
            [
                f"- Estimated physical error rate: `{feasibility_precheck.get('estimated_physical_error_rate', 0.0):.3e}`",
                f"- Requested target qubits / logicals: `{feasibility_precheck.get('requested_target_qubits')}` / `{feasibility_precheck.get('requested_target_logical_qubits')}`",
                f"- Warnings: `{json.dumps(feasibility_precheck.get('warnings', []), ensure_ascii=False)}`",
                *family_lines,
            ]
        )

    layout_exploration_section = "- Layout exploration summary unavailable."
    if layout_exploration is not None:
        stages = layout_exploration.get("stages", [])
        layout_exploration_section = "\n".join(
            [
                f"- Selected candidate key: `{layout_exploration.get('selected_candidate_key')}`",
                f"- Selected topology variant: `{layout_exploration.get('selected_variant')}`",
                f"- Explored layout stages: `{len(stages)}`",
            ]
        )

    paper_section = "- Paper crawl disabled for this run."
    if paper_knowledge:
        paper_lines = [
            f"- Crawled papers: `{len(paper_knowledge.papers)}`",
            f"- Semantic focus: `{', '.join(paper_knowledge.semantic_focus_terms)}`",
            f"- Recommended cavity Q: `{paper_knowledge.recommended_cavity_q:.1f}`",
            f"- Recommended waveguide width (um): `{paper_knowledge.recommended_waveguide_width_um:.3f}`",
            f"- Recommended temperature (K): `{paper_knowledge.recommended_temp_k:.2f}`",
            f"- Recommended pitch (um): `{paper_knowledge.recommended_pitch_um:.2f}`",
            f"- Topic counts: `{json.dumps(paper_knowledge.topic_counts)}`",
        ]
        if result.paper_bias_audit:
            paper_lines.append(f"- Bias audit warnings: `{json.dumps(result.paper_bias_audit.get('warnings', []), ensure_ascii=False)}`")
        paper_section = "\n".join(paper_lines)

    paper_provenance_section = "- No paper-seed provenance attached."
    if result.seed_provenance:
        applied_bindings = []
        for entry in result.seed_provenance[:6]:
            for binding in entry.get("paper_recommendation_bindings", []):
                if binding.get("applied_in_seed"):
                    applied_bindings.append(
                        {
                            "candidate_key": entry.get("candidate_key"),
                            "parameter": binding.get("parameter"),
                            "candidate_value": binding.get("candidate_value"),
                            "recommended_value": binding.get("recommended_value"),
                        }
                    )
        paper_provenance_section = "\n".join(
            [
                f"- Seed provenance entries: `{len(result.seed_provenance)}`",
                f"- Sample provenance: `{json.dumps(result.seed_provenance[:4], ensure_ascii=False)}`",
                f"- Applied recommendation bindings: `{json.dumps(applied_bindings[:6], ensure_ascii=False)}`",
            ]
        )

    paper_bias_section = "- Paper bias audit unavailable."
    if result.paper_bias_audit:
        paper_bias_section = "\n".join(
            [
                f"- Topic counts: `{json.dumps(result.paper_bias_audit.get('topic_counts', {}), ensure_ascii=False)}`",
                f"- Architecture counts: `{json.dumps(result.paper_bias_audit.get('architecture_counts', {}), ensure_ascii=False)}`",
                f"- Actual architecture ratios: `{json.dumps(result.paper_bias_audit.get('actual_architecture_ratios', {}), ensure_ascii=False)}`",
                f"- Expected architecture ratios: `{json.dumps(result.paper_bias_audit.get('expected_architecture_ratios', {}), ensure_ascii=False)}`",
                f"- Warnings: `{json.dumps(result.paper_bias_audit.get('warnings', []), ensure_ascii=False)}`",
            ]
        )

    neural_section = "- No frozen neural surrogate data attached."
    if selected_ranked and selected_ranked.neural_signals:
        neural = selected_ranked.neural_signals
        neural_section = "\n".join(
            [
                f"- Design quality: `{neural.design_quality:.4f}`",
                f"- Architecture affinity: `{neural.architecture_affinity:.4f}`",
                f"- Prototype affinity: `{neural.prototype_affinity:.4f}`",
                f"- Confidence: `{neural.confidence:.4f}`",
                f"- Latency risk: `{neural.latency_risk:.4f}`",
                f"- Crosstalk risk: `{neural.crosstalk_risk:.4f}`",
                f"- Suggested pitch delta (um): `{neural.pitch_delta_um:.3f}`",
                f"- Suggested bus / line / shield deltas: `{neural.optical_bus_delta}`, `{neural.microwave_line_delta}`, `{neural.shielding_delta}`",
            ]
        )

    world_model_section = "- No world-model rollout attached."
    if selected_ranked and getattr(selected_ranked, "world_model_signals", None):
        world_model = selected_ranked.world_model_signals
        world_model_lines = [
            f"- Terminal quality: `{world_model.terminal_quality:.4f}`",
            f"- Schedule / logical / routing / thermal pressure: `{world_model.schedule_pressure:.4f}` / `{world_model.logical_pressure:.4f}` / `{world_model.routing_pressure:.4f}` / `{world_model.thermal_pressure:.4f}`",
            f"- Confidence: `{world_model.confidence:.4f}`",
            f"- Recommended qubit / factory-lane delta: `{world_model.recommended_qubit_delta}` / `{world_model.recommended_factory_lane_delta}`",
            f"- Recommended pitch delta (um): `{world_model.recommended_pitch_delta_um:.3f}`",
            f"- Rollout actions: `{json.dumps([step['action'] for step in [item.to_dict() for item in world_model.rollout]])}`",
        ]
        active_feedback = getattr(result, "seed_feedback", {}) or {}
        if active_feedback.get("active"):
            world_model_lines.append(f"- Bound next-round seed feedback: `{json.dumps(active_feedback, ensure_ascii=False)}`")
        world_model_section = "\n".join(world_model_lines)

    topology_section = "- Topology plan unavailable."
    if advanced_refinement.bundles:
        topology = advanced_refinement.bundles[0].topology
        topology_section = "\n".join(
            [
                f"- Optical topology: `{topology.get('optical_topology')}`",
                f"- Control topology: `{topology.get('control_topology')}`",
                f"- Readout topology: `{topology.get('readout_topology')}`",
                f"- Mesh pitch (um): `{topology.get('power_mesh_pitch_um')}`",
                f"- Shield pitch (um): `{topology.get('shield_pitch_um')}`",
                f"- Complexity score: `{topology.get('layout_complexity_score')}`",
                f"- Reasoning: `{json.dumps(topology.get('reasoning', []), ensure_ascii=False)}`",
            ]
        )

    qec_section = "- QEC plan unavailable."
    if selected_qec_plan:
        qec_section = "\n".join(
            [
                f"- Code family: `{selected_qec_plan.code_family}`",
                f"- Code distance: `{selected_qec_plan.code_distance}`",
                f"- Target / achievable logical qubits: `{selected_qec_plan.target_logical_qubits}` / `{selected_qec_plan.achievable_logical_qubits}`",
                f"- Physical qubits per logical: `{selected_qec_plan.physical_qubits_per_logical}`",
                f"- Physical qubits per target logical: `{selected_qec_plan.physical_qubits_per_target_logical:.2f}`",
                f"- Estimated required total qubits: `{selected_qec_plan.estimated_required_total_qubits}`",
                f"- Physical error rate: `{selected_qec_plan.physical_error_rate:.6e}`",
                f"- Logical error rate: `{selected_qec_plan.logical_error_rate:.6e}`",
                f"- Distance escalation triggered / candidate: `{selected_qec_plan.escalation_rule_triggered}` / `{selected_qec_plan.escalated_distance_candidate}`",
                f"- Syndrome / decoder / logical cycle ns: `{selected_qec_plan.syndrome_cycle_ns:.2f}` / `{selected_qec_plan.decoder_latency_ns:.2f}` / `{selected_qec_plan.logical_cycle_ns:.2f}`",
                f"- Logical success probability: `{selected_qec_plan.logical_success_probability:.6f}`",
                f"- Decoder locality score: `{selected_qec_plan.decoder_locality_score:.4f}`",
                f"- Surgery throughput ops/us: `{selected_qec_plan.surgery_throughput_ops_per_us:.4f}`",
                f"- Magic-state rate / factories: `{selected_qec_plan.magic_state_rate_per_us:.4f}` / `{len(selected_qec_plan.magic_state_factories)}`",
                f"- Schedule makespan / critical path ns: `{selected_qec_plan.schedule_makespan_ns:.2f}` / `{selected_qec_plan.schedule_critical_path_ns:.2f}`",
                f"- Factory / decoder utilization: `{selected_qec_plan.factory_utilization:.4f}` / `{selected_qec_plan.decoder_utilization:.4f}`",
                f"- Logical scheduled ops / factory batches: `{len(selected_qec_plan.logical_schedule)}` / `{len(selected_qec_plan.factory_timelines)}`",
                f"- Violations: `{json.dumps(selected_qec_plan.violations)}`",
                f"- Feasibility warnings: `{json.dumps(selected_qec_plan.feasibility_warnings, ensure_ascii=False)}`",
            ]
        )

    robust_section = "- Fast robust scenario summary unavailable."
    if selected_ranked and getattr(selected_ranked, "robust_signals", None):
        robust = selected_ranked.robust_signals
        robust_section = "\n".join(
            [
                f"- Scenario / logical / schedule pass rate: `{robust.scenario_pass_rate:.4f}` / `{robust.logical_pass_rate:.4f}` / `{robust.schedule_pass_rate:.4f}`",
                f"- Gate / readout floor: `{robust.gate_floor:.6f}` / `{robust.readout_floor:.6f}`",
                f"- Worst latency ns: `{robust.worst_latency_ns:.2f}`",
                f"- Worst logical error rate: `{robust.worst_logical_error_rate:.6e}`",
            ]
        )

    interpretation_section = "- Interpretation unavailable."
    health = "strong"
    if selected_qec_plan and selected_qec_plan.violations:
        health = "logical-limited"
    elif metrics.constraint_violations:
        health = "physically-limited"
    interpretation_lines = [f"- Overall health: `{health}`"]
    if selected_qec_plan:
        if selected_qec_plan.achievable_logical_qubits >= spec.target_logical_qubits:
            interpretation_lines.append("- Logical resource count meets or exceeds the requested target.")
        else:
            interpretation_lines.append("- Logical resource count is below the requested target, so physical scaling or lighter logical targets are needed.")
        if selected_qec_plan.logical_error_rate <= spec.target_logical_error_rate:
            interpretation_lines.append("- Logical error budget is satisfied under the selected QEC family.")
        else:
            interpretation_lines.append("- Logical error budget is not yet satisfied; the design is currently constrained by physical error rate and QEC overhead.")
        interpretation_lines.append(f"- Decoder locality and schedule utilization are `{selected_qec_plan.decoder_locality_score:.3f}` and `{selected_qec_plan.decoder_utilization:.3f}`, indicating how concentrated the logical control load is.")
    if layout_consistency.closure_pass:
        interpretation_lines.append("- Post-layout consistency checks pass, so generated geometry is internally coherent with the QEC/topology plan.")
    else:
        interpretation_lines.append(f"- Post-layout consistency flags remain: `{json.dumps(layout_consistency.missing_items)}`.")
    interpretation_section = "\n".join(interpretation_lines)

    signoff_section = "- Signoff manifest not yet attached."
    if signoff_manifest is not None:
        handoff_ready_text = "attached" if handoff_bundle else "not attached"
        signoff_section = "\n".join(
            [
                f"- Run ID: `{signoff_manifest.run_id}`",
                f"- Candidate key: `{signoff_manifest.candidate_key}`",
                f"- Handoff ready: `{signoff_manifest.handoff_ready}`",
                f"- Handoff bundle: `{handoff_ready_text}`",
                f"- Report/GDS linkage: `{signoff_manifest.report_references_run_id}` / `{signoff_manifest.report_references_gds}` / `{signoff_manifest.report_references_candidate_key}` / `{signoff_manifest.report_references_die_area}`",
                f"- Hierarchy cells / references: `{signoff_manifest.hierarchy_cell_count}` / `{signoff_manifest.hierarchy_reference_count}`",
                f"- Metrics vs layout die area mm2: `{signoff_manifest.metrics_die_area_mm2:.4f}` / `{signoff_manifest.layout_die_area_mm2:.4f}` (delta `{signoff_manifest.area_delta_mm2:.6f}`)",
                f"- Report SHA / GDS SHA: `{signoff_manifest.report_sha256[:12]}` / `{signoff_manifest.gds_sha256[:12]}`",
                f"- Artifact registry: `{signoff_manifest.artifact_registry_path}`",
                f"- Handoff manifest: `{signoff_manifest.handoff_bundle_path}`",
                f"- Signoff version: `{signoff_manifest.signoff_version}`",
                f"- Signoff schema: `{signoff_manifest.signoff_schema_path}`",
                f"- Timestamp: `{signoff_manifest.timestamp}`",
                f"- Gates: `{signoff_manifest.gate_count}` total, `{signoff_manifest.gates_passed}` passed, `{signoff_manifest.gates_failed}` failed",
                f"- Critical gates pass: `{signoff_manifest.critical_gates_pass}`",
                f"- DRC all pass: `{signoff_manifest.drc_all_pass}`",
                f"- Design vs layout match: `{signoff_manifest.design_vs_layout_match}`",
                f"- Handoff authorized: `{signoff_manifest.handoff_authorized}`",
                f"- Handoff quarantined / flag: `{signoff_manifest.handoff_quarantined}` / `{signoff_manifest.quarantine_flag}`",
            ]
        )

    artifact_section = "- Artifact registry unavailable."
    if handoff_bundle is not None:
        artifact_section = "\n".join(
            [
                f"- Handoff package dir: `{handoff_bundle.get('package_dir')}`",
                f"- Handoff package manifest: `{handoff_bundle.get('package_manifest')}`",
                f"- Handoff package SHA: `{str(handoff_bundle.get('package_sha256', ''))[:12]}`",
                f"- Quarantined / reason: `{handoff_bundle.get('quarantined')}` / `{handoff_bundle.get('quarantine_reason')}`",
                f"- Immutable artifacts: `{json.dumps(handoff_bundle.get('immutable_artifacts', {}))}`",
            ]
        )

    design_vs_layout_section = "- Design-vs-layout diff unavailable."
    if design_vs_layout is not None:
        constraint_diff_rows = []
        sim_mismatch = next((item for item in design_vs_layout.mismatches if item.check_name == "simulation_constraints"), None)
        if sim_mismatch is not None:
            for diff in sim_mismatch.details.get("constraint_diffs", []):
                constraint_diff_rows.append(
                    f"| {diff['name']} | {diff['comparison']} | {diff['target']:.6g} | {diff['actual']:.6g} | {diff['delta']:.6g} | {diff['violated']} |"
                )
        constraint_diff_table = "\n".join(constraint_diff_rows) if constraint_diff_rows else "| none | - | - | - | - | - |"
        design_vs_layout_section = "\n".join(
            [
                f"- All match: `{design_vs_layout.all_match}`",
                f"- Checks performed: `{design_vs_layout.checks_performed}`",
                f"- Mismatches: `{json.dumps([item.to_dict() for item in design_vs_layout.mismatches], ensure_ascii=False)}`",
                "",
                "| Parameter | Rule | Target | Actual | Delta | Violated |",
                "| --- | --- | --- | --- | --- | --- |",
                constraint_diff_table,
            ]
        )

    signoff_gates_section = "- Signoff gate report unavailable."
    if signoff_report is not None:
        gate_rows = []
        for gate in signoff_report.gates:
            status = "PASS" if gate.passed else "FAIL"
            gate_rows.append(f"| {status} | {gate.gate_name} | {gate.category} | {gate.severity} | {gate.message} |")
        gate_table = "\n".join(gate_rows) if gate_rows else "| - | - | - | - | - |"
        signoff_gates_section = "\n".join([
            f"- Signoff version: `{signoff_report.signoff_version}`",
            f"- All gates pass: `{signoff_report.all_gates_pass}`",
            f"- Critical gates pass: `{signoff_report.critical_gates_pass}`",
            f"- **Handoff authorized: `{signoff_report.handoff_authorized}`**",
            "",
            "| Status | Gate | Category | Severity | Message |",
            "| --- | --- | --- | --- | --- |",
            gate_table,
        ])

    advanced_rows = []
    for bundle in advanced_refinement.bundles:
        advanced_rows.append(
            f"| {bundle.candidate_key} | {bundle.base_score:.4f} | {bundle.composite_score:.4f} | {bundle.placement.placement_score:.4f} | {bundle.dense_crosstalk.effective_crosstalk_db:.3f} | {bundle.qutip.single_gate_state_fidelity:.6f} | {bundle.qutip.two_qubit_state_fidelity:.6f} | {bundle.control.phase_margin_deg:.2f} | {bundle.noise.rms_jitter_ps:.3f} | {bundle.power_grid.worst_drop_mV:.3f} | {bundle.entanglement.bell_fidelity:.4f} | {bundle.error_channels.decoder_failure_proxy:.4f} | {bundle.routing.congestion_ratio:.4f} | {bundle.optical.net_collection_efficiency:.4f} | {bundle.geometry.min_spacing_um:.4f} | {bundle.photon.empirical_readout_fidelity:.6f} | {bundle.thermal.peak_temp_rise_k:.6f} | {bundle.microwave.center_s21_db:.4f} |"
        )
    advanced_table = "\n".join(advanced_rows) if advanced_rows else "| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
    return f"""# NV-Center QSoC EDA Report

## Project

- Run ID: `{run_id}`
- Design name: `{spec.design_name}`
- Application: `{spec.application}`
- Chosen architecture: `{best.architecture}`
- Target qubits: `{spec.target_qubits}`
- Implemented qubits: `{best.qubits}`
- Target logical qubits: `{spec.target_logical_qubits}`

## Compute Backend

{compute_section}

## Quality Profile

{quality_section}

## Feasibility Pre-check

{feasibility_section}

## Algorithm

1. Normalize the user spec and convert soft requirements into explicit optimization targets.
2. Build architecture-conditioned seeds across `sensor_dense`, `hybrid_router`, and `network_node` families.
3. Evaluate each seed analytically with coherence, microwave, photonic, thermal, yield, and routing equations.
4. Run beam-search style refinement with high-diversity mutations over pitch, buses, resonators, muxing, layers, and cavity Q.
5. Inject dense placement, stochastic cross-talk, and frozen-neural surrogate signals directly into the optimization score.
6. Re-score the frontier using Monte Carlo perturbations on fabrication spread, temperature drift, field drift, gain, and readout noise.
7. Crawl NV-center papers with Scrapling, extract priors, and use them to bias architecture and device variables.
8. Let the topology controller reshape optical, control, and readout macro topology around hotspot regions.
9. Re-rank the Pareto-like frontier with open-source simulations: QuTiP gate dynamics, scikit-rf microwave extraction, sparse thermal solve, control-loop response, phase-noise PSD, power-grid IR drop, routing-graph congestion analysis, optical resonator spectrum modeling, geometry-density DRC, and photon-count Monte Carlo.
10. Synthesize a denser chip-style layout with pad ring, seal ring, power mesh, via farms, resonator tiles, grating couplers, control/readout macros, hotspot keepouts, and fill shapes.
11. Emit `layout.json`, `layout.svg`, `layout_preview.png`, direct `layout.gds`, `layout.oas`, and `layout_klayout.py` for downstream signoff work.

## Strong Formula Stack

{formula_block}

## Requirement Decomposition

{requirements_section}

## Paper Knowledge

{paper_section}

## Paper Provenance

{paper_provenance_section}

## Paper Bias Audit

{paper_bias_section}

## Frozen Neural Surrogate

{neural_section}

## World Model

{world_model_section}

## Robust Optimization

{robust_section}

## Topology Controller

{topology_section}

## Layout Exploration

{layout_exploration_section}

## QEC And Logical Qubits

{qec_section}

## Interpretation

{interpretation_section}

## Signoff

{signoff_section}

## Artifact Management

{artifact_section}

## Signoff Gates

{signoff_gates_section}

## Design Vs Layout Diff

{design_vs_layout_section}

## Best Candidate Parameters

| Parameter | Value |
| --- | --- |
{_candidate_table(best)}

## Best Candidate Metrics

| Metric | Value |
| --- | --- |
{_metrics_table(metrics)}

## Monte Carlo Robustness

- Trials: `{mc.trials}`
- Pass rate: `{mc.pass_rate:.4f}`
- Gate fidelity mean / p05: `{mc.gate_fidelity_mean:.6f}` / `{mc.gate_fidelity_p05:.6f}`
- Readout fidelity mean / p05: `{mc.readout_fidelity_mean:.6f}` / `{mc.readout_fidelity_p05:.6f}`
- T2 mean / p05 (us): `{mc.t2_mean_us:.2f}` / `{mc.t2_p05_us:.2f}`
- Power mean (mW): `{mc.power_mean_mw:.2f}`
- Yield mean: `{mc.yield_mean:.6f}`
- Robustness mean: `{mc.robustness_mean:.6f}`

## Open-Source Simulation Refinement

| Candidate | Base score | Composite | Place score | XT dB | 1Q QuTiP | 2Q QuTiP | Phase margin | Jitter ps | IR drop mV | Bell F | Dec fail | Routing cong. | Optical eta | Min spacing | Photon MC | Peak dT (K) | Center S21 (dB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{advanced_table}

## Layout Parasitics

- Total route length (um): `{layout_parasitics.total_route_length_um:.2f}`
- Route density: `{layout_parasitics.route_density:.4f}`
- Max coupling risk: `{layout_parasitics.max_coupling_risk:.6f}`
- Optical route length (um): `{layout_parasitics.optical_route_length_um:.2f}`
- Microwave route length (um): `{layout_parasitics.microwave_route_length_um:.2f}`
- Min feature spacing (um): `{layout_parasitics.min_feature_spacing_um:.4f}`
- Metal density min / max: `{layout_parasitics.metal_density_min:.4f}` / `{layout_parasitics.metal_density_max:.4f}`
- Overlap violations: `{layout_parasitics.overlap_violations}`

## Layout Consistency

- Closure pass: `{layout_consistency.closure_pass}`
- Patch count match: `{layout_consistency.patch_count_match}`
- Surgery route coverage: `{layout_consistency.surgery_route_coverage}`
- Factory route coverage: `{layout_consistency.factory_route_coverage}`
- Pad ring complete: `{layout_consistency.pad_ring_complete}`
- Power mesh present: `{layout_consistency.power_mesh_present}`
- Hotspot keepout present: `{layout_consistency.hotspot_keepout_present}`
- Logical schedule labeled: `{layout_consistency.logical_schedule_labeled}`
- Hierarchy present: `{layout_consistency.hierarchy_present}`
- Missing items: `{json.dumps(layout_consistency.missing_items)}`
- Recommendations: `{json.dumps(layout_consistency.recommendations)}`

## Layout Outputs

- Layout JSON: `{layout_paths['layout_json']}`
- Layout SVG: `{layout_paths['layout_svg']}`
- Layout preview PNG: `{layout_paths['layout_preview_png']}`
- Direct GDS: `{layout_paths['gds']}`
- Direct OAS: `{layout_paths['oas']}`
- KLayout macro: `{layout_paths['klayout_py']}`

## Search Log

```json
{_json_dumps(result.search_log, indent=2)}
```
"""


def write_report(output_dir: Path, report_text: str) -> str:
    path = output_dir / "design_report.md"
    path.write_text(report_text, encoding="utf-8")
    return str(path)
