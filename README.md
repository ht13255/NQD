# NV-Center QSoC EDA

End-to-end Python EDA pipeline that takes an NV-center QSoC design spec, searches multiple architecture families, evaluates physics and implementation tradeoffs, then emits floorplan/layout artifacts.

## Features

- Normalizes user design specs for NV-center based QSoC targets.
- Parses natural-language requirements with Ollama when available, with heuristic fallback when it is not.
- Searches `sensor_dense`, `hybrid_router`, and `network_node` architectures.
- Evaluates coherence, optical readout, microwave control, thermal load, routing, area, yield, and robustness.
- Runs Monte Carlo perturbation analysis for fabrication and operating spread.
- Crawls NV-center paper pages with Scrapling and converts them into design priors.
- Builds probabilistic dense-placement plans and evaluates stochastic high-density crosstalk before final layout generation.
- Injects dense-placement and cross-talk signals into the main optimization loop, not only the final refinement stage.
- Uses a frozen neural surrogate and prototype-memory design proxy to steer optimization without any weight training.
- Lets the frozen neural surrogate directly reshape layout topology: optical trunks, control domains, readout tree, shielding pitch, mesh pitch, and macro segmentation.
- Iteratively relocates macro topology around dense cross-talk hotspots with keepout regions, shifted banks, and segmented detector placement.
- Adds QEC and logical-qubit planning with code-family selection, code distance estimation, logical error-rate modeling, decoder timing, and logical patch synthesis.
- Adds decoder-locality-aware logical scheduling, lattice-surgery channel planning, and magic-state factory timelines.
- Uses a frozen world-model style rollout proxy to score future schedule/logical pressure and steer repair actions without training.
- Dynamically selects QEC families per spec/area/latency/error budget instead of forcing a single fixed code.
- Adapts wiring density, redundancy, cluster count, and chip area pressure from the requested spec.
- Adds entanglement-link and richer error-channel simulation to strengthen networked/logical design realism.
- Re-ranks top candidates with open-source simulations: QuTiP, scikit-rf, sparse thermal solving, routing-graph congestion, optical spectrum modeling, control-loop response, phase-noise PSD, geometry-density analysis, dense crosstalk Monte Carlo, and photon-count Monte Carlo.
- Launches with a GUI by default when run without a spec path.
- Generates layout.json, layout.svg, layout_preview.png, layout.gds, layout.oas, layout_klayout.py, optimization.json, and design_report.md.

## Quick Start

### GUI-first launch

```bash
python -m nvqsoc_eda.cli
```

### Natural-language parsing with Ollama/fallback

```bash
python -m nvqsoc_eda.cli --requirements "Design an NV-center quantum repeater with 64 qubits, low latency under 200 ns, high gate fidelity, and strong cross-talk robustness." --out outputs/nl_run
```

### Team-specific terminology/profile injection

```bash
python -m nvqsoc_eda.cli --requirements-file team_request.txt --team-profile team_profile.json --team-notes "Use our internal terms: cryo-control island, shield fence, pad ring, optical trunk." --out outputs/team_run
```

### Headless run

```bash
python -m nvqsoc_eda.cli examples/quantum_repeater.json --out outputs/quantum_repeater
```

### Research-augmented run with Scrapling paper crawl and stronger refinement

```bash
python -m nvqsoc_eda.cli examples/quantum_repeater.json --out outputs/quantum_repeater_research --crawl-papers --paper-limit 4 --advanced-top-k 2 --mc-trials 64 --generations 4 --beam-width 5 --mutations 4
```

### Or install locally

```bash
pip install -e .
nvqsoc-eda examples/magnetometer.json --out outputs/magnetometer
nvqsoc-eda-gui
```

## Main Outputs

- `normalized_spec.json`: validated design input.
- `optimization.json`: best candidate, frontier, and search history.
- `layout.json`: machine-readable geometry.
- `layout.svg`: rendered floorplan and route visualization.
- `layout_preview.png`: GUI-friendly raster preview.
- `layout.gds`: direct GDSII export written with `gdstk`.
- `layout.oas`: direct OASIS export when supported by the installed `gdstk` build.
- `layout_klayout.py`: KLayout macro that reproduces the layout in KLayout Python.
- `paper_knowledge.json`: Scrapling/arXiv-derived priors and crawled paper metadata.
- `requirements_analysis.json`: natural-language decomposition and normalized spec.
- `topology_plan.json`: topology synthesized from dense signals, neural surrogate, and requirement intent.
- `qec_plan.json`: logical-qubit and QEC plan including code family, distance, decoder timing, overhead, and logical patches.
- `design_report.md`: unified human-readable report that folds requirements, paper priors, neural/world-model reasoning, QEC, schedule, and layout analysis into one file.
- `advanced/*.png`: microwave, thermal, and photon-statistics plots.
- `design_report.md`: detailed report with equations and Monte Carlo summary.

## Spec Format

See `examples/quantum_repeater.json` and `examples/magnetometer.json`.

Key fields:

- `design_name`
- `application`
- `target_qubits`
- `max_die_area_mm2`
- `max_power_mw`
- `operating_temp_k`
- `magnetic_field_mT`
- `target_t2_us`
- `target_gate_fidelity`
- `target_readout_fidelity`
- `max_latency_ns`
- `min_yield`
- `routing_preference`
- `objective_weights`

## Installation

```bash
git clone https://github.com/yourusername/nv-center-eda.git
cd nv-center-eda
pip install -e .
```

## Usage

See the Quick Start section above.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

- Thanks to the NV-center research community.
- Built with Python.
