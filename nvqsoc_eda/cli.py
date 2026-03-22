from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline
from .requirements import parse_design_requirements


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NV-center QSoC end-to-end EDA pipeline")
    parser.add_argument("spec", nargs="?", help="Path to JSON spec")
    parser.add_argument("--out", default="outputs/run", help="Output directory")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--mc-trials", type=int, default=256, help="Monte Carlo trial count")
    parser.add_argument("--generations", type=int, default=7, help="Search generations")
    parser.add_argument("--beam-width", type=int, default=8, help="Beam width")
    parser.add_argument("--mutations", type=int, default=7, help="Mutations per parent")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    parser.add_argument("--requirements", help="Natural-language requirements string to parse into a spec")
    parser.add_argument("--requirements-file", help="Path to a text/markdown file containing natural-language requirements")
    parser.add_argument("--ollama-model", default="llama3.1:8b", help="Ollama model used for natural-language parsing")
    parser.add_argument("--team-profile", help="Optional JSON file describing team terminology aliases and template hints")
    parser.add_argument("--team-notes", default="", help="Optional team-specific terminology or template notes appended to the natural-language parser")
    parser.add_argument("--parse-only", action="store_true", help="Only parse natural-language requirements and print the decomposition JSON")
    parser.add_argument("--crawl-papers", action="store_true", help="Use Scrapling to crawl NV-center paper pages and infer design priors")
    parser.add_argument("--paper-limit", type=int, default=6, help="Maximum papers to crawl")
    parser.add_argument("--advanced-top-k", type=int, default=3, help="Top frontier candidates for open-source simulation refinement")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    requirements_text = None
    if args.requirements_file:
        requirements_text = Path(args.requirements_file).read_text(encoding="utf-8")
    elif args.requirements:
        requirements_text = args.requirements

    if args.gui or (not args.spec and requirements_text is None):
        from .gui import main as gui_main

        gui_main()
        return

    requirements_bundle = None
    spec_path = Path(args.spec) if args.spec else None
    spec_data = None
    if requirements_text is not None:
        requirements_bundle = parse_design_requirements(
            requirements_text,
            model=args.ollama_model,
            team_profile_path=args.team_profile,
            team_notes=args.team_notes,
        )
        if args.parse_only:
            print(json.dumps(requirements_bundle.to_dict(), indent=2, ensure_ascii=False))
            return
        spec_data = requirements_bundle.normalized_spec

    summary = run_pipeline(
        spec_path=spec_path,
        spec_data=spec_data,
        output_dir=Path(args.out),
        seed=args.seed,
        monte_carlo_trials=args.mc_trials,
        generations=args.generations,
        beam_width=args.beam_width,
        mutations_per_parent=args.mutations,
        crawl_papers=args.crawl_papers,
        paper_limit=args.paper_limit,
        advanced_top_k=args.advanced_top_k,
        requirements_bundle=requirements_bundle,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
