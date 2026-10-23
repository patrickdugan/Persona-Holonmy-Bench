"""Command-line entrypoint for experiment generation, execution, and analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from . import __version__

from .analysis import analyze, read_jsonl, to_markdown
from .activations import activation_report, capture
from .interventions import intervene_hf, intervention_report, intervention_markdown, _read_cases
from .sweep import sweep_hf, sweep_report, sweep_markdown, sweep_plan
from .sweep_plot import render_sweep_svg
from .freegen import freegen_hf, write_blind_export, read_blind_report
from .attribution_graph import graph_plan_file, graph_hf
from .perc_bridge import select_file as perc_select_file, audit_file as perc_audit_file
from .perc_holonomy import holonomy_file as perc_holonomy_file
from .perc_components import (
    component_plan_file, component_ablation_hf, component_report_file,
)
from .sae_pipeline import capture_plan, collect_hf, fit_topk, export_feature_map
from .backends import make_backend
from .protocol import ALL_CONDITIONS, CONDITIONS, PHASES, PROBES, VARIATIONS, build_cases
from .plotting import render_svg
from .pilot import plan_pilot, run_stage, STAGES


def _csv(value: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    items = tuple(x.strip() for x in value.split(",") if x.strip())
    if not items:
        raise argparse.ArgumentTypeError(f"{label} must include at least one value")
    bad = [x for x in items if x not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown {label}: {bad}; valid choices: {', '.join(allowed)}"
        )
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError(f"Repeated {label} value")
    return items


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    probe_ids = [p[0] for p in PROBES]
    parser.add_argument("--variation", choices=VARIATIONS, default="static",
                        help="Use seeded_v1 for independent randomized transcript content")
    parser.add_argument("--seed", type=int, default=41,
                        help="First randomization seed (default: 41)")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Distinct seeds to run (default: 1)")
    parser.add_argument("--conditions", type=lambda x: _csv(x, ALL_CONDITIONS, "condition"),
                        default=CONDITIONS,
                        help="CSV conditions; default: the four original conditions "
                             "(lexical_match is opt-in)")
    parser.add_argument("--phases", type=lambda x: _csv(x, PHASES, "phase"),
                        default=("baseline", "peak", "exit", "washout_2"),
                        help="CSV checkpoints; default: baseline,peak,exit,washout_2")
    parser.add_argument("--probes", type=lambda x: _csv(x, probe_ids, "probe"),
                        default=("flaky_ci", "parser_validator"),
                        help="CSV probe IDs; default: flaky_ci,parser_validator")


def _cases(args: argparse.Namespace):
    chosen = set(args.probes)
    return build_cases(
        repetitions=args.repetitions,
        seed=args.seed, variation=args.variation,
        conditions=args.conditions,
        phases=args.phases,
        probes=tuple(p for p in PROBES if p[0] in chosen),
    )


def command_generate(args: argparse.Namespace) -> None:
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for case in _cases(args):
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    print(f"Generated {count} cases at {path}")


def command_run(args: argparse.Namespace) -> None:
    if args.backend != "mock" and not args.model:
        raise SystemExit("--model is required for a non-mock backend")
    model = args.model or "synthetic-mock-fixture"
    backend = make_backend(
        args.backend,
        model=model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        api_key_env=args.api_key_env,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = list(_cases(args))
    if not args.sequential:
        random.Random(args.execution_seed).shuffle(cases)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            start = time.monotonic()
            result = backend.generate(case.messages, seed=case.seed)
            record = {
                **case.to_dict(),
                "schema_version": 1,
                "backend": args.backend,
                "model": model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "response": result.text,
                "generation_meta": result.meta,
                "execution_seed": args.execution_seed if not args.sequential else None,
                "execution_order": count,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            count += 1
            if args.verbose:
                print(f"[{count}] {case.case_id}: {result.text[:100]!r}")
    print(f"Wrote {count} generations to {path}")
    if args.backend == "mock":
        print("These are SYNTHETIC fixture responses, not model measurements.")


def command_analyze(args: argparse.Namespace) -> None:
    report = analyze(read_jsonl(args.input))
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    if report["synthetic"]:
        print("Warning: synthetic fixture responses; no empirical inference.")


def command_plot(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(report), encoding="utf-8")
    print(f"Wrote SVG plot to {output}")


def command_capture(args: argparse.Namespace) -> None:
    count = capture(
        input_path=args.input, output_path=args.output, model_name=args.model,
        layers=args.layers, device=args.device, dtype=args.dtype,
        allow_download=args.allow_download, trust_remote_code=args.trust_remote_code,
        max_input_tokens=args.max_input_tokens, max_cases=args.max_cases,
    )
    print(f"Saved {count} full hidden-state records to {args.output}")


def command_activation_report(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = activation_report(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved descriptive activation comparisons to {output}")



def command_intervene(args: argparse.Namespace) -> None:
    count = intervene_hf(
        input_path=args.input, output_path=args.output, model_name=args.model,
        layer=args.layer, mode=args.mode, alpha=args.alpha,
        source_conditions=tuple(args.sources.split(",")),
        target_condition=args.target, phases=tuple(args.phases.split(",")),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        max_pairs=args.max_pairs, max_total_tokens=args.max_total_tokens,
        device=args.device, dtype=args.dtype,
        allow_download=args.allow_download, trust_remote_code=args.trust_remote_code,
        self_check=not args.skip_self_check,
    )
    print(f"Wrote {count} causal intervention rows to {args.output}")


def command_intervention_report(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = intervention_report(rows)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    prefix.with_suffix(".md").write_text(intervention_markdown(report), encoding="utf-8")
    print(f"Wrote {prefix.with_suffix('.json')} and {prefix.with_suffix('.md')}")


def command_sweep(args: argparse.Namespace) -> None:
    count = sweep_hf(
        input_path=args.input, output_path=args.output, model_name=args.model,
        layers=tuple(x.strip() for x in args.layers.split(",") if x.strip()),
        alphas=tuple(float(x.strip()) for x in args.alphas.split(",") if x.strip()),
        source_conditions=tuple(x.strip() for x in args.sources.split(",") if x.strip()),
        reference_condition=args.reference,
        phases=tuple(x.strip() for x in args.phases.split(",") if x.strip()),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        max_total_tokens=args.max_total_tokens,
        device=args.device, dtype=args.dtype,
        allow_download=args.allow_download, trust_remote_code=args.trust_remote_code,
        self_check=not args.skip_self_check,
        heldout_probe_ids=tuple(x.strip() for x in args.heldout_probes.split(",") if x.strip()),
    )
    print(f"Wrote {count} held-out causal sweep observations to {args.output}")


def command_sweep_plan(args: argparse.Namespace) -> None:
    """Estimate forward invocations and freeze the split before model loading."""
    path = Path(args.input)
    cases = _read_cases(path)
    report = sweep_plan(
        cases=cases,
        layers=tuple(x.strip() for x in args.layers.split(",") if x.strip()),
        alphas=tuple(float(x.strip()) for x in args.alphas.split(",") if x.strip()),
        source_conditions=tuple(x.strip() for x in args.sources.split(",") if x.strip()),
        reference_condition=args.reference,
        phases=tuple(x.strip() for x in args.phases.split(",") if x.strip()),
        holdout_fraction=args.holdout_fraction,
        split_seed=args.split_seed,
        heldout_probe_ids=tuple(x.strip() for x in args.heldout_probes.split(",") if x.strip()),
        self_check=not args.skip_self_check,
    )
    report["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Estimated {report['estimated_total_model_forwards']} model forward calls "
          f"across {len(report['layers_requested'])} layer(s), "
          f"{report['n_test_pairs']} held-out pairs. Wrote {output}")

def command_sweep_report(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = sweep_report(rows)
    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(sweep_markdown(report), encoding="utf-8")
    print(f"Wrote paired causal specificity report to {output.with_suffix('.md')}")


def command_sweep_plot(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    svg = render_sweep_svg(report, source=args.source, phase=args.phase, alpha=args.alpha)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    print(f"Wrote held-out layer comparison to {path}")



def command_freegen(args: argparse.Namespace) -> None:
    count = freegen_hf(
        input_path=args.input, output_path=args.output, model_name=args.model,
        layer=args.layer, alphas=tuple(float(x) for x in args.alphas.split(",") if x.strip()),
        source_conditions=tuple(x.strip() for x in args.sources.split(",") if x.strip()),
        reference_condition=args.reference,
        phases=tuple(x.strip() for x in args.phases.split(",") if x.strip()),
        heldout_probe_ids=tuple(x.strip() for x in args.heldout_probes.split(",") if x.strip()),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        max_total_tokens=args.max_total_tokens, max_new_tokens=args.max_new_tokens,
        device=args.device, dtype=args.dtype, allow_download=args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote {count} free-generated validation responses to {args.output}")


def command_blind_export(args: argparse.Namespace) -> None:
    csv_path, key_path = write_blind_export(args.input, args.output_prefix, seed=args.blind_seed)
    print(f"Rater sheet: {csv_path}; keep mapping PRIVATE: {key_path}")


def command_blind_report(args: argparse.Namespace) -> None:
    report = read_blind_report(args.ratings, args.key, args.output)
    print(f"Reported {report['n_trials']} fully rated trials to {args.output}")



def command_perc_select(args: argparse.Namespace) -> None:
    report = perc_select_file(args.graph, args.output, n_shuffles=args.shuffles,
                              n_random=args.random_controls, seed=args.seed,
                              z_min=args.z_min)
    print(f"Frozen {len(report['selected_nodes']['sieve'])} graph nodes "
          f"at c*={report['sieve']['c_star']:.4g}; saved {args.output}")


def command_perc_audit(args: argparse.Namespace) -> None:
    report = perc_audit_file(args.manifest, args.sweep, args.output,
                             source=args.source, phase=args.phase,
                             alpha=args.alpha, mode=args.mode)
    print(f"Audited {report['selected_block_budget']} selected decoder blocks "
          f"on {report['n_scored_seeds']} held-out seeds; wrote {args.output}")


def command_perc_holonomy(args: argparse.Namespace) -> None:
    report = perc_holonomy_file(args.activations, args.output,
                                layer=args.layer,
                                conditions=tuple(x.strip() for x in args.conditions.split(',')),
                                phases=tuple(x.strip() for x in args.phases.split(',')),
                                rank=args.rank, min_overlap=args.min_overlap,
                                epsilon=args.epsilon, seed=args.seed,
                                null_shuffles=args.null_shuffles)
    print(f"Analyzed {report['holonomy']['cycles_recorded']} certified frame loops "
          f"across {len(report['nodes'])} states; wrote {args.output}")

def command_perc_components_plan(args: argparse.Namespace) -> None:
    result = component_plan_file(
        args.manifest, args.feature_map, args.input, args.output,
        model_name=args.model, source=args.source, reference=args.reference,
        phase=args.phase, heldout_probes=tuple(x.strip() for x in args.heldout_probes.split(",") if x.strip()),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        self_check=not args.skip_self_check,
    )
    print(f"Planned {result['n_test_cases']} held-out component trials, "
          f"{result['estimated_model_forwards']} forwards; saved {args.output}")


def command_perc_components(args: argparse.Namespace) -> None:
    count = component_ablation_hf(
        manifest_path=args.manifest, feature_path=args.feature_map,
        input_path=args.input, output_path=args.output,
        model_name=args.model, source=args.source, reference=args.reference,
        phase=args.phase,
        heldout_probes=tuple(x.strip() for x in args.heldout_probes.split(",") if x.strip()),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        alpha=args.alpha, max_total_tokens=args.max_total_tokens,
        self_check=not args.skip_self_check,
        device=args.device, dtype=args.dtype, allow_download=args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Recorded {count} held-out selector ablation rows in {args.output}")


def command_perc_components_report(args: argparse.Namespace) -> None:
    report = component_report_file(args.input, args.output)
    print(f"Reported {report['n_selectors']} selectors across "
          f"{report['n_seed_clusters']} seeds to {args.output}")



def command_sae_capture_plan(args: argparse.Namespace) -> None:
    report = capture_plan(_read_cases(args.input), phase=args.phase,
        heldout_probes=tuple(x for x in args.heldout_probes.split(",") if x),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        sae_validation_fraction=args.sae_validation_fraction)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"SAE capture plan: {report['n_selected_cases']} training-only cases; saved {path}")


def command_sae_capture(args: argparse.Namespace) -> None:
    report = collect_hf(input_path=args.input, output_dir=args.output_dir,
        model_name=args.model, checkpoint_revision=args.revision,
        layers=args.layers, phase=args.phase,
        heldout_probes=tuple(x for x in args.heldout_probes.split(",") if x),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        sae_validation_fraction=args.sae_validation_fraction,
        max_input_tokens=args.max_input_tokens,
        max_tokens_per_case=args.max_tokens_per_case,
        seed=args.seed, device=args.device, dtype=args.dtype,
        allow_download=args.allow_download, trust_remote_code=args.trust_remote_code)
    print(f"Captured {len(report['blocks'])} SAE layers using training-only cases in {args.output_dir}")


def command_sae_fit(args: argparse.Namespace) -> None:
    registry = fit_topk(capture_manifest=args.capture, output_dir=args.output_dir,
        n_latents=args.n_latents, k=args.k, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device)
    print(f"Trained {len(registry['blocks'])} PercInterp TopK SAEs in {args.output_dir}")
    for block, details in registry['blocks'].items():
        print(f"  block {block}: validation FVU={details['validation_fvu']:.4f}; "
              f"coverage={details['validation_feature_coverage']:.3f}")


def command_sae_feature_map(args: argparse.Namespace) -> None:
    result = export_feature_map(registry_path=args.registry,
        selection_path=args.selection, output_path=args.output)
    print(f"Mapped {len(result['features'])} selected nodes to pinned SAEs in {args.output}")

def command_sae_attribution_plan(args: argparse.Namespace) -> None:
    report = graph_plan_file(cases_path=args.input, registry_path=args.registry,
        model_name=args.model, output_path=args.output, layers=args.layers,
        top_features=args.top_features, phase=args.phase,
        heldout_probes=tuple(x.strip() for x in args.heldout_probes.split(",") if x.strip()),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed)
    print(f"Attribution plan: {report['n_training_pairs']} training pairs, "
          f"at most {report['estimated_model_forwards_upper_bound']} forwards; {args.output}")


def command_sae_attribution(args: argparse.Namespace) -> None:
    result = graph_hf(cases_path=args.input, registry_path=args.registry,
        model_name=args.model, output_path=args.output, layers=args.layers,
        top_features=args.top_features, phase=args.phase,
        heldout_probes=tuple(x.strip() for x in args.heldout_probes.split(",") if x.strip()),
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        device=args.device, dtype=args.dtype, max_total_tokens=args.max_total_tokens,
        allow_download=args.allow_download, trust_remote_code=args.trust_remote_code)
    print(f"Measured {len(result['edges'])} signed training-only edges "
          f"using {result['diagnostics']['total_model_forwards']} forwards; {args.output}")


def command_pilot_plan(args: argparse.Namespace) -> None:
    result = plan_pilot(output_dir=args.output_dir, model=args.model,
        checkpoint_revision=args.revision, n_seeds=args.repetitions,
        first_seed=args.first_seed, layers=args.layers,
        top_features=args.top_features, random_controls=args.random_controls,
        max_model_forwards=args.max_model_forwards,
        max_total_tokens=args.max_total_tokens,
        max_tokens_per_case=args.max_tokens_per_case)
    print(f"Frozen pilot plan: {args.output_dir}, "
          f"{result['estimated_graph_forwards_upper_bound']} estimated graph forwards; "
          "zero model calls")


def command_pilot_stage(args: argparse.Namespace) -> None:
    result = run_stage(plan_path=args.plan, stage=args.stage,
                       device=args.device, dtype=args.dtype)
    print(f"Pilot {args.stage}: {result['status']}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phb", description=f"Persona Holonomy Bench, version {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Export scripted cases as JSONL")
    _add_protocol_args(generate)
    generate.add_argument("--output", default="runs/cases.jsonl")
    generate.set_defaults(func=command_generate)

    run = sub.add_parser("run", help="Generate model responses for the cases")
    _add_protocol_args(run)
    run.add_argument("--backend", choices=("mock", "ollama", "openai-compatible"),
                     default="mock")
    run.add_argument("--model", default=None,
                     help="Model identifier required for non-mock backends")
    run.add_argument("--base-url", default=None,
                     help="Explicit backend URL; defaults to loopback/local endpoints")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    run.add_argument("--api-key-env", default="OPENAI_API_KEY",
                     help="Env var holding optional compatible API key")
    run.add_argument("--output", default="runs/responses.jsonl")
    run.add_argument("--execution-seed", type=int, default=1729,
                     help="Fixed seed for randomized inference order (default: 1729)")
    run.add_argument("--sequential", action="store_true",
                     help="Disable default execution-order randomization")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=command_run)

    analysis = sub.add_parser("analyze", help="Write paired JSON and Markdown reports")
    analysis.add_argument("--input", required=True)
    analysis.add_argument("--output-prefix", default="runs/report")
    analysis.set_defaults(func=command_analyze)

    plot = sub.add_parser("plot", help="Render a summary report as an SVG trajectory")
    plot.add_argument("--report", required=True, help="Path to phb analyze JSON report")
    plot.add_argument("--output", default="runs/trajectory.svg")
    plot.set_defaults(func=command_plot)

    cap = sub.add_parser("capture", help="Record optional open-weight hidden-state vectors")
    cap.add_argument("--model", required=True,
                     help="Local/cached Hugging Face chat model or explicit model ID")
    cap.add_argument("--input", required=True, help="Cases JSONL from phb generate")
    cap.add_argument("--output", default="runs/activations.jsonl")
    cap.add_argument("--layers", default="last", help="Hidden-state indexes, e.g. 0,8,last")
    cap.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"),
                     default="auto")
    cap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                     default="auto")
    cap.add_argument("--max-input-tokens", type=int, default=4096)
    cap.add_argument("--max-cases", type=int, default=None)
    cap.add_argument("--allow-download", action="store_true",
                     help="Opt in to contacting model hub for missing weights")
    cap.add_argument("--trust-remote-code", action="store_true",
                     help="Opt in to executing custom code supplied by model")
    cap.set_defaults(func=command_capture)

    act = sub.add_parser("activation-report", help="Compare matched hidden-state vectors")
    act.add_argument("--input", required=True, help="Output JSONL from phb capture")
    act.add_argument("--output", default="runs/activation-report.json")
    act.set_defaults(func=command_activation_report)
    inv = sub.add_parser("intervene", help="Causally patch local decoder-block outputs")
    inv.add_argument("--model", required=True, help="Cached/local Hugging Face causal chat model")
    inv.add_argument("--input", required=True, help="Cases JSONL from phb generate")
    inv.add_argument("--output", default="runs/interventions.jsonl")
    inv.add_argument("--mode", choices=("rank1", "transplant"), default="rank1")
    inv.add_argument("--layer", default="last", help="Decoder block index (zero-based), or last")
    inv.add_argument("--sources", default="ordered,neutral", help="Comma-separated donor conditions")
    inv.add_argument("--target", default="tail_matched", help="Recipient condition")
    inv.add_argument("--phases", default="exit,washout_2", help="Comma-separated phases")
    inv.add_argument("--alpha", type=float, default=1.0, help="Intervention strength [0,1]")
    inv.add_argument("--holdout-fraction", type=float, default=0.4,
                     help="Fraction of independent seeds held out for rank1")
    inv.add_argument("--split-seed", type=int, default=20261022)
    inv.add_argument("--max-pairs", type=int, default=None,
                     help="Limit evaluated source-to-target pairs")
    inv.add_argument("--max-total-tokens", type=int, default=4096,
                     help="Hard limit includes prefix and forced continuation")
    inv.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    inv.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"),
                     default="auto")
    inv.add_argument("--allow-download", action="store_true")
    inv.add_argument("--trust-remote-code", action="store_true")
    inv.add_argument("--skip-self-check", action="store_true",
                     help="Skip extra unmodified-target identity patch test")
    inv.set_defaults(func=command_intervene)

    inv_report = sub.add_parser("intervention-report", help="Seed-clustered causal readout report")
    inv_report.add_argument("--input", required=True)
    inv_report.add_argument("--output-prefix", default="runs/intervention-report")
    inv_report.set_defaults(func=command_intervention_report)
    sweep = sub.add_parser("sweep", help="Held-out multi-layer rank-one injection/erasure study")
    sweep.add_argument("--model", required=True, help="Cached/local Transformers causal chat model")
    sweep.add_argument("--input", required=True, help="Seed-varied cases JSONL")
    sweep.add_argument("--output", default="runs/sweep.jsonl")
    sweep.add_argument("--layers", default="0,mid,last",
                       help="Zero-based decoder blocks: e.g. 0,4,last (mid is accepted)")
    sweep.add_argument("--alphas", default="0,0.5,1", help="Comma-separated strengths within [0,1]")
    sweep.add_argument("--sources", default="ordered,neutral")
    sweep.add_argument("--heldout-probes", default="",
                       help="CSV readout IDs excluded from direction training; must be present in JSONL")
    sweep.add_argument("--reference", default="tail_matched")
    sweep.add_argument("--phases", default="exit")
    sweep.add_argument("--holdout-fraction", type=float, default=.4)
    sweep.add_argument("--split-seed", type=int, default=20261022)
    sweep.add_argument("--max-total-tokens", type=int, default=4096)
    sweep.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    sweep.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"),
                       default="auto")
    sweep.add_argument("--allow-download", action="store_true")
    sweep.add_argument("--trust-remote-code", action="store_true")
    sweep.add_argument("--skip-self-check", action="store_true")
    sweep.set_defaults(func=command_sweep)

    plan = sub.add_parser("sweep-plan", help="Offline split audit and GPU forward-call estimate")
    plan.add_argument("--input", required=True, help="Seed-varied case JSONL")
    plan.add_argument("--output", default="runs/sweep-plan.json")
    plan.add_argument("--layers", default="0,mid,last")
    plan.add_argument("--alphas", default="0,0.5,1")
    plan.add_argument("--sources", default="ordered,neutral")
    plan.add_argument("--reference", default="tail_matched")
    plan.add_argument("--phases", default="exit")
    plan.add_argument("--heldout-probes", default="")
    plan.add_argument("--holdout-fraction", type=float, default=.4)
    plan.add_argument("--split-seed", type=int, default=20261022)
    plan.add_argument("--skip-self-check", action="store_true")
    plan.set_defaults(func=command_sweep_plan)

    sr = sub.add_parser("sweep-report", help="Paired candidate-vs-random causal specificity")
    sr.add_argument("--input", required=True)
    sr.add_argument("--output-prefix", default="runs/sweep-report")
    sr.set_defaults(func=command_sweep_report)
    sp = sub.add_parser("sweep-plot", help="Visualize causal specificity across blocks")
    sp.add_argument("--report", required=True)
    sp.add_argument("--output", default="runs/layer-sweep.svg")
    sp.add_argument("--source", default="ordered")
    sp.add_argument("--phase", default="exit")
    sp.add_argument("--alpha", type=float, default=1.)
    sp.set_defaults(func=command_sweep_plot)
    freegen = sub.add_parser("freegen", help="Held-out causal free-generation validation")
    freegen.add_argument("--model", required=True, help="Locally cached HF causal chat checkpoint")
    freegen.add_argument("--input", required=True, help="Seeded case JSONL")
    freegen.add_argument("--output", default="runs/freegen.jsonl")
    freegen.add_argument("--layer", default="mid")
    freegen.add_argument("--alphas", default="1")
    freegen.add_argument("--sources", default="ordered")
    freegen.add_argument("--reference", default="tail_matched")
    freegen.add_argument("--phases", default="exit")
    freegen.add_argument("--heldout-probes", default="parser_validator")
    freegen.add_argument("--holdout-fraction", type=float, default=.4)
    freegen.add_argument("--split-seed", type=int, default=20261022)
    freegen.add_argument("--max-new-tokens", type=int, default=64)
    freegen.add_argument("--max-total-tokens", type=int, default=4096)
    freegen.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    freegen.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    freegen.add_argument("--allow-download", action="store_true")
    freegen.add_argument("--trust-remote-code", action="store_true")
    freegen.set_defaults(func=command_freegen)

    blind = sub.add_parser("blind-export", help="Shuffle free generations for masked manual ratings")
    blind.add_argument("--input", required=True, help="JSONL from phb freegen")
    blind.add_argument("--output-prefix", default="runs/blind")
    blind.add_argument("--blind-seed", type=int, default=1729)
    blind.set_defaults(func=command_blind_export)

    rating = sub.add_parser("blind-report", help="Unblind completed ratings and calculate paired effects")
    rating.add_argument("--ratings", required=True)
    rating.add_argument("--key", required=True, help="Private mapping from blind-export")
    rating.add_argument("--output", default="runs/blind-report.json")
    rating.set_defaults(func=command_blind_report)

    perc = sub.add_parser("perc-select", help="Freeze PercInterp backbone and graph-only budget-matched selections")
    perc.add_argument("--graph", required=True, help="Direct-attribution graph JSON (see examples/percinterp)")
    perc.add_argument("--output", default="runs/perc-selection.json")
    perc.add_argument("--shuffles", type=int, default=32)
    perc.add_argument("--random-controls", type=int, default=12)
    perc.add_argument("--seed", type=int, default=20261022)
    perc.add_argument("--z-min", type=float, default=3.0)
    perc.set_defaults(func=command_perc_select)

    perc_audit = sub.add_parser("perc-audit", help="Compare frozen PercInterp block selectors on held-out PHB sweep data")
    perc_audit.add_argument("--manifest", required=True, help="Frozen JSON output of perc-select")
    perc_audit.add_argument("--sweep", required=True, help="PHB causal sweep JSONL, held out")
    perc_audit.add_argument("--output", default="runs/perc-block-audit.json")
    perc_audit.add_argument("--source", default="ordered")
    perc_audit.add_argument("--phase", default="exit")
    perc_audit.add_argument("--alpha", type=float, default=1.)
    perc_audit.add_argument("--mode", choices=("inject", "erase"), default="inject")
    perc_audit.set_defaults(func=command_perc_audit)

    perc_h = sub.add_parser("perc-holonomy", help="Frame-transport cycle diagnostics via PercInterp")
    perc_h.add_argument("--activations", required=True, help="JSONL produced by phb capture")
    perc_h.add_argument("--output", default="runs/perc-frame-holonomy.json")
    perc_h.add_argument("--layer", default="hs_8", help="One hidden-state index, e.g. hs_8")
    perc_h.add_argument("--conditions", default="ordered,tail_matched")
    perc_h.add_argument("--phases", default="peak,exit,washout_2")
    perc_h.add_argument("--rank", type=int, default=2)
    perc_h.add_argument("--min-overlap", type=float, default=0.1)
    perc_h.add_argument("--epsilon", type=float, default=0.05)
    perc_h.add_argument("--null-shuffles", type=int, default=16,
                        help="Within-phase condition-label swaps as a permutation null")
    perc_h.add_argument("--seed", type=int, default=20261022)
    perc_h.set_defaults(func=command_perc_holonomy)
    comp_plan = sub.add_parser(
        "perc-components-plan", help="Offline frozen SAE feature-map audit and forward-cost preflight")
    comp = sub.add_parser(
        "perc-components", help="Held-out TopK SAE feature-contribution ablations with matched nulls")
    for cmd in (comp_plan, comp):
        cmd.add_argument("--manifest", required=True, help="Frozen output of phb perc-select")
        cmd.add_argument("--feature-map", required=True, help="Pinned TopKSAE checkpoint and latent map JSON")
        cmd.add_argument("--input", required=True, help="Seed-distinct case JSONL")
        cmd.add_argument("--model", required=True, help="Exact model ID in graph/feature provenance")
        cmd.add_argument("--source", default="ordered")
        cmd.add_argument("--reference", default="tail_matched")
        cmd.add_argument("--phase", default="exit")
        cmd.add_argument("--heldout-probes", default="parser_validator")
        cmd.add_argument("--holdout-fraction", type=float, default=.4)
        cmd.add_argument("--split-seed", type=int, default=20261022)
        cmd.add_argument("--skip-self-check", action="store_true")
    comp_plan.add_argument("--output", default="runs/perc-components-plan.json")
    comp_plan.set_defaults(func=command_perc_components_plan)
    comp.add_argument("--output", default="runs/perc-components.jsonl")
    comp.add_argument("--alpha", type=float, default=1.)
    comp.add_argument("--max-total-tokens", type=int, default=4096)
    comp.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    comp.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    comp.add_argument("--allow-download", action="store_true")
    comp.add_argument("--trust-remote-code", action="store_true")
    comp.set_defaults(func=command_perc_components)

    cr = sub.add_parser("perc-components-report", help="Paired seed-clustered PercInterp component-ablation report")
    cr.add_argument("--input", required=True)
    cr.add_argument("--output", default="runs/perc-components-report.json")
    cr.set_defaults(func=command_perc_components_report)

    sae_plan = sub.add_parser("sae-capture-plan", help="Offline training-only SAE data split audit")
    for cmd in (sae_plan,):
        cmd.add_argument("--input", required=True)
        cmd.add_argument("--phase", default="exit")
        cmd.add_argument("--heldout-probes", default="parser_validator")
        cmd.add_argument("--holdout-fraction", type=float, default=.4)
        cmd.add_argument("--split-seed", type=int, default=20261022)
        cmd.add_argument("--sae-validation-fraction", type=float, default=.25)
    sae_plan.add_argument("--output", default="runs/sae-capture-plan.json")
    sae_plan.set_defaults(func=command_sae_capture_plan)

    sae_cap = sub.add_parser("sae-capture", help="Capture bounded token activations for train-only SAE fitting")
    sae_cap.add_argument("--input", required=True)
    sae_cap.add_argument("--model", required=True)
    sae_cap.add_argument("--revision", required=True, help="Immutable model commit or weights digest")
    sae_cap.add_argument("--output-dir", required=True)
    sae_cap.add_argument("--layers", default="0,mid,last")
    sae_cap.add_argument("--phase", default="exit")
    sae_cap.add_argument("--heldout-probes", default="parser_validator")
    sae_cap.add_argument("--holdout-fraction", type=float, default=.4)
    sae_cap.add_argument("--split-seed", type=int, default=20261022)
    sae_cap.add_argument("--sae-validation-fraction", type=float, default=.25)
    sae_cap.add_argument("--max-input-tokens", type=int, default=4096)
    sae_cap.add_argument("--max-tokens-per-case", type=int, default=64)
    sae_cap.add_argument("--seed", type=int, default=20261022)
    sae_cap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    sae_cap.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    sae_cap.add_argument("--allow-download", action="store_true")
    sae_cap.add_argument("--trust-remote-code", action="store_true")
    sae_cap.set_defaults(func=command_sae_capture)

    sae_fit = sub.add_parser("sae-fit", help="Train PercInterp TopKSAE checkpoints and validation report")
    sae_fit.add_argument("--capture", required=True, help="capture-manifest.json generated by sae-capture")
    sae_fit.add_argument("--output-dir", required=True)
    sae_fit.add_argument("--n-latents", type=int, default=0, help="Dictionary size (0 means 4x hidden width)")
    sae_fit.add_argument("--k", type=int, default=8)
    sae_fit.add_argument("--steps", type=int, default=100)
    sae_fit.add_argument("--batch-size", type=int, default=32)
    sae_fit.add_argument("--lr", type=float, default=.0003)
    sae_fit.add_argument("--seed", type=int, default=20261022)
    sae_fit.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    sae_fit.set_defaults(func=command_sae_fit)

    sae_map = sub.add_parser("sae-feature-map", help="Bind frozen PercInterp selected nodes to SAE registry")
    sae_map.add_argument("--registry", required=True)
    sae_map.add_argument("--selection", required=True)
    sae_map.add_argument("--output", required=True)
    sae_map.set_defaults(func=command_sae_feature_map)

    attr_plan = sub.add_parser("sae-attribution-plan",
        help="Offline provenance and forward-cost gate for signed SAE influence edges")
    attr = sub.add_parser("sae-attribution",
        help="Training-only single-latent finite-difference influence graph")
    for cmd in (attr_plan, attr):
        cmd.add_argument("--input", required=True, help="Seeded PHB cases JSONL")
        cmd.add_argument("--registry", required=True, help="SAE registry from sae-fit")
        cmd.add_argument("--model", required=True, help="Exact cached HF model / registry identity")
        cmd.add_argument("--layers", default="all", help="Ascending SAE decoder blocks or 'all'")
        cmd.add_argument("--top-features", type=int, default=8,
                         help="Candidate latents per SAE block, ranked on training contrast only")
        cmd.add_argument("--phase", default="exit")
        cmd.add_argument("--heldout-probes", default="parser_validator")
        cmd.add_argument("--holdout-fraction", type=float, default=.4)
        cmd.add_argument("--split-seed", type=int, default=20261022)
    attr_plan.add_argument("--output", default="runs/sae-attribution-plan.json")
    attr_plan.set_defaults(func=command_sae_attribution_plan)
    attr.add_argument("--output", default="runs/sae-attribution-graph.json")
    attr.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    attr.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    attr.add_argument("--max-total-tokens", type=int, default=4096)
    attr.add_argument("--allow-download", action="store_true")
    attr.add_argument("--trust-remote-code", action="store_true")
    attr.set_defaults(func=command_sae_attribution)

    pilot = sub.add_parser("pilot-plan", help="Freeze an offline, budget-capped Qwen/PercInterp pilot")
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--model", required=True, help="Exact local Hugging Face checkpoint directory")
    pilot.add_argument("--revision", required=True, help="Immutable 40-hex commit or 64-hex weights digest")
    pilot.add_argument("--repetitions", type=int, default=8)
    pilot.add_argument("--first-seed", type=int, default=100)
    pilot.add_argument("--layers", default="mid,last")
    pilot.add_argument("--top-features", type=int, default=4)
    pilot.add_argument("--random-controls", type=int, default=4)
    pilot.add_argument("--max-model-forwards", type=int, default=250)
    pilot.add_argument("--max-total-tokens", type=int, default=1536)
    pilot.add_argument("--max-tokens-per-case", type=int, default=32)
    pilot.set_defaults(func=command_pilot_plan)

    stage = sub.add_parser("pilot-stage", help="Run one explicit verified phase of a frozen pilot")
    stage.add_argument("--plan", required=True, help="Frozen pilot plan.json")
    stage.add_argument("--stage", choices=STAGES, required=True)
    stage.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    stage.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    stage.set_defaults(func=command_pilot_stage)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    if getattr(args, "repetitions", 1) < 1:
        parser.error("--repetitions must be positive")
    if getattr(args, "max_tokens", 1) < 1:
        parser.error("--max-tokens must be positive")
    if getattr(args, "temperature", 0) < 0:
        parser.error("--temperature must be >= 0")
    args.func(args)


if __name__ == "__main__":
    main()
