"""Command-line entrypoint for experiment generation, execution, and analysis."""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .analysis import analyze, read_jsonl, to_markdown
from .activations import activation_report, capture
from .interventions import intervene_hf, intervention_report, intervention_markdown
from .sweep import sweep_hf, sweep_report, sweep_markdown
from .sweep_plot import render_sweep_svg
from .backends import make_backend
from .protocol import CONDITIONS, PHASES, PROBES, VARIATIONS, build_cases
from .plotting import render_svg


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
    parser.add_argument("--conditions", type=lambda x: _csv(x, CONDITIONS, "condition"),
                        default=CONDITIONS, help="CSV conditions; default: all")
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
    )
    print(f"Wrote {count} held-out causal sweep observations to {args.output}")


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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phb", description="Persona Holonomy Bench, version 0.5"
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
