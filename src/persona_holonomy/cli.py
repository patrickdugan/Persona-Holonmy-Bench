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
from .backends import make_backend
from .protocol import CONDITIONS, PHASES, PROBES, build_cases
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
        seed=args.seed,
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phb", description="Persona Holonomy Bench, version 0.1"
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
