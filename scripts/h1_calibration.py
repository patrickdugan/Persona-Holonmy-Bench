"""H1 calibration: does an ordered persona history carry over into plain questions?

Unpatched model only. Scores the forced-continuation style margin for every
case and, at the chosen generation phases, a greedy answer with the lexical
style scorer. Seeds must come from a calibration range that no confirmatory
plan uses. Not evidence for any preregistered hypothesis.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from persona_holonomy.inference import signflip_pvalue
from persona_holonomy.interventions import CausalProbe
from persona_holonomy.protocol import ALL_PROBES, build_cases
from persona_holonomy.scoring import score_response

METRICS = ("margin", "first_token_margin", "later_token_margin",
           "copy_free_margin", "copy_free_first_token_margin", "copy_free_later_token_margin",
           "copyable_margin", "copyable_first_token_margin", "copyable_later_token_margin",
           "persona_index")
# The copy-free readout averages the package's `targets.COPY_FREE_TARGETS`
# registers. Calibrations before 2026-10-24 used a single calibration-only
# "novel" target per probe instead (script at commit 9a2c1e6).


def _csv(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def summarize(rows: list[dict], *, contrasts: tuple[tuple[str, str], ...]) -> dict:
    cells = {(r["seed"], r["probe_id"], r["phase"], r["condition"]): r for r in rows}
    phases = sorted({r["phase"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    probes = sorted({r["probe_id"] for r in rows})
    out: dict = {"n_seeds": len(seeds), "probes": probes, "contrasts": {}, "conditions": {}}
    for phase in phases:
        for condition in sorted({r["condition"] for r in rows}):
            chosen = [r for r in rows if r["phase"] == phase and r["condition"] == condition]
            entry = {m: statistics.mean(r[m] for r in chosen)
                     for m in METRICS + ("prefix_tokens",)
                     if all(r.get(m) is not None for r in chosen)}
            if all("genre_intrusion" in r for r in chosen):
                entry["genre_intrusion_rate"] = statistics.mean(r["genre_intrusion"] for r in chosen)
            out["conditions"][f"{phase}/{condition}"] = entry
        for left, right in contrasts:
            key = f"{phase}/{left}-{right}"
            result = {}
            for metric in METRICS + ("prefix_tokens",):
                pairs = {(s, p): (cells.get((s, p, phase, left), {}).get(metric),
                                  cells.get((s, p, phase, right), {}).get(metric))
                         for s in seeds for p in probes}
                if any(a is None or b is None for a, b in pairs.values()):
                    continue  # metric not measured in this phase (e.g. no generation)
                per_seed = [statistics.mean(pairs[s, p][0] - pairs[s, p][1] for p in probes)
                            for s in seeds]
                per_probe = {p: [pairs[s, p][0] - pairs[s, p][1] for s in seeds] for p in probes}
                result[metric] = {
                    "mean": statistics.mean(per_seed),
                    "sd_seed": statistics.stdev(per_seed) if len(per_seed) > 1 else None,
                    "positive_seeds": sum(v > 0 for v in per_seed),
                    "signflip": signflip_pvalue(per_seed),
                    "per_probe_mean": {p: statistics.mean(v) for p, v in per_probe.items()},
                }
            out["contrasts"][key] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF snapshot directory")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--conditions", default="ordered,tail_matched,neutral,permuted")
    parser.add_argument("--contrasts", default="",
                        help="Extra CSV of left-right condition pairs, e.g. lexical_match-neutral")
    parser.add_argument("--phases", default="baseline,exit,washout_2")
    parser.add_argument("--generate-phases", default="exit")
    parser.add_argument("--probes", default="flaky_ci,parser_validator,stale_cache")
    parser.add_argument("--genres", default="gothic")
    parser.add_argument("--system-variants", default="default")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Render the chat template with enable_thinking=False")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    probe_ids = _csv(args.probes)
    cases = [c.to_dict() for c in build_cases(
        repetitions=args.repetitions, seed=args.seed, variation="seeded_v1",
        conditions=_csv(args.conditions), phases=_csv(args.phases),
        probes=tuple(p for p in ALL_PROBES if p[0] in probe_ids),
        genres=_csv(args.genres), system_variants=_csv(args.system_variants))]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), local_files_only=True).to("cuda")
    template_kwargs = {"enable_thinking": False} if args.no_thinking else None
    runner = CausalProbe(model, tokenizer, layer="last", device="cuda",
                         max_total_tokens=args.max_total_tokens,
                         chat_template_kwargs=template_kwargs)
    generate_phases = set(_csv(args.generate_phases))
    started = time.time()
    rows = []
    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for number, case in enumerate(cases, 1):
            score = runner.all_margins(case["messages"], probe_id=case["probe_id"],
                                       genre=case["genre"])
            row = {k: case[k] for k in ("case_id", "seed", "probe_id", "phase", "condition",
                                       "genre", "system_variant")}
            row.update({k: score.get(k) for k in ("margin", "first_token_margin",
                                                  "later_token_margin", "prefix_sha256")})
            row["prefix_tokens"] = score["prefix_tokens"]
            row.update({k: v for k, v in score.items()
                        if k.startswith(("copy_free_", "copyable_"))})
            if case["phase"] in generate_phases:
                generated = runner.generate(case["messages"], max_new_tokens=args.max_new_tokens)
                row["response"] = generated["response"]
                row.update(score_response(generated["response"]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if number % 25 == 0:
                print(f"{number}/{len(cases)} cases, {time.time() - started:.0f}s", flush=True)
    contrasts = tuple(("ordered", c) for c in _csv(args.conditions) if c != "ordered") + tuple(
        tuple(pair.split("-", 1)) for pair in _csv(args.contrasts))
    summary = {
        "schema": "phb.h1_calibration.v1",
        "model": args.model, "revision": args.revision, "dtype": args.dtype,
        "chat_template_kwargs": template_kwargs,
        "seeds": [args.seed, args.seed + args.repetitions - 1],
        "n_cases": len(cases), "elapsed_seconds": round(time.time() - started, 1),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "groups": {f"{g}/{v}": summarize(
            [r for r in rows if r["genre"] == g and r["system_variant"] == v],
            contrasts=contrasts)
            for g in _csv(args.genres) for v in _csv(args.system_variants)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {len(cases)} cases in {summary['elapsed_seconds']}s -> {out}")


if __name__ == "__main__":
    main()
