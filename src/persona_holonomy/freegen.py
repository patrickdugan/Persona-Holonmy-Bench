"""Held-out free-generation validation for causal persona interventions.

This is a separate behavioral readout of rank-one directions fitted on training
histories. Greedy outputs are blinded and scored by independent raters before
unmasking candidate, orthogonal, and baseline identities.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .inference import holm_adjust, signflip_pvalue
from .interventions import (
    CausalProbe, _orthogonal_random, _read_cases, decoder_blocks,
    prepare_pairs, select_block,
)
from .sweep import _partition_pairs

CONTROLS = frozenset(("baseline", "candidate", "orthogonal"))


def run_freegen(
    *, cases: list[dict[str, Any]], model: Any, tokenizer: Any,
    layer: str = "mid", alphas: tuple[float, ...] = (1.,),
    source_conditions: tuple[str, ...] = ("ordered",),
    reference_condition: str = "tail_matched",
    phases: tuple[str, ...] = ("exit",),
    holdout_fraction: float = .4, split_seed: int = 20261022,
    heldout_probe_ids: tuple[str, ...] = ("parser_validator",),
    max_new_tokens: int = 64, max_total_tokens: int = 4096,
    device: str = "cpu", model_name: str = "unknown",
) -> list[dict[str, Any]]:
    """Free-generate baseline and directional interventions on unseen cases.

    Each trial contains three *deterministic* outcomes. These outputs must be
    blinded and scored independently: built-in keywords are not a valid
    substitute for an external style/answer-quality assessment.
    """
    if not alphas or len(set(alphas)) != len(alphas) or any(
        not math.isfinite(a) or not 0 < a <= 1 for a in alphas
    ):
        raise ValueError("Free-generation alphas must be unique within (0,1]")
    if not heldout_probe_ids:
        raise ValueError("Free-generation validation requires unseen heldout probes")
    _, blocks = decoder_blocks(model)
    block_index = select_block(layer, len(blocks))
    pairs = prepare_pairs(cases, source_conditions=source_conditions,
                          target_condition=reference_condition, phases=phases)
    train, test, training, evaluation, train_probes, eval_probes = _partition_pairs(
        pairs, holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probe_ids=heldout_probe_ids)
    model.eval()
    runner = CausalProbe(model, tokenizer, layer=str(block_index), device=device,
                         max_total_tokens=max_total_tokens)
    vectors = {}

    def capture(case):
        key = case["case_id"]
        if key not in vectors:
            vectors[key] = runner.capture_vector(case["messages"]).cpu()
        return vectors[key]

    learned = {}
    for condition in source_conditions:
        matched = [(donor, ref) for donor, ref in training if donor["condition"] == condition]
        if not matched:
            raise ValueError(f"No training histories for source condition {condition}")
        direction = runner.torch.stack([
            capture(donor) - capture(ref) for donor, ref in matched
        ]).float().mean(0)
        norm = float(runner.torch.linalg.vector_norm(direction))
        if norm < 1e-10:
            raise ValueError(f"Zero direction at decoder block {block_index}")
        center = runner.torch.stack([capture(ref) for _, ref in matched]).float().mean(0)
        learned[condition] = direction, center

    generations = {}
    rows = []
    for donor, recipient in evaluation:
        condition = donor["condition"]
        direction, center = learned[condition]
        unit = direction / runner.torch.linalg.vector_norm(direction)
        for mode, case in (("inject", recipient), ("erase", donor)):
            case_id = case["case_id"]
            if case_id not in generations:
                generations[case_id] = runner.generate(
                    case["messages"], max_new_tokens=max_new_tokens)
            baseline = generations[case_id]
            if mode == "inject":
                candidate = {"vector": direction, "mode": "add"}
                norm = float(runner.torch.linalg.vector_norm(direction))
            else:
                projection = float(runner.torch.dot(
                    (capture(donor) - center).float(), unit))
                candidate = {"vector": direction, "mode": "erase", "reference": center}
                norm = abs(projection)
            digest = hashlib.sha256(
                f"freegen|{split_seed}|{block_index}|{condition}|{case_id}|{mode}".encode()
            ).digest()
            seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
            orthogonal = _orthogonal_random(direction, seed=seed, torch=runner.torch)
            if mode == "erase":
                orthogonal = -projection * orthogonal / runner.torch.linalg.vector_norm(orthogonal)
            for alpha in alphas:
                trial = f"{block_index}|{condition}|{mode}|{case['seed']}|{case['probe_id']}|{case['phase']}|{alpha:g}"
                interventions = {
                    "baseline": None,
                    "candidate": candidate,
                    "orthogonal": {"vector": orthogonal, "mode": "add"},
                }
                for control, parameters in interventions.items():
                    response = (baseline if parameters is None else runner.generate(
                        case["messages"], alpha=alpha,
                        max_new_tokens=max_new_tokens, **parameters))
                    rows.append({
                        "schema_version": 1, "experiment": "heldout_freegen",
                        "trial_id": trial, "model": model_name,
                        "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
                        "layer": block_index, "block_path": runner.block_path,
                        "alpha": alpha, "mode": mode, "control": control,
                        "direction_source": condition,
                        "reference_condition": reference_condition,
                        "recipient_condition": case["condition"],
                        "case_id": case_id, "seed": case["seed"],
                        "probe_id": case["probe_id"], "phase": case["phase"],
                        "probe_text": case["messages"][-1]["content"],
                        "prefix_sha256": response["prefix_sha256"],
                        "training_seeds": list(train), "test_seeds": list(test),
                        "training_probe_ids": sorted(train_probes),
                        "evaluation_probe_ids": sorted(eval_probes),
                        "patch_l2_norm": 0. if control == "baseline" else alpha * norm,
                        "response": response["response"],
                        "generated_tokens": response["generated_tokens"],
                    })
    return rows


def freegen_hf(*, input_path: str, output_path: str, model_name: str,
               device: str = "auto", dtype: str = "auto",
               allow_download: bool = False, trust_remote_code: bool = False,
               **kwargs) -> int:
    """Run using locally cached HF weights unless downloads are authorized."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers with .[mechanistic]") from exc
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if getattr(torch.backends.mps, "is_available", lambda: False)()
                  else "cpu")
    if device not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unsupported device: {device}")
    if dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    load_kwargs = {"local_files_only": not allow_download,
                   "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        **load_kwargs).to(device)
    data = Path(input_path).read_bytes()
    rows = run_freegen(cases=_read_cases(input_path), model=model,
                       tokenizer=tokenizer, device=device, model_name=model_name, **kwargs)
    sha = hashlib.sha256(data).hexdigest()
    for row in rows:
        row["input_sha256"] = sha
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def blind_export(rows: list[dict[str, Any]], *, seed: int = 1729,
                 include_zero_dose: bool = False) -> tuple[list[dict], dict]:
    """Create shuffled, anonymized human-rating rows and a private key.

    The rating file includes only an opaque sample identifier, the neutral
    question, and the generated answer. Do not share the key with raters.
    """
    if not rows:
        raise ValueError("Cannot blind an empty generation file")
    entries = []
    mapping = {}
    trials = defaultdict(dict)
    for row in rows:
        if row.get("experiment") != "heldout_freegen":
            raise ValueError("Expected heldout_freegen records")
        if row["alpha"] == 0 and not include_zero_dose:
            continue
        trial = str(row["trial_id"])
        control = row["control"]
        if control not in CONTROLS or control in trials[trial]:
            raise ValueError(f"Unexpected or repeated control in trial {trial}")
        trials[trial][control] = row
    for trial, controls in sorted(trials.items()):
        if set(controls) != CONTROLS:
            raise ValueError(f"Trial {trial} missing baseline/candidate/orthogonal control")
        # The three continuations must share an identical prompt, prefix,
        # configuration and seed. A mismatched triplet is not a paired trial.
        matched_fields = ("probe_text", "prefix_sha256", "case_id", "model",
                          "seed", "probe_id", "layer", "mode", "alpha",
                          "phase", "direction_source")
        for field in matched_fields:
            observed = {json.dumps(controls[c].get(field), sort_keys=True)
                        for c in CONTROLS}
            if len(observed) != 1:
                raise ValueError(f"Trial {trial} has mismatched {field}")
        if any(controls[c].get("training_seeds") != controls["baseline"].get("training_seeds")
               or controls[c].get("test_seeds") != controls["baseline"].get("test_seeds")
               for c in CONTROLS):
            raise ValueError(f"Trial {trial} has mismatched split provenance")
        for control in sorted(CONTROLS):
            row = controls[control]
            digest = hashlib.sha256(f"{seed}|{trial}|{control}".encode()).hexdigest()[:20]
            sample_id = f"S-{digest}"
            entries.append({"sample_id": sample_id, "question": row["probe_text"],
                            "response": row["response"], "style_rating_0_to_4": "",
                            "answer_quality_0_to_4": "", "notes": ""})
            mapping[sample_id] = {
                "trial_id": trial, "control": control, "seed": row["seed"],
                "probe_id": row["probe_id"], "direction_source": row["direction_source"],
                "phase": row["phase"], "layer": row["layer"],
                "mode": row["mode"], "alpha": row["alpha"],
            }
    if not entries:
        raise ValueError("No rated trials selected")
    random.Random(seed).shuffle(entries)
    return entries, {"schema_version": 1, "blind_seed": seed,
                     "n_samples": len(entries), "lookup": mapping}


def blind_report(ratings: list[dict], key: dict) -> dict[str, Any]:
    """Unblind only after all samples have been scored independently."""
    mapping = key["lookup"]
    scored = {}
    for row in ratings:
        sid = row["sample_id"]
        if sid not in mapping or sid in scored:
            raise ValueError("Unknown or repeated sample identifier")
        values = {}
        for field in ("style_rating_0_to_4", "answer_quality_0_to_4"):
            raw = row.get(field, "")
            if raw is None or str(raw).strip() == "":
                raise ValueError("All blind ratings must be complete before unblinding")
            try:
                value = float(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid rating: {field}") from exc
            if not math.isfinite(value) or not 0 <= value <= 4:
                raise ValueError(f"Rating {field} must fall in [0,4]")
            values[field] = value
        scored[sid] = values
    if set(scored) != set(mapping):
        raise ValueError("All blinded ratings must be complete before reporting")
    by_trial = defaultdict(dict)
    for sid, scores in scored.items():
        meta = mapping[sid]
        trial, control = meta["trial_id"], meta["control"]
        if control in by_trial[trial]:
            raise ValueError("Duplicate rating control within trial")
        by_trial[trial][control] = (meta, scores)
    by_cell = defaultdict(lambda: defaultdict(list))
    for trial, observations in by_trial.items():
        if set(observations) != CONTROLS:
            raise ValueError(f"Incomplete trial {trial}")
        meta = observations["candidate"][0]
        for control in CONTROLS:
            current = observations[control][0]
            if any(current[k] != meta[k] for k in (
                "trial_id", "seed", "layer", "mode", "alpha", "probe_id",
                "phase", "direction_source",
            )):
                raise ValueError("Control metadata mismatch")
        cell = (meta["layer"], meta["mode"], meta["direction_source"],
                meta["phase"], meta["alpha"])
        orientation = 1 if meta["mode"] == "inject" else -1
        values = {name: observations[name][1] for name in CONTROLS}
        by_cell[cell][meta["seed"]].append({
            "style_specificity": orientation * (values["candidate"]["style_rating_0_to_4"] -
                                                values["orthogonal"]["style_rating_0_to_4"]),
            "quality_specificity": values["candidate"]["answer_quality_0_to_4"] -
                                   values["orthogonal"]["answer_quality_0_to_4"],
            "style_vs_baseline": orientation * (values["candidate"]["style_rating_0_to_4"] -
                                                values["baseline"]["style_rating_0_to_4"]),
        })
    summaries = []
    for (layer, mode, source, phase, alpha), seeds in sorted(by_cell.items()):
        style = [statistics.mean(p["style_specificity"] for p in probes)
                 for _, probes in sorted(seeds.items())]
        quality = [statistics.mean(p["quality_specificity"] for p in probes)
                   for _, probes in sorted(seeds.items())]
        base = [statistics.mean(p["style_vs_baseline"] for p in probes)
                for _, probes in sorted(seeds.items())]
        p = signflip_pvalue(style)
        summaries.append({"layer": layer, "mode": mode, "direction_source": source,
                          "phase": phase, "alpha": alpha, "n_seeds": len(seeds),
                          "n_trials": sum(len(probes) for probes in seeds.values()),
                          "mean_style_specificity": statistics.mean(style),
                          "mean_style_vs_baseline": statistics.mean(base),
                          "mean_answer_quality_specificity": statistics.mean(quality),
                          "paired_signflip_p_value": p["p_value"],
                          "paired_signflip_method": p["method"],
                          "holm_p_value": None})
    adjustment = holm_adjust([x["paired_signflip_p_value"] for x in summaries])
    for summary, corrected in zip(summaries, adjustment):
        summary["holm_p_value"] = corrected
    return {"schema_version": 1, "experiment": "blind_freegen_report",
            "n_samples": len(scored), "n_trials": len(by_trial),
            "hypothesis_family_size": len(summaries), "summaries": summaries,
            "limitations": [
                "Ratings depend on independent human judgments and may be subjective.",
                "Seed-level sign-flip tests require exchangeable signs under the null.",
                "Fixed prompts, scripted assistant history, and greedy decoding limit generality.",
                "A significant style shift does not establish a VPD feature or geometric holonomy.",
            ]}


def write_blind_export(input_path: str, output_prefix: str, *, seed: int = 1729) -> tuple[Path, Path]:
    rows = _read_cases(input_path)
    sheet, key = blind_export(rows, seed=seed)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.parent / (prefix.name + "-ratings.csv")
    key_path = prefix.parent / (prefix.name + "-PRIVATE-key.json")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return csv_path, key_path


def read_blind_report(ratings_path: str, key_path: str, output_path: str) -> dict:
    with Path(ratings_path).open("r", encoding="utf-8", newline="") as fh:
        ratings = list(csv.DictReader(fh))
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    report = blind_report(ratings, key)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
