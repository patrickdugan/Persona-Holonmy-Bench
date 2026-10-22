"""Frozen, explicitly staged PHB/PercInterp workstation pilot.

This is an orchestration and reproducibility tool, not a new attribution
estimator. The pilot deliberately requires a separate affirmative CLI call for
each stage. It neither downloads model weights nor runs a GPU job at planning.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from .protocol import PROBES, build_cases
from .sae_pipeline import capture_plan

SCHEMA = "phb.workstation_pilot.v1"
STAGES = ("capture", "fit", "graph", "select", "map", "evaluate")
OUTPUTS = {
    "capture": ("capture/capture-manifest.json",),
    "fit": ("saes/sae-registry.json",),
    "graph": ("attribution-plan.json", "attribution-graph.json"),
    "select": ("perc-selection.json",),
    "map": ("feature-map.json",),
    "evaluate": ("component-plan.json", "component-ablation.jsonl", "component-report.json"),
}


def file_sha256(filename: Path) -> str:
    h = hashlib.sha256()
    with filename.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".pilot-", delete=False) as temp:
        json.dump(value, temp, indent=2, ensure_ascii=False, allow_nan=False)
        temp.write("\n")
        name = temp.name
    os.replace(name, path)


def plan_pilot(*, output_dir: str | Path, model: str,
               checkpoint_revision: str, n_seeds: int = 8,
               first_seed: int = 100, layers: str = "mid,last",
               top_features: int = 4, max_model_forwards: int = 250,
               max_total_tokens: int = 1536,
               max_tokens_per_case: int = 32,
               random_controls: int = 4) -> dict[str, Any]:
    """Freeze PHB cases and a resource-bounded experiment plan, fully offline."""
    if not model.strip() or not checkpoint_revision.strip():
        raise ValueError("Explicit model path and pinned checkpoint revision are required")
    # A mutable branch name gives an irreproducible model identity.
    if not re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", checkpoint_revision):
        raise ValueError("--revision must be a 40-character commit or 64-character SHA256 hex digest")
    if not 8 <= n_seeds <= 1000 or top_features < 1 or random_controls < 1:
        raise ValueError("Require >=8 seeds and positive feature/random-control counts")
    if not 1 <= max_tokens_per_case <= max_total_tokens or max_model_forwards < 1:
        raise ValueError("Token and model-forward budgets must be positive")
    aliases = [v.strip() for v in layers.split(",")]
    if len(aliases) < 2 or len(set(aliases)) != len(aliases) or any(
            not re.fullmatch(r"[0-9]+|mid|last", v) for v in aliases):
        raise ValueError("Select at least two distinct decoder block aliases")
    if top_features > 256:
        raise ValueError("top_features cannot exceed 256")
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Pilot directory is already populated; use a fresh location")
    model = str(Path(model).expanduser().resolve())
    probe_names = ("flaky_ci", "parser_validator")
    probes = tuple(p for p in PROBES if p[0] in probe_names)
    cases = [case.to_dict() for case in build_cases(
        repetitions=n_seeds, seed=first_seed, variation="seeded_v1",
        conditions=("ordered", "tail_matched"), phases=("exit",), probes=probes)]
    split = capture_plan(cases, heldout_probes=("parser_validator",))
    # Offline estimate for the graph phase; actual candidate counts can be lower.
    max_graph = len(split["phb_train_seeds"]) * (2 + len(aliases) * top_features)
    if max_graph > max_model_forwards:
        raise ValueError(f"Graph preflight estimates {max_graph} forwards; cap is {max_model_forwards}")
    root.mkdir(parents=True, exist_ok=True)
    cases_path = root / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    plan = {
        "schema": SCHEMA, "status": "offline_preflight_only",
        "model": model, "checkpoint_revision": checkpoint_revision.lower(),
        "n_seeds": n_seeds, "first_seed": first_seed, "layers": aliases,
        "top_features": top_features, "random_controls": random_controls,
        "max_model_forwards_per_stage": max_model_forwards,
        "max_total_tokens": max_total_tokens,
        "max_tokens_per_case": max_tokens_per_case,
        "case_file_sha256": file_sha256(cases_path),
        "capture_split": {k: split[k] for k in (
            "phb_train_seeds", "phb_test_seeds_excluded", "phb_training_probes",
            "phb_test_probes_excluded", "sae_fit_seeds", "sae_validation_seeds")},
        "estimated_graph_forwards_upper_bound": max_graph,
        "stages": list(STAGES),
        "limitations": [
            "Planning loads no weights, performs no inference and makes no downloads.",
            "The first pilot has too few held-out seeds for strong statistical claims.",
            "Checkpoint revision is an operator declaration; verify it against local weight provenance.",
            "The SAE is fitted on narrow scripted prompts; general feature fidelity remains untested.",
            "If the trained graph has no measured input-output path, stop and report that negative result.",
        ],
    }
    _json(root / "plan.json", plan)
    _json(root / "progress.json", {"schema": SCHEMA,
                                    "plan_sha256": file_sha256(root / "plan.json"),
                                    "completed": {}})
    return plan


def _load_plan(path: str | Path):
    filename = Path(path).expanduser().resolve()
    plan = json.loads(filename.read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA or plan.get("stages") != list(STAGES):
        raise ValueError("Unexpected or incompatible pilot plan")
    root = filename.parent
    if file_sha256(root / "cases.jsonl") != plan["case_file_sha256"]:
        raise ValueError("Frozen cases.jsonl has changed; refuse continuation")
    return root, plan


def _commands(stage: str, root: Path, plan: dict, device: str, dtype: str):
    p = lambda name: str(root / name)
    model = plan["model"]
    common = ["--heldout-probes", "parser_validator"]
    base = [sys.executable, "-m", "persona_holonomy"]
    if stage == "capture":
        return [base + ["sae-capture", "--input", p("cases.jsonl"), "--model", model,
                        "--revision", plan["checkpoint_revision"], "--output-dir", p("capture"),
                        "--layers", ",".join(plan["layers"]), "--max-input-tokens",
                        str(plan["max_total_tokens"]), "--max-tokens-per-case",
                        str(plan["max_tokens_per_case"]), "--device", device,
                        "--dtype", dtype, *common]]
    if stage == "fit":
        return [base + ["sae-fit", "--capture", p("capture/capture-manifest.json"),
                        "--output-dir", p("saes"), "--n-latents", "0", "--k", "8",
                        "--steps", "100", "--batch-size", "32", "--device", "cpu"]]
    if stage == "graph":
        shared = ["--input", p("cases.jsonl"), "--registry", p("saes/sae-registry.json"),
                  "--model", model, "--layers", "all", "--top-features",
                  str(plan["top_features"]), *common]
        return [base + ["sae-attribution-plan", *shared, "--output", p("attribution-plan.json")],
                base + ["sae-attribution", *shared, "--max-total-tokens",
                        str(plan["max_total_tokens"]), "--device", device, "--dtype", dtype,
                        "--output", p("attribution-graph.json")]]
    if stage == "select":
        return [base + ["perc-select", "--graph", p("attribution-graph.json"),
                        "--shuffles", "32", "--random-controls",
                        str(plan["random_controls"]), "--output", p("perc-selection.json")]]
    if stage == "map":
        return [base + ["sae-feature-map", "--registry", p("saes/sae-registry.json"),
                        "--selection", p("perc-selection.json"), "--output", p("feature-map.json")]]
    if stage == "evaluate":
        shared = ["--manifest", p("perc-selection.json"), "--feature-map", p("feature-map.json"),
                  "--input", p("cases.jsonl"), "--model", model, *common]
        return [base + ["perc-components-plan", *shared, "--output", p("component-plan.json")],
                base + ["perc-components", *shared, "--device", device, "--dtype", dtype,
                        "--max-total-tokens", str(plan["max_total_tokens"]),
                        "--output", p("component-ablation.jsonl")],
                base + ["perc-components-report", "--input", p("component-ablation.jsonl"),
                        "--output", p("component-report.json")]]
    raise ValueError(f"Unknown stage: {stage}")


def _local_model(path: str):
    root = Path(path)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("Real GPU stages require a local HF directory with config.json")
    if not any(root.glob("*.safetensors")) and not any(root.glob("*.bin")):
        raise ValueError("Local HF checkpoint directory has no weight files")
    if not any((root / filename).exists() for filename in (
            "tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Local HF checkpoint directory has no tokenizer files")


def _check_budget(filename: Path, key: str, limit: int):
    report = json.loads(filename.read_text(encoding="utf-8"))
    forwards = report.get(key)
    if type(forwards) is not int or forwards < 0 or forwards > limit:
        raise ValueError(f"Preflight {key}={forwards} exceeds/invalidates budget {limit}")
    return forwards


def run_stage(*, plan_path: str | Path, stage: str, device: str = "cuda",
              dtype: str = "float16", runner=subprocess.run):
    """Run one opt-in stage, verifying frozen inputs and prior output digests."""
    if stage not in STAGES:
        raise ValueError(f"Stage must be one of {STAGES}")
    if device not in ("cpu", "cuda", "mps") or dtype not in (
            "float32", "float16", "bfloat16"):
        raise ValueError("Unrecognized device/dtype")
    root, plan = _load_plan(plan_path)
    progress_path = root / "progress.json"
    state = json.loads(progress_path.read_text(encoding="utf-8"))
    if (state.get("schema") != SCHEMA or not isinstance(state.get("completed"), dict)
            or state.get("plan_sha256") != file_sha256(root / "plan.json")):
        raise ValueError("Pilot plan changed or progress metadata is invalid")
    index = STAGES.index(stage)
    for prev in STAGES[:index]:
        if prev not in state["completed"]:
            raise ValueError(f"Finish stage {prev} before {stage}")
    # Always verify all finished stages, including artifacts a later stage reads.
    for finished, artifacts in state["completed"].items():
        if finished not in OUTPUTS or not isinstance(artifacts, dict):
            raise ValueError("Progress contains unknown stage/artifact metadata")
        if set(artifacts) != set(OUTPUTS[finished]):
            raise ValueError(f"Incomplete metadata for completed {finished}")
        for relative, digest in artifacts.items():
            file = root / relative
            if not file.is_file() or file_sha256(file) != digest:
                raise ValueError(f"Finished {finished} artifact changed: {relative}")
    if stage in state["completed"]:
        return {"status": "previously_completed_verified", "stage": stage}
    if any((root / name).exists() for name in OUTPUTS[stage]) or (
            stage == "capture" and (root / "capture").exists()) or (
            stage == "fit" and (root / "saes").exists()):
        raise FileExistsError("Partial or unregistered stage output exists; inspect before retry")
    if stage in ("capture", "graph", "evaluate"):
        _local_model(plan["model"])
    if stage == "select":
        graph = json.loads((root / "attribution-graph.json").read_text(encoding="utf-8"))
        if not graph.get("diagnostics", {}).get("has_measured_source_to_sink_path"):
            raise ValueError("Measured influence graph does not span; record negative result")
    commands = _commands(stage, root, plan, device, dtype)
    for index, command in enumerate(commands):
        runner(command, check=True, cwd=str(root))
        if stage == "graph" and index == 0:
            _check_budget(root / "attribution-plan.json", "estimated_model_forwards_upper_bound",
                          plan["max_model_forwards_per_stage"])
        if stage == "evaluate" and index == 0:
            _check_budget(root / "component-plan.json", "estimated_model_forwards",
                          plan["max_model_forwards_per_stage"])
    completed = {}
    for relative in OUTPUTS[stage]:
        file = root / relative
        if not file.is_file():
            raise RuntimeError(f"Stage {stage} succeeded without producing {relative}")
        completed[relative] = file_sha256(file)
    state["completed"][stage] = completed
    _json(progress_path, state)
    return {"status": "completed", "stage": stage, "artifacts": completed}
