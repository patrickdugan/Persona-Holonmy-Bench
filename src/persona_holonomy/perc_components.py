"""Held-out SAE-latent contribution ablations for frozen PercInterp selections.

This is a *feature-level* perturbation of the contribution predicted by a
pretrained TopK SAE at the baseline prefix activation. It is not a retraining
or causal proof that the selected SAE latents faithfully represent the source
attribution graph. The residual reconstruction error is kept unchanged.

Public interchange schema: ``phb.percinterp.feature_map.v1``. The external
PercInterp graph supplies selectors; this module never reselects using test data.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from .inference import holm_adjust, signflip_pvalue
from .interventions import (
    CausalProbe, _capture_hook, _orthogonal_random, _patch_hook,
    _read_cases, decoder_blocks, prepare_pairs,
)
from .perc_bridge import SELECT_SCHEMA
from .sweep import _partition_pairs

FEATURE_MAP_SCHEMA = "phb.percinterp.feature_map.v1"
COMPONENT_SCHEMA = "phb.percinterp.component_ablation.v1"
REPORT_SCHEMA = "phb.percinterp.component_report.v1"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _model_identity(manifest: dict, feature_map: dict, *, model_name: str,
                    tokenizer_name: str | None) -> None:
    provenance = manifest.get("provenance", {})
    if provenance.get("model") != model_name or feature_map.get("model") != model_name:
        raise ValueError("Graph, feature map and target model identities must agree")
    for value in (provenance.get("tokenizer"), feature_map.get("tokenizer")):
        if value is not None and tokenizer_name is not None and value != tokenizer_name:
            raise ValueError("Graph/feature-map tokenizer does not match target")
    if (provenance.get("tokenizer") is not None and feature_map.get("tokenizer") is not None
            and provenance["tokenizer"] != feature_map["tokenizer"]):
        raise ValueError("Graph and feature-map tokenizer IDs differ")


def _sha256_json(obj: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_plan(manifest: dict, feature_map: dict, *, model_name: str,
                   tokenizer_name: str | None = None) -> dict[str, dict[int, tuple[int, ...]]]:
    """Validate frozen selections and per-block latent budgets without Torch."""
    if manifest.get("schema") != SELECT_SCHEMA:
        raise ValueError("Expected frozen PercInterp selection manifest")
    if feature_map.get("schema") != FEATURE_MAP_SCHEMA:
        raise ValueError(f"Feature map requires schema {FEATURE_MAP_SCHEMA}")
    _model_identity(manifest, feature_map, model_name=model_name,
                    tokenizer_name=tokenizer_name)
    if feature_map.get("selection_graph_sha256") != manifest.get("graph_sha256"):
        raise ValueError("Feature map must be pinned to the selection graph SHA256")
    selected = manifest.get("selected_nodes")
    features = feature_map.get("features")
    checkpoints = feature_map.get("checkpoints")
    if not isinstance(selected, dict) or "sieve" not in selected or not selected["sieve"]:
        raise ValueError("PercInterp manifest must have a nonempty sieve selection")
    if not isinstance(features, dict) or not isinstance(checkpoints, dict):
        raise ValueError("Feature map must contain features and checkpoints")
    if not isinstance(feature_map.get("checkpoint_revision"), str) or not feature_map["checkpoint_revision"].strip():
        raise ValueError("Pin the immutable model checkpoint revision in feature_map.checkpoint_revision")
    if not all(isinstance(k, str) and k.isdigit() for k in checkpoints):
        raise ValueError("Checkpoint keys must be zero-based decoder block numbers")
    seen_locations: set[tuple[int, int]] = set()
    declared_blocks = manifest.get("node_to_decoder_block")
    if not isinstance(declared_blocks, dict):
        raise ValueError("Frozen selection must preserve node_to_decoder_block mappings")
    for name, location in features.items():
        if not isinstance(name, str) or not name or not isinstance(location, dict):
            raise ValueError("Feature IDs and locations must be explicit")
        block, latent = location.get("block"), location.get("latent")
        if type(block) is not int or type(latent) is not int or block < 0 or latent < 0:
            raise ValueError("Every feature needs nonnegative integer block/latent indices")
        if str(block) not in checkpoints or (block, latent) in seen_locations:
            raise ValueError("Unmapped checkpoint or duplicate feature location")
        seen_locations.add((block, latent))
        graph_block = declared_blocks.get(name)
        if graph_block is None or graph_block != block:
            raise ValueError(f"Feature {name} has a missing or different graph decoder_block")
    for block, entry in checkpoints.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"Checkpoint {block} must specify a local path")
        if type(entry.get("top_k")) is not int or entry["top_k"] < 1:
            raise ValueError(f"Checkpoint {block} requires positive top_k")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"Checkpoint {block} needs lowercase SHA256")
    frozen: dict[str, dict[int, tuple[int, ...]]] = {}
    sieve_budgets = None
    for selector, members in selected.items():
        if not isinstance(selector, str) or not isinstance(members, list) or len(set(members)) != len(members):
            raise ValueError("Selected features must be unique lists")
        by_block: dict[int, list[int]] = defaultdict(list)
        for name in members:
            if name not in features:
                raise ValueError(f"Selected node {name!r} lacks a feature-map location")
            location = features[name]
            by_block[location["block"]].append(location["latent"])
        frozen[selector] = {b: tuple(sorted(v)) for b, v in sorted(by_block.items())}
        budgets = {b: len(v) for b, v in frozen[selector].items()}
        if selector == "sieve":
            sieve_budgets = budgets
        elif budgets != sieve_budgets:
            raise ValueError(f"Selector {selector} has unequal per-decoder-block latent budget")
    if not frozen["sieve"]:
        raise ValueError("Sieve must select at least one mapped SAE latent")
    return frozen


def _partition(manifest: dict, cases: list[dict], *, source: str,
               reference: str, phase: str, holdout_fraction: float,
               split_seed: int, heldout_probes: tuple[str, ...]):
    if not heldout_probes:
        raise ValueError("A component-ablation run requires a disjoint held-out probe")
    pairs = prepare_pairs(cases, source_conditions=(source,),
                          target_condition=reference, phases=(phase,))
    train, test, training, evaluation, train_probes, eval_probes = _partition_pairs(
        pairs, holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probe_ids=heldout_probes)
    provenance = manifest.get("provenance", {})
    if set(provenance.get("train_seeds", [])) != set(train) or not train:
        raise ValueError("Graph training seeds must equal the frozen PHB training split")
    if set(provenance.get("training_probe_ids", [])) != set(train_probes):
        raise ValueError("Graph training probes must equal the frozen PHB training probes")
    if not set(train).isdisjoint(test) or not set(train_probes).isdisjoint(eval_probes):
        raise ValueError("Seed/probe leakage detected")
    return train, test, evaluation, train_probes, eval_probes


def plan_component_ablation(*, manifest: dict, feature_map: dict, cases: list[dict],
                            model_name: str, tokenizer_name: str | None = None,
                            source: str = "ordered", reference: str = "tail_matched",
                            phase: str = "exit", holdout_fraction: float = .4,
                            split_seed: int = 20261022,
                            heldout_probes: tuple[str, ...] = ("parser_validator",),
                            self_check: bool = True,
                            map_root: str | Path = ".",
                            check_sha256: bool = True) -> dict[str, Any]:
    """Cheap metadata/hashes and forward-call budget; no GPU or Torch needed."""
    frozen = _validate_plan(manifest, feature_map, model_name=model_name,
                            tokenizer_name=tokenizer_name)
    train, test, evaluation, train_probes, eval_probes = _partition(
        manifest, cases, source=source, reference=reference, phase=phase,
        holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probes=heldout_probes)
    hashes = {}
    if check_sha256:
        root = Path(map_root)
        for block in sorted({b for mapping in frozen.values() for b in mapping}):
            entry = feature_map["checkpoints"][str(block)]
            path = root / entry["path"]
            if not path.is_file() or (digest := _hash_file(path)) != entry["sha256"]:
                raise ValueError(f"SAE checkpoint missing or SHA256 mismatch at block {block}: {path}")
            hashes[str(block)] = digest
    n = len(evaluation)
    # One activation capture per case; two scores per baseline, candidate,
    # and orthogonal selector per case. Optional sham is two more forwards.
    per_case = 1 + 2 + 4 * len(frozen) + (2 if self_check else 0)
    return {
        "schema": "phb.percinterp.component_plan.v1",
        "model": model_name,
        "checkpoint_revision": feature_map["checkpoint_revision"],
        "graph_sha256": manifest["graph_sha256"],
        "feature_map_sha256": _sha256_json(feature_map),
        "checkpoint_sha256": hashes,
        "source": source, "reference": reference, "phase": phase,
        "train_seeds": list(train), "test_seeds": list(test),
        "train_probe_ids": sorted(train_probes), "test_probe_ids": sorted(eval_probes),
        "n_test_cases": n,
        "selector_budgets": {s: {str(b): len(v) for b, v in loc.items()}
                             for s, loc in frozen.items()},
        "n_unique_component_sets": len({tuple((b, v) for b, v in loc.items())
                                        for loc in frozen.values()}),
        "estimated_model_forwards": n * per_case,
        "caveat": "Forwards are not FLOPs, elapsed time, VRAM, or proof of graph faithfulness.",
    }


def _sae_tensors(entry: dict, *, root: str | Path, torch: Any, device: str,
                 check_hash: bool = True) -> dict[str, Any]:
    path = Path(root) / entry["path"]
    if not path.is_file():
        raise ValueError(f"Missing local SAE checkpoint: {path}")
    if check_hash and _hash_file(path) != entry["sha256"]:
        raise ValueError(f"SAE checkpoint SHA256 changed: {path}")
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Install safetensors to read this checkpoint") from exc
        state = load_file(str(path), device="cpu")
    elif suffix == ".pt":
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            config = state.get("config", {})
            if isinstance(config, dict) and config.get("k") != entry["top_k"]:
                raise ValueError("Checkpoint TopKSAE config.k differs from feature-map top_k")
            state = state["state_dict"]
    else:
        raise ValueError("SAE checkpoint must use .safetensors or trusted weights-only .pt")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain state_dict-like tensor keys")
    required = ("W_enc", "W_dec", "b_enc", "b_dec", "scale")
    if any(k not in state for k in required):
        raise ValueError("SAE must contain PercInterp TopKSAE W_enc/W_dec/biases/scale")
    if not all(isinstance(state[k], torch.Tensor) for k in required):
        raise ValueError("All SAE weights must be torch tensors")
    sae = {k: state[k].to(device=device, dtype=torch.float32).detach() for k in required}
    wenc, wdec, benc, bdec, scale = (sae[k] for k in required)
    if (wenc.ndim != 2 or wdec.shape != wenc.T.shape or
            benc.shape != (wenc.shape[1],) or bdec.shape != (wenc.shape[0],) or
            scale.numel() != 1 or not (1 <= entry["top_k"] <= wenc.shape[1])):
        raise ValueError("Checkpoint SAE dimensions or top_k inconsistent")
    if not all(torch.isfinite(t).all().item() for t in sae.values()) or float(scale) <= 0:
        raise ValueError("SAE contains nonfinite tensors or nonpositive scale")
    sae["top_k"] = entry["top_k"]
    return sae


def _selected_contribution(vector: Any, indices: tuple[int, ...], sae: dict,
                           *, torch: Any):
    """Contribution of baseline-active TopK latents, preserving reconstruction error."""
    wenc, wdec = sae["W_enc"], sae["W_dec"]
    if vector.ndim != 1 or vector.numel() != wenc.shape[0]:
        raise ValueError("Decoder hidden size differs from SAE dictionary")
    if not indices or min(indices) < 0 or max(indices) >= wenc.shape[1]:
        raise ValueError("Selected latent ID outside SAE dictionary")
    x = vector.to(device=wenc.device, dtype=torch.float32)
    pre = (x * sae["scale"] - sae["b_dec"]) @ wenc + sae["b_enc"]
    values, ids = pre.topk(sae["top_k"])
    active = torch.zeros_like(pre).scatter(0, ids, torch.nn.functional.relu(values))
    selected = torch.tensor(indices, dtype=torch.long, device=wenc.device)
    return (active[selected] @ wdec[selected]) / sae["scale"]


def _capture_all(*, model: Any, runner: CausalProbe, blocks: Any,
                 layers: tuple[int, ...], messages: list[dict]):
    prefix = runner._prefix(messages)
    states: dict[int, dict] = {}
    with ExitStack() as stack, runner.torch.inference_mode():
        for index in layers:
            state: dict[str, Any] = {}
            stack.enter_context(_capture_hook(blocks[index], state))
            states[index] = state
        model(**prefix, use_cache=False, return_dict=True)
    if any(s["calls"] != 1 for s in states.values()):
        raise RuntimeError("Feature activation capture must invoke each decoder block once")
    return {i: state["vector"].float() for i, state in states.items()}


def _readout_with_patches(*, runner: CausalProbe, messages: list[dict], probe_id: str,
                          blocks: Any, patches: dict[int, Any], alpha: float):
    prefix = runner._prefix(messages)
    position = int(prefix["input_ids"].shape[1]) - 1
    counters = {}
    with ExitStack() as stack:
        for block in sorted(patches):
            state = {}
            stack.enter_context(_patch_hook(blocks[block], position=position,
                vector=patches[block], alpha=alpha, mode="add", state=state))
            counters[block] = state
        score = runner.style_margin(messages, probe_id=probe_id)
    if any(s["calls"] != 2 for s in counters.values()):
        raise RuntimeError("Feature patch did not run once per forced continuation")
    return score


def _random_null(delta: Any, *, seed: int, torch: Any):
    if float(torch.linalg.vector_norm(delta)) < 1e-9:
        return torch.zeros_like(delta)
    return _orthogonal_random(delta, seed=seed, torch=torch)


def run_component_ablation(*, manifest: dict, feature_map: dict,
                           cases: list[dict], model: Any, tokenizer: Any,
                           model_name: str, map_root: str | Path,
                           source: str = "ordered", reference: str = "tail_matched",
                           phase: str = "exit", heldout_probes: tuple[str, ...] = ("parser_validator",),
                           holdout_fraction: float = .4, split_seed: int = 20261022,
                           alpha: float = 1., max_total_tokens: int = 4096,
                           device: str = "cpu", self_check: bool = True,
                           chat_template_kwargs: dict | None = None) -> list[dict[str, Any]]:
    """Ablate SAE-latent contributions, compared to norm-matched orthogonal nulls.

    Frozen feature contributions are computed from a baseline forward pass per
    held-out prompt. A simultaneous patch across decoder blocks subtracts the
    saved contribution at the final prefix token. It never refits the SAE.
    """
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite within (0,1]")
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "unknown"))
    plan = plan_component_ablation(
        manifest=manifest, feature_map=feature_map, cases=cases,
        model_name=model_name, tokenizer_name=tokenizer_name,
        source=source, reference=reference, phase=phase,
        holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probes=heldout_probes, map_root=map_root, self_check=self_check)
    frozen = _validate_plan(manifest, feature_map, model_name=model_name,
                            tokenizer_name=tokenizer_name)
    _, _, evaluation, _, _ = _partition(
        manifest, cases, source=source, reference=reference, phase=phase,
        holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probes=heldout_probes)
    model.eval()
    torch = __import__("torch")
    path, blocks = decoder_blocks(model)
    all_blocks = tuple(sorted({b for selector in frozen.values() for b in selector}))
    if not all_blocks or max(all_blocks) >= len(blocks):
        raise ValueError("SAE feature maps contain unavailable decoder blocks")
    sae_by_block = {
        b: _sae_tensors(feature_map["checkpoints"][str(b)], root=map_root,
                        torch=torch, device=device)
        for b in all_blocks
    }
    runner = CausalProbe(model, tokenizer, layer=str(all_blocks[0]),
                         device=device, max_total_tokens=max_total_tokens,
                         chat_template_kwargs=chat_template_kwargs)
    out = []
    cache: dict[tuple, dict] = {}
    for donor, _recipient in evaluation:
        case = donor   # Determine necessity on the held-out *persona* condition.
        key = (case["seed"], case["probe_id"], case["phase"])
        if key in cache:
            raise ValueError("Repeated evaluation seed/probe/phase cell")
        baseline_vectors = _capture_all(model=model, runner=runner, blocks=blocks,
                                        layers=all_blocks, messages=case["messages"])
        baseline = runner.style_margin(case["messages"], probe_id=case["probe_id"])
        cache[key] = baseline
        selected_by_set: dict[tuple, tuple[dict[int, Any], dict[int, Any]]] = {}
        for selector, locations in frozen.items():
            identity = tuple(sorted(locations.items()))
            if identity not in selected_by_set:
                patch = {}
                null = {}
                for b, indices in locations.items():
                    delta = _selected_contribution(baseline_vectors[b], indices,
                                                   sae_by_block[b], torch=torch).detach().cpu()
                    patch[b] = -delta
                    signature = f"{split_seed}|{case['case_id']}|{b}|{indices}"
                    seed = int.from_bytes(hashlib.sha256(signature.encode()).digest()[:8], "big") % (2**63 - 1)
                    null[b] = _random_null(-delta, seed=seed, torch=torch).detach().cpu()
                selected_by_set[identity] = (patch, null)
            patch, null = selected_by_set[identity]
            if self_check and selector == "sieve":
                sham = _readout_with_patches(runner=runner, messages=case["messages"],
                    probe_id=case["probe_id"], blocks=blocks, patches=patch, alpha=0.)
                if abs(sham["margin"] - baseline["margin"]) > 1e-5:
                    raise RuntimeError("Alpha-zero patched self-check changed model predictions")
            candidate = _readout_with_patches(runner=runner,
                messages=case["messages"], probe_id=case["probe_id"],
                blocks=blocks, patches=patch, alpha=alpha)
            orthogonal = _readout_with_patches(runner=runner,
                messages=case["messages"], probe_id=case["probe_id"],
                blocks=blocks, patches=null, alpha=alpha)
            norm = {str(b): float(torch.linalg.vector_norm(v)) * alpha for b, v in patch.items()}
            null_norm = {str(b): float(torch.linalg.vector_norm(v)) * alpha for b, v in null.items()}
            if any(abs(norm[b] - null_norm[b]) > 1e-4 * max(1., norm[b]) for b in norm):
                raise RuntimeError("Candidate and orthogonal null patch norms diverged")
            out.append({
                **({"chat_template_kwargs": chat_template_kwargs}
                   if chat_template_kwargs is not None else {}),
                "schema": COMPONENT_SCHEMA, "model": model_name,
                "tokenizer": tokenizer_name, "checkpoint_revision": feature_map["checkpoint_revision"],
                "block_path": path, "selection_graph_sha256": manifest["graph_sha256"],
                "feature_map_sha256": plan["feature_map_sha256"],
                "sae_sha256": plan["checkpoint_sha256"],
                "source": source, "reference": reference, "phase": phase,
                "selector": selector, "selected_node_ids": manifest["selected_nodes"][selector],
                "latent_budget": {str(b): len(indices) for b, indices in locations.items()},
                "seed": case["seed"], "probe_id": case["probe_id"],
                "case_id": case["case_id"], "prefix_sha256": baseline["prefix_sha256"],
                "train_seeds": plan["train_seeds"], "test_seeds": plan["test_seeds"],
                "train_probe_ids": plan["train_probe_ids"], "test_probe_ids": plan["test_probe_ids"],
                "alpha": alpha, "candidate_patch_norms": norm, "orthogonal_patch_norms": null_norm,
                "baseline_margin": baseline["margin"],
                "candidate_margin": candidate["margin"],
                "orthogonal_margin": orthogonal["margin"],
                "baseline_first_token": baseline["first_token_margin"],
                "candidate_first_token": candidate["first_token_margin"],
                "orthogonal_first_token": orthogonal["first_token_margin"],
                "baseline_later_token": baseline["later_token_margin"],
                "candidate_later_token": candidate["later_token_margin"],
                "orthogonal_later_token": orthogonal["later_token_margin"],
                "candidate_effect": baseline["margin"] - candidate["margin"],
                "orthogonal_effect": baseline["margin"] - orthogonal["margin"],
                "specificity": orthogonal["margin"] - candidate["margin"],
            })
    return out


def component_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired selector comparison with seed as the randomization unit."""
    if not rows or any(r.get("schema") != COMPONENT_SCHEMA for r in rows):
        raise ValueError("Expected component-ablation JSONL rows")
    identity = {(r["model"], r["tokenizer"], r["selection_graph_sha256"],
                 r["feature_map_sha256"], tuple(sorted(r["train_seeds"])),
                 tuple(sorted(r["test_seeds"])), tuple(sorted(r["train_probe_ids"])),
                 tuple(sorted(r["test_probe_ids"])), r["alpha"], r["phase"])
                for r in rows}
    if len(identity) != 1:
        raise ValueError("Mixed model, frozen selection, splits or intervention strengths")
    by_key = defaultdict(dict)
    for r in rows:
        key = (r["seed"], r["probe_id"])
        name = r["selector"]
        if name in by_key[key]:
            raise ValueError("Duplicate selector on held-out seed/probe")
        if not math.isfinite(r["specificity"]):
            raise ValueError("Non-finite causal specificity")
        by_key[key][name] = r
    if not by_key or not all("sieve" in cells for cells in by_key.values()):
        raise ValueError("Incomplete matched selector cells: missing sieve observations")
    if {seed for seed, _ in by_key} != set(rows[0]["test_seeds"]):
        raise ValueError("Incomplete held-out seed coverage")
    if {probe for _, probe in by_key} != set(rows[0]["test_probe_ids"]):
        raise ValueError("Incomplete held-out probe coverage")
    names = set(next(iter(by_key.values())))
    if any(set(cells) != names for cells in by_key.values()):
        raise ValueError("Incomplete matched selector cells")
    for (seed, probe), cell in by_key.items():
        if len({(r["case_id"], r["prefix_sha256"], r["source"], r["reference"])
                for r in cell.values()}) != 1:
            raise ValueError("Compared selectors on mismatched held-out prefixes")
        for r in cell.values():
            if any(abs(value - r["orthogonal_patch_norms"].get(block, float("nan"))) >
                   1e-4 * max(1.0, value)
                   for block, value in r["candidate_patch_norms"].items()):
                raise ValueError("Feature and orthogonal null patch norms differ")
            if set(r["candidate_patch_norms"]) != set(r["orthogonal_patch_norms"]):
                raise ValueError("Feature/null decoder block coverage differs")
    frozen_sets = {}
    for name in names:
        selected_ids = {tuple(sorted(cell[name]["selected_node_ids"])) for cell in by_key.values()}
        if len(selected_ids) != 1:
            raise ValueError("Selected component identities changed across held-out cells")
        frozen_sets[name] = next(iter(selected_ids))
    seeds = sorted({seed for seed, _ in by_key})
    for seed in seeds:
        probes = {probe for (s, probe) in by_key if s == seed}
        if probes != {probe for (s, probe) in by_key if s == seeds[0]}:
            raise ValueError("Every test seed must have the same probe coverage")
    results = {}
    per_seed = defaultdict(dict)
    for name in sorted(names):
        for seed in seeds:
            data = [cells[name]["specificity"] for (s, _), cells in by_key.items()
                    if s == seed]
            per_seed[name][seed] = float(statistics.mean(data))
        values = [per_seed[name][seed] for seed in seeds]
        detailed = [cell[name] for cell in by_key.values()]
        first_token_specificity = [r["orthogonal_first_token"] - r["candidate_first_token"]
                                   for r in detailed]
        later_token_specificity = [r["orthogonal_later_token"] - r["candidate_later_token"]
                                   for r in detailed if r["candidate_later_token"] is not None
                                   and r["orthogonal_later_token"] is not None]
        exemplar = next(cells[name] for cells in by_key.values())
        results[name] = {
            "n_seed_clusters": len(seeds), "mean_specificity": float(statistics.mean(values)),
            "seed_specificities": {str(s): per_seed[name][s] for s in seeds},
            "latent_budget": exemplar["latent_budget"],
            "selected_node_ids": exemplar["selected_node_ids"],
            "within_selector_signflip": signflip_pvalue(values),
            "mean_first_token_specificity": float(statistics.mean(first_token_specificity)),
            "mean_later_token_specificity": (
                float(statistics.mean(later_token_specificity)) if later_token_specificity else None),
            "patch_active_fraction": float(statistics.mean(
                int(any(norm > 1e-9 for norm in r["candidate_patch_norms"].values()))
                for r in detailed)),
        }
    tests = []
    comparison_names = [name for name in sorted(names) if name != "sieve"]
    for name in comparison_names:
        diff = [per_seed["sieve"][seed] - per_seed[name][seed] for seed in seeds]
        results[name]["mean_sieve_minus_selector"] = float(statistics.mean(diff))
        results[name]["paired_signflip"] = signflip_pvalue(diff)
        results[name]["identical_to_sieve"] = (
            frozen_sets[name] == frozen_sets["sieve"])
        tests.append(results[name]["paired_signflip"]["p_value"])
    for name, adjusted in zip(comparison_names, holm_adjust(tests)):
        results[name]["holm_p_value"] = adjusted
        if results[name]["identical_to_sieve"]:
            if any(abs(per_seed[name][seed] - per_seed["sieve"][seed]) > 1e-8 for seed in seeds):
                raise ValueError("Identical selected features produced different causal results")
    return {
        "schema": REPORT_SCHEMA,
        "n_seed_clusters": len(seeds), "n_test_cases": len(by_key),
        "n_selectors": len(names), "model": rows[0]["model"],
        "selection_graph_sha256": rows[0]["selection_graph_sha256"],
        "feature_map_sha256": rows[0]["feature_map_sha256"],
        "selector_comparisons": results,
        "causal_readout": "baseline-frozen TopK SAE decoder contribution subtraction; positive specificity predicts persona-supporting features",
        "limitations": [
            "Uses pre-intervention SAE activations, with reconstruction error preserved; not full counterfactual re-encoding.",
            "Subtracts feature contributions at the final prefix token only, rather than ablating all generated tokens.",
            "SAE feature attribution fidelity and source graph construction require separate verification.",
            "Random orthogonal controls match per-block displacement norm, not every distributional characteristic.",
            "Fixed style completions may disagree with the model's free-response behavior.",
            "Selection on held-out labels invalidates the randomization test.",
        ],
    }


def component_plan_file(manifest_path: str, feature_path: str, input_path: str,
                        output_path: str, **kwargs):
    manifest = _load_json(manifest_path)
    features = _load_json(feature_path)
    cases = _read_cases(input_path)
    result = plan_component_ablation(manifest=manifest, feature_map=features,
              cases=cases, map_root=Path(feature_path).parent, **kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def component_ablation_hf(*, manifest_path: str, feature_path: str,
                         input_path: str, output_path: str, model_name: str,
                         device: str = "auto", dtype: str = "auto",
                         allow_download: bool = False,
                         trust_remote_code: bool = False,
                         chat_template_kwargs: dict | None = None, **kwargs) -> int:
    # Validate metadata and local SHA before loading model weights.
    manifest, features, cases = (_load_json(manifest_path), _load_json(feature_path),
                                 _read_cases(input_path))
    preflight = {k: kwargs[k] for k in ("source", "reference", "phase",
                 "heldout_probes", "holdout_fraction", "split_seed", "self_check")
                 if k in kwargs}
    plan_component_ablation(manifest=manifest, feature_map=features, cases=cases,
              model_name=model_name, map_root=Path(feature_path).parent, **preflight)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch/transformers using .[mechanistic]") from exc
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if getattr(torch.backends.mps, "is_available", lambda: False)()
                  else "cpu")
    if device not in ("cpu", "cuda", "mps") or dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Unsupported device or dtype")
    kwargs_load = {"local_files_only": not allow_download,
                   "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs_load)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        **kwargs_load).to(device)
    results = run_component_ablation(manifest=manifest, feature_map=features,
        chat_template_kwargs=chat_template_kwargs,
        cases=cases, model=model, tokenizer=tokenizer, model_name=model_name,
        map_root=Path(feature_path).parent, device=device, **kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
    return len(results)


def component_report_file(input_path: str, output_path: str):
    rows = _read_cases(input_path)
    result = component_report(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result
