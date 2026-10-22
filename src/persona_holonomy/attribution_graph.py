"""Bounded, training-only, signed finite-difference SAE influence graph exporter.

One edge measures a *local intervention response*, not a correlation:
remove a donor-minus-reference SAE latent contribution at a decoder block's
last prompt token and observe the induced change in a later block's SAE latent
or the next-token style log-odds. We average signed effects over PHB training
histories, never using held-out seeds or held-out question families.

This is context-dependent single-feature causal attribution, NOT a gradient EAP
estimator, a proof of additivity, or attribution of the entire generated reply.
The graph is restricted to adjacent selected decoder blocks to avoid silently
counting skip-path effects twice. Refit/freezing against held-out outcomes is
forbidden. Model weights are loaded locally unless explicitly opted in.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .interventions import (
    CausalProbe, STYLE_COMPLETIONS, _capture_hook, _patch_hook,
    decoder_blocks, prepare_pairs,
)
from .perc_components import _hash_file, _load_json, _sae_tensors
from .sae_pipeline import REGISTRY_SCHEMA, capture_plan

GRAPH_SCHEMA = "phb.percinterp.attribution.v1"
PLAN_SCHEMA = "phb.percinterp.attribution_plan.v1"
ESTIMATOR = "single_feature_contrast_finite_difference.v1"


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def graph_plan(*, cases: list[dict], registry: dict, model_name: str,
               registry_root: str | Path = ".", tokenizer_name: str | None = None,
               layers: str = "all", top_features: int = 8,
               heldout_probes: tuple[str, ...] = ("parser_validator",),
               phase: str = "exit", source: str = "ordered",
               reference: str = "tail_matched", holdout_fraction: float = .4,
               split_seed: int = 20261022) -> dict[str, Any]:
    """Offline cost/provenance gate. Requires local SAE files, no model loading."""
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("Expected PHB SAE registry produced by sae-fit")
    if registry.get("model") != model_name or not registry.get("checkpoint_revision"):
        raise ValueError("SAE registry model identity or revision mismatch")
    if tokenizer_name is not None and registry.get("tokenizer") != tokenizer_name:
        raise ValueError("Tokenizer identity differs from SAE registry")
    if type(top_features) is not int or not 1 <= top_features <= 256:
        raise ValueError("top_features must be an integer in [1,256]")
    plan = capture_plan(cases, heldout_probes=heldout_probes, phase=phase,
        source=source, reference=reference, holdout_fraction=holdout_fraction,
        split_seed=split_seed)
    checks = (("phb_train_seeds", "phb_train_seeds"),
              ("phb_test_seeds_excluded", "phb_test_seeds_excluded"),
              ("phb_training_probes", "phb_training_probes"),
              ("phb_test_probes_excluded", "phb_test_probes_excluded"))
    for left, right in checks:
        if set(registry.get(left, ())) != set(plan[right]):
            raise ValueError(f"SAE registry {left} does not match frozen PHB split")
    available = registry.get("blocks")
    if not isinstance(available, dict) or len(available) < 2:
        raise ValueError("At least two trained SAE decoder blocks are required")
    if any(not isinstance(k, str) or not k.isdigit() for k in available):
        raise ValueError("SAE registry decoder block keys must be integers")
    chosen = sorted(int(k) for k in available) if layers == "all" else []
    if layers != "all":
        for item in layers.split(","):
            item = item.strip()
            if not item.isdigit():
                raise ValueError("--layers must be 'all' or comma-separated SAE block indexes")
            chosen.append(int(item))
    if len(chosen) < 2 or len(set(chosen)) != len(chosen) or chosen != sorted(chosen):
        raise ValueError("Choose two or more distinct ascending SAE blocks")
    if any(str(i) not in available for i in chosen):
        raise ValueError("Requested block has no trained SAE in the registry")
    root = Path(registry_root)
    checksums = {}
    widths = {}
    for b in chosen:
        record = available[str(b)]
        if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
                or type(record.get("n_latents")) is not int
                or type(record.get("top_k")) is not int):
            raise ValueError("SAE registry block lacks checkpoint metadata")
        if record["n_latents"] < 1 or not 1 <= record["top_k"] <= record["n_latents"]:
            raise ValueError("Invalid trained SAE dictionary size or top_k")
        filename = root / record["path"]
        if (not filename.is_file() or filename.suffix not in (".pt", ".safetensors")
                or _hash_file(filename) != record.get("sha256")):
            raise ValueError(f"SAE checkpoint missing or SHA256 mismatch at block {b}")
        checksums[str(b)] = record["sha256"]
        widths[str(b)] = record["n_latents"]
    # capture_plan already checks duplicated/invalid paired-case structure.
    paired = prepare_pairs(cases, source_conditions=(source,),
        target_condition=reference, phases=(phase,))
    training = [(donor, ref) for donor, ref in paired
                if donor["seed"] in plan["phb_train_seeds"]
                and donor["probe_id"] in plan["phb_training_probes"]]
    actual_seeds = {d["seed"] for d, _ in training}
    actual_probes = {d["probe_id"] for d, _ in training}
    if (actual_seeds != set(plan["phb_train_seeds"])
            or actual_probes != set(plan["phb_training_probes"])):
        raise ValueError("Attribution training pairs have incomplete seed/probe coverage")
    per_layer_cap = {str(b): min(top_features, widths[str(b)]) for b in chosen}
    return {
        "schema": PLAN_SCHEMA, "model": model_name,
        "tokenizer": registry["tokenizer"],
        "checkpoint_revision": registry["checkpoint_revision"],
        "source": source, "reference": reference, "phase": phase,
        "registry_sha256": _json_sha(registry),
        "cases_sha256": _json_sha(cases),
        "layers": chosen, "top_features": top_features,
        "candidate_caps": per_layer_cap, "checkpoint_sha256": checksums,
        "train_seeds": sorted(actual_seeds),
        "test_seeds_excluded": sorted(plan["phb_test_seeds_excluded"]),
        "training_probe_ids": sorted(actual_probes),
        "test_probe_ids_excluded": sorted(plan["phb_test_probes_excluded"]),
        "n_training_pairs": len(training),
        "estimated_model_forwards_upper_bound": len(training) * (2 + sum(per_layer_cap.values())),
        "estimator": ESTIMATOR,
        "caveats": [
            "Forward counts omit tokenizer and SAE encoding cost and are not a VRAM or time estimate.",
            "First SAE stage is the set of source vertices; PercInterp selects interior features.",
            "Signed finite differences are contextual total responses over adjacent sampled layers.",
            "The output readout is next-token style log-odds, not complete-response quality.",
            "All candidates and edge weights are fitted on PHB training seeds and probes only.",
        ],
    }


def _encode(vector: Any, sae: dict, torch: Any):
    """Exactly match PercInterp TopKSAE's encoder and ReLU TopK convention."""
    if vector.ndim != 1 or vector.numel() != sae["W_enc"].shape[0]:
        raise ValueError("Decoder width differs from trained SAE")
    x = vector.to(device=sae["W_enc"].device, dtype=torch.float32)
    pre = (x * sae["scale"] - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    values, ids = pre.topk(sae["top_k"])
    return torch.zeros_like(pre).scatter(0, ids, torch.nn.functional.relu(values))


def _style_tokens(tokenizer: Any, probe: str) -> tuple[int, int]:
    if probe not in STYLE_COMPLETIONS:
        raise ValueError(f"No registered style readout for technical probe {probe}")
    target = STYLE_COMPLETIONS[probe]
    plain = tokenizer.encode(target["plain"], add_special_tokens=False)
    theatrical = tokenizer.encode(target["theatrical"], add_special_tokens=False)
    if not plain or not theatrical or plain[0] == theatrical[0]:
        raise ValueError("The two style continuations must have distinct first token IDs")
    return int(plain[0]), int(theatrical[0])


def _capture_once(*, model: Any, runner: CausalProbe, blocks: Any,
                  layers: tuple[int, ...], messages: list[dict],
                  patch_block: int | None = None, patch_vector=None):
    """One prefill forward, capturing block outputs after optional patch."""
    prefix = runner._prefix(messages)
    snapshots: dict[int, dict] = {}
    with ExitStack() as stack, runner.torch.inference_mode():
        patched_state = None
        if patch_block is not None:
            patched_state = {}
            stack.enter_context(_patch_hook(blocks[patch_block],
                position=int(prefix["input_ids"].shape[1]) - 1,
                vector=patch_vector, alpha=1., mode="add", state=patched_state))
        for index in layers:
            state: dict = {}
            stack.enter_context(_capture_hook(blocks[index], state))
            snapshots[index] = state
        outputs = model(**prefix, use_cache=False, return_dict=True)
    if any(state["calls"] != 1 for state in snapshots.values()):
        raise RuntimeError("Attribution hook must capture each decoder block exactly once")
    if patched_state is not None and patched_state["calls"] != 1:
        raise RuntimeError("Attribution patch must execute exactly once")
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != prefix["input_ids"].shape[1]:
        raise ValueError("Causal model returned an unexpected logits shape")
    return {index: snapshots[index]["vector"].float().cpu()
            for index in layers}, logits[0, -1, :].detach().float().cpu()


def measure_graph(*, cases: list[dict], registry: dict,
                  model: Any, tokenizer: Any, model_name: str,
                  registry_root: str | Path = ".", layers: str = "all",
                  top_features: int = 8,
                  heldout_probes: tuple[str, ...] = ("parser_validator",),
                  phase: str = "exit", source: str = "ordered",
                  reference: str = "tail_matched", holdout_fraction: float = .4,
                  split_seed: int = 20261022, max_total_tokens: int = 4096,
                  device: str = "cpu") -> dict[str, Any]:
    """Compute context-specific interventional edges on training cells only."""
    import torch
    preflight = graph_plan(cases=cases, registry=registry, model_name=model_name,
        tokenizer_name=str(getattr(tokenizer, "name_or_path", "unknown")),
        registry_root=registry_root, layers=layers, top_features=top_features,
        heldout_probes=heldout_probes, phase=phase, source=source,
        reference=reference, holdout_fraction=holdout_fraction,
        split_seed=split_seed)
    _, blocks = decoder_blocks(model)
    chosen = tuple(preflight["layers"])
    if chosen[-1] >= len(blocks):
        raise ValueError("SAE registry names a decoder block missing in the model")
    saes = {}
    for index in chosen:
        item = registry["blocks"][str(index)]
        saes[index] = _sae_tensors({"path": item["path"],
            "sha256": item["sha256"], "top_k": item["top_k"]},
            root=registry_root, torch=torch, device="cpu")
        if saes[index]["W_enc"].shape[1] != item["n_latents"]:
            raise ValueError("SAE registry dictionary width differs from checkpoint")
        if (not math.isfinite(item.get("validation_fvu", float("nan")))
                or item["validation_fvu"] < 0):
            raise ValueError("SAE registry is missing valid held-out reconstruction FVU")
    runner = CausalProbe(model, tokenizer, layer=str(chosen[0]),
                         device=device, max_total_tokens=max_total_tokens)
    model.eval()
    paired = prepare_pairs(cases, source_conditions=(source,),
        target_condition=reference, phases=(phase,))
    training = [(donor, ref) for donor, ref in paired
                if donor["seed"] in preflight["train_seeds"]
                and donor["probe_id"] in preflight["training_probe_ids"]]
    cached = []
    contrasts = {b: torch.zeros(saes[b]["W_enc"].shape[1]) for b in chosen}
    for donor, ref in training:
        donor_vec, donor_logits = _capture_once(model=model, runner=runner,
            blocks=blocks, layers=chosen, messages=donor["messages"])
        ref_vec, _ = _capture_once(model=model, runner=runner,
            blocks=blocks, layers=chosen, messages=ref["messages"])
        d_act = {b: _encode(donor_vec[b], saes[b], torch) for b in chosen}
        r_act = {b: _encode(ref_vec[b], saes[b], torch) for b in chosen}
        delta = {b: d_act[b] - r_act[b] for b in chosen}
        for b in chosen:
            contrasts[b] += delta[b].abs()
        style_ids = _style_tokens(tokenizer, donor["probe_id"])
        if max(style_ids) >= donor_logits.numel():
            raise ValueError("Style token ID is outside the model output vocabulary")
        baseline_style = float((donor_logits[style_ids[1]] - donor_logits[style_ids[0]]).item())
        cached.append((donor, d_act, delta, baseline_style, style_ids))
    nodes = {}
    for stage, b in enumerate(chosen):
        avg = contrasts[b] / len(training)
        candidates = [i for i in range(len(avg)) if float(avg[i]) > 1e-8]
        top = sorted(candidates, key=lambda i: (-float(avg[i]), i))[:top_features]
        if not top:
            raise ValueError(f"No persona-contrast SAE features active at block {b}")
        nodes[b] = tuple(top)
    result_nodes = []
    for stage, b in enumerate(chosen):
        for latent in nodes[b]:
            result_nodes.append({"id": f"sae:{b}:{latent}",
                "stage": stage, "decoder_block": b})
    result_nodes.append({"id": "output:first_token_style_logodds",
                         "stage": len(chosen)})
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    counters = {"forwards": 2 * len(training), "skipped_zero_contrast": 0}
    for donor, original, delta, baseline_style, (plain, theatrical) in cached:
        for stage, b in enumerate(chosen):
            for latent in nodes[b]:
                shift = (-delta[b][latent] / saes[b]["scale"] *
                         saes[b]["W_dec"][latent]).detach().cpu()
                src = f"sae:{b}:{latent}"
                if float(torch.linalg.vector_norm(shift)) <= 1e-9:
                    counters["skipped_zero_contrast"] += 1
                    effects = [0.] * (len(nodes[chosen[stage + 1]]) if stage + 1 < len(chosen) else 1)
                else:
                    patched_vectors, patched_logits = _capture_once(
                        model=model, runner=runner, blocks=blocks, layers=chosen,
                        messages=donor["messages"], patch_block=b, patch_vector=shift)
                    counters["forwards"] += 1
                    if stage + 1 < len(chosen):
                        target_b = chosen[stage + 1]
                        new_acts = _encode(patched_vectors[target_b], saes[target_b], torch)
                        effects = [(original[target_b][j] - new_acts[j]).item()
                                   for j in nodes[target_b]]
                    else:
                        changed = float((patched_logits[theatrical] - patched_logits[plain]).item())
                        effects = [baseline_style - changed]
                targets = ([f"sae:{chosen[stage + 1]}:{j}" for j in nodes[chosen[stage + 1]]]
                           if stage + 1 < len(chosen) else ["output:first_token_style_logodds"])
                for dst, effect in zip(targets, effects):
                    if not math.isfinite(float(effect)):
                        raise ValueError("Nonfinite feature influence in training graph")
                    scores[(src, dst)].append(float(effect))
    edges = []
    for (src, dst), values in sorted(scores.items()):
        if len(values) != len(training):
            raise RuntimeError("Missing training-cell effect for an attribution edge")
        mean = sum(values) / len(values)
        if abs(mean) <= 1e-12:
            continue  # PercInterp drops zero-weight edges.
        edges.append({"source": src, "target": dst,
                      "signed_weight": mean,
                      "mean_absolute_effect": sum(map(abs, values)) / len(values),
                      "n_training_pairs": len(values),
                      "positive_effect_fraction": sum(v > 0 for v in values) / len(values)})
    if not edges or not any(e["target"] == "output:first_token_style_logodds" for e in edges):
        raise ValueError("No nonzero signed influence to style output; report as a null result")
    # The directed graph may still have no complete first-stage-to-output
    # pathway. Preserve the graph as a negative diagnostic rather than
    # hallucinating missing influence edges to satisfy a downstream sieve.
    adjacent: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adjacent[e["source"]].add(e["target"])
    reachable = set(f"sae:{chosen[0]}:{i}" for i in nodes[chosen[0]])
    frontier = list(reachable)
    for vertex in frontier:
        for target in sorted(adjacent.get(vertex, ())):
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    spans = "output:first_token_style_logodds" in reachable
    return {
        "schema": GRAPH_SCHEMA,
        "provenance": {
            "edge_semantics": "direct_attribution",
            "estimator": ESTIMATOR,
            "model": model_name, "tokenizer": preflight["tokenizer"],
            "checkpoint_revision": registry["checkpoint_revision"],
            "sae_registry_sha256": _json_sha(registry),
            "sae_checkpoints_sha256": preflight["checkpoint_sha256"],
            "cases_sha256": preflight["cases_sha256"],
            "train_seeds": preflight["train_seeds"],
            "training_probe_ids": preflight["training_probe_ids"],
            "test_seeds_excluded": preflight["test_seeds_excluded"],
            "test_probe_ids_excluded": preflight["test_probe_ids_excluded"],
            "source": source, "reference": reference, "phase": phase,
            "output_metric": "logit(theatrical_first_token)-logit(plain_first_token)",
            "patch": "subtract ordered-minus-tail TopK SAE decoded latent contribution at final prefix token",
            "edge_aggregation": "arithmetic mean of signed single-latent finite differences over training pairs",
        },
        "nodes": result_nodes, "edges": edges,
        "sources": [f"sae:{chosen[0]}:{i}" for i in nodes[chosen[0]]],
        "sinks": ["output:first_token_style_logodds"],
        "diagnostics": {
            "n_train_pairs": len(training), "candidate_blocks": list(chosen),
            "candidate_counts": {str(b): len(v) for b, v in nodes.items()},
            "total_model_forwards": counters["forwards"],
            "zero_contrast_patches_skipped": counters["skipped_zero_contrast"],
            "edges_emitted": len(edges),
            "has_measured_source_to_sink_path": spans,
            "graph_status": "spanning" if spans else "no_measured_path",
            "max_sae_validation_fvu": max(registry["blocks"][str(b)]["validation_fvu"] for b in chosen),
            "training_case_id_hashes": [_json_sha(d["case_id"]) for d, *_ in cached],
        },
        "limitations": [
            "Adjacent-stage signed intervention responses are local, context-dependent total effects, not a unique additive path decomposition.",
            "First-stage SAE features are graph sources, not available as interior PercInterp selector nodes.",
            "Fixed first-token style log-odds may differ from free-generation behavior.",
            "Signed effects can cancel across training histories; zero-mean edges are omitted.",
            "Only training seeds/probes influence feature candidate ranking and edge weights.",
        ],
    }


def graph_plan_file(*, cases_path: str, registry_path: str, output_path: str,
                    model_name: str, **kwargs) -> dict:
    from .interventions import _read_cases
    cases = _read_cases(cases_path)
    registry = _load_json(registry_path)
    result = graph_plan(cases=cases, registry=registry, model_name=model_name,
                        registry_root=Path(registry_path).parent, **kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def graph_hf(*, cases_path: str, registry_path: str, output_path: str,
             model_name: str, device: str = "auto", dtype: str = "auto",
             allow_download: bool = False, trust_remote_code: bool = False,
             **kwargs) -> dict:
    """No implicit downloads; refuse invalid provenance before loading model."""
    from .interventions import _read_cases
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("Attribution graph output exists; choose a new output")
    cases = _read_cases(cases_path)
    registry = _load_json(registry_path)
    graph_plan(cases=cases, registry=registry, model_name=model_name,
               registry_root=Path(registry_path).parent, **kwargs)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install .[mechanistic] to run the pretrained graph estimator") from exc
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if getattr(torch.backends.mps, "is_available", lambda: False)()
                  else "cpu")
    if device not in ("cpu", "cuda", "mps") or dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Unsupported device or dtype")
    options = {"local_files_only": not allow_download, "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        **options).to(device)
    result = measure_graph(cases=cases, registry=registry, model=model,
                           tokenizer=tokenizer, model_name=model_name,
                           registry_root=Path(registry_path).parent,
                           device=device, **kwargs)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n", encoding="utf-8")
    return result
