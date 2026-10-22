"""Optional PHB ↔ PercInterp bridge for *direct* attribution graphs.

PercInterp selects graph components without looking at held-out PHB outcomes.
The downstream layer audit checks a weaker, explicitly layer-level proxy:
PHB v0.6 intervenes on entire block residual vectors, not individual nodes.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from .inference import holm_adjust, signflip_pvalue


GRAPH_SCHEMA = "phb.percinterp.attribution.v1"
SELECT_SCHEMA = "phb.percinterp.selection.v1"
AUDIT_SCHEMA = "phb.percinterp.layer_audit.v1"


def _library():
    try:
        import numpy as np
        from percinterp.interop import from_attribution_edges
        from percinterp.measure import profile, rank_entry
        from percinterp.sieve import sieve, degree_entry
    except ImportError as exc:
        raise RuntimeError(
            "PercInterp is optional. Install from the verified project: "
            "pip install 'git+https://github.com/MoralityLabAI/percolation-interp' "
            "(Python >=3.11)"
        ) from exc
    return np, from_attribution_edges, profile, rank_entry, sieve, degree_entry


def _read_graph(graph_data: dict[str, Any]):
    if graph_data.get("schema") != GRAPH_SCHEMA:
        raise ValueError(f"Expected graph schema {GRAPH_SCHEMA}")
    nodes = graph_data.get("nodes")
    edges = graph_data.get("edges")
    if not isinstance(nodes, list) or len(nodes) < 3 or not isinstance(edges, list) or not edges:
        raise ValueError("Require a nonempty attribution graph with at least three nodes")
    if not all(isinstance(n, dict) for n in nodes):
        raise ValueError("Each node must be an object")
    ids = [n.get("id") for n in nodes]
    stages = [n.get("stage") for n in nodes]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Node IDs must be unique, nonempty strings")
    if any(type(s) is not int or s < 0 for s in stages) or len(set(stages)) < 3:
        raise ValueError("The directed influence graph must have at least three stages")
    mapping = {}
    for node in nodes:
        b = node.get("decoder_block")
        if b is not None and (type(b) is not int or b < 0):
            raise ValueError("decoder_block must be a nonnegative explicit integer")
        mapping[node["id"]] = b
    direct = []
    for edge in edges:
        if not isinstance(edge, dict) or not {"source", "target", "signed_weight"} <= edge.keys():
            raise ValueError("Each edge requires source, target, and signed_weight")
        direct.append((edge["source"], edge["target"], edge["signed_weight"]))
    provenance = graph_data.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("edge_semantics") != "direct_attribution":
        raise ValueError("Provenance must explicitly label edges as direct_attribution")
    # Recorded values are an explicit assertion by the exporter, not cryptographic proof.
    training = provenance.get("train_seeds", [])
    if not isinstance(training, list) or any(type(s) is not int for s in training) or len(set(training)) != len(training):
        raise ValueError("provenance.train_seeds must be a list of distinct integers")
    return ids, stages, mapping, direct, provenance


def select_attribution_graph(graph_data: dict[str, Any], *, n_shuffles: int = 32,
                             n_random: int = 12, seed: int = 20261022,
                             z_min: float = 3.) -> dict[str, Any]:
    """Frozen PercInterp percolation + local and random budget-matched selectors.

    All selection and shuffle metrics are graph-only. This function neither
    infers direct edges from PHB activations nor looks at evaluation outcomes.
    """
    if n_shuffles < 4 or n_random < 1 or not 0 <= n_random <= 1000:
        raise ValueError("Require at least four shuffles and one random selector")
    if not math.isfinite(z_min) or z_min < 0:
        raise ValueError("z_min must be finite and nonnegative")
    np, from_edges, profile, rank_entry, sieve, degree_entry = _library()
    node_ids, stages, block_mapping, direct, provenance = _read_graph(graph_data)
    imported = from_edges(node_ids, stages, direct,
                          sources=graph_data.get("sources"), sinks=graph_data.get("sinks"))
    g = imported.graph
    if g.n_edges == 0:
        raise ValueError("Attribution graph has no nonzero-weight edges")
    group = g.edge_layer("dst")
    entry = rank_entry(g.weight, group)
    spanning = profile(g, entry).p_span
    if not math.isfinite(spanning):
        raise ValueError("Attribution graph has no source-to-sink path")
    internal = ~(g.sources | g.sinks)
    if int(internal.sum()) == 0:
        raise ValueError("Graph lacks interior nodes for component selection")
    result = sieve(g, g.weight, groups=group, n_shuffles=n_shuffles,
                   rng=np.random.default_rng(seed), z_min=z_min)
    if not (np.isfinite(result.c_star) and result.c_star >= 0):
        raise ValueError("No valid percolation occupation")
    degree_profile = profile(g, degree_entry(g, g.weight, group))
    effective_c_star = float(result.c_star)
    selection_fallback = False
    selected = np.asarray(result.backbone, dtype=bool) & internal
    if not selected.any():
        # Sparse / null-like graphs may have no interior backbone at the
        # learned-connectivity peak. A spanning fallback is explicitly flagged.
        if not math.isfinite(degree_profile.p_span):
            raise ValueError("No spanning backbone even at full occupation")
        effective_c_star = max(effective_c_star, float(degree_profile.p_span))
        selected = np.asarray(degree_profile.node_entry <= effective_c_star, dtype=bool) & internal
        selection_fallback = True
        if not selected.any():
            raise ValueError("No interior backbone even at the spanning threshold")
    # Same per-stage quota for the sieve, local baselines, and each random draw.
    node_stages = g.node_layer
    names = imported.node_ids
    kept = degree_entry(g, g.weight, group) <= effective_c_star
    incoming = np.bincount(g.dst[kept], minlength=g.n_nodes)
    outgoing = np.bincount(g.src[kept], minlength=g.n_nodes)
    local_degree = incoming + outgoing
    weighted_degree = np.bincount(g.src, weights=g.weight, minlength=g.n_nodes) + \
        np.bincount(g.dst, weights=g.weight, minlength=g.n_nodes)
    selectors: dict[str, list[str]] = {"sieve": [names[i] for i in np.flatnonzero(selected)],
                                       "local_degree": [], "top_attribution": []}
    rng = random.Random(seed)
    random_sets: list[str] = [f"random_{k:03d}" for k in range(n_random)]
    for key in random_sets:
        selectors[key] = []
    stage_quotas = {}
    for stage in sorted(set(node_stages[internal].tolist())):
        eligible = [int(i) for i in np.flatnonzero(internal & (node_stages == stage))]
        quota = int(selected[eligible].sum())
        stage_quotas[str(stage)] = quota
        def top(score):
            return sorted(eligible, key=lambda i: (-float(score[i]), names[i]))[:quota]
        selectors["local_degree"].extend(names[i] for i in top(local_degree))
        selectors["top_attribution"].extend(names[i] for i in top(weighted_degree))
        for key in random_sets:
            selectors[key].extend(names[i] for i in rng.sample(eligible, quota))
    selectors = {key: sorted(ids) for key, ids in selectors.items()}
    blocks = {key: sorted({block_mapping[node] for node in chosen
                           if block_mapping[node] is not None}) for key, chosen in selectors.items()}
    digest = hashlib.sha256(json.dumps(graph_data, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema": SELECT_SCHEMA, "source_schema": GRAPH_SCHEMA,
        "graph_sha256": digest,
        "provenance": provenance,
        "algorithm": "percinterp.sieve with dst-stage degree-entry and within-dst-stage weight shuffles",
        "n_nodes": g.n_nodes, "n_edges": g.n_edges,
        "n_nonzero_signed_edges": len(imported.signed_weight),
        "dropped_zero_edges": imported.dropped_zero_edges,
        "signed_negative_edges": int((imported.signed_weight < 0).sum()),
        "directed_p_span_ranked": float(spanning),
        "sieve": {"c_star": float(effective_c_star),
                  "raw_c_star": float(result.c_star),
                  "spanning_threshold_fallback": selection_fallback,
                  "z_peak": float(result.z_peak),
                  "peak_learned_connectivity": float(result.peak),
                  "shuffle_p_value": float(result.p_value),
                  "null_shuffles": n_shuffles, "z_min": z_min,
                  "curve": {"c": result.cs.tolist(), "backbone": result.real.tolist(),
                            "shuffled_mean": result.shuffles.mean(0).tolist()}},
        "stage_quotas": stage_quotas,
        "selected_nodes": selectors,
        "selected_decoder_blocks": blocks,
        "node_to_decoder_block": block_mapping,
        "random_seed": seed,
        "comparators": ["local_degree", "top_attribution", *random_sets],
        "selector_identical_to_sieve": {k: v == selectors["sieve"] for k, v in selectors.items()},
        "status": "graph_only_candidate_selection",
        "limitations": [
            "Edges must come from a valid upstream *direct* attribution estimator; PHB hidden vectors are not edges.",
            "The sieve retains absolute edge weight for ranking; signed weights are tracked separately.",
            "A graph-only p-value is not a causal intervention p-value.",
            "Decoder-block mappings are explicit, not inferred from topological stages.",
            "Decoder-block steering is weaker evidence than per-component patching.",
            "Selector design and graph provenance must be frozen before held-out outcomes are inspected.",
        ],
    }


def audit_sweep_by_block(manifest: dict[str, Any], rows: list[dict[str, Any]], *,
                         source: str = "ordered", phase: str = "exit",
                         alpha: float = 1., mode: str = "inject") -> dict[str, Any]:
    """Paired *block-level proxy* comparison; cannot attribute causal effects to graph nodes.

    Each selected decoder block must have both candidate and orthogonal rows for
    all the same held-out seed/probe pairs. This is a cross-module audit only.
    """
    if manifest.get("schema") != SELECT_SCHEMA:
        raise ValueError("Expected a frozen PercInterp selection manifest")
    if not rows or not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Need sweep observations and a nonzero valid alpha")
    if mode not in ("inject", "erase"):
        raise ValueError("Mode must be inject or erase")
    provenance = manifest.get("provenance", {})
    train_from_graph = provenance.get("train_seeds")
    if not isinstance(train_from_graph, list) or not train_from_graph:
        raise ValueError("Layer audit requires graph provenance.train_seeds to prevent leakage")
    selected = manifest.get("selected_decoder_blocks", {})
    if not selected or not selected.get("sieve"):
        raise ValueError("No explicitly mapped sieve decoder blocks; provide decoder_block on graph nodes")
    for selector, blocks in selected.items():
        if len(blocks) != len(selected["sieve"]):
            raise ValueError(f"Selector {selector} has a different decoder-block budget; cannot compare")
    required = {int(b) for blocks in selected.values() for b in blocks}
    filtered = [r for r in rows if r.get("direction_source") == source
                and r.get("phase") == phase and r.get("mode") == mode
                and r.get("alpha") == alpha and r.get("layer") in required]
    if not filtered:
        raise ValueError("No matching PHB held-out sweep rows")
    models = {(r.get("model"), r.get("tokenizer"), r.get("reference_condition")) for r in filtered}
    if len(models) != 1:
        raise ValueError("Mixed model/tokenizer/reference in sweep")
    split_configs = {(tuple(r.get("training_seeds", [])), tuple(r.get("test_seeds", [])),
                      tuple(r.get("training_probe_ids", [])), tuple(r.get("evaluation_probe_ids", [])))
                     for r in filtered}
    if len(split_configs) != 1:
        raise ValueError("Sweep contains inconsistent seed/probe partitions")
    train, test, train_probe, eval_probe = next(iter(split_configs))
    if set(train) != set(train_from_graph) or not test or set(train) & set(test):
        raise ValueError("Graph training seeds must match sweep training split, disjoint from held-out seeds")
    if not set(train_probe).isdisjoint(eval_probe):
        raise ValueError("Training and held-out probe domains overlap")
    observations: dict[tuple[int, int, str], dict[str, dict]] = defaultdict(dict)
    for r in filtered:
        if r.get("control") not in ("candidate", "orthogonal"):
            raise ValueError("Unknown intervention control")
        key = (r["layer"], r["seed"], r["probe_id"])
        if r["seed"] not in test or r["probe_id"] not in eval_probe:
            raise ValueError("Sweep includes unregistered test seed or probe")
        if r["control"] in observations[key]:
            raise ValueError("Duplicate matched control row")
        observations[key][r["control"]] = r
    for key, pair in observations.items():
        if set(pair) != {"candidate", "orthogonal"}:
            raise ValueError(f"Unpaired causal controls at {key}")
        a, b = pair["candidate"], pair["orthogonal"]
        if (a.get("recipient_case_id"), a.get("prefix_sha256")) != \
           (b.get("recipient_case_id"), b.get("prefix_sha256")):
            raise ValueError("Candidate and orthogonal were scored on different recipient prompts")
    if {k[0] for k in observations} != required:
        raise ValueError("Sweep is missing one or more preselected decoder blocks")
    # Every decoder block must be measured on the *same* seed/probe cases.
    expected = {(seed, probe) for layer, seed, probe in observations if layer == min(required)}
    if not expected or {seed for seed, _ in expected} != set(test) or \
       any({(seed, probe) for layer, seed, probe in observations if layer == block}
           != expected for block in required):
        raise ValueError("Unequal test units across selected decoder blocks")
    effect = {key: float(pair["candidate"]["oriented_effect"] -
                         pair["orthogonal"]["oriented_effect"])
              for key, pair in observations.items()}
    if not all(math.isfinite(v) for v in effect.values()):
        raise ValueError("Non-finite paired causal specificity")
    by_seed = {}
    for selector, blocks in selected.items():
        per_seed = {}
        for seed in sorted({seed for seed, _ in expected}):
            probes = sorted(p for s, p in expected if s == seed)
            per_seed[seed] = statistics.mean(effect[(int(block), seed, probe)]
                                             for block in blocks for probe in probes)
        by_seed[selector] = per_seed
    sieve_score = by_seed["sieve"]
    summaries = {}
    tests = []
    keys = [k for k in selected if k != "sieve"]
    for selector in selected:
        scores = by_seed[selector]
        item = {"blocks": selected[selector], "n_seed_clusters": len(scores),
                "mean_specificity": float(statistics.mean(scores.values())),
                "seed_means": {str(k): float(v) for k, v in scores.items()}}
        if selector != "sieve":
            differences = [sieve_score[seed] - scores[seed] for seed in sorted(scores)]
            item["mean_sieve_minus_selector"] = float(statistics.mean(differences))
            item["signflip"] = signflip_pvalue(differences)
            tests.append(item["signflip"]["p_value"])
        summaries[selector] = item
    adjusted = holm_adjust(tests)
    for key, p in zip(keys, adjusted):
        summaries[key]["holm_p_value"] = p
    return {
        "schema": AUDIT_SCHEMA, "selection_graph_sha256": manifest["graph_sha256"],
        "model": next(iter(models))[0], "tokenizer": next(iter(models))[1],
        "direction_source": source, "phase": phase, "mode": mode, "alpha": alpha,
        "train_seeds": sorted(train), "test_seeds": sorted(test),
        "train_probe_ids": sorted(train_probe), "test_probe_ids": sorted(eval_probe),
        "n_scored_seeds": len(sieve_score), "n_scored_probes": len(expected),
        "selected_block_budget": len(selected["sieve"]),
        "selector_comparisons": summaries,
        "interpretation": "Held-out decoder-*block* intervention proxy; selected graph nodes were not individually ablated.",
        "limitations": [
            "A layer selected by many components is still one block intervention; node-level causal claims require component patching.",
            "Comparing selected blocks does not prove the PercInterp graph's edge attribution was faithful.",
            "Probe and seed independence, attribution provenance and preregistration remain external obligations.",
            "Selector scores share the same evaluation probes and are paired by held-out seed.",
        ],
    }


def select_file(input_path: str, output_path: str, **kwargs) -> dict:
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = select_attribution_graph(raw, **kwargs)
    result["graph_file_sha256"] = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def audit_file(manifest_path: str, sweep_path: str, output_path: str, **kwargs) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(sweep_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    result = audit_sweep_by_block(manifest, rows, **kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
