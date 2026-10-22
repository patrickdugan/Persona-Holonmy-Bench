"""PHB activation-frame adapter for PercInterp's discrete holonomy experiment.

Defines a state graph *explicitly*: nodes are (condition, phase) cells; each
node receives a rank-r PCA frame estimated from matched seed/probe final-token
hidden vectors at one transformer hidden-state index. Edges join adjacent
phases within condition and the corresponding phase across two conditions.
Polar Procrustes aligns frames; cycle spectra are invariant under local
orthonormal basis changes. This is a specified geometric diagnostic, not a
claim that model computation follows this connection or that it has memory.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "phb.percinterp.frame_holonomy.v1"


def analyze_frames(records: list[dict[str, Any]], *, layer: str,
                   conditions: tuple[str, str] = ("ordered", "tail_matched"),
                   phases: tuple[str, ...] = ("peak", "exit", "washout_2"),
                   rank: int = 2, min_overlap: float = 0.1,
                   epsilon: float = 0.05, seed: int = 20261022,
                   null_shuffles: int = 16) -> dict[str, Any]:
    try:
        import numpy as np
        from percinterp.holonomy import (
            analyze_connection, frame_edge_transports, signature,
        )
        from percinterp.measure import rank_entry
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional PercInterp package from MoralityLabAI/percolation-interp "
            "(Python >=3.11)."
        ) from exc
    if not records or len(conditions) != 2 or len(set(conditions)) != 2:
        raise ValueError("Require two different conditions and nonempty activation rows")
    if len(phases) < 2 or len(set(phases)) != len(phases):
        raise ValueError("Require two or more distinct, ordered phases")
    if type(rank) is not int or rank < 1 or not 0 <= min_overlap <= 1 or \
       not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Invalid rank, overlap, or epsilon")
    if type(null_shuffles) is not int or not 0 <= null_shuffles <= 1000:
        raise ValueError("null_shuffles must be in 0..1000")
    models = {(r.get("model"), r.get("tokenizer")) for r in records}
    if len(models) != 1:
        raise ValueError("Activation rows must share the same model and tokenizer")
    state: dict[tuple[str, str], dict[tuple[int, str], list[float]]] = defaultdict(dict)
    input_hashes: dict[tuple[str, str], dict[tuple[int, str], str]] = defaultdict(dict)
    for r in records:
        if r.get("schema_version") != 1:
            raise ValueError("Expected PHB activation capture schema_version 1")
        cond, phase = r["condition"], r["phase"]
        if cond not in conditions or phase not in phases:
            continue
        vec = r.get("vectors", {}).get(layer)
        if vec is None:
            raise ValueError(f"Missing hidden-state vector {layer} at {cond}/{phase}")
        key = (r["seed"], r["probe_id"])
        node = (cond, phase)
        if key in state[node]:
            raise ValueError(f"Duplicate seed/probe observation at {node}/{key}")
        state[node][key] = vec
        input_hashes[node][key] = r.get("input_ids_sha256", "")
    nodes = [(condition, phase) for condition in conditions for phase in phases]
    if any(not state[node] for node in nodes):
        raise ValueError("Every requested condition/phase cell needs activation observations")
    sample_sets = [set(state[node]) for node in nodes]
    if any(keys != sample_sets[0] for keys in sample_sets[1:]):
        raise ValueError("All state cells must have exactly the same seed/probe samples")
    samples = sorted(sample_sets[0])
    seeds = sorted({s for s, _ in samples})
    if len(seeds) < rank + 1 or len(samples) < rank + 1:
        raise ValueError("Need at least rank+1 distinct seeds and samples for PCA frames")
    if len(samples) > 50000:
        raise ValueError("Too many samples for dense PCA; use a controlled subset")
    def fit_frames(observations):
        ambient_dim = None
        frames, explained = [], []
        for node in nodes:
            X = np.asarray([observations[node][key] for key in samples], dtype=np.float64)
            if X.ndim != 2 or not np.isfinite(X).all():
                raise ValueError("Hidden vectors must be rectangular and finite")
            if ambient_dim is None:
                ambient_dim = X.shape[1]
            if X.shape[1] != ambient_dim or ambient_dim < rank:
                raise ValueError("All hidden vectors must share ambient dimension >= rank")
            centered = X - X.mean(axis=0, keepdims=True)
            _, svals, vh = np.linalg.svd(centered, full_matrices=False)
            if len(svals) < rank or svals[rank - 1] <= max(1e-10, svals[0] * 1e-8):
                raise ValueError(f"Rank-deficient activation frame at {node}")
            frames.append(vh[:rank].T)
            explained.append(float((svals[:rank] ** 2).sum() / (svals ** 2).sum()))
        return np.stack(frames), explained, ambient_dim

    frames_arr, energy, ambient = fit_frames(state)
    src, dst, kinds = [], [], []
    P = len(phases)
    for k in range(2):
        for j in range(P - 1):
            src.append(k * P + j)
            dst.append(k * P + j + 1)
            kinds.append("time_adjacent")
    for j in range(P):
        src.append(j)
        dst.append(P + j)
        kinds.append("condition_matched")
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    transports, reliable, overlap = frame_edge_transports(
        frames_arr, src, dst, min_overlap=min_overlap)
    # Bond occupation uses ranked frame overlaps; certify weak links separately.
    entry = rank_entry(overlap)
    result = analyze_connection(len(nodes), src, dst, entry, transports,
                                reliable=reliable, epsilon=epsilon, max_recorded_cycles=256)
    # Independently rotate each local orthonormal basis. Loop eigenphases and
    # distance to identity must be unchanged, even when individual matrices vary.
    rng = np.random.default_rng(seed)
    rotated = []
    for F in frames_arr:
        Q, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
        rotated.append(F @ Q)
    transports_g, reliable_g, _ = frame_edge_transports(
        np.stack(rotated), src, dst, min_overlap=min_overlap)
    if not np.array_equal(reliable_g, reliable):
        raise RuntimeError("Gauge change altered overlap certification")
    gauge = analyze_connection(len(nodes), src, dst, entry, transports_g,
                               reliable=reliable, epsilon=epsilon, max_recorded_cycles=256)
    if len(gauge["loops"]) != len(result["loops"]):
        raise RuntimeError("Gauge change altered the loop set")
    error = 0.0
    for original, changed in zip(result["loops"], gauge["loops"]):
        if original["steps"] and changed["steps"]:
            # Compare spectral and Frobenius invariants, *not* raw H matrices.
            error = max(error,
                        abs(original["frobenius_distance"] - changed["frobenius_distance"]),
                        max(abs(a - b) for a, b in zip(
                            original["eigenphases_degrees"], changed["eigenphases_degrees"])))
    if error > 1e-5:
        raise RuntimeError(f"Frame holonomy gauge-invariance regression: {error}")
    # Condition-label permutation null. Swap *whole seed clusters*, including
    # all probes and phases. A seed can contribute multiple strongly correlated
    # probe rows and a longitudinal trajectory; treating them as independent
    # randomization units would exaggerate the apparent permutation sample size.
    # Every null gets its *own* frame fit and link ranking; weak edges are
    # certified using the same min_overlap as the observed experiment.
    def loop_strength(local_frames):
        tq, rel, ov = frame_edge_transports(local_frames, src, dst,
                                             min_overlap=min_overlap)
        nf = analyze_connection(len(nodes), src, dst, rank_entry(ov), tq,
                                reliable=rel, epsilon=epsilon,
                                max_recorded_cycles=256)
        return max((float(c["frobenius_distance"]) for c in nf["loops"]),
                   default=0.0), int(nf["cycles_recorded"])

    rng_null = np.random.default_rng(seed + 1)
    null_strengths = []
    null_cycles = []
    swap_mask_hashes = []
    if null_shuffles:
        for _ in range(null_shuffles):
            scrambled = {node: dict(values) for node, values in state.items()}
            swapped_seed = {s: rng_null.random() < .5 for s in seeds}
            # Record the *seed-cluster* permutation, not each correlated probe.
            # This can be replayed and tested independently of PCA roundoff.
            swap_mask_hashes.append(hashlib.sha256(json.dumps(
                [s for s in seeds if swapped_seed[s]], separators=(",", ":")
            ).encode("utf-8")).hexdigest())
            for phase in phases:
                first, second = (conditions[0], phase), (conditions[1], phase)
                for sample in samples:
                    if swapped_seed[sample[0]]:
                        scrambled[first][sample], scrambled[second][sample] = (
                            state[second][sample], state[first][sample])
            # Scrambled frames can be rank-deficient in degenerate synthetic
            # data; report an explicit error rather than fabricating a null.
            null_frames, _, _ = fit_frames(scrambled)
            strength, cycle_count = loop_strength(null_frames)
            null_strengths.append(strength)
            null_cycles.append(cycle_count)
    observed_strength = max((float(c["frobenius_distance"]) for c in result["loops"]),
                            default=0.0)
    shuffled_p = ((1 + sum(v >= observed_strength - 1e-12 for v in null_strengths)) /
                  (1 + len(null_strengths))) if null_strengths else None

    fields = [{"id": f"{c}:{p}", "condition": c, "phase": p,
               "rank_energy_fraction": energy[i]}
              for i, (c, p) in enumerate(nodes)]
    result.pop("src", None)
    result.pop("dst", None)
    result.pop("entry", None)
    return {
        "schema": SCHEMA, "model": next(iter(models))[0],
        "tokenizer": next(iter(models))[1], "hidden_state": layer,
        "ambient_dimension": ambient, "frame_rank": rank,
        "n_matched_seed_probe_samples": len(samples),
        "n_distinct_seeds": len(seeds), "seeds": seeds,
        "probe_ids": sorted({p for _, p in samples}),
        "nodes": fields,
        "edges": [{"source": fields[int(a)]["id"], "target": fields[int(b)]["id"],
                   "kind": kind, "weakest_singular_overlap": float(w),
                   "rank_occupation": float(e), "transport_certified": bool(cert)}
                  for a, b, kind, w, e, cert in zip(src, dst, kinds, overlap, entry, reliable)],
        "connection": "rank-r PCA frame, polar(Procrustes(dst.T @ src)) on explicit state-graph edges",
        "min_overlap": min_overlap, "epsilon": epsilon,
        "holonomy": {k: result[k] for k in ("ps", "largest", "betti1", "certified_betti1",
                                               "cycles_recorded", "total_certified_cycles", "truncated",
                                               "p_h", "p_h_known", "loops")},
        "gauge_invariance_max_error": error,
        "gauge_test_seed": seed,
        "history_label_swap_null": {
            "scheme": "whole-seed condition-label swap shared across all probes and phases",
            "permutation_unit": "seed_cluster",
            "n_permutation_units": len(seeds),
            "n_shuffles": null_shuffles,
            "observed_max_certified_loop_frobenius": observed_strength,
            "null_max_loop_frobenius": null_strengths,
            "null_certified_cycle_counts": null_cycles,
            "null_seed_swap_masks_sha256": swap_mask_hashes,
            "one_sided_permutation_p_value": shuffled_p,
            "p_value_resolution": 1 / (1 + null_shuffles) if null_shuffles else None,
        },
        "status": "exploratory_state_frame_transport",
        "limitations": [
            "Frames are empirical PCA bases of last-prefix-token vectors, not a model-defined state connection.",
            "Token positions differ between conversational histories; activations can encode sequence length and text.",
            "Holonomy is well-defined for this frame/connection choice; it does not establish persistent model identity.",
            "Seed-level label swaps preserve within-seed probe dependence and phase trajectories.",
            "Condition-label permutation is a within-dataset exploratory null; independent-seed replication is required.",
        ],
    }


def holonomy_file(input_path: str, output_path: str, **kwargs) -> dict:
    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = analyze_frames(rows, **kwargs)
    report["input_sha256"] = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
