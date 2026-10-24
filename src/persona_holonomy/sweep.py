"""Held-out layer and dose-response tests for a candidate persona direction.

An identified VPD or SAE feature requires more than these residual-space tests.
We preregister source/recipient conditions, layer indices, and doses; learn
activation-mean differences on *training transcript seeds only* and evaluate:

1. Sufficiency-like injection: h_tail <- h_tail + alpha * d.
2. Necessity-like projection erasure: h_ordered <- h_ordered - alpha
      <h_ordered - mean(h_tail_train), unit(d)> * unit(d).

Each is compared with norm-matched orthogonal-direction nulls at the same target token.
The erasure comparison is calibrated to the *realized* projection magnitude,
not the norm of the training difference. Outcomes are registered alternative
continuation log-likelihood margins, split into first and later token terms.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from .analysis import _bootstrap_ci
from .inference import signflip_pvalue, holm_adjust
from .interventions import (
    CausalProbe, STYLE_COMPLETIONS, _orthogonal_random, _read_cases, prepare_pairs, select_block,
    split_seeds,
)


def _fingerprint(case: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        case["messages"][:-1], sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def split_and_guard(pairs, *, holdout_fraction: float, split_seed: int):
    """Keep complete transcript histories out of the opposite split."""
    train, test = split_seeds(
        (recipient["seed"] for _, recipient in pairs),
        holdout_fraction=holdout_fraction, split_seed=split_seed,
    )
    identities = {"train": set(), "test": set()}
    for donor, recipient in pairs:
        split = "train" if recipient["seed"] in train else "test"
        for case in (donor, recipient):
            identities[split].add(_fingerprint(case))
    if identities["train"] & identities["test"]:
        raise ValueError("Train/test history leakage: regenerate with --variation seeded_v1")
    return train, test



def _partition_pairs(pairs, *, holdout_fraction: float, split_seed: int,
                     heldout_probe_ids: tuple[str, ...]):
    """Canonical train/test split for model runs and offline preflight."""
    train, test = split_and_guard(pairs, holdout_fraction=holdout_fraction,
                                  split_seed=split_seed)
    available = {target["probe_id"] for _, target in pairs}
    if len(set(heldout_probe_ids)) != len(heldout_probe_ids):
        raise ValueError("Held-out probe IDs must be unique")
    if not set(heldout_probe_ids) <= available:
        raise ValueError("Held-out probes must occur in the input cases")
    training_probes = available - set(heldout_probe_ids)
    if not training_probes:
        raise ValueError("Probe holdout must leave at least one training probe")
    eval_probes = set(heldout_probe_ids) if heldout_probe_ids else available
    selected = [(source, target) for source, target in pairs
                if target["seed"] in test and target["probe_id"] in eval_probes]
    training = [(source, target) for source, target in pairs
                if target["seed"] in train and target["probe_id"] in training_probes]
    if not selected or not training:
        raise ValueError("Probe/seed split produced an empty training or evaluation set")
    return train, test, training, selected, training_probes, eval_probes


def _validate_readout_nulls(readout: str, n_nulls: int) -> None:
    if readout not in ("style", "copy_free"):
        raise ValueError("readout must be style or copy_free")
    if isinstance(n_nulls, bool) or not isinstance(n_nulls, int) or n_nulls < 1:
        raise ValueError("n_nulls must be an integer >= 1")


def sweep_plan(*, cases: list[dict[str, Any]], layers: tuple[str, ...],
               alphas: tuple[float, ...], source_conditions: tuple[str, ...],
               reference_condition: str = "tail_matched",
               phases: tuple[str, ...] = ("exit",),
               holdout_fraction: float = .4, split_seed: int = 20261022,
               heldout_probe_ids: tuple[str, ...] = (),
               self_check: bool = True, readout: str = "style",
               n_nulls: int = 1) -> dict[str, Any]:
    """Offline forward-call count assuming caching in ``run_sweep``.

    Invocations differ greatly in token length. The count is not a FLOP, peak
    VRAM, elapsed time, or token count estimate.
    """
    _validate_readout_nulls(readout, n_nulls)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Layers must be nonempty and unique")
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("Alphas must be nonempty and unique")
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas):
        raise ValueError("Alphas must be finite and within [0,1]")
    pairs = prepare_pairs(cases, source_conditions=source_conditions,
                          target_condition=reference_condition, phases=phases)
    train, test, training, selected, train_probes, eval_probes = _partition_pairs(
        pairs, holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probe_ids=heldout_probe_ids)
    base_cases = {case["case_id"] for donor, ref in selected for case in (donor, ref)}
    capture_cases = {case["case_id"] for donor, ref in training for case in (donor, ref)}
    capture_cases.update(donor["case_id"] for donor, _ in selected)
    if self_check:
        capture_cases.update(base_cases)
    nonzero_alphas = sum(a != 0 for a in alphas)
    continuations = 5 if readout == "copy_free" else 2
    patches_per_layer = len(selected) * 2 * (1 + n_nulls) * continuations * nonzero_alphas
    identity_per_layer = continuations * len(base_cases) if self_check else 0
    capture_per_layer = len(capture_cases)
    baseline_all_layers = continuations * len(base_cases)
    total_forwards = len(layers) * (
        patches_per_layer + identity_per_layer + capture_per_layer
    ) + baseline_all_layers
    return {
        "schema_version": 1,
        "experiment": "causal_sweep_preflight",
        "readout": readout,
        "n_nulls": n_nulls,
        "n_input_cases": len(cases),
        "sources": list(source_conditions),
        "reference_condition": reference_condition,
        "phases": list(phases),
        "layers_requested": list(layers),
        "alphas": list(alphas),
        "train_seeds": list(train),
        "test_seeds": list(test),
        "train_probe_ids": sorted(train_probes),
        "test_probe_ids": sorted(eval_probes),
        "probe_holdout": bool(heldout_probe_ids),
        "n_training_pairs": len(training),
        "n_test_pairs": len(selected),
        "n_distinct_baseline_cases": len(base_cases),
        "n_captures_per_layer": capture_per_layer,
        "n_self_patch_scoring_forwards_per_layer": identity_per_layer,
        "n_intervention_scoring_forwards_per_layer": patches_per_layer,
        "n_baseline_scoring_forwards": baseline_all_layers,
        "estimated_total_model_forwards": total_forwards,
        "max_history_characters": max(
            sum(len(m["content"]) for m in c["messages"]) for c in cases
        ),
        "caveats": [
            "Forward-call counts mirror current caching and are not time or VRAM estimates.",
            "Maximum history characters excludes serialized chat-template tokens.",
            "Layer aliases are only validated against a model when actually running the sweep.",
            "Long teacher-forced scoring passes cost more than short activation-only passes.",
        ],
    }

def _signed_effect(mode: str, difference: float) -> float:
    # Positive means more theatrical for injection, less theatrical for erasure.
    return difference if mode == "inject" else -difference


def _safe_delta(patched: dict, baseline: dict, field: str):
    if patched.get(field) is None or baseline.get(field) is None:
        return None
    return float(patched[field] - baseline[field])


def run_sweep(
    *, cases: list[dict[str, Any]], model: Any, tokenizer: Any,
    layers: tuple[str, ...] = ("0", "last"),
    alphas: tuple[float, ...] = (0., .5, 1.),
    source_conditions: tuple[str, ...] = ("ordered", "neutral"),
    reference_condition: str = "tail_matched",
    phases: tuple[str, ...] = ("exit",),
    holdout_fraction: float = .4,
    split_seed: int = 20261022,
    max_total_tokens: int = 4096,
    device: str = "cpu", model_name: str = "unknown",
    self_check: bool = True,
    heldout_probe_ids: tuple[str, ...] = (),
    chat_template_kwargs: dict | None = None,
    readout: str = "style", n_nulls: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate preregistered layers/doses with a single loaded model.

    A layer is captured once per distinct case, and baseline scores are reused
    across all layers and doses. All evaluation is restricted to held-out seeds.
    Copy-free scoring averages four theatrical registers against the plain
    continuation. Orthogonal rows average ``n_nulls`` independently seeded nulls;
    null zero and the default style row schema retain their original behavior.
    """
    _validate_readout_nulls(readout, n_nulls)
    if not layers or not alphas or not phases:
        raise ValueError("Provide at least one layer, alpha and phase")
    if len(set(source_conditions)) != len(source_conditions):
        raise ValueError("Donor source conditions must be unique")
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas):
        raise ValueError("Alphas must be finite values in [0,1]")
    if len(set(alphas)) != len(alphas):
        raise ValueError("Alphas must be unique")
    pairs = prepare_pairs(cases, source_conditions=source_conditions,
                          target_condition=reference_condition, phases=phases)
    model.eval()
    # Validate aliases that resolve to the same physical decoder block.
    from .interventions import decoder_blocks
    _, blocks = decoder_blocks(model)
    indices = [select_block(name, len(blocks)) for name in layers]
    if len(set(indices)) != len(indices):
        raise ValueError("Layer selection includes a repeated decoder block")
    train, test, training, selected, training_probes, evaluation_probes = _partition_pairs(
        pairs, holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probe_ids=heldout_probe_ids)
    baselines: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    for block_index in indices:
        runner = CausalProbe(model, tokenizer, layer=str(block_index),
                             device=device, max_total_tokens=max_total_tokens,
                             chat_template_kwargs=chat_template_kwargs)
        activations = {}

        def activation(case):
            key = case["case_id"]
            if key not in activations:
                activations[key] = runner.capture_vector(case["messages"]).cpu()
            return activations[key]

        def read_margin(case, **patch):
            if readout == "style":
                return runner.style_margin(
                    case["messages"], probe_id=case["probe_id"], **patch)
            score = runner.copy_free_margin(
                case["messages"], probe_id=case["probe_id"], **patch)
            return {
                **score,
                "margin": score["copy_free_margin"],
                "first_token_margin": score["copy_free_first_token_margin"],
                "later_token_margin": score["copy_free_later_token_margin"],
            }

        def baseline(case):
            key = case["case_id"]
            if key not in baselines:
                score = read_margin(case)
                if readout == "copy_free":
                    # Metadata only: no additional model forwards. The theatrical
                    # token count is averaged across the scored registers.
                    prefix_ids = runner._prefix(case["messages"])["input_ids"]
                    score.update({
                        "plain_tokens": len(tokenizer.encode(
                            STYLE_COMPLETIONS[case["probe_id"]]["plain"],
                            add_special_tokens=False)),
                        "theatrical_tokens": statistics.mean(
                            r["tokens"] for r in score["copy_free_by_register"].values()),
                        "prefix_sha256": hashlib.sha256(struct.pack(
                            f"<{prefix_ids.numel()}q",
                            *prefix_ids.detach().cpu().reshape(-1).tolist(),
                        )).hexdigest(),
                    })
                baselines[key] = score
            return baselines[key]

        directions = {}
        for condition in source_conditions:
            matched = [(donor, target) for donor, target in training
                       if donor["condition"] == condition]
            delta = runner.torch.stack([
                activation(donor) - activation(target) for donor, target in matched
            ]).float().mean(0)
            length = float(runner.torch.linalg.vector_norm(delta))
            if length <= 1e-10:
                raise ValueError(f"Zero direction for {condition}, block {block_index}")
            center = runner.torch.stack([activation(target) for _, target in matched]).float().mean(0)
            directions[condition] = (delta, center)

        check_delta: dict[str, float | None] = {}
        for donor, recipient in selected:
            condition = donor["condition"]
            direction, center = directions[condition]
            unit = direction / runner.torch.linalg.vector_norm(direction)
            for mode, tested_case in (("inject", recipient), ("erase", donor)):
                case_id = tested_case["case_id"]
                score = baseline(tested_case)
                check_key = f"{block_index}:{case_id}"
                if check_key not in check_delta:
                    if self_check:
                        own = activation(tested_case)
                        patched_self = read_margin(
                            tested_case,
                            vector=own, mode="transplant", alpha=1.,
                        )
                        check_delta[check_key] = patched_self["margin"] - score["margin"]
                    else:
                        check_delta[check_key] = None
                if mode == "erase":
                    displacement = (activation(donor) - center).float()
                    amplitude = float(runner.torch.dot(displacement, unit))
                    intervention = {"vector": direction, "reference": center, "mode": "erase"}
                    shift_norm = abs(amplitude)
                else:
                    amplitude = float(runner.torch.linalg.vector_norm(direction))
                    intervention = {"vector": direction, "mode": "add"}
                    shift_norm = amplitude
                # Reproducible, *paired* null for each recipient and source.
                digest = hashlib.sha256(
                    f"{split_seed}|{block_index}|{condition}|{tested_case['case_id']}|{mode}".encode()
                ).digest()
                null_vectors = []
                for null_index in range(n_nulls):
                    # Keep the original seed exactly for null zero.
                    indexed_digest = (digest if null_index == 0 else hashlib.sha256(
                        digest + f"|{null_index}".encode()).digest())
                    null_seed = int.from_bytes(indexed_digest[:8], "big") % (2**63 - 1)
                    orth = _orthogonal_random(direction, seed=null_seed, torch=runner.torch)
                    if mode == "erase":
                        # Same per-target shift L2 norm as projection erasure.
                        null_vector = (-amplitude * orth /
                                       runner.torch.linalg.vector_norm(orth))
                    else:
                        null_vector = orth
                    null_vectors.append(null_vector)

                for alpha in alphas:
                    for control in ("candidate", "orthogonal"):
                        null_scores = []
                        if alpha == 0:
                            patched = score
                            if control == "orthogonal":
                                null_scores = [score] * n_nulls
                        elif control == "candidate":
                            patched = read_margin(
                                tested_case,
                                alpha=alpha, **intervention,
                            )
                        else:
                            null_scores = [read_margin(
                                tested_case, vector=vector, mode="add", alpha=alpha,
                            ) for vector in null_vectors]
                            patched = null_scores[0]
                        effect = patched["margin"] - score["margin"]
                        first_delta = _safe_delta(patched, score, "first_token_margin")
                        later_delta = _safe_delta(patched, score, "later_token_margin")
                        null_effects = [s["margin"] - score["margin"] for s in null_scores]
                        if control == "orthogonal" and n_nulls > 1:
                            patched = {"margin": statistics.mean(s["margin"] for s in null_scores)}
                            effect = statistics.mean(null_effects)
                            deltas = [[_safe_delta(s, score, field) for s in null_scores]
                                      for field in ("first_token_margin", "later_token_margin")]
                            first_delta, later_delta = [
                                statistics.mean(values) if None not in values else None
                                for values in deltas]
                        results.append({
                            **({"chat_template_kwargs": chat_template_kwargs}
                               if chat_template_kwargs is not None else {}),
                            "schema_version": 2,
                            "experiment": "direction_sweep",
                            "model": model_name,
                            "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
                            "block_path": runner.block_path,
                            "layer": block_index,
                            "alpha": float(alpha),
                            "mode": mode,
                            "control": control,
                            "negative_control": control == "orthogonal",
                            "direction_source": condition,
                            "reference_condition": reference_condition,
                            "recipient_condition": tested_case["condition"],
                            "source_case_id": donor["case_id"],
                            "reference_case_id": recipient["case_id"],
                            "recipient_case_id": case_id,
                            "seed": tested_case["seed"],
                            "probe_id": tested_case["probe_id"],
                            "phase": tested_case["phase"],
                            "training_seeds": list(train),
                            "test_seeds": list(test),
                            "training_probe_ids": sorted(training_probes),
                            "evaluation_probe_ids": sorted(evaluation_probes),
                            "probe_holdout": bool(heldout_probe_ids),
                            "direction_l2_norm": amplitude if mode == "inject" else float(runner.torch.linalg.vector_norm(direction)),
                            "patch_l2_norm": alpha * shift_norm,
                            "signed_projection": amplitude if mode == "erase" else None,
                            "baseline_margin": score["margin"],
                            "patched_margin": patched["margin"],
                            "effect": effect,
                            "oriented_effect": _signed_effect(mode, effect),
                            "first_token_effect": first_delta,
                            "later_token_effect": later_delta,
                            "self_patch_delta": check_delta[check_key],
                            "plain_tokens": score["plain_tokens"],
                            "theatrical_tokens": score["theatrical_tokens"],
                            "prefix_sha256": score["prefix_sha256"],
                            **({"readout": readout} if readout != "style" else {}),
                            **({"null_effects": null_effects, "n_nulls": n_nulls}
                               if control == "orthogonal" and n_nulls > 1 else {}),
                        })
    return results


def sweep_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive seed-clustered contrasts, pairing every candidate with null."""
    if not rows:
        raise ValueError("Empty sweep")
    readouts = {r.get("readout", "style") for r in rows}
    if len(readouts) != 1:
        raise ValueError("Sweep report cannot mix readouts")
    readout = next(iter(readouts))
    identity = {(r["model"], r["tokenizer"], r["reference_condition"]) for r in rows}
    if len(identity) != 1:
        raise ValueError("Sweep report requires one model, tokenizer and reference condition")
    probe_settings = {
        (tuple(r.get("training_probe_ids", ())),
         tuple(r.get("evaluation_probe_ids", ())),
         bool(r.get("probe_holdout", False))) for r in rows
    }
    if len(probe_settings) != 1:
        raise ValueError("Sweep report cannot mix probe holdout configurations")
    train_probes, eval_probes, probe_holdout = next(iter(probe_settings))
    observed = set()
    indexed = {}
    for row in rows:
        key = (row["layer"], row["alpha"], row["mode"], row["direction_source"],
               row["phase"], row["seed"], row["probe_id"], row["control"])
        if key in observed:
            raise ValueError(f"Duplicate sweep cell {key}")
        observed.add(key)
        indexed[key] = row
    groups = defaultdict(lambda: defaultdict(list))
    max_abs_self = 0.
    n_self = 0
    for key, row in indexed.items():
        if row["self_patch_delta"] is not None:
            n_self += 1
            max_abs_self = max(max_abs_self, abs(row["self_patch_delta"]))
        if row["control"] != "candidate":
            continue
        other = indexed.get((*key[:-1], "orthogonal"))
        if other is None:
            raise ValueError(f"Unpaired candidate and orthogonal null at {key}")
        record = (row["layer"], row["mode"], row["direction_source"],
                  row["phase"], row["alpha"])
        groups[record][row["seed"]].append((row, other))
    summaries = []
    for key, by_seed in sorted(groups.items()):
        measured = []
        nulls = []
        differences = []
        first = []
        later = []
        for seed, probes in sorted(by_seed.items()):
            measured.append(statistics.mean(r["oriented_effect"] for r, _ in probes))
            nulls.append(statistics.mean(n["oriented_effect"] for _, n in probes))
            differences.append(statistics.mean(
                r["oriented_effect"] - n["oriented_effect"] for r, n in probes))
            first.append(statistics.mean(
                _signed_effect(key[1], r["first_token_effect"]) for r, _ in probes
                if r["first_token_effect"] is not None))
            tails = [_signed_effect(key[1], r["later_token_effect"]) for r, _ in probes
                     if r["later_token_effect"] is not None]
            if tails:
                later.append(statistics.mean(tails))
        layer, mode, source, phase, alpha = key
        inference = signflip_pvalue(differences)
        summaries.append({
            "layer": layer, "mode": mode, "direction_source": source,
            "phase": phase, "alpha": alpha,
            "n_seeds": len(by_seed),
            "n_paired_probes": sum(len(v) for v in by_seed.values()),
            "mean_oriented_effect": statistics.mean(measured),
            "mean_orthogonal_null": statistics.mean(nulls),
            "mean_paired_specificity": statistics.mean(differences),
            "seed_bootstrap_95pct_ci": _bootstrap_ci(differences),
            "paired_signflip_p_value": inference["p_value"],
            "paired_signflip_method": inference["method"],
            "paired_signflip_null_draws": inference["null_draws"],
            "holm_familywise_p_value": None,
            "mean_first_token_oriented_effect": statistics.mean(first),
            "mean_later_token_oriented_effect": statistics.mean(later) if later else None,
        })
    # Register each nonzero layer/mode/source/phase/dose cell as one hypothesis.
    # alpha=0 is an identity test and excluded from multiplicity correction.
    hypothesis_indices = [i for i, row in enumerate(summaries) if row["alpha"] > 0]
    adjusted = holm_adjust([summaries[i]["paired_signflip_p_value"]
                            for i in hypothesis_indices])
    for i, value in zip(hypothesis_indices, adjusted):
        summaries[i]["holm_familywise_p_value"] = value
    alphas = sorted({r["alpha"] for r in rows})
    zero = [r for r in rows if r["alpha"] == 0]
    zero_error = max((abs(r["effect"]) for r in zero), default=None)
    model_name, tokenizer_name, ref = next(iter(identity))
    return {
        "schema_version": 2,
        "model": model_name,
        **({"readout": readout} if readout != "style" else {}),
        "tokenizer": tokenizer_name,
        "reference_condition": ref,
        "layers": sorted({r["layer"] for r in rows}),
        "alphas": alphas,
        "modes": sorted({r["mode"] for r in rows}),
        "n_records": len(rows),
        "n_test_seeds": len({r["seed"] for r in rows}),
        "n_training_seeds": len(rows[0]["training_seeds"]),
        "training_probe_ids": list(train_probes),
        "evaluation_probe_ids": list(eval_probes),
        "probe_holdout": probe_holdout,
        "hypothesis_family_size": len(hypothesis_indices),
        "hypothesis_family": "all reported layer/mode/source/phase/alpha>0 contrasts",
        "max_abs_self_patch_delta": max_abs_self if n_self else None,
        "max_abs_alpha_zero_effect": zero_error,
        "identity_check_passed_1e_3": max_abs_self <= 1e-3 if n_self else None,
        "summaries": summaries,
        "limitations": [
            "Residual-stream rank-one projection is not SAE-feature ablation or VPD attribution.",
            "All cases use fixed scripted assistant turns; these are not interactive model rollouts.",
            "Style readout uses different completion wordings and lengths; prefer external response grading for validation.",
            "Training and evaluation transcripts differ by seed and share genre and templated syntax; probes may repeat unless held out.",
            "A single last-prefix-token hook primarily changes the first suffix prediction in final blocks.",
            "Bootstrap intervals resample transcript seeds; sign-flips operate on seed-averaged paired effects.",
            "Exact sign-flip p-values assume exchangeability of signs under the null, which must be justified for the design.",
            "Holm adjustment covers all reported nonzero-dose contrasts and does not rescue post-hoc layer selection.",
            "Use --heldout-probes for entirely unseen technical readout questions; seeded holdout alone reuses probe text.",
            "Inspect unregistered layers as exploratory; a best-layer finding requires a fresh validation split.",
            "No model consciousness, model-identity switch, or geometric holonomy is implied.",
        ],
    }


def sweep_markdown(report: dict[str, Any]) -> str:
    text = ["# Persona Holonomy Bench: causal layer sweep", "",
            f"Model: `{report['model']}`; reference: `{report['reference_condition']}`",
            *([f"Readout: `{report['readout']}`"] if "readout" in report else []),
            f"Layers: {report['layers']}; strengths: {report['alphas']}",
            f"Training seeds: {report['n_training_seeds']}; held-out: {report['n_test_seeds']}",
            f"Train probes: {report['training_probe_ids']}; evaluation probes: {report['evaluation_probe_ids']}; probe holdout: {report['probe_holdout']}",
            f"Holm multiplicity family: {report['hypothesis_family_size']} nonzero-dose comparisons",
            f"Identity max |Δ|: {report['max_abs_self_patch_delta']}; alpha-zero max |Δ|: {report['max_abs_alpha_zero_effect']}",
            "", "Positive **oriented effect** means more Gothic preference after injection or less after erasure.",
            "Specificity subtracts the norm-matched orthogonal null on the same seed/probe.", "",
            "| Layer | Mode | Donor | Phase | α | Seeds | Pairs | Effect | Null | Specificity | 95% CI | First-token effect | Later-token effect | Sign-flip p | Holm p |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
           ]
    for s in report["summaries"]:
        tail = s["mean_later_token_oriented_effect"]
        text.append(f"| {s['layer']} | {s['mode']} | {s['direction_source']} | {s['phase']} | {s['alpha']:.2f} | "
                    f"{s['n_seeds']} | {s['n_paired_probes']} | {s['mean_oriented_effect']:+.4f} | "
                    f"{s['mean_orthogonal_null']:+.4f} | {s['mean_paired_specificity']:+.4f} | "
                    f"{s['seed_bootstrap_95pct_ci'] or 'n/a'} | "
                    f"{s['mean_first_token_oriented_effect']:+.4f} | "
                    f"{tail if tail is not None else 'n/a'} | "
                    f"{s['paired_signflip_p_value']:.4f} | "
                    f"{s['holm_familywise_p_value'] if s['holm_familywise_p_value'] is not None else 'n/a'} |")
    text.extend(["", "## Interpretation limits", ""])
    text.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(text) + "\n"


def sweep_hf(
    *, input_path: str, output_path: str, model_name: str,
    device: str = "auto", dtype: str = "auto", allow_download: bool = False,
    trust_remote_code: bool = False, chat_template_kwargs: dict | None = None,
    readout: str = "style", n_nulls: int = 1,
    **kwargs,
) -> int:
    """Opt-in local model load. Requires no network by default."""
    _validate_readout_nulls(readout, n_nulls)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install mechanistic dependencies with pip install -e '.[mechanistic]'") from exc
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends.mps, "is_available", lambda: False)()
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
    rows = run_sweep(cases=_read_cases(input_path), model=model, tokenizer=tokenizer,
                     model_name=model_name, device=device,
                     readout=readout, n_nulls=n_nulls,
                     chat_template_kwargs=chat_template_kwargs, **kwargs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
