"""Optional open-weight activation capture and descriptive vector comparisons.

Capture deliberately records model outputs only at the *final input position* of
an independently evaluated, fully scripted chat prefix. It does not implement
transport, SAE feature identification, or causal intervention.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_layers(value: str, n: int) -> tuple[int, ...]:
    """Hidden-state index 0 = embeddings, indexes 1..n-1 = layer outputs."""
    choices = [s.strip() for s in value.split(",")]
    if not choices or any(not c for c in choices):
        raise ValueError("--layers must contain at least one valid hidden-state index")
    values: list[int] = []
    for choice in choices:
        index = n - 1 if choice == "last" else int(choice)
        if not 0 <= index < n:
            raise ValueError(f"Hidden-state index {index} outside valid range 0..{n-1}")
        if index not in values:
            values.append(index)
    return tuple(values)


def cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        raise ValueError("Activation vectors must be nonempty and have equal dimensions")
    aa = math.sqrt(sum(x * x for x in a))
    bb = math.sqrt(sum(x * x for x in b))
    if aa == 0 or bb == 0:
        raise ValueError("Zero-norm vector cannot be compared using cosine distance")
    similarity = sum(x * y for x, y in zip(a, b)) / (aa * bb)
    return max(0.0, 1.0 - min(1.0, max(-1.0, similarity)))


def _vectors(records: list[dict[str, Any]]) -> dict[tuple, list[float]]:
    index: dict[tuple, list[float]] = {}
    for row in records:
        if row.get("schema_version") != 1:
            raise ValueError("Activation JSONL schema_version must equal 1")
        for layer, vector in row["vectors"].items():
            key = (row["condition"], row["phase"], row["seed"], row["probe_id"], layer)
            if key in index:
                raise ValueError(f"Duplicate activation vector key: {key}")
            index[key] = vector
    return index


def activation_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive distances; never interprets these as geometric holonomy."""
    if not records:
        raise ValueError("Cannot analyze an empty activation capture")
    models = {(row.get("model"), row.get("tokenizer")) for row in records}
    if len(models) != 1:
        raise ValueError("Cannot compare activation vectors across different models/tokenizers")
    indexed = _vectors(records)
    baseline: dict[tuple[str, str, str], list[float]] = {}
    comparisons: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (condition, phase, seed, probe, layer), vector in indexed.items():
        if phase != "baseline":
            key = (condition, "baseline", seed, probe, layer)
            if key in indexed:
                comparisons[("baseline_drift", condition, phase, layer)].append(
                    cosine_distance(vector, indexed[key])
                )
        if condition == "ordered":
            key = ("tail_matched", phase, seed, probe, layer)
            if key in indexed:
                comparisons[("tail_matched_distance", "ordered", phase, layer)].append(
                    cosine_distance(vector, indexed[key])
                )
    values = []
    for (metric, condition, phase, layer), dists in sorted(comparisons.items()):
        values.append({
            "metric": metric,
            "condition": condition,
            "phase": phase,
            "layer": layer,
            "n_pairs": len(dists),
            "mean_cosine_distance": round(statistics.mean(dists), 8),
        })
    return {
        "schema_version": 1,
        "model": next(iter(models))[0],
        "tokenizer": next(iter(models))[1],
        "n_records": len(records),
        "comparisons": values,
        "limitations": [
            "Unaligned token positions and transcript length can affect activation distances.",
            "Last-prefix-token activations summarize transcript conditioning, not a persistent recurrent state.",
            "Cosine distances are descriptive; they do not define parallel transport or holonomy.",
            "Activation correlations alone do not identify causal mechanisms or VPD components.",
        ],
    }


def capture(
    *, input_path: str, output_path: str, model_name: str,
    layers: str = "last", device: str = "auto", dtype: str = "auto",
    allow_download: bool = False, trust_remote_code: bool = False,
    max_input_tokens: int = 4096, max_cases: int | None = None,
) -> int:
    """Capture selected hidden states on a local/cached Hugging Face chat model.

    Optional heavy dependencies are imported only in this command. User must
    explicitly allow network downloads and execution of custom model code.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional extra: pip install -e '.[mechanistic]'") from exc

    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if hasattr(torch.backends, "mps")
                  and torch.backends.mps.is_available() else "cpu")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported device: {device}")
    if dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    resolved_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive")

    kwargs = {"local_files_only": not allow_download,
              "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=resolved_dtype, **kwargs
    ).to(device)
    model.eval()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(input_path).open(encoding="utf-8") as source, out_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list):
                raise ValueError("Expected generated case JSONL with messages as an array")
            # Transformers v4 and v5 have slightly different return conventions.
            encoded = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            if isinstance(encoded, torch.Tensor):
                inputs = {"input_ids": encoded.to(device)}
            else:
                inputs = {k: v.to(device) for k, v in encoded.items()
                          if isinstance(v, torch.Tensor)}
            if "input_ids" not in inputs:
                raise ValueError("Chat template did not return input_ids")
            tokens = int(inputs["input_ids"].shape[-1])
            if tokens > max_input_tokens:
                raise ValueError(
                    f"Case {row['case_id']} contains {tokens} tokens; "
                    f"limit is {max_input_tokens}. Increase the limit explicitly."
                )
            with torch.inference_mode():
                response = model(**inputs, output_hidden_states=True,
                                 use_cache=False, return_dict=True)
            hidden = response.hidden_states
            if not hidden:
                raise RuntimeError("Model returned no hidden states")
            chosen = parse_layers(layers, len(hidden))
            vectors = {f"hs_{i}": hidden[i][0, -1].detach().float().cpu().tolist()
                       for i in chosen}
            token_hash = hashlib.sha256(
                json.dumps(inputs["input_ids"].cpu().tolist(), separators=(",", ":")).encode()
            ).hexdigest()
            output.write(json.dumps({
                "schema_version": 1,
                "case_id": row["case_id"],
                "condition": row["condition"], "phase": row["phase"],
                "seed": row["seed"], "probe_id": row["probe_id"],
                "model": model_name,
                "tokenizer": str(tokenizer.name_or_path),
                "actual_model_parameter_dtype": str(next(model.parameters()).dtype),
                "device": device, "requested_dtype": dtype,
                "token_count": tokens, "input_ids_sha256": token_hash,
                "vectors": vectors,
            }, ensure_ascii=False) + "\n")
            output.flush()
            count += 1
            if max_cases is not None and count >= max_cases:
                break
    return count
