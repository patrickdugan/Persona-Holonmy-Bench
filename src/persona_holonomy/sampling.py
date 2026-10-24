"""Sampled behavioral rates for Q5, using the existing lexical/style proxies."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .chat import render_chat
from .interventions import _read_cases
from .scoring import score_response


def sample_rates(
    *, cases, model, tokenizer, n_samples=20, temperature=1.0, top_p=1.0,
    max_new_tokens=64, seed=20261025, device="cpu", max_total_tokens=4096,
    chat_template_kwargs=None,
) -> list[dict]:
    """Draw one batch per case and score only its generated continuations.

    ``seed`` in each output row is the case's experiment seed. The separate
    ``sampling_seed`` records the base used to derive a case-local Torch seed:
    the first eight bytes (big endian) of SHA256("{seed}|{case_id}").
    """
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive for sampling")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must fall in (0, 1]")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if max_total_tokens < 2:
        raise ValueError("max_total_tokens must be >= 2")
    if not callable(getattr(model, "generate", None)):
        raise ValueError("Model must implement generate() for sampling")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Sampling requires torch: install .[mechanistic]") from exc

    model.eval()
    rows = []
    for case in cases:
        # Keep rendering, tensor handling and attention identical to
        # CausalProbe._prefix, without requiring access to decoder blocks.
        encoded = render_chat(
            tokenizer, case["messages"], chat_template_kwargs=chat_template_kwargs,
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = {"input_ids": encoded}
        if "input_ids" not in encoded:
            raise ValueError("Chat template must yield input_ids")
        ids = encoded["input_ids"].to(device)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Expected single nonempty tokenized conversation")
        n_prefix = int(ids.shape[1])
        if n_prefix >= max_total_tokens:
            raise ValueError(f"Prefix length {n_prefix} exceeds token budget")
        if n_prefix + max_new_tokens > max_total_tokens:
            raise ValueError("Generation exceeds total token budget; shorten prompt or max_new_tokens")
        attention = encoded.get("attention_mask")
        attention = torch.ones_like(ids) if attention is None else attention.to(device)
        generation_kwargs = dict(
            input_ids=ids, attention_mask=attention, do_sample=True,
            temperature=temperature, top_p=top_p, num_return_sequences=n_samples,
            max_new_tokens=max_new_tokens, num_beams=1, use_cache=True,
        )
        padding = getattr(tokenizer, "pad_token_id", None)
        if padding is None:
            padding = getattr(tokenizer, "eos_token_id", None)
        if isinstance(padding, list):
            padding = padding[0] if padding else None
        if padding is not None:
            generation_kwargs["pad_token_id"] = padding
        digest = hashlib.sha256(f"{seed}|{case['case_id']}".encode("utf-8")).digest()
        torch.manual_seed(int.from_bytes(digest[:8], "big"))
        with torch.inference_mode():
            output = model.generate(**generation_kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        if sequences.ndim != 2 or sequences.shape[0] != n_samples:
            raise ValueError("Expected n_samples generated token sequences")
        if sequences.shape[1] < n_prefix or not torch.equal(
                sequences[:, :n_prefix], ids.expand(n_samples, -1)):
            raise ValueError("Generate output must include the unchanged prefix")
        samples = []
        for sequence in sequences:
            response = tokenizer.decode(
                sequence[n_prefix:].detach().cpu().tolist(), skip_special_tokens=True)
            samples.append({"response": response, **score_response(response)})
        rows.append({
            **{key: case[key] for key in ("case_id", "seed", "probe_id", "phase", "condition")},
            **{key: case[key] for key in ("genre", "system_variant") if key in case},
            "sampling_seed": seed,
            "n_samples": n_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "genre_intrusion_rate": sum(s["genre_intrusion"] for s in samples) / n_samples,
            "theatrical_rate": sum(
                s["exclamations"] > 0 or s["uppercase_words"] >= 2 for s in samples
            ) / n_samples,
            "mean_persona_index": sum(s["persona_index"] for s in samples) / n_samples,
            "samples": samples,
            "prefix_tokens": n_prefix,
            **({"chat_template_kwargs": chat_template_kwargs}
               if chat_template_kwargs is not None else {}),
        })
    return rows


def sample_rates_hf(
    *, input_path, output_path, model_name, device="auto", dtype="auto",
    allow_download=False, trust_remote_code=False, chat_template_kwargs=None,
    conditions=None, phases=None, **kwargs: Any,
) -> int:
    """Load a cached/local HF model, optionally filter cases, and write JSONL."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -e '.[mechanistic]'") from exc
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends.mps, "is_available", lambda: False)()
                  else "cpu")
    if device not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unsupported device: {device}")
    if dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    cases = [case for case in _read_cases(input_path)
             if (conditions is None or case["condition"] in conditions)
             and (phases is None or case["phase"] in phases)]
    load_kwargs = {"local_files_only": not allow_download,
                   "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        **load_kwargs,
    ).to(device)
    rows = sample_rates(
        cases=cases, model=model, tokenizer=tokenizer, device=device,
        chat_template_kwargs=chat_template_kwargs, **kwargs,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
