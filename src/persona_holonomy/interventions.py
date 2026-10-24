"""Causal last-prefix-token interventions for locally loaded causal LMs.

The unit of intervention is a *decoder-block output residual vector* at the
assistant-generation boundary. We change that vector during a teacher-forced
continuation and measure the within-completion log-likelihood change.

Two modes:
* transplant: replace target residual with one from a matched donor history;
* rank1: learn a single mean-difference direction on *training seeds only*,
  then inject it into held-out targets (alongside a norm-matched random null).

This is causal for the hooked tensor and chosen log-likelihood readout; it is
not an identified SAE feature, evidence of model identity, or holonomy.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import statistics
import struct
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .chat import render_chat

# Both options answer the SAME technical question. They differ mainly in style.
# These are pre-registered targets; do not tune after viewing results.
STYLE_COMPLETIONS: dict[str, dict[str, str]] = {
    "flaky_ci": {
        "plain": "CI runs can fail when timing or dependencies change. Compare logs, environments, and pinned versions to find the cause.",
        "theatrical": "ACT IV: THE FLAKY CI BOSS! Timing or dependencies have changed. QUEST UNLOCKED: Compare logs, environments, and pinned versions!",
    },
    "parser_validator": {
        "plain": "A parser turns incoming data into a structured representation. A validator checks whether that structure satisfies the required rules.",
        "theatrical": "ACT IV: THE API GATEKEEPER! The parser assembles incoming data into a structure. QUEST UNLOCKED: Validate that it follows the rules!",
    },
    "stale_cache": {
        "plain": "Compare the cached value's age and version with the underlying source. Refresh it when the two no longer agree.",
        "theatrical": "ACT IV: THE STALE CACHE! Compare the cached value's age and version with its source. QUEST UNLOCKED: Refresh anything out of date!",
    },
    "retry_backoff": {
        "plain": "Exponential backoff increases the delay between retries to ease server load. Jitter spreads clients' retries apart so they do not all retry together.",
        "theatrical": "ACT IV: THE RETRY STORM! Exponential backoff increases retry delays to ease server load. QUEST UNLOCKED: Add jitter so clients do not retry together!",
    },
    "schema_migration": {
        "plain": "Add compatible schema changes first and migrate existing data. Move clients to the new schema before removing old fields they still need.",
        "theatrical": "ACT IV: THE SCHEMA GATEKEEPER! Add compatible changes and migrate existing data. QUEST UNLOCKED: Move clients before removing old fields they need!",
    },
    "memory_leak": {
        "plain": "Track memory usage over time under a steady workload. Compare heap snapshots to find objects that accumulate without being released.",
        "theatrical": "ACT IV: THE MEMORY HOARD! Track memory usage under a steady workload. QUEST UNLOCKED: Compare heap snapshots for objects never released!",
    },
    "rate_limit": {
        "plain": "Honor the server's retry delay after a rate limit response. Reduce request frequency and coordinate clients to stay within the allowed quota.",
        "theatrical": "ACT IV: THE QUOTA GATEKEEPER! Honor the server's retry delay. QUEST UNLOCKED: Reduce request frequency and coordinate clients within the quota!",
    },
    "feature_flag": {
        "plain": "Enable the change for a small group first and monitor errors and performance. Expand gradually if results are healthy, and disable the flag if problems appear.",
        "theatrical": "ACT IV: THE ROLLOUT TRIAL! Enable a small group and monitor errors and performance. QUEST UNLOCKED: Expand gradually, or disable the flag if problems appear!",
    },
    "log_rotation": {
        "plain": "Rotate logs by age or size so one file cannot grow indefinitely. Retain only a bounded set of old files to prevent logs from filling the disk.",
        "theatrical": "ACT IV: THE ENDLESS LOG! Rotate logs by age or size to bound each file. QUEST UNLOCKED: Limit retained old files before they fill the disk!",
    },
}


def prepare_pairs(
    cases: list[dict[str, Any]], *, source_conditions: tuple[str, ...],
    target_condition: str, phases: tuple[str, ...],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Strictly match by (seed, probe, phase); fail on absent or duplicate cells."""
    if not source_conditions or len(source_conditions) != len(set(source_conditions)):
        raise ValueError("Provide unique source conditions")
    if target_condition in source_conditions:
        raise ValueError("Source conditions must differ from target condition")
    indexed: dict[tuple, dict[str, Any]] = {}
    for row in cases:
        for name in ("seed", "probe_id", "phase", "condition", "case_id", "messages"):
            if name not in row:
                raise ValueError(f"Case missing required field {name}")
        key = (row["seed"], row["probe_id"], row["phase"], row["condition"])
        if key in indexed:
            raise ValueError(f"Duplicate case cell: {key}")
        if not row["messages"] or row["messages"][-1]["role"] != "user":
            raise ValueError(f"Case {row['case_id']} must end with user probe")
        indexed[key] = row
    pairs = []
    for (seed, probe, phase, condition), target in sorted(indexed.items()):
        if condition != target_condition or phase not in phases:
            continue
        if probe not in STYLE_COMPLETIONS:
            raise ValueError(f"No registered style continuations for probe: {probe}")
        for source_condition in source_conditions:
            source = indexed.get((seed, probe, phase, source_condition))
            if source is None:
                raise ValueError(f"Missing matched source: {source_condition}/{phase}/{probe}/s{seed}")
            if (source["messages"][0] != target["messages"][0]
                    or source["messages"][-1] != target["messages"][-1]):
                raise ValueError("Matched case has a different system message or probe")
            pairs.append((source, target))
    if not pairs:
        raise ValueError("No matching cases for requested sources, target and phases")
    return pairs


def split_seeds(seeds: Iterable[int], *, holdout_fraction: float = .4,
                split_seed: int = 20261022) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split *seeds*, not probes, to avoid correlated train/test leakage."""
    unique = sorted(set(seeds))
    if not 0 < holdout_fraction < 1:
        raise ValueError("Holdout fraction must be strictly between zero and one")
    if len(unique) < 2:
        raise ValueError("Rank-one estimation requires >=2 distinct seeds")
    randomized = unique.copy()
    random.Random(split_seed).shuffle(randomized)
    n_test = max(1, min(len(unique) - 1, round(len(unique) * holdout_fraction)))
    test = set(randomized[:n_test])
    return tuple(s for s in unique if s not in test), tuple(s for s in unique if s in test)


def decoder_blocks(model: Any) -> tuple[str, Any]:
    """Resolve common HF causal model decoder ModuleLists; never guess silently."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers",
                 "transformer.layers", "model.decoder.layers"):
        current = model
        for attr in path.split("."):
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None and len(current) > 0 and all(
            callable(getattr(layer, "register_forward_hook", None)) for layer in current
        ):
            return path, current
    raise ValueError("Unsupported architecture: decoder blocks not found. "
                     "Expected model.layers, transformer.h or gpt_neox.layers")


def select_block(layer: str, n: int) -> int:
    """Unlike hidden-state indexes in `capture`, blocks are zero-indexed."""
    if layer == "last":
        return n - 1
    if layer == "mid":
        return n // 2
    try:
        i = int(layer)
    except ValueError as exc:
        raise ValueError("--layer must be a non-negative block index or 'last'") from exc
    if not 0 <= i < n:
        raise ValueError(f"Block index {i} outside 0..{n-1}")
    return i


def _unpack_hidden(output: Any):
    main = output[0] if isinstance(output, (tuple, list)) else output
    if not hasattr(main, "ndim") or main.ndim != 3:
        raise ValueError("Expected block activation of shape [batch, tokens, hidden]")
    if main.shape[0] != 1:
        raise ValueError("Causal intervention supports batch size 1")
    return main


def _repack_hidden(output: Any, modified: Any):
    if isinstance(output, tuple):
        return (modified, *output[1:])
    if isinstance(output, list):
        return [modified, *output[1:]]
    return modified


@contextmanager
def _capture_hook(block: Any, state: dict[str, Any]):
    def hook(_module, _inputs, output):
        hidden = _unpack_hidden(output)
        state["vector"] = hidden[0, -1, :].detach().clone()
        state["calls"] += 1
    state["calls"] = 0
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _patch_hook(block: Any, *, position: int, vector: Any,
                alpha: float, mode: str, state: dict[str, Any], reference=None,
                once: bool = False):
    def hook(_module, _inputs, output):
        # Cached generation calls decoder blocks again for individual suffix
        # tokens. Patch only the first (prefix prefill) invocation if requested.
        if once and state["calls"]:
            return output
        hidden = _unpack_hidden(output)
        if position >= hidden.shape[1]:
            raise ValueError("Prefix boundary lies outside hooked block output")
        donor = vector.to(device=hidden.device, dtype=hidden.dtype)
        if donor.shape != hidden.shape[-1:]:
            raise ValueError("Source and target activation dimensions differ")
        modified = hidden.clone()
        if mode == "transplant":
            modified[0, position, :] = (
                (1 - alpha) * hidden[0, position, :] + alpha * donor
            )
        elif mode == "add":
            modified[0, position, :] = hidden[0, position, :] + alpha * donor
        elif mode == "erase":
            if reference is None:
                raise ValueError("Projection erasure requires a reference activation")
            center = reference.to(device=hidden.device, dtype=hidden.dtype)
            if center.shape != donor.shape:
                raise ValueError("Erasure reference dimensions differ from activation")
            norm = donor.float().norm()
            if float(norm) < 1e-10:
                raise ValueError("Cannot erase a zero-length direction")
            unit = donor / norm.to(dtype=donor.dtype)
            displacement = hidden[0, position, :] - center
            removed = (displacement.float() * unit.float()).sum().to(hidden.dtype) * unit
            modified[0, position, :] = hidden[0, position, :] - alpha * removed
        else:
            raise ValueError(f"Unknown intervention mode: {mode}")
        state["calls"] += 1
        return _repack_hidden(output, modified)
    state["calls"] = 0
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class CausalProbe:
    """Teacher-forced readout and causal block-output modification.

    The patch is installed only while scoring, at the *last prefix position*,
    which precedes every scored continuation. Transformer causal masking means
    all later suffix tokens can attend to that patched prefix activation.
    """

    def __init__(self, model: Any, tokenizer: Any, *, layer: str = "last",
                 device: str = "cpu", max_total_tokens: int = 4096,
                 chat_template_kwargs: dict | None = None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Causal interventions require torch: install .[mechanistic]") from exc
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs
        self.device = device
        self.max_total_tokens = max_total_tokens
        self.block_path, blocks = decoder_blocks(model)
        self.layer = select_block(layer, len(blocks))
        self.block = blocks[self.layer]
        if max_total_tokens < 2:
            raise ValueError("max_total_tokens must be >= 2")

    def _prefix(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        encoded = render_chat(
            self.tokenizer, messages, chat_template_kwargs=self.chat_template_kwargs,
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        if isinstance(encoded, self.torch.Tensor):
            encoded = {"input_ids": encoded}
        if "input_ids" not in encoded:
            raise ValueError("Chat template must yield input_ids")
        ids = encoded["input_ids"].to(self.device)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Expected single nonempty tokenized conversation")
        if ids.shape[1] >= self.max_total_tokens:
            raise ValueError(f"Prefix length {ids.shape[1]} exceeds token budget")
        attention = encoded.get("attention_mask")
        if attention is None:
            attention = self.torch.ones_like(ids)
        else:
            attention = attention.to(self.device)
        return {"input_ids": ids, "attention_mask": attention}

    def capture_vector(self, messages: list[dict[str, str]]):
        """Read exact hooked decoder-block output at the assistant boundary."""
        prefix = self._prefix(messages)
        state: dict[str, Any] = {}
        with self.torch.inference_mode(), _capture_hook(self.block, state):
            self.model(**prefix, use_cache=False, return_dict=True)
        if state["calls"] != 1:
            raise RuntimeError(f"Expected one block hook call; got {state['calls']}")
        return state["vector"].float()


    def generate(self, messages: list[dict[str, str]], *, vector=None,
                 mode: str = "add", reference=None, alpha: float = 1.,
                 max_new_tokens: int = 96) -> dict[str, Any]:
        """Greedy, free-generation check with a prefill-only activation patch.

        Generation is a secondary validation readout, distinct from the forced
        continuation margin. The hook must patch only the prefix prefill: with
        KV caching, later forward calls usually contain a single new token.
        """
        if not callable(getattr(self.model, "generate", None)):
            raise ValueError("Model must implement generate() for free generation")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must fall in [0,1]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        prefix = self._prefix(messages)
        n_prefix = int(prefix["input_ids"].shape[1])
        if n_prefix + max_new_tokens > self.max_total_tokens:
            raise ValueError("Generation exceeds total token budget; shorten prompt or max_new_tokens")
        padding = getattr(self.tokenizer, "pad_token_id", None)
        if padding is None:
            padding = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(padding, list):
            padding = padding[0] if padding else None
        kwargs = dict(**prefix, max_new_tokens=max_new_tokens, do_sample=False,
                      num_beams=1, use_cache=True)
        if padding is not None:
            kwargs["pad_token_id"] = padding
        state: dict[str, Any] = {}
        with self.torch.inference_mode():
            if vector is None or alpha == 0:
                output = self.model.generate(**kwargs)
            else:
                with _patch_hook(self.block, position=n_prefix - 1,
                                 vector=vector, reference=reference, alpha=alpha,
                                 mode=mode, state=state, once=True):
                    output = self.model.generate(**kwargs)
                if state["calls"] != 1:
                    raise RuntimeError("Expected one prefix-prefill patch during generation")
        sequences = output.sequences if hasattr(output, "sequences") else output
        if sequences.ndim != 2 or sequences.shape[0] != 1:
            raise ValueError("Expected one generated token sequence")
        if sequences.shape[1] < n_prefix or not self.torch.equal(
                sequences[0, :n_prefix], prefix["input_ids"][0]):
            raise ValueError("Generate output must include the unchanged prefix")
        generated_ids = sequences[0, n_prefix:].detach().cpu().tolist()
        return {
            "response": self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            "generated_tokens": len(generated_ids),
            "prefix_tokens": n_prefix,
            "prefix_sha256": hashlib.sha256(struct.pack(
                f"<{n_prefix}q", *prefix["input_ids"].detach().cpu()[0].tolist(),
            )).hexdigest(),
        }

    def _suffix_ids(self, completion: str):
        token_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not token_ids:
            raise ValueError("Style continuation tokenized to empty sequence")
        return self.torch.tensor([token_ids], device=self.device, dtype=self.torch.long)

    def logprob(self, messages: list[dict[str, str]], completion: str,
                *, vector=None, mode: str = "transplant", alpha: float = 1.0,
                prefix=None, reference=None) -> tuple[float, int, float, float | None]:
        prefix = prefix if prefix is not None else self._prefix(messages)
        ids = prefix["input_ids"]
        suffix = self._suffix_ids(completion)
        prefix_len, suffix_len = int(ids.shape[1]), int(suffix.shape[1])
        if prefix_len + suffix_len > self.max_total_tokens:
            raise ValueError(f"Case requires {prefix_len + suffix_len} tokens; "
                             f"max is {self.max_total_tokens}")
        inputs = {"input_ids": self.torch.cat((ids, suffix), dim=1),
                  "attention_mask": self.torch.cat((
                      prefix["attention_mask"], self.torch.ones_like(suffix)
                  ), dim=1)}
        state: dict[str, Any] = {}
        if vector is not None and not 0 <= alpha <= 1:
            raise ValueError("alpha must fall in [0, 1]")
        with self.torch.inference_mode():
            if vector is None:
                logits = self.model(**inputs, use_cache=False, return_dict=True).logits
            else:
                with _patch_hook(self.block, position=prefix_len - 1,
                                 vector=vector, alpha=alpha, mode=mode,
                                 state=state, reference=reference):
                    logits = self.model(**inputs, use_cache=False, return_dict=True).logits
                if state["calls"] != 1:
                    raise RuntimeError("Hook did not run exactly once during scoring")
            relevant = logits[0, prefix_len - 1:prefix_len + suffix_len - 1]
            log_probs = relevant.float().log_softmax(-1)
            picked = log_probs.gather(-1, suffix[0].unsqueeze(1)).squeeze(1)
            tail = (float(picked[1:].mean().item()) if suffix_len > 1 else None)
            return float(picked.mean().item()), suffix_len, float(picked[0].item()), tail

    def score_continuations(
        self, messages: list[dict[str, str]], completions: list[str], *,
        vector=None, mode: str = "transplant", alpha: float = 1.0,
        reference=None, prefix=None,
    ) -> list[tuple[float, int, float, float | None]]:
        """Score suffixes with one patched prefill and one unpatched batch.

        As in logprob, the hook changes only the final prefix block output.
        Later layers cache that change; suffix tokens never need the hook.
        Models without a cache-aware forward signature use logprob instead.
        """
        if not completions:
            return []
        prefix = prefix if prefix is not None else self._prefix(messages)
        patch = dict(vector=vector, mode=mode, alpha=alpha, reference=reference)
        try:
            parameters = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in parameters.values())
        cache_keywords = ("past_key_values", "use_cache", "position_ids")
        if not all(accepts_kwargs or (
            name in parameters and parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY
        ) for name in cache_keywords):
            return [self.logprob(messages, completion, prefix=prefix, **patch)
                    for completion in completions]

        torch = self.torch
        suffixes = [self._suffix_ids(completion)[0] for completion in completions]
        prefix_len = int(prefix["input_ids"].shape[1])
        lengths = [int(suffix.shape[0]) for suffix in suffixes]
        for length in lengths:
            if prefix_len + length > self.max_total_tokens:
                raise ValueError(f"Case requires {prefix_len + length} tokens; "
                                 f"max is {self.max_total_tokens}")
        if vector is not None and not 0 <= alpha <= 1:
            raise ValueError("alpha must fall in [0, 1]")

        with torch.inference_mode():
            if vector is None:
                output = self.model(**prefix, use_cache=True, return_dict=True)
            else:
                state: dict[str, Any] = {}
                with _patch_hook(self.block, position=prefix_len - 1,
                                 state=state, **patch):
                    output = self.model(**prefix, use_cache=True, return_dict=True)
                if state["calls"] != 1:
                    raise RuntimeError("Hook did not run exactly once during scoring")
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError("Cache-aware model returned no past_key_values")
            first_log_probs = output.logits[0, -1].float().log_softmax(-1)
            del output

            batch_size, width = len(suffixes), max(lengths)
            repeat_cache = getattr(cache, "batch_repeat_interleave", None)
            if callable(repeat_cache):
                # Transformers 5.x DynamicCache mutates in place and returns None.
                repeat_cache(batch_size)
            elif isinstance(cache, (tuple, list)):
                cache = tuple(tuple(value.repeat_interleave(batch_size, dim=0)
                                    for value in layer) for layer in cache)
            else:
                raise TypeError("Unsupported prefix cache: cannot repeat batch dimension")

            # Token zero is only filler; its right-padded positions are masked.
            tokens = torch.zeros((batch_size, width), dtype=torch.long, device=self.device)
            suffix_mask = torch.zeros_like(tokens)
            for i, suffix in enumerate(suffixes):
                tokens[i, :lengths[i]] = suffix
                suffix_mask[i, :lengths[i]] = 1
            attention_mask = torch.cat((
                prefix["attention_mask"].expand(batch_size, -1), suffix_mask,
            ), dim=1)
            position_ids = torch.arange(prefix_len, prefix_len + width,
                                        device=self.device).unsqueeze(0).expand(batch_size, -1)
            # The prefix hook has already been removed, including for batch size > 1.
            output = self.model(input_ids=tokens, attention_mask=attention_mask,
                                position_ids=position_ids, past_key_values=cache,
                                use_cache=True, return_dict=True)
            first = first_log_probs[tokens[:, 0]]
            later = output.logits[:, :-1].float().log_softmax(-1).gather(
                -1, tokens[:, 1:].unsqueeze(-1),
            ).squeeze(-1)
            scores = []
            for i, length in enumerate(lengths):
                picked = torch.cat((first[i:i + 1], later[i, :length - 1]))
                tail = float(picked[1:].mean().item()) if length > 1 else None
                scores.append((float(picked.mean().item()), length,
                               float(picked[0].item()), tail))
            return scores

    def style_margin(self, messages: list[dict[str, str]], *, probe_id: str,
                     vector=None, mode="transplant", alpha=1.0,
                     reference=None) -> dict[str, Any]:
        prefix = self._prefix(messages)
        targets = STYLE_COMPLETIONS[probe_id]
        plain, theatrical = self.score_continuations(
            messages, [targets["plain"], targets["theatrical"]], vector=vector,
            mode=mode, alpha=alpha, prefix=prefix, reference=reference,
        )
        return self._style_margin_result(prefix, plain, theatrical)

    @staticmethod
    def _style_margin_result(prefix, plain, styled) -> dict[str, Any]:
        ordinary, n_plain, ordinary_first, ordinary_tail = plain
        theatrical, n_theatrical, theatrical_first, theatrical_tail = styled
        return {
            "margin": theatrical - ordinary,
            "first_token_margin": theatrical_first - ordinary_first,
            "later_token_margin": (theatrical_tail - ordinary_tail
                                   if theatrical_tail is not None and ordinary_tail is not None
                                   else None),
            "plain_mean_logprob": ordinary,
            "theatrical_mean_logprob": theatrical,
            "plain_tokens": n_plain,
            "theatrical_tokens": n_theatrical,
            "prefix_tokens": int(prefix["input_ids"].shape[1]),
            "prefix_sha256": hashlib.sha256(
                struct.pack(
                    f"<{prefix['input_ids'].numel()}q",
                    *prefix["input_ids"].detach().cpu().reshape(-1).tolist(),
                )
            ).hexdigest(),
        }

    def copyable_margin(self, messages: list[dict[str, str]], *, probe_id: str,
                        genre="gothic", vector=None, mode="transplant", alpha=1.0,
                        reference=None) -> dict[str, Any]:
        """Genre-specific phrase-sharing theatrical-minus-plain margin for Q3."""
        from .targets import copyable_target
        prefix = self._prefix(messages)
        plain, copyable = self.score_continuations(
            messages, [STYLE_COMPLETIONS[probe_id]["plain"], copyable_target(genre, probe_id)],
            vector=vector, mode=mode, alpha=alpha, prefix=prefix, reference=reference)
        return self._copyable_margin_result(plain, copyable)

    @staticmethod
    def _copyable_margin_result(ordinary, styled) -> dict[str, Any]:
        plain, _, plain_first, plain_tail = ordinary
        copyable, _, copyable_first, copyable_tail = styled
        return {
            "copyable_margin": copyable - plain,
            "copyable_first_token_margin": copyable_first - plain_first,
            "copyable_later_token_margin": (copyable_tail - plain_tail
                                            if copyable_tail is not None and plain_tail is not None
                                            else None),
        }

    def copy_free_margin(self, messages: list[dict[str, str]], *, probe_id: str,
                         registers: tuple[str, ...] | None = None, vector=None,
                         mode="transplant", alpha=1.0, reference=None) -> dict[str, Any]:
        """Theatrical-minus-plain margin on targets sharing no phrase with any history.

        Averages the per-register margins (see `targets.COPY_FREE_TARGETS`), so it
        cannot be driven by copying phrases from the conversation.
        """
        from .targets import COPY_FREE_TARGETS, REGISTERS
        prefix = self._prefix(messages)
        registers = registers or REGISTERS
        scores = self.score_continuations(
            messages, [STYLE_COMPLETIONS[probe_id]["plain"],
                       *(COPY_FREE_TARGETS[probe_id][r] for r in registers)],
            vector=vector, mode=mode, alpha=alpha, prefix=prefix, reference=reference)
        return self._copy_free_margin_result(scores[0], zip(registers, scores[1:]))

    @staticmethod
    def _copy_free_margin_result(ordinary, scored_registers) -> dict[str, Any]:
        plain, _, plain_first, plain_tail = ordinary
        by_register = {}
        for register, (mean, n, first, tail) in scored_registers:
            by_register[register] = {
                "margin": mean - plain,
                "first_token_margin": first - plain_first,
                "later_token_margin": (tail - plain_tail
                                       if tail is not None and plain_tail is not None else None),
                "tokens": n,
            }
        later = [v["later_token_margin"] for v in by_register.values()]
        return {
            "copy_free_margin": statistics.mean(v["margin"] for v in by_register.values()),
            "copy_free_first_token_margin": statistics.mean(
                v["first_token_margin"] for v in by_register.values()),
            "copy_free_later_token_margin": (statistics.mean(later)
                                             if None not in later else None),
            "copy_free_by_register": by_register,
        }

    def all_margins(self, messages: list[dict[str, str]], *, probe_id: str,
                    genre: str = "gothic", **patch) -> dict[str, Any]:
        """Score all registered readouts in one shared-prefix batch."""
        from .targets import COPY_FREE_TARGETS, REGISTERS, copyable_target
        prefix = patch.pop("prefix", None)
        prefix = prefix if prefix is not None else self._prefix(messages)
        targets = STYLE_COMPLETIONS[probe_id]
        scores = self.score_continuations(
            messages, [targets["plain"], targets["theatrical"],
                       *(COPY_FREE_TARGETS[probe_id][r] for r in REGISTERS),
                       copyable_target(genre, probe_id)], prefix=prefix, **patch)
        return {
            **self._style_margin_result(prefix, scores[0], scores[1]),
            **self._copy_free_margin_result(scores[0], zip(REGISTERS, scores[2:-1])),
            **self._copyable_margin_result(scores[0], scores[-1]),
        }


def _orthogonal_random(direction: Any, *, seed: int, torch: Any):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    d = direction.detach().float().cpu()
    norm = float(torch.linalg.vector_norm(d))
    if norm == 0:
        raise ValueError("Cannot normalize zero-length activation direction")
    noise = torch.randn(d.shape, generator=generator)
    unit = d / norm
    orth = noise - torch.dot(noise, unit) * unit
    n = float(torch.linalg.vector_norm(orth))
    if n < 1e-9:
        raise ValueError("Random null orthogonalization is numerically degenerate")
    return orth * (norm / n)


def _read_cases(path: str | Path):
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid case JSONL at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {number} must be a JSON object")
        cases.append(row)
    return cases


def run_interventions(
    *, cases: list[dict[str, Any]], model: Any, tokenizer: Any,
    source_conditions: tuple[str, ...] = ("ordered", "neutral"),
    target_condition: str = "tail_matched",
    phases: tuple[str, ...] = ("exit", "washout_2"),
    layer: str = "last", mode: str = "rank1", alpha: float = 1.0,
    holdout_fraction: float = .4, split_seed: int = 20261022,
    max_pairs: int | None = None, max_total_tokens: int = 4096,
    device: str = "cpu", self_check: bool = True,
    model_name: str = "unknown",
    chat_template_kwargs: dict | None = None,
) -> list[dict[str, Any]]:
    """One causal intervention experiment; exactly one new model is loaded outside.

    Rank-one directions are learned only on training seeds and applied only
    to disjoint held-out seeds. In transplant mode every matched pair may run.
    """
    if mode not in ("rank1", "transplant"):
        raise ValueError("mode must be rank1 or transplant")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must fall in [0, 1]")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    pairs = prepare_pairs(cases, source_conditions=source_conditions,
                          target_condition=target_condition, phases=phases)
    model.eval()
    runner = CausalProbe(model, tokenizer, layer=layer, device=device,
                         max_total_tokens=max_total_tokens,
                         chat_template_kwargs=chat_template_kwargs)
    train: tuple[int, ...] = ()
    test: tuple[int, ...] = ()
    if mode == "rank1":
        train, test = split_seeds((target["seed"] for _, target in pairs),
                                  holdout_fraction=holdout_fraction,
                                  split_seed=split_seed)
        # Prevent a misleading "held-out" result on the old static protocol,
        # where changing the seed does not change ordered/tail history text.
        def history_id(case):
            return hashlib.sha256(json.dumps(case["messages"][:-1],
                sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        train_histories = {history_id(case) for source, target in pairs
                           if target["seed"] in train for case in (source, target)}
        test_histories = {history_id(case) for source, target in pairs
                          if target["seed"] in test for case in (source, target)}
        if train_histories & test_histories:
            raise ValueError("Train/test history leakage: identical scripted conversations "
                             "appear in both splits. Generate with --variation seeded_v1.")
    # One activation capture per distinct case. Keep on CPU until use.
    vectors: dict[str, Any] = {}
    def activation(case):
        case_id = case["case_id"]
        if case_id not in vectors:
            vectors[case_id] = runner.capture_vector(case["messages"]).cpu()
        return vectors[case_id]

    directions: dict[str, Any] = {}
    if mode == "rank1":
        for source_condition in source_conditions:
            diffs = [activation(source) - activation(target)
                     for source, target in pairs
                     if target["seed"] in train and source["condition"] == source_condition]
            if not diffs:
                raise ValueError(f"No training samples for {source_condition}")
            direction = runner.torch.stack(diffs).mean(dim=0)
            if float(runner.torch.linalg.vector_norm(direction)) < 1e-10:
                raise ValueError(f"Zero mean direction for {source_condition}; cannot steer")
            directions[source_condition] = direction
    chosen = [(source, target) for source, target in pairs
              if mode == "transplant" or target["seed"] in test]
    if max_pairs is not None:
        chosen = chosen[:max_pairs]
    if not chosen:
        raise ValueError("No evaluation cases available")

    results: list[dict[str, Any]] = []
    cached_baselines: dict[str, dict] = {}
    cached_self: dict[str, float | None] = {}
    for source, target in chosen:
        target_id = target["case_id"]
        if target_id not in cached_baselines:
            cached_baselines[target_id] = runner.style_margin(
                target["messages"], probe_id=target["probe_id"])
        baseline = cached_baselines[target_id]
        if target_id not in cached_self:
            if self_check:
                own = activation(target)
                self_patch = runner.style_margin(
                    target["messages"], probe_id=target["probe_id"],
                    vector=own, mode="transplant", alpha=1.0)
                cached_self[target_id] = self_patch["margin"] - baseline["margin"]
            else:
                cached_self[target_id] = None
        donor = (activation(source) if mode == "transplant"
                 else directions[source["condition"]])
        intervention = ("transplant" if mode == "transplant" else "add")
        patched = runner.style_margin(
            target["messages"], probe_id=target["probe_id"],
            vector=donor, mode=intervention, alpha=alpha)
        effect = patched["margin"] - baseline["margin"]
        pair = {
            **({"chat_template_kwargs": chat_template_kwargs}
               if chat_template_kwargs is not None else {}),
            "schema_version": 1,
            "experiment": mode,
            "model": model_name,
            "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
            "layer": runner.layer,
            "block_path": runner.block_path,
            "source_condition": source["condition"],
            "target_condition": target["condition"],
            "source_case_id": source["case_id"],
            "target_case_id": target_id,
            "seed": target["seed"],
            "probe_id": target["probe_id"],
            "phase": target["phase"],
            "alpha": alpha,
            "n_training_seeds": len(train) if mode == "rank1" else None,
            "training_seeds": list(train) if mode == "rank1" else None,
            "test_seeds": list(test) if mode == "rank1" else None,
            "direction_l2_norm": float(runner.torch.linalg.vector_norm(donor)),
            "baseline_margin": baseline["margin"],
            "patched_margin": patched["margin"],
            "effect": effect,
            "self_patch_delta": cached_self[target_id],
            "plain_tokens": baseline["plain_tokens"],
            "theatrical_tokens": baseline["theatrical_tokens"],
            "prefix_tokens": baseline["prefix_tokens"],
            "prefix_sha256": baseline["prefix_sha256"],
            "negative_control": False,
        }
        results.append(pair)
        if mode == "rank1" and source["condition"] == source_conditions[0]:
            # Norm-matched, orthogonal, deterministic null direction.
            random_direction = _orthogonal_random(
                donor, seed=split_seed + 1000003 * target["seed"]
                + sum(ord(c) for c in target_id), torch=runner.torch)
            null = runner.style_margin(
                target["messages"], probe_id=target["probe_id"],
                vector=random_direction, mode="add", alpha=alpha)
            results.append({**pair, "source_condition": "random_direction",
                            "source_case_id": None, "negative_control": True,
                            "patched_margin": null["margin"],
                            "effect": null["margin"] - baseline["margin"]})
    return results


def intervention_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Seed-clustered descriptive effects, plus matched donor-vs-control tests."""
    if not rows:
        raise ValueError("Empty intervention run")
    from .analysis import _bootstrap_ci
    identities = {(r["model"], r["tokenizer"], r["layer"], r["experiment"],
                   r["target_condition"], r["alpha"]) for r in rows}
    if len(identities) != 1:
        raise ValueError("Report requires one model, layer, intervention and target condition")
    seen: set[tuple] = set()
    groups: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    observed_self: dict[str, float] = {}
    for row in rows:
        key = (row["seed"], row["probe_id"], row["phase"], row["source_condition"])
        if key in seen:
            raise ValueError(f"Duplicate intervention row {key}")
        seen.add(key)
        groups[(row["source_condition"], row["phase"])][row["seed"]].append(row["effect"])
        if row["self_patch_delta"] is not None:
            observed_self[row["target_case_id"]] = row["self_patch_delta"]
    summaries = []
    for (condition, phase), by_seed in sorted(groups.items()):
        values = [statistics.mean(v) for _, v in sorted(by_seed.items())]
        summaries.append({
            "source_condition": condition,
            "phase": phase,
            "n_seeds": len(values),
            "n_pairs": sum(map(len, by_seed.values())),
            "mean_effect": statistics.mean(values),
            "seed_bootstrap_95pct_ci": _bootstrap_ci(values),
        })
    primary = next((r["source_condition"] for r in rows
                    if not r["negative_control"]), None)
    indexed = {(r["phase"], r["seed"], r["probe_id"], r["source_condition"]):r["effect"]
               for r in rows}
    contrasts = []
    for secondary in sorted({r["source_condition"] for r in rows} - {primary}):
        for phase in sorted({r["phase"] for r in rows}):
            by_seed: dict[int, list[float]] = defaultdict(list)
            for (p, seed, probe, condition), effect in indexed.items():
                if p == phase and condition == primary:
                    other = indexed.get((p, seed, probe, secondary))
                    if other is not None:
                        by_seed[seed].append(effect - other)
            if by_seed:
                values = [statistics.mean(v) for _, v in sorted(by_seed.items())]
                contrasts.append({
                    "phase": phase,
                    "contrast": f"{primary} minus {secondary}",
                    "n_seeds": len(values),
                    "n_paired_probes": sum(len(v) for v in by_seed.values()),
                    "mean_effect_difference": statistics.mean(values),
                    "seed_bootstrap_95pct_ci": _bootstrap_ci(values),
                })
    model, tokenizer, layer, mode, target, alpha = next(iter(identities))
    maximum_self = max(map(abs, observed_self.values()), default=None)
    return {
        "schema_version": 1,
        "model": model, "tokenizer": tokenizer, "layer": layer,
        "experiment": mode, "target_condition": target, "alpha": alpha,
        "n_records": len(rows), "n_targets": len(observed_self) if observed_self else len({r["target_case_id"] for r in rows}),
        "max_abs_self_patch_delta": maximum_self,
        "self_patch_check_passed_1e_3": (maximum_self <= 1e-3 if maximum_self is not None else None),
        "groups": summaries, "contrasts": contrasts,
        "limitations": [
            "This causally perturbs one block output at the final prompt token, not an identified SAE feature.",
            "Style margins compare mean token log probabilities of two pre-registered continuations; length and wording remain confounders.",
            "Rank-one candidates are mean residual differences; train/test splits separate seeds but not model pretraining.",
            "Multiple donor patches do not isolate one latent mechanism without targeted ablations and controls.",
            "Last-prefix-token intervention is not a persistent hidden state, path transport, or geometric holonomy.",
            "Bootstrap intervals resample seeds; fewer than five test seeds produce no interval.",
        ],
    }


def intervention_markdown(report: dict[str, Any]) -> str:
    lines = ["# Persona Holonomy Bench: intervention report", "",
             f"Model: `{report['model']}`; block: `{report['layer']}`; mode: `{report['experiment']}`",
             "", f"Records: {report['n_records']}; evaluated targets: {report['n_targets']}",
             "", f"Max absolute self-patch logit-margin error: {report['max_abs_self_patch_delta']}",
             "", "## Within-target causal effects", "",
             "| Source | Phase | Pairs | Seeds | Mean Δ log-likelihood margin | 95% seed bootstrap |",
             "|---|---|---:|---:|---:|---|"]
    for row in report["groups"]:
        lines.append(f"| {row['source_condition']} | {row['phase']} | {row['n_pairs']} | "
                     f"{row['n_seeds']} | {row['mean_effect']:+.5f} | "
                     f"{row['seed_bootstrap_95pct_ci'] or 'insufficient seeds'} |")
    lines.extend(["", "## Paired donor contrasts", "",
                  "| Phase | Contrast | Paired probes | Seeds | Difference | 95% seed bootstrap |",
                  "|---|---|---:|---:|---:|---|"])
    for row in report["contrasts"]:
        lines.append(f"| {row['phase']} | {row['contrast']} | {row['n_paired_probes']} | "
                     f"{row['n_seeds']} | {row['mean_effect_difference']:+.5f} | "
                     f"{row['seed_bootstrap_95pct_ci'] or 'insufficient seeds'} |")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {x}" for x in report["limitations"])
    return "\n".join(lines) + "\n"


def intervene_hf(
    *, input_path: str, output_path: str, model_name: str,
    device: str = "auto", dtype: str = "auto", allow_download: bool = False,
    trust_remote_code: bool = False, chat_template_kwargs: dict | None = None,
    **kwargs: Any,
) -> int:
    """Load a cached/local Hugging Face model, run causal tests, save JSONL."""
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
    load_kwargs = {"local_files_only": not allow_download,
                   "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        **load_kwargs,
    ).to(device)
    results = run_interventions(cases=_read_cases(input_path), model=model,
                                tokenizer=tokenizer, device=device,
                                model_name=model_name,
                                chat_template_kwargs=chat_template_kwargs, **kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in results:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(results)
