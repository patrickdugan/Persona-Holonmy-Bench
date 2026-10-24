"""CPU float32 equivalence checks for prefix-cached scoring; no downloads."""
from __future__ import annotations

import hashlib
import statistics
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from persona_holonomy.interventions import CausalProbe, STYLE_COMPLETIONS
from persona_holonomy.targets import COPY_FREE_TARGETS, REGISTERS, copyable_target
from test_interventions import ToyTokenizer, toy_model


class ByteTokenizer(ToyTokenizer):
    @staticmethod
    def encode(text, add_special_tokens=False):
        return list(text.encode("utf-8"))


MESSAGES = [{"role": "user", "content": "How can we check a stale cache?"}]
COMPLETIONS = ["x", "two", "a considerably longer suffix", "café"]


@pytest.fixture(scope="module")
def probe():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(713)
    config = transformers.Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=4096, attention_dropout=0.0,
        attn_implementation="eager",
    )
    model = transformers.Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
    return CausalProbe(model, ByteTokenizer(), layer="1", max_total_tokens=4096)


def patch_options(probe, mode):
    if mode is None:
        return {}
    torch = probe.torch
    generator = torch.Generator(device="cpu").manual_seed(315)
    options = dict(vector=torch.randn(64, generator=generator) * .1, mode=mode, alpha=.65)
    if mode == "erase":
        options["reference"] = torch.randn(64, generator=generator) * .02
    return options


def assert_scores_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got[1] == want[1]
        for index in (0, 2, 3):
            if want[index] is None:
                assert got[index] is None
            else:
                assert got[index] == pytest.approx(want[index], abs=1e-5, rel=0)


def assert_result_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_result_equal(actual[key], value)
        elif isinstance(value, float):
            assert actual[key] == pytest.approx(value, abs=1e-5, rel=0)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("mode", [None, "add", "transplant", "erase"])
def test_cached_scores_match_full_forwards(probe, mode):
    options = patch_options(probe, mode)
    expected = [probe.logprob(MESSAGES, text, **options) for text in COMPLETIONS]
    # A middle-block patch affects later layers' prefix K/V and suffix logits.
    with patch.object(probe.model, "forward", wraps=probe.model.forward) as forward:
        actual = probe.score_continuations(MESSAGES, COMPLETIONS, **options)
    assert_scores_equal(actual, expected)
    assert actual[0][1] == 1 and actual[0][3] is None
    assert forward.call_count == 2
    prefix_call, suffix_call = forward.call_args_list
    prefix_len = prefix_call.kwargs["input_ids"].shape[1]
    assert prefix_call.kwargs["use_cache"] is True
    tokens = suffix_call.kwargs["input_ids"]
    assert tokens.shape == (len(COMPLETIONS), max(s[1] for s in expected))
    mask = suffix_call.kwargs["attention_mask"]
    assert bool(mask[:, :prefix_len].eq(1).all())
    for i, (_, length, _, _) in enumerate(expected):
        assert bool(mask[i, prefix_len:prefix_len + length].eq(1).all())
        assert bool(mask[i, prefix_len + length:].eq(0).all())
    positions = suffix_call.kwargs["position_ids"]
    assert positions.tolist() == [list(range(prefix_len, prefix_len + tokens.shape[1]))] * len(COMPLETIONS)


def test_only_single_token_completions(probe):
    texts = ["a", "b"]
    assert_scores_equal(probe.score_continuations(MESSAGES, texts),
                        [probe.logprob(MESSAGES, text) for text in texts])


def per_completion_margins(probe, *, genre, options):
    probe_id = "stale_cache"
    plain = probe.logprob(MESSAGES, STYLE_COMPLETIONS[probe_id]["plain"], **options)
    styled = probe.logprob(MESSAGES, STYLE_COMPLETIONS[probe_id]["theatrical"], **options)
    copyable = probe.logprob(MESSAGES, copyable_target(genre, probe_id), **options)

    def differences(score):
        return dict(margin=score[0] - plain[0], first_token_margin=score[2] - plain[2],
                    later_token_margin=score[3] - plain[3]
                    if score[3] is not None and plain[3] is not None else None)

    prefix_ids = probe._prefix(MESSAGES)["input_ids"][0].tolist()
    style = {
        **differences(styled), "plain_mean_logprob": plain[0],
        "theatrical_mean_logprob": styled[0], "plain_tokens": plain[1],
        "theatrical_tokens": styled[1], "prefix_tokens": len(prefix_ids),
        "prefix_sha256": hashlib.sha256(struct.pack(f"<{len(prefix_ids)}q", *prefix_ids)).hexdigest(),
    }
    by_register = {}
    for register in REGISTERS:
        score = probe.logprob(MESSAGES, COPY_FREE_TARGETS[probe_id][register], **options)
        by_register[register] = {**differences(score), "tokens": score[1]}
    copy_free = {
        **{f"copy_free_{key}": statistics.mean(row[key] for row in by_register.values())
           for key in ("margin", "first_token_margin", "later_token_margin")},
        "copy_free_by_register": by_register,
    }
    return style, copy_free, {f"copyable_{key}": value
                              for key, value in differences(copyable).items()}


@pytest.mark.parametrize("mode,genre", [(None, "gothic"), ("transplant", "pirate")])
def test_margin_helpers_and_union(probe, mode, genre):
    options = patch_options(probe, mode)
    expected = per_completion_margins(probe, genre=genre, options=options)
    with patch.object(probe, "score_continuations", wraps=probe.score_continuations) as score:
        style = probe.style_margin(MESSAGES, probe_id="stale_cache", **options)
        copy_free = probe.copy_free_margin(MESSAGES, probe_id="stale_cache", **options)
        copyable = probe.copyable_margin(MESSAGES, probe_id="stale_cache", genre=genre, **options)
        union = probe.all_margins(MESSAGES, probe_id="stale_cache", genre=genre, **options)
    assert score.call_count == 4
    assert [len(call.args[1]) for call in score.call_args_list] == [2, 5, 2, 7]
    for actual, want in zip((style, copy_free, copyable), expected):
        assert_result_equal(actual, want)
    assert_result_equal(union, {**style, **copy_free, **copyable})
    subset = probe.copy_free_margin(MESSAGES, probe_id="stale_cache",
                                    registers=(REGISTERS[0],), **options)
    assert subset["copy_free_margin"] == pytest.approx(
        copy_free["copy_free_by_register"][REGISTERS[0]]["margin"], abs=1e-5, rel=0)


@pytest.mark.parametrize("patched", [False, True])
def test_existing_toy_model_falls_back(patched):
    model, tokenizer = toy_model()
    runner = CausalProbe(model, tokenizer, layer="0")
    options = dict(vector=runner.torch.ones(12), mode="add") if patched else {}
    expected = [runner.logprob(MESSAGES, text, **options) for text in COMPLETIONS]
    with patch.object(runner, "logprob", wraps=runner.logprob) as logprob:
        actual = runner.score_continuations(MESSAGES, COMPLETIONS, **options)
    assert logprob.call_count == len(COMPLETIONS)
    assert_scores_equal(actual, expected)


def test_legacy_tuple_cache(probe):
    # Bridge Qwen's modern cache to the legacy representation at the API boundary.
    from transformers.cache_utils import DynamicCache

    forward = probe.model.forward

    def legacy_forward(*, input_ids, attention_mask=None, past_key_values=None,
                       position_ids=None, use_cache=False, return_dict=True):
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple)
            assert all(k.shape[0] == len(COMPLETIONS) for k, _ in past_key_values)
            cache = DynamicCache()
            for index, (keys, values) in enumerate(past_key_values):
                cache.update(keys, values, index)
            past_key_values = cache
        output = forward(input_ids=input_ids, attention_mask=attention_mask,
                         past_key_values=past_key_values, position_ids=position_ids,
                         use_cache=use_cache, return_dict=return_dict)
        cache = output.past_key_values
        legacy = tuple((layer.keys, layer.values) for layer in cache.layers)
        return SimpleNamespace(logits=output.logits, past_key_values=legacy)

    expected = [probe.logprob(MESSAGES, text) for text in COMPLETIONS]
    with patch.object(probe.model, "forward", new=legacy_forward):
        assert_scores_equal(probe.score_continuations(MESSAGES, COMPLETIONS), expected)


def test_forward_errors_are_not_hidden_by_fallback(probe):
    def broken_forward(*, input_ids, attention_mask=None, past_key_values=None,
                       position_ids=None, use_cache=False, return_dict=True):
        raise TypeError("internal model error")

    with patch.object(probe.model, "forward", new=broken_forward):
        with patch.object(probe, "logprob") as fallback:
            with pytest.raises(TypeError, match="internal model error"):
                probe.score_continuations(MESSAGES, COMPLETIONS)
        fallback.assert_not_called()


def test_budget_and_input_validation(probe):
    prefix = probe._prefix(MESSAGES)
    prefix_len = int(prefix["input_ids"].shape[1])
    limited = CausalProbe(probe.model, probe.tokenizer, layer="1", max_total_tokens=prefix_len + 3)
    # Equality is allowed; both supplied-prefix and rendered-prefix paths enforce the limit.
    assert_scores_equal(limited.score_continuations(MESSAGES, ["abc"]),
                        [limited.logprob(MESSAGES, "abc")])
    for supplied in (None, prefix):
        with patch.object(probe.model, "forward", wraps=probe.model.forward) as forward:
            with pytest.raises(ValueError, match=f"Case requires {prefix_len + 4} tokens"):
                limited.score_continuations(MESSAGES, ["a", "abcd"], prefix=supplied)
        forward.assert_not_called()
    too_short = CausalProbe(probe.model, probe.tokenizer, max_total_tokens=prefix_len)
    with pytest.raises(ValueError, match="token budget"):
        too_short.score_continuations(MESSAGES, ["a"])
    with pytest.raises(ValueError, match="empty sequence"):
        probe.score_continuations(MESSAGES, [""])
    with pytest.raises(ValueError, match="alpha"):
        probe.score_continuations(MESSAGES, ["a"], vector=probe.torch.ones(64), alpha=1.1)
    assert probe.score_continuations(MESSAGES, []) == []


def test_supplied_prefix_is_reusable(probe):
    prefix = probe._prefix(MESSAGES)
    original = {name: tensor.clone() for name, tensor in prefix.items()}
    with patch.object(probe, "_prefix", side_effect=AssertionError("prefix re-rendered")):
        first = probe.score_continuations(MESSAGES, COMPLETIONS, prefix=prefix)
        second = probe.score_continuations(MESSAGES, COMPLETIONS, prefix=prefix)
    assert_scores_equal(first, second)
    assert all(probe.torch.equal(prefix[name], tensor) for name, tensor in original.items())
