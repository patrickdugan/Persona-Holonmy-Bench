"""Sampled decoding with fixed token outputs and no checkpoint downloads."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from persona_holonomy.sampling import sample_rates, sample_rates_hf
from persona_holonomy.scoring import score_response
from test_interventions import ToyTokenizer, toy_model


RESPONSES = (
    "A plain answer.",
    "quest unlocked",
    "Great!",
    "LOUD WORDS",
    "ACT IV: BOLD WORDS!",
)


class SamplingTokenizer(ToyTokenizer):
    pad_token_id = None
    eos_token_id = 0

    def __init__(self):
        self.render_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.render_calls.append((messages, kwargs))
        return super().apply_chat_template(messages, **kwargs)

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        # A single fixed generated token maps to each response. Decoding a
        # prefix as well would fail this assertion.
        assert len(ids) == 1
        return RESPONSES[ids[0] - 500]


class SamplingModel:
    def __init__(self, torch):
        self.torch = torch
        self.calls = []
        self.evaluating = False
        self.device = None

    def eval(self):
        self.evaluating = True
        return self

    def to(self, device):
        self.device = device
        return self

    def generate(self, **kwargs):
        assert self.evaluating
        assert self.torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        count = kwargs["num_return_sequences"]
        suffix = self.torch.tensor([[500 + i % len(RESPONSES)] for i in range(count)])
        return self.torch.cat((kwargs["input_ids"].repeat(count, 1), suffix), dim=1)


@pytest.fixture
def sampling_toy():
    torch = pytest.importorskip("torch")
    return SamplingModel(torch), SamplingTokenizer()


def case(case_id="a", condition="ordered", phase="exit"):
    return {
        "case_id": case_id, "seed": 9200, "probe_id": "flaky_ci",
        "condition": condition, "phase": phase,
        # Put persona markers in the prompt to detect accidental prefix scoring.
        "messages": [{"role": "user", "content": "QUEST UNLOCKED! Explain CI."}],
    }


@pytest.mark.parametrize("options", [None, {}, {"enable_thinking": False}])
def test_rates_and_generation_arguments(sampling_toy, options):
    model, tokenizer = sampling_toy
    current = {**case(), "genre": "noir", "system_variant": "plain"}
    rows = sample_rates(
        cases=[current], model=model, tokenizer=tokenizer, n_samples=5,
        temperature=.8, top_p=.9, max_new_tokens=7, seed=123,
        max_total_tokens=128, chat_template_kwargs=options,
    )
    assert len(rows) == 1
    row = rows[0]
    for key in ("case_id", "seed", "probe_id", "phase", "condition", "genre", "system_variant"):
        assert row[key] == current[key]
    assert row["sampling_seed"] == 123
    assert row["n_samples"] == 5
    assert row["temperature"] == .8
    assert row["top_p"] == .9
    assert row["max_new_tokens"] == 7
    assert row["genre_intrusion_rate"] == pytest.approx(.4)
    assert row["theatrical_rate"] == pytest.approx(.6)
    assert row["mean_persona_index"] == pytest.approx(.33)
    assert row["samples"] == [{"response": text, **score_response(text)} for text in RESPONSES]
    assert ("chat_template_kwargs" in row) is (options is not None)
    if options is not None:
        assert row["chat_template_kwargs"] == options
    assert len(model.calls) == 1
    call = model.calls[0]
    assert {key: call[key] for key in (
        "do_sample", "temperature", "top_p", "num_return_sequences",
        "max_new_tokens", "num_beams", "use_cache", "pad_token_id",
    )} == {
        "do_sample": True, "temperature": .8, "top_p": .9,
        "num_return_sequences": 5, "max_new_tokens": 7,
        "num_beams": 1, "use_cache": True, "pad_token_id": 0,
    }
    assert row["prefix_tokens"] == call["input_ids"].shape[1]
    assert tokenizer.render_calls == [(current["messages"], {
        "tokenize": True, "add_generation_prompt": True,
        "return_tensors": "pt", "return_dict": True, **(options or {}),
    })]


def test_seed_is_case_local_and_order_independent(sampling_toy, monkeypatch):
    torch = pytest.importorskip("torch")
    model, tokenizer = sampling_toy
    seeds = []
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    cases = [case("a"), case("b")]
    for ordered in (cases, cases[::-1]):
        sample_rates(cases=ordered, model=model, tokenizer=tokenizer, seed=17)
    a, b, repeated_b, repeated_a = seeds
    assert a == repeated_a
    assert b == repeated_b
    assert a != b
    assert seeds[:2] == [
        int.from_bytes(hashlib.sha256(f"17|{c['case_id']}".encode()).digest()[:8], "big")
        for c in cases
    ]
    assert all(0 <= value < 2**64 for value in seeds)
    sample_rates(cases=[cases[0]], model=model, tokenizer=tokenizer, seed=18)
    assert seeds[-1] != a


@pytest.mark.parametrize("tensor_only", [False, True])
def test_prefix_matches_causal_probe_and_accepts_sequence_wrapper(sampling_toy, tensor_only):
    from persona_holonomy.interventions import CausalProbe

    torch = pytest.importorskip("torch")
    model, tokenizer = sampling_toy
    original_render = tokenizer.apply_chat_template

    def render(messages, **kwargs):
        encoded = original_render(messages, **kwargs)
        if tensor_only:
            return encoded["input_ids"]
        encoded["attention_mask"][0, 0] = 0
        return encoded

    tokenizer.apply_chat_template = render
    original_generate = model.generate
    model.generate = lambda **kwargs: SimpleNamespace(sequences=original_generate(**kwargs))
    probe_model, _ = toy_model()
    options = {"enable_thinking": False}
    probe = CausalProbe(probe_model, tokenizer, chat_template_kwargs=options)
    expected = probe._prefix(case()["messages"])
    row = sample_rates(cases=[case()], model=model, tokenizer=tokenizer,
                       chat_template_kwargs=options)[0]
    for key in ("input_ids", "attention_mask"):
        assert torch.equal(model.calls[0][key], expected[key])
    assert row["prefix_tokens"] == expected["input_ids"].shape[1]
    assert row["n_samples"] == 20
    assert row["temperature"] == row["top_p"] == 1.0
    assert row["max_new_tokens"] == 64
    assert row["sampling_seed"] == 20261025
    assert "genre" not in row and "system_variant" not in row


@pytest.mark.parametrize("kwargs,match", [
    ({"n_samples": 0}, "n_samples"),
    ({"temperature": 0}, "temperature"),
    ({"temperature": float("nan")}, "temperature"),
    ({"top_p": 1.1}, "top_p"),
    ({"top_p": 0}, "top_p"),
    ({"max_new_tokens": 0}, "max_new_tokens"),
    ({"max_total_tokens": 1}, "max_total_tokens"),
    ({"max_total_tokens": 4}, "token budget"),
    ({"max_new_tokens": 128, "max_total_tokens": 128}, "total token budget"),
])
def test_invalid_sampling_settings(sampling_toy, kwargs, match):
    model, tokenizer = sampling_toy
    with pytest.raises(ValueError, match=match):
        sample_rates(cases=[case()], model=model, tokenizer=tokenizer, **kwargs)
    assert not model.calls


@pytest.mark.parametrize("bad_output", ["count", "prefix"])
def test_rejects_malformed_generation(sampling_toy, bad_output):
    model, tokenizer = sampling_toy
    generate = model.generate

    def malformed(**kwargs):
        sequences = generate(**kwargs)
        if bad_output == "count":
            return sequences[:1]
        sequences[0, 0] = 999
        return sequences

    model.generate = malformed
    with pytest.raises(ValueError, match="n_samples" if bad_output == "count" else "unchanged prefix"):
        sample_rates(cases=[case()], model=model, tokenizer=tokenizer)


@pytest.mark.parametrize("conditions,phases,expected", [
    (None, None, ["a", "b", "c", "d"]),
    (("ordered",), None, ["a", "b"]),
    (None, ("peak",), ["a", "c"]),
    (("ordered",), ("exit",), ["b"]),
    ((), None, []),
])
def test_hf_filtering_loading_and_jsonl(
        sampling_toy, monkeypatch, tmp_path, conditions, phases, expected):
    torch = pytest.importorskip("torch")
    model, tokenizer = sampling_toy
    loads = []

    def load_tokenizer(name, **kwargs):
        loads.append(("tokenizer", name, kwargs))
        return tokenizer

    def load_model(name, **kwargs):
        loads.append(("model", name, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    cases = [case("a", "ordered", "peak"), case("b", "ordered", "exit"),
             case("c", "neutral", "peak"), case("d", "neutral", "exit")]
    source = tmp_path / "cases.jsonl"
    source.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    output = tmp_path / "nested" / "rates.jsonl"
    options = {"enable_thinking": False}
    count = sample_rates_hf(
        input_path=source, output_path=output, model_name="toy", conditions=conditions,
        phases=phases, chat_template_kwargs=options, n_samples=5, max_new_tokens=3,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == len(rows) == len(expected)
    assert [row["case_id"] for row in rows] == expected
    assert all(row["chat_template_kwargs"] == options for row in rows)
    assert all(row["n_samples"] == 5 and row["max_new_tokens"] == 3 for row in rows)
    assert len(model.calls) == count
    assert model.device == "cpu"
    assert loads == [
        ("tokenizer", "toy", {"local_files_only": True, "trust_remote_code": False}),
        ("model", "toy", {"dtype": "auto", "local_files_only": True, "trust_remote_code": False}),
    ]


def test_hf_download_opt_in_and_explicit_dtype(sampling_toy, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    model, tokenizer = sampling_toy
    loads = []

    def loader(value):
        def load(name, **kwargs):
            loads.append((name, kwargs))
            return value
        return load

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=loader(tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader(model)),
    ))
    source = tmp_path / "cases.jsonl"
    source.write_text(json.dumps(case()) + "\n", encoding="utf-8")
    assert sample_rates_hf(
        input_path=source, output_path=tmp_path / "rates.jsonl", model_name="toy",
        device="cpu", dtype="float32", allow_download=True, trust_remote_code=True,
        n_samples=1,
    ) == 1
    assert loads == [
        ("toy", {"local_files_only": False, "trust_remote_code": True}),
        ("toy", {"dtype": torch.float32, "local_files_only": False, "trust_remote_code": True}),
    ]


@pytest.mark.parametrize("no_thinking", [False, True])
def test_cli_parses_and_forwards(monkeypatch, no_thinking):
    from persona_holonomy import cli, sampling

    calls = []

    def fake_adapter(**kwargs):
        calls.append(kwargs)
        return 2

    monkeypatch.setattr(sampling, "sample_rates_hf", fake_adapter)
    argv = [
        "sample-rates", "--model", "toy", "--input", "cases.jsonl", "--output", "rates.jsonl",
        "--n-samples", "7", "--temperature", ".8", "--top-p", ".9", "--max-new-tokens", "12",
        "--seed", "17", "--conditions", "ordered,lexical_match", "--phases", "peak,exit",
        "--device", "cpu", "--dtype", "float32", "--max-total-tokens", "128",
        "--allow-download", "--trust-remote-code",
    ]
    if no_thinking:
        argv.append("--no-thinking")
    parsed = cli.make_parser().parse_args(argv)
    assert parsed.func is cli.command_sample_rates
    cli.main(argv)
    assert calls == [{
        "input_path": "cases.jsonl", "output_path": "rates.jsonl", "model_name": "toy",
        "n_samples": 7, "temperature": .8, "top_p": .9, "max_new_tokens": 12, "seed": 17,
        "conditions": ("ordered", "lexical_match"), "phases": ("peak", "exit"),
        "device": "cpu", "dtype": "float32", "max_total_tokens": 128,
        "allow_download": True, "trust_remote_code": True,
        "chat_template_kwargs": {"enable_thinking": False} if no_thinking else None,
    }]


def test_cli_defaults():
    from persona_holonomy.cli import make_parser

    args = make_parser().parse_args(["sample-rates", "--model", "toy", "--input", "cases.jsonl"])
    assert args.n_samples == 20
    assert args.temperature == args.top_p == 1.0
    assert args.max_new_tokens == 64
    assert args.seed == 20261025
    assert args.conditions is args.phases is None
    assert args.device == args.dtype == "auto"
    assert args.max_total_tokens == 4096
    assert not args.allow_download and not args.trust_remote_code and not args.no_thinking
