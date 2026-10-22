"""Causal test double with real Torch hooks; no model downloads."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from persona_holonomy.interventions import (
    STYLE_COMPLETIONS, _orthogonal_random, decoder_blocks,
    intervention_markdown, intervention_report, prepare_pairs,
    run_interventions, select_block, split_seeds,
)
from persona_holonomy.protocol import PROBES, build_cases


class ToyTokenizer:
    name_or_path = "tiny-local-fixture-tokenizer"

    @staticmethod
    def encode(text, add_special_tokens=False):
        # Deterministic, tiny word-based vocabulary (bounded at 256).
        # We intentionally keep ACT and QUEST distinct style markers.
        return [10 + (sum((i + 1) * ord(c) for i, c in enumerate(word)) % 200)
                for word in text.split()]

    def apply_chat_template(self, messages, **kwargs):
        torch = pytest.importorskip("torch")
        ids = [1]
        for row in messages:
            ids.append({"system": 2, "user": 3, "assistant": 4}[row["role"]])
            ids.extend(self.encode(row["content"]))
        ids.append(5)
        return {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


def toy_model():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(92)

    class CausalBlock(nn.Module):
        def forward(self, x):
            # Causal exponential state with order dependence, no future leakage.
            previous = torch.zeros_like(x[:, 0, :])
            outputs = []
            for i in range(x.shape[1]):
                previous = .99 * previous + x[:, i, :]
                outputs.append(previous)
            return torch.stack(outputs, 1)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([CausalBlock(), CausalBlock()])

    class ToyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(256, 12)
            self.model = Decoder()
            self.lm_head = nn.Linear(12, 256, bias=False)

        def forward(self, *, input_ids, attention_mask=None,
                    use_cache=False, return_dict=True):
            x = self.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            return SimpleNamespace(logits=self.lm_head(x))

    return ToyLM().eval(), ToyTokenizer()


def samples(reps=1, phases=("exit",), variation="static"):
    return [case.to_dict() for case in build_cases(
        repetitions=reps, seed=9,
        conditions=("ordered", "tail_matched", "neutral"),
        phases=phases, probes=(PROBES[0],), variation=variation,
    )]


def test_pairs_strict_and_split():
    cases = samples()
    pairs = prepare_pairs(cases, source_conditions=("ordered", "neutral"),
                          target_condition="tail_matched", phases=("exit",))
    assert len(pairs) == 2
    assert {s["condition"] for s, _ in pairs} == {"ordered", "neutral"}
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_pairs(cases + [cases[0]], source_conditions=("ordered",),
                      target_condition="tail_matched", phases=("exit",))
    with pytest.raises(ValueError, match="Missing matched"):
        prepare_pairs(cases[1:2], source_conditions=("ordered",),
                      target_condition="tail_matched", phases=("exit",))
    with pytest.raises(ValueError, match=">=2"):
        split_seeds([1])
    train, test = split_seeds([1, 2, 3, 4, 5], split_seed=17)
    assert len(test) == 2 and len(train) == 3
    assert set(test).isdisjoint(train)
    assert set(train) | set(test) == {1, 2, 3, 4, 5}


def test_block_resolution_and_bounds():
    model, _ = toy_model()
    path, blocks = decoder_blocks(model)
    assert path == "model.layers" and len(blocks) == 2
    assert select_block("last", 2) == 1
    assert select_block("0", 2) == 0
    with pytest.raises(ValueError, match="outside"):
        select_block("2", 2)
    with pytest.raises(ValueError, match="Unsupported architecture"):
        decoder_blocks(object())


def test_transplant_alpha_zero_and_self_patch():
    model, tok = toy_model()
    rows = run_interventions(
        cases=samples(), model=model, tokenizer=tok,
        mode="transplant", source_conditions=("ordered",),
        phases=("exit",), max_total_tokens=256, model_name="toy",
        alpha=0,
    )
    assert len(rows) == 1
    assert abs(rows[0]["effect"]) < 1e-6
    assert abs(rows[0]["self_patch_delta"]) < 1e-6
    # The intervention must have a causal effect when turned on.
    changed = run_interventions(
        cases=samples(), model=model, tokenizer=tok,
        mode="transplant", source_conditions=("ordered",),
        phases=("exit",), max_total_tokens=256, model_name="toy",
    )
    assert abs(changed[0]["effect"]) > 1e-7
    assert abs(changed[0]["self_patch_delta"]) < 1e-6
    assert changed[0]["block_path"] == "model.layers"
    assert changed[0]["prefix_sha256"] == rows[0]["prefix_sha256"]
    assert changed[0]["plain_tokens"] > 0
    report = intervention_report(changed)
    assert report["self_patch_check_passed_1e_3"]
    assert "causal" in intervention_markdown(report).lower()


def test_rank1_is_seed_heldout_and_has_null_control():
    model, tok = toy_model()
    rows = run_interventions(
        cases=samples(reps=4, variation="seeded_v1"), model=model, tokenizer=tok,
        mode="rank1", source_conditions=("ordered", "neutral"),
        phases=("exit",), max_total_tokens=256, model_name="toy",
    )
    train, test = split_seeds(range(9, 13))
    assert len(rows) == len(test) * 3  # ordered, neutral and randomized null
    assert all(row["seed"] in test for row in rows)
    assert all(row["seed"] not in train for row in rows)
    assert {row["source_condition"] for row in rows} == {
        "ordered", "neutral", "random_direction"}
    assert len({row["target_case_id"] for row in rows}) == len(test)
    assert all(abs(row["self_patch_delta"]) < 1e-6 for row in rows)
    null = [row for row in rows if row["negative_control"]]
    assert len(null) == len(test)
    orig = [row for row in rows if row["source_condition"] == "ordered"]
    for r, n in zip(orig, null):
        assert n["direction_l2_norm"] == pytest.approx(r["direction_l2_norm"])
        assert n["baseline_margin"] == r["baseline_margin"]
    report = intervention_report(rows)
    assert len(report["groups"]) == 3
    assert len(report["contrasts"]) == 2
    assert report["self_patch_check_passed_1e_3"]


def test_orthogonal_null_and_tuple_preserving_hook():
    torch = pytest.importorskip("torch")
    from persona_holonomy.interventions import _patch_hook
    direction = torch.arange(1, 13, dtype=torch.float32)
    null = _orthogonal_random(direction, seed=9, torch=torch)
    # Compare scalar values: recent PyTorch tensors do not consistently
    # implement pytest.approx's numeric coercion across versions.
    cosine = float(torch.dot(direction, null)) / (
        float(torch.linalg.vector_norm(direction)) *
        float(torch.linalg.vector_norm(null))
    )
    assert cosine == pytest.approx(0.0, abs=1e-6)
    assert float(torch.linalg.vector_norm(direction)) == pytest.approx(
        float(torch.linalg.vector_norm(null)))

    class TupleBlock(torch.nn.Module):
        def forward(self, x):
            return x * 2, "cache"
    block = TupleBlock()
    state = {}
    x = torch.zeros((1, 3, 12))
    with _patch_hook(block, position=2, vector=direction, alpha=1,
                     mode="transplant", state=state):
        output = block(x)
    assert output[1] == "cache"
    assert torch.allclose(output[0][0, -1], direction)
    assert state["calls"] == 1
    assert torch.equal(block(x)[0], x)


def test_input_validation_without_loading_model():
    cases = samples()
    wrong_probe = copy.deepcopy(cases)
    for row in wrong_probe:
        if row["condition"] == "ordered":
            row["messages"][-1]["content"] = "unexpected different probe"
            break
    with pytest.raises(ValueError, match="different system message or probe"):
        prepare_pairs(wrong_probe, source_conditions=("ordered",),
                      target_condition="tail_matched", phases=("exit",))
    model, tok = toy_model()
    with pytest.raises(ValueError, match="requires >=2"):
        run_interventions(cases=cases, model=model, tokenizer=tok, mode="rank1")
    with pytest.raises(ValueError, match="history leakage"):
        run_interventions(cases=samples(reps=3), model=model, tokenizer=tok, mode="rank1")
    with pytest.raises(ValueError, match="alpha"):
        run_interventions(cases=cases, model=model, tokenizer=tok,
                          mode="transplant", alpha=1.5)
    with pytest.raises(ValueError, match="token budget"):
        run_interventions(cases=cases, model=model, tokenizer=tok,
                          mode="transplant", max_total_tokens=12)


def test_report_detects_duplicate_and_mixed():
    base = dict(schema_version=1, experiment="transplant", model="toy",
                tokenizer="tok", layer=1, target_condition="tail_matched",
                alpha=1., source_condition="ordered", seed=1, probe_id="flaky_ci",
                phase="exit", effect=.2, self_patch_delta=0.,
                target_case_id="t1", negative_control=False)
    report = intervention_report([base])
    assert report["n_records"] == 1
    assert report["groups"][0]["seed_bootstrap_95pct_ci"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        intervention_report([base, base])
    with pytest.raises(ValueError, match="one model"):
        intervention_report([base, {**base, "model":"other", "seed": 2}])
