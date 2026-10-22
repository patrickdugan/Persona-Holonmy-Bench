"""SAE acquisition and provenance checks; local toy model, no downloads."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from persona_holonomy.protocol import PROBES, build_cases
from persona_holonomy.sae_pipeline import (
    DATA_SCHEMA, REGISTRY_SCHEMA, _check_capture, _choose_tokens, _resolve_layers,
    capture_plan, collect_activations, export_feature_map, fit_topk,
)
from persona_holonomy.perc_components import _hash_file, _sae_tensors, _validate_plan
from persona_holonomy.cli import make_parser


def cases():
    return [case.to_dict() for case in build_cases(
        repetitions=8, seed=100, variation="seeded_v1",
        conditions=("ordered", "tail_matched"), phases=("exit",),
        probes=PROBES[:2])]


class ToyTokenizer:
    name_or_path = "toy-tokenizer"

    def apply_chat_template(self, messages, **_kwargs):
        import torch
        data = [1]
        for msg in messages:
            data.append({"system": 2, "user": 3, "assistant": 4}[msg["role"]])
            data += [5 + sum(map(ord, w)) % 32 for w in msg["content"].split()]
        data.append(6)
        return {"input_ids": torch.tensor([data]),
                "attention_mask": torch.ones(1, len(data), dtype=torch.long)}


def toy_model():
    import torch
    torch.manual_seed(33)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return x + self.linear(x)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(), Block(), Block()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(40, 10)
            self.model = Decoder()
            self.head = torch.nn.Linear(10, 40)

        def forward(self, *, input_ids, attention_mask=None,
                    use_cache=False, return_dict=True):
            x = self.embedding(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            return SimpleNamespace(logits=self.head(x))

    return Model().eval()


def test_plan_split_and_reproducibility():
    plan = capture_plan(cases())
    assert plan["schema"] == DATA_SCHEMA
    assert set(plan["phb_train_seeds"]).isdisjoint(plan["phb_test_seeds_excluded"])
    assert set(plan["sae_fit_seeds"]).isdisjoint(plan["sae_validation_seeds"])
    assert set(plan["phb_training_probes"]) == {"flaky_ci"}
    assert set(plan["phb_test_probes_excluded"]) == {"parser_validator"}
    assert {c["probe_id"] for c in plan["selected_cases"]} == {"flaky_ci"}
    assert {c["condition"] for c in plan["selected_cases"]} == {"ordered", "tail_matched"}
    assert plan == capture_plan(cases())
    with pytest.raises(ValueError, match="held-out"):
        capture_plan(cases(), heldout_probes=())
    with pytest.raises(ValueError, match="training probe"):
        capture_plan(cases(), heldout_probes=("flaky_ci", "parser_validator"))
    with pytest.raises(ValueError, match="Duplicate case"):
        capture_plan([*cases(), copy.deepcopy(cases()[0])])
    with pytest.raises(ValueError, match="different system|leakage|histories"):
        capture_plan([dict(c, case_id=c["case_id"] + ".new", messages=cases()[0]["messages"])
                      for c in cases()])


def test_token_subsampling_and_layer_aliases():
    assert _choose_tokens(7, 3, 11)[-1] == 6
    assert _choose_tokens(3, 5, 11) == [0, 1, 2]
    assert _choose_tokens(7, 3, 11) == _choose_tokens(7, 3, 11)
    assert _resolve_layers("0,mid,last", 5) == (0, 2, 4)
    with pytest.raises(ValueError, match="distinct"):
        _resolve_layers("last,2", 3)


def captured(tmp_path):
    torch = pytest.importorskip("torch")
    return collect_activations(cases=cases(), model=toy_model(),
        tokenizer=ToyTokenizer(), model_name="toy", checkpoint_revision="toy-checkpoint-sha",
        layers="0,last", output_dir=tmp_path / "capture", max_input_tokens=1024,
        max_tokens_per_case=9)


def test_torch_capture_and_integrity(tmp_path):
    np = pytest.importorskip("numpy")
    manifest = captured(tmp_path)
    assert set(manifest["blocks"]) == {"0", "2"}
    assert manifest["model"] == "toy"
    assert all(v["hidden_width"] == 10 for v in manifest["blocks"].values())
    assert (tmp_path / "capture" / "capture-manifest.json").exists()
    for block in manifest["blocks"].values():
        with np.load(tmp_path / "capture" / block["path"], allow_pickle=False) as v:
            assert v["train"].shape[1] == v["validation"].shape[1] == 10
            assert 0 < len(v["train"])
            assert 0 < len(v["validation"])
        assert _hash_file(tmp_path / "capture" / block["path"]) == block["sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        captured(tmp_path)
    bad = copy.deepcopy(manifest)
    bad["sae_fit_seeds"] = bad["phb_test_seeds_excluded"]
    with pytest.raises(ValueError, match="disjoint"):
        _check_capture(bad, tmp_path / "capture")
    bad = copy.deepcopy(manifest)
    bad["blocks"]["0"]["sha256"] = "x" * 64
    with pytest.raises(ValueError, match="hash"):
        _check_capture(bad, tmp_path / "capture")


def test_trained_real_percinterp_sae_and_feature_map(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("percinterp.sae")
    captured(tmp_path)
    registry = fit_topk(capture_manifest=str(tmp_path / "capture" / "capture-manifest.json"),
        output_dir=str(tmp_path / "saes"), n_latents=12, k=3,
        steps=8, batch_size=8, seed=123)
    assert registry["schema"] == REGISTRY_SCHEMA
    assert set(registry["blocks"]) == {"0", "2"}
    assert all(v["validation_fvu"] >= 0 and v["validation_feature_coverage"] >= 0
               for v in registry["blocks"].values())
    ckpt = registry["blocks"]["0"]
    loaded = _sae_tensors({"path": ckpt["path"], "sha256": ckpt["sha256"],
                            "top_k": ckpt["top_k"]}, root=tmp_path / "saes",
                          torch=torch, device="cpu")
    assert loaded["W_enc"].shape == (10, 12)
    train_seeds = registry["phb_train_seeds"]
    manifest = {
        "schema": "phb.percinterp.selection.v1",
        "graph_sha256": "0" * 64,
        "provenance": {"model": "toy", "tokenizer": "toy-tokenizer",
                       "train_seeds": train_seeds,
                       "training_probe_ids": registry["phb_training_probes"]},
        "selected_nodes": {
            "sieve": ["sae:0:2", "sae:2:2"],
            "local_degree": ["sae:0:3", "sae:2:1"],
        },
        "node_to_decoder_block": {"sae:0:2": 0, "sae:2:2": 2,
                                  "sae:0:3": 0, "sae:2:1": 2},
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(manifest))
    output = tmp_path / "fmap" / "feature-map.json"
    mapping = export_feature_map(registry_path=str(tmp_path / "saes" / "sae-registry.json"),
                                 selection_path=str(selection), output_path=str(output))
    assert set(mapping["features"]) == set(manifest["node_to_decoder_block"])
    _validate_plan(manifest, mapping, model_name="toy", tokenizer_name="toy-tokenizer")
    assert _sae_tensors(mapping["checkpoints"]["2"],root=output.parent,
                        torch=torch,device="cpu")["W_enc"].shape == (10,12)
    wrong = copy.deepcopy(manifest)
    wrong["provenance"]["train_seeds"] = [999]
    selection.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="provenance"):
        export_feature_map(registry_path=str(tmp_path / "saes" / "sae-registry.json"),
                           selection_path=str(selection), output_path=str(output))


def test_cli_contract():
    parser = make_parser()
    args = parser.parse_args(["sae-capture-plan", "--input", "runs/cases.jsonl"])
    assert args.command == "sae-capture-plan"
    assert args.heldout_probes == "parser_validator"
    args = parser.parse_args(["sae-fit", "--capture", "runs/capture.json", "--output-dir", "runs/saes"])
    assert args.n_latents == 0 and args.steps == 100
    args = parser.parse_args(["sae-capture", "--input", "runs/cases.jsonl",
                              "--model", "cache/model", "--revision", "sha", "--output-dir", "runs/sae"])
    assert args.allow_download is False
