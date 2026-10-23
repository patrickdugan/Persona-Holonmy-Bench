"""Real PyTorch finite-difference edge tests; synthetic decoder and SAEs only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from persona_holonomy.attribution_graph import (
    ESTIMATOR, GRAPH_SCHEMA, PLAN_SCHEMA, _encode, _style_tokens,
    graph_hf, graph_plan, graph_plan_file, measure_graph,
)
from persona_holonomy.perc_components import _hash_file
from persona_holonomy.protocol import PROBES, build_cases
from persona_holonomy.sae_pipeline import capture_plan
from test_interventions import toy_model


def fixture(tmp_path, *, n=8):
    torch = pytest.importorskip("torch")
    model, tokenizer = toy_model()
    cases = [case.to_dict() for case in build_cases(
        repetitions=n, seed=100, variation="seeded_v1",
        conditions=("ordered", "tail_matched"), phases=("exit",),
        probes=PROBES[:2])]
    split = capture_plan(cases)
    blocks = {}
    for b in (0, 1):
        state = {
            "W_enc": torch.eye(12), "W_dec": torch.eye(12),
            "b_enc": torch.ones(12) * 400,
            "b_dec": torch.zeros(12), "scale": torch.tensor(1.),
        }
        output = tmp_path / f"block-{b}.pt"
        torch.save({"config": {"d": 12, "n_latents": 12, "k": 12},
                    "state_dict": state}, output)
        blocks[str(b)] = {"path": output.name, "sha256": _hash_file(output),
            "n_latents": 12, "hidden_width": 12, "top_k": 12,
            "validation_fvu": .1}
    registry = {
        "schema": "phb.sae.registry.v1", "model": "toy",
        "tokenizer": tokenizer.name_or_path,
        "checkpoint_revision": "toy-immutable-commit",
        "phb_train_seeds": split["phb_train_seeds"],
        "phb_test_seeds_excluded": split["phb_test_seeds_excluded"],
        "phb_training_probes": split["phb_training_probes"],
        "phb_test_probes_excluded": split["phb_test_probes_excluded"],
        "blocks": blocks,
    }
    return model, tokenizer, cases, registry


def test_offline_plan_no_model_execution(tmp_path):
    _model, tok, cases, registry = fixture(tmp_path)
    plan = graph_plan(cases=cases, registry=registry,
                      model_name="toy", tokenizer_name=tok.name_or_path,
                      registry_root=tmp_path, top_features=3)
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["layers"] == [0, 1]
    assert len(plan["train_seeds"]) == 5
    assert len(plan["test_seeds_excluded"]) == 3
    assert plan["estimated_model_forwards_upper_bound"] == 5 * (2 + 6)
    assert plan["training_probe_ids"] == ["flaky_ci"]
    assert plan["test_probe_ids_excluded"] == ["parser_validator"]
    assert plan["estimator"] == ESTIMATOR
    assert graph_plan(cases=cases, registry=registry, model_name="toy",
                      registry_root=tmp_path, top_features=3) == plan


def test_plan_rejects_bad_registry_provenance_and_hash(tmp_path):
    _, tok, cases, registry = fixture(tmp_path)
    def plan(r, **kw):
        return graph_plan(cases=cases, registry=r, model_name="toy",
                          registry_root=tmp_path, **kw)
    with pytest.raises(ValueError, match="model identity"):
        graph_plan(cases=cases, registry=registry, model_name="other",
                   registry_root=tmp_path)
    with pytest.raises(ValueError, match="Tokenizer identity"):
        plan(registry, tokenizer_name="wrong")
    wrong = copy.deepcopy(registry)
    wrong["phb_train_seeds"] = wrong["phb_test_seeds_excluded"]
    with pytest.raises(ValueError, match="split"):
        plan(wrong)
    wrong = copy.deepcopy(registry)
    wrong["blocks"]["0"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        plan(wrong)
    with pytest.raises(ValueError, match="two or more"):
        plan(registry, layers="0")
    with pytest.raises(ValueError, match="ascending"):
        plan(registry, layers="1,0")
    with pytest.raises(ValueError, match="training probe|training pairs|histories"):
        plan(registry, heldout_probes=("flaky_ci", "parser_validator"))
    with pytest.raises(ValueError, match="top_features"):
        plan(registry, top_features=0)


def test_encoder_topk_and_token_readout(tmp_path):
    torch = pytest.importorskip("torch")
    model, tok, _, _ = fixture(tmp_path)
    x = torch.arange(4, dtype=torch.float32)
    sae = {"W_enc": torch.eye(4), "W_dec": torch.eye(4),
           "b_enc": torch.zeros(4), "b_dec": torch.zeros(4),
           "scale": torch.tensor(1.), "top_k": 2}
    assert _encode(x, sae, torch).tolist() == [0., 0., 2., 3.]
    a, b = _style_tokens(tok, "flaky_ci")
    assert a != b
    class Broken:
        encode = staticmethod(lambda *_args, **_kw: [1])
    with pytest.raises(ValueError, match="distinct"):
        _style_tokens(Broken(), "flaky_ci")


def test_signed_interventions_export_training_graph(tmp_path):
    model, tok, cases, registry = fixture(tmp_path)
    graph = measure_graph(cases=cases, registry=registry, model=model,
        tokenizer=tok, model_name="toy", registry_root=tmp_path,
        top_features=3, max_total_tokens=512)
    assert graph["schema"] == GRAPH_SCHEMA
    assert graph["provenance"]["edge_semantics"] == "direct_attribution"
    assert graph["provenance"]["estimator"] == ESTIMATOR
    assert graph["provenance"]["train_seeds"] == registry["phb_train_seeds"]
    assert graph["provenance"]["test_seeds_excluded"] == registry["phb_test_seeds_excluded"]
    assert graph["provenance"]["training_probe_ids"] == ["flaky_ci"]
    assert graph["provenance"]["test_probe_ids_excluded"] == ["parser_validator"]
    assert {n["stage"] for n in graph["nodes"]} == {0, 1, 2}
    assert len(graph["sources"]) == 3
    assert len(graph["sinks"]) == 1
    assert len(graph["edges"]) > 0
    assert graph["diagnostics"]["graph_status"] == "spanning"
    assert graph["diagnostics"]["has_measured_source_to_sink_path"]
    assert any(e["target"] == graph["sinks"][0] for e in graph["edges"])
    stage = {n["id"]: n["stage"] for n in graph["nodes"]}
    for edge in graph["edges"]:
        assert stage[edge["source"]] + 1 == stage[edge["target"]]
        assert edge["n_training_pairs"] == len(registry["phb_train_seeds"])
        assert edge["mean_absolute_effect"] >= abs(edge["signed_weight"]) - 1e-7
    assert graph["diagnostics"]["total_model_forwards"] <= (
        len(registry["phb_train_seeds"]) * 8)
    again = measure_graph(cases=cases, registry=registry, model=model,
        tokenizer=tok, model_name="toy", registry_root=tmp_path,
        top_features=3, max_total_tokens=512)
    assert graph == again
    assert all(len(hash_) == 64 for hash_ in graph["diagnostics"]["training_case_id_hashes"])
    # The exporter emits exactly what the earlier PercInterp reader accepts.
    from persona_holonomy.perc_bridge import _read_graph
    ids, stages, mapped, direct, provenance = _read_graph(graph)
    assert len(ids) == len(stages) and len(direct) == len(graph["edges"])
    assert mapped[graph["sources"][0]] == 0


def test_hf_entrypoint_accepts_cli_token_budget(tmp_path, monkeypatch):
    # sae-attribution forwards --max-total-tokens; the preflight must not reject it.
    transformers = pytest.importorskip("transformers")
    model, tok, cases, registry = fixture(tmp_path)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *_a, **_k: tok)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *_a, **_k: model)
    cases_path, registry_path = tmp_path / "cases.jsonl", tmp_path / "sae-registry.json"
    cases_path.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    graph = graph_hf(cases_path=str(cases_path), registry_path=str(registry_path),
        output_path=str(tmp_path / "graph.json"), model_name="toy", device="cpu",
        dtype="float32", top_features=3, max_total_tokens=512)
    assert graph == measure_graph(cases=cases, registry=registry, model=model,
        tokenizer=tok, model_name="toy", registry_root=tmp_path,
        top_features=3, max_total_tokens=512)


def test_offline_plan_cli_contract(tmp_path):
    from persona_holonomy.cli import make_parser
    parser = make_parser()
    args = parser.parse_args(["sae-attribution-plan", "--input", "runs/cases.jsonl",
                              "--registry", "runs/saes/sae-registry.json", "--model", "toy"])
    assert args.command == "sae-attribution-plan"
    assert args.top_features == 8
    args = parser.parse_args(["sae-attribution", "--input", "runs/cases.jsonl",
                              "--registry", "runs/saes/sae-registry.json", "--model", "toy"])
    assert args.allow_download is False and args.top_features == 8


def test_optional_real_percinterp_adapter(tmp_path):
    pytest.importorskip("percinterp")
    from persona_holonomy.perc_bridge import select_attribution_graph
    from percinterp.interop import from_attribution_edges
    from percinterp.measure import rank_entry, profile
    model, tok, cases, registry = fixture(tmp_path)
    graph = measure_graph(cases=cases, registry=registry, model=model,
        tokenizer=tok, model_name="toy", registry_root=tmp_path,
        top_features=3, max_total_tokens=512)
    imported = from_attribution_edges(
        [n["id"] for n in graph["nodes"]],
        [n["stage"] for n in graph["nodes"]],
        [(e["source"], e["target"], e["signed_weight"]) for e in graph["edges"]],
        sources=graph["sources"], sinks=graph["sinks"])
    profile_result = profile(imported.graph,
        rank_entry(imported.graph.weight, imported.graph.edge_layer("dst")))
    assert float(profile_result.p_span) <= 1.0
    frozen = select_attribution_graph(graph, n_shuffles=8, n_random=2)
    assert frozen["schema"] == "phb.percinterp.selection.v1"
    assert frozen["selected_nodes"]["sieve"]
    assert all(n.startswith("sae:1:") for n in frozen["selected_nodes"]["sieve"])
