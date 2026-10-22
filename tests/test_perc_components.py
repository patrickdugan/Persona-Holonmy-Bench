"""SAE latent-level intervention tests using safe, tiny local tensor checkpoints."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from persona_holonomy.perc_components import (
    COMPONENT_SCHEMA, FEATURE_MAP_SCHEMA, _selected_contribution,
    _validate_plan, component_report, plan_component_ablation,
    run_component_ablation, _hash_file,
)
from persona_holonomy.interventions import prepare_pairs
from persona_holonomy.sweep import _partition_pairs
from persona_holonomy.protocol import PROBES, build_cases
from test_interventions import toy_model


def cases_fixture(n=5):
    return [r.to_dict() for r in build_cases(
        repetitions=n, seed=100, variation="seeded_v1",
        conditions=("ordered", "tail_matched"), phases=("exit",),
        probes=(PROBES[0], PROBES[1]),
    )]


def specimen(tmp_path, *, n=5):
    torch = pytest.importorskip("torch")
    model, tokenizer = toy_model()
    cases = cases_fixture(n=n)
    pairs = prepare_pairs(cases, source_conditions=("ordered",),
                          target_condition="tail_matched", phases=("exit",))
    train, test, _, _, train_probes, eval_probes = _partition_pairs(
        pairs, holdout_fraction=.4, split_seed=20261022,
        heldout_probe_ids=("parser_validator",))
    checkpoint_map = {}
    for b in (0, 1):
        path = tmp_path / f"block-{b}.pt"
        d = 12
        state = {
            "W_enc": torch.eye(d), "W_dec": torch.eye(d),
            "b_enc": torch.ones(d) * 20, "b_dec": torch.zeros(d),
            "scale": torch.tensor(1., dtype=torch.float32),
        }
        torch.save({"config": {"d": d, "n_latents": d, "k": d},
                    "state_dict": state}, path)
        checkpoint_map[str(b)] = {"path": path.name, "sha256": _hash_file(path), "top_k": d}
    manifest = {
        "schema": "phb.percinterp.selection.v1", "graph_sha256": "f" * 64,
        "provenance": {"model": "toy", "tokenizer": tokenizer.name_or_path,
                       "train_seeds": list(train), "training_probe_ids": sorted(train_probes)},
        "selected_nodes": {
            "sieve": ["a", "c"], "local_degree": ["b", "c"],
            "top_attribution": ["a", "d"], "random_000": ["b", "d"],
            "duplicate_selector": ["a", "c"],
        },
        "node_to_decoder_block": {"a": 0, "b": 0, "c": 1, "d": 1},
    }
    features = {
        "schema": FEATURE_MAP_SCHEMA, "model": "toy", "tokenizer": tokenizer.name_or_path,
        "selection_graph_sha256": manifest["graph_sha256"],
        "checkpoint_revision": "toy-test-fixed-92", "checkpoints": checkpoint_map,
        "features": {"a": {"block": 0, "latent": 0}, "b": {"block": 0, "latent": 1},
                     "c": {"block": 1, "latent": 0}, "d": {"block": 1, "latent": 1}},
    }
    return model, tokenizer, cases, manifest, features, train, test


def test_topk_feature_contribution_keeps_reconstruction_error(tmp_path):
    torch = pytest.importorskip("torch")
    _, _, _, _, fmap, _, _ = specimen(tmp_path)
    from persona_holonomy.perc_components import _sae_tensors
    sae = _sae_tensors(fmap["checkpoints"]["0"], root=tmp_path, torch=torch, device="cpu")
    x = torch.arange(12, dtype=torch.float32)
    contrib = _selected_contribution(x, (2, 4), sae, torch=torch)
    assert contrib.shape == (12,)
    assert contrib[2] == pytest.approx(22.)
    assert contrib[4] == pytest.approx(24.)
    assert contrib[0] == pytest.approx(0.)
    with pytest.raises(ValueError, match="outside"):
        _selected_contribution(x, (12,), sae, torch=torch)


def test_preflight_and_budget(tmp_path):
    _, tok, cases, manifest, fmap, train, test = specimen(tmp_path)
    plan = plan_component_ablation(manifest=manifest, feature_map=fmap,
        cases=cases, model_name="toy", tokenizer_name=tok.name_or_path,
        map_root=tmp_path)
    assert plan["train_seeds"] == list(train)
    assert plan["test_seeds"] == list(test)
    assert plan["train_probe_ids"] == ["flaky_ci"]
    assert plan["test_probe_ids"] == ["parser_validator"]
    assert plan["n_unique_component_sets"] == 4
    assert plan["n_test_cases"] == len(test)
    assert plan["estimated_model_forwards"] > len(test) * 10
    wrong = copy.deepcopy(manifest)
    wrong["provenance"]["train_seeds"].append(1000)
    with pytest.raises(ValueError, match="training seeds"):
        plan_component_ablation(manifest=wrong, feature_map=fmap,
            cases=cases, model_name="toy", map_root=tmp_path)
    wrong = copy.deepcopy(fmap)
    wrong["checkpoints"]["0"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        plan_component_ablation(manifest=manifest, feature_map=wrong,
            cases=cases, model_name="toy", map_root=tmp_path)


def test_reject_bad_mappings_and_budget(tmp_path):
    _, tok, cases, manifest, fmap, _, _ = specimen(tmp_path)
    wrong = copy.deepcopy(fmap)
    wrong["features"]["b"]["latent"] = 0
    with pytest.raises(ValueError, match="duplicate feature location"):
        _validate_plan(manifest, wrong, model_name="toy")
    wrong = copy.deepcopy(fmap)
    wrong["features"]["c"]["block"] = 0
    wrong["features"]["c"]["latent"] = 9
    with pytest.raises(ValueError, match="different graph decoder_block"):
        _validate_plan(manifest, wrong, model_name="toy")
    wrong = copy.deepcopy(manifest)
    wrong["selected_nodes"]["random_000"] = ["b"]
    with pytest.raises(ValueError, match="unequal per-decoder-block"):
        _validate_plan(wrong, fmap, model_name="toy")
    wrong = copy.deepcopy(fmap)
    wrong["checkpoint_revision"] = ""
    with pytest.raises(ValueError, match="checkpoint revision"):
        _validate_plan(manifest, wrong, model_name="toy")


def test_causal_component_ablation_paired_and_deterministic(tmp_path):
    model, tok, cases, manifest, fmap, train, test = specimen(tmp_path)
    rows = run_component_ablation(manifest=manifest, feature_map=fmap,
        cases=cases, model=model, tokenizer=tok, model_name="toy",
        map_root=tmp_path, max_total_tokens=512)
    assert len(rows) == len(test) * len(manifest["selected_nodes"])
    assert all(r["schema"] == COMPONENT_SCHEMA for r in rows)
    assert all(r["seed"] in test and r["seed"] not in train for r in rows)
    assert all(r["probe_id"] == "parser_validator" for r in rows)
    assert all(r["alpha"] == 1 for r in rows)
    assert any(abs(r["candidate_effect"]) > 1e-5 for r in rows)
    for row in rows:
        assert row["candidate_patch_norms"].keys() == row["orthogonal_patch_norms"].keys()
        for b in row["candidate_patch_norms"]:
            assert row["candidate_patch_norms"][b] == pytest.approx(row["orthogonal_patch_norms"][b], rel=1e-5)
    by_case = {(r["case_id"], r["selector"]): r for r in rows}
    for r in rows:
        if r["selector"] == "sieve":
            twin = by_case[(r["case_id"], "duplicate_selector")]
            assert twin["specificity"] == pytest.approx(r["specificity"], abs=1e-7)
            assert twin["candidate_margin"] == pytest.approx(r["candidate_margin"], abs=1e-7)
    report = component_report(rows)
    assert report["schema"].endswith("component_report.v1")
    assert report["n_seed_clusters"] == len(test)
    assert report["selector_comparisons"]["duplicate_selector"]["identical_to_sieve"]
    assert report["selector_comparisons"]["duplicate_selector"]["mean_sieve_minus_selector"] == pytest.approx(0.)
    assert "holm_p_value" in report["selector_comparisons"]["local_degree"]


def test_report_rejects_incomplete_and_duplicate(tmp_path):
    model, tok, cases, manifest, fmap, _, _ = specimen(tmp_path, n=4)
    rows = run_component_ablation(manifest=manifest, feature_map=fmap,
        cases=cases, model=model, tokenizer=tok, model_name="toy",
        map_root=tmp_path, max_total_tokens=512)
    with pytest.raises(ValueError, match="Incomplete"):
        component_report(rows[:1] + rows[2:])
    with pytest.raises(ValueError, match="Duplicate"):
        component_report(rows + rows[:1])
    changed = copy.deepcopy(rows)
    changed[0]["model"] = "another-model"
    with pytest.raises(ValueError, match="Mixed"):
        component_report(changed)


def test_checkpoint_config_mismatch_fails(tmp_path):
    torch = pytest.importorskip("torch")
    _, _, cases, manifest, fmap, _, _ = specimen(tmp_path)
    wrong = copy.deepcopy(fmap)
    wrong["checkpoints"]["0"]["top_k"] = 3
    from persona_holonomy.perc_components import _sae_tensors
    with pytest.raises(ValueError, match="config.k"):
        _sae_tensors(wrong["checkpoints"]["0"], root=tmp_path, torch=torch, device="cpu")


def test_report_rejects_mismatched_recipients_and_missing_seed(tmp_path):
    model, tok, cases, manifest, fmap, _, _ = specimen(tmp_path)
    rows = run_component_ablation(manifest=manifest, feature_map=fmap,
        cases=cases, model=model, tokenizer=tok, model_name="toy",
        map_root=tmp_path, max_total_tokens=512)
    different_prompt = copy.deepcopy(rows)
    different_prompt[2]["prefix_sha256"] = "wrong-reused-prefix"
    with pytest.raises(ValueError, match="mismatched held-out prefixes"):
        component_report(different_prompt)
    first_seed = rows[0]["seed"]
    truncated = [r for r in rows if r["seed"] != first_seed]
    with pytest.raises(ValueError, match="Incomplete held-out seed coverage"):
        component_report(truncated)
    wrong_norm = copy.deepcopy(rows)
    wrong_norm[0]["orthogonal_patch_norms"]["0"] += 9999
    with pytest.raises(ValueError, match="patch norms differ"):
        component_report(wrong_norm)


def test_percinterp_topk_parity_when_installed(tmp_path):
    """Gate a real PercInterp TopKSAE weight roundtrip in integration CI."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("percinterp")
    from percinterp.sae import TopKSAE
    from persona_holonomy.perc_components import _sae_tensors
    torch.manual_seed(41)
    sae = TopKSAE(d=8, n_latents=16, k=3)
    x = torch.randn(10, 8)
    sae.set_scale(x)
    path = tmp_path / "perc-real.pt"
    torch.save(sae.state(), path)
    loaded = _sae_tensors({"path": path.name, "sha256": _hash_file(path), "top_k": 3},
                          root=tmp_path, torch=torch, device="cpu")
    reference_latents = sae.encode(x[0])
    selected = (0, 1, 2, 3, 4)
    expected = reference_latents[list(selected)] @ sae.W_dec[list(selected)] / sae.scale
    observed = _selected_contribution(x[0], selected, loaded, torch=torch)
    assert torch.allclose(expected, observed, atol=1e-5)


def test_component_preflight_cli_without_loading_model(tmp_path, capsys):
    import json
    from persona_holonomy.cli import main
    _, _, cases, manifest, fmap, _, _ = specimen(tmp_path)
    src = tmp_path / "cases.jsonl"
    src.write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps(manifest), encoding="utf-8")
    fmap_file = tmp_path / "feature-map.json"
    fmap_file.write_text(json.dumps(fmap), encoding="utf-8")
    out = tmp_path / "plan.json"
    main(["perc-components-plan", "--manifest", str(sel),
          "--feature-map", str(fmap_file), "--input", str(src),
          "--model", "toy", "--output", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"].endswith("component_plan.v1")
    assert report["checkpoint_sha256"]["0"] == fmap["checkpoints"]["0"]["sha256"]
    assert "Planned" in capsys.readouterr().out
