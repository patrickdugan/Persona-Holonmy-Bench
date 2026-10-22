"""PercInterp cross-library contract and held-out layer audit tests."""
import copy
import json

import pytest

from persona_holonomy.perc_bridge import (
    GRAPH_SCHEMA, SELECT_SCHEMA, _read_graph,
    audit_sweep_by_block, select_attribution_graph,
)


def graph_fixture():
    return {
        "schema": GRAPH_SCHEMA,
        "provenance": {"edge_semantics": "direct_attribution", "train_seeds": [1, 2, 3],
                       "estimator": "synthetic_unit_test_only", "model": "toy",
                       "training_probe_ids": ["old_probe"], "tokenizer": "toy"},
        "nodes": [
            {"id": "input", "stage": 0},
            {"id": "a", "stage": 1, "decoder_block": 0},
            {"id": "b", "stage": 1, "decoder_block": 1},
            {"id": "c", "stage": 2, "decoder_block": 2},
            {"id": "d", "stage": 2, "decoder_block": 3},
            {"id": "output", "stage": 3},
        ],
        "edges": [
            {"source": x, "target": y, "signed_weight": w}
            for x, y, w in (
                ("input", "a", .9), ("input", "b", .3),
                ("a", "c", .88), ("a", "d", -.2),
                ("b", "c", .1), ("b", "d", .67),
                ("c", "output", .81), ("d", "output", -.3),
            )
        ],
        "sources": ["input"], "sinks": ["output"],
    }


def manifest_fixture():
    return {
        "schema": SELECT_SCHEMA, "graph_sha256": "fixture-hash",
        "provenance": {"train_seeds": [1, 2, 3], "training_probe_ids": ["old_probe"],
                       "model": "toy", "tokenizer": "toy"},
        "selected_decoder_blocks": {
            "sieve": [0, 2], "local_degree": [1, 2],
            "top_attribution": [0, 3], "random_000": [1, 3],
        },
    }


def rows_fixture():
    data = []
    for layer in range(4):
        for seed in (4, 5, 6):
            for probe in ("new_probe",):
                for control in ("candidate", "orthogonal"):
                    data.append({
                        "model": "toy", "tokenizer": "toy", "reference_condition": "tail_matched",
                        "direction_source": "ordered", "phase": "exit", "mode": "inject", "alpha": 1.,
                        "layer": layer, "seed": seed, "probe_id": probe, "control": control,
                        "training_seeds": [1, 2, 3], "test_seeds": [4, 5, 6],
                        "training_probe_ids": ["old_probe"], "evaluation_probe_ids": ["new_probe"],
                        "recipient_case_id": f"case-{seed}-{probe}",
                        "prefix_sha256": f"digest-{seed}-{probe}",
                        "oriented_effect": (layer + 1) * .1 if control == "candidate" else .02,
                    })
    return data


def test_graph_requires_direct_attribution_and_explicit_layer_mapping():
    g = graph_fixture()
    ids, stages, mapping, direct, provenance = _read_graph(g)
    assert mapping["c"] == 2
    assert direct[3] == ("a", "d", -.2)
    bad = copy.deepcopy(g)
    bad["provenance"]["edge_semantics"] = "cosine_similarity"
    with pytest.raises(ValueError, match="direct_attribution"):
        _read_graph(bad)
    bad = copy.deepcopy(g)
    bad["nodes"][1]["stage"] = True
    with pytest.raises(ValueError, match="stage"):
        _read_graph(bad)
    bad = copy.deepcopy(g)
    bad["nodes"][1]["decoder_block"] = "0"
    with pytest.raises(ValueError, match="decoder_block"):
        _read_graph(bad)


def test_percinterp_actual_selection_contract():
    pytest.importorskip("percinterp")
    report = select_attribution_graph(graph_fixture(), n_shuffles=6, n_random=3, seed=9)
    assert report["schema"] == SELECT_SCHEMA
    assert report["n_nodes"] == 6
    assert report["n_edges"] == 8
    assert report["signed_negative_edges"] == 2
    assert len(report["selected_nodes"]["sieve"]) >= 2
    assert set(report["selected_nodes"]) == {
        "sieve", "local_degree", "top_attribution", "random_000", "random_001", "random_002"
    }
    for group in report["selected_nodes"].values():
        assert {len([n for n in group if n in ("a", "b")]),
                len([n for n in group if n in ("c", "d")])} <= {0, 1, 2}
    assert all(set(report["selected_nodes"][k]) <= {"a", "b", "c", "d"}
               for k in report["selected_nodes"])
    again = select_attribution_graph(graph_fixture(), n_shuffles=6, n_random=3, seed=9)
    assert again["selected_nodes"] == report["selected_nodes"]
    assert again["graph_sha256"] == report["graph_sha256"]
    assert report["sieve"]["spanning_threshold_fallback"] in (True, False)


def test_block_audit_pairs_same_cases_and_clusters_by_seed():
    report = audit_sweep_by_block(manifest_fixture(), rows_fixture())
    assert report["schema"].endswith("layer_audit.v1")
    assert report["n_scored_seeds"] == 3
    assert report["selected_block_budget"] == 2
    assert report["selector_comparisons"]["sieve"]["mean_specificity"] == pytest.approx(.18)
    assert report["selector_comparisons"]["local_degree"]["mean_specificity"] == pytest.approx(.23)
    assert report["selector_comparisons"]["local_degree"]["signflip"]["n_seed_clusters"] == 3
    assert "holm_p_value" in report["selector_comparisons"]["local_degree"]


def test_block_audit_refuses_overlap_unbalanced_controls_and_missing_layers():
    m, r = manifest_fixture(), rows_fixture()
    wrong = copy.deepcopy(m)
    wrong["provenance"]["train_seeds"] = [1, 4]
    with pytest.raises(ValueError, match="Graph training seeds"):
        audit_sweep_by_block(wrong, r)
    with pytest.raises(ValueError, match="Unpaired"):
        audit_sweep_by_block(m, r[:-1])
    with pytest.raises(ValueError, match="missing"):
        audit_sweep_by_block(m, [x for x in r if x["layer"] != 3])
    mismatched = copy.deepcopy(m)
    mismatched["selected_decoder_blocks"]["local_degree"] = [1]
    with pytest.raises(ValueError, match="different decoder-block budget"):
        audit_sweep_by_block(mismatched, r)
    duplicated = r + [r[0]]
    with pytest.raises(ValueError, match="Duplicate"):
        audit_sweep_by_block(m, duplicated)
    wrong_prefix = copy.deepcopy(r)
    wrong_prefix[0]["prefix_sha256"] = "tampered"
    with pytest.raises(ValueError, match="different recipient"):
        audit_sweep_by_block(m, wrong_prefix)


def test_block_audit_rejects_model_and_probe_provenance_mismatch():
    original = manifest_fixture()
    rows = rows_fixture()
    wrong = copy.deepcopy(original)
    wrong["provenance"]["model"] = "other-checkpoint"
    with pytest.raises(ValueError, match="model does not match"):
        audit_sweep_by_block(wrong, rows)
    wrong = copy.deepcopy(original)
    wrong["provenance"]["training_probe_ids"] = ["new_probe"]
    with pytest.raises(ValueError, match="Graph training probes"):
        audit_sweep_by_block(wrong, rows)
    wrong = copy.deepcopy(original)
    wrong["provenance"]["tokenizer"] = "other-tokenizer"
    with pytest.raises(ValueError, match="tokenizer does not match"):
        audit_sweep_by_block(wrong, rows)
    wrong = copy.deepcopy(original)
    del wrong["provenance"]["training_probe_ids"]
    with pytest.raises(ValueError, match="provenance.training_probe_ids"):
        audit_sweep_by_block(wrong, rows)


def test_block_equivalent_selector_is_explicitly_identified():
    manifest = manifest_fixture()
    manifest["selected_decoder_blocks"]["local_degree"] = [2, 0]
    report = audit_sweep_by_block(manifest, rows_fixture())
    comp = report["selector_comparisons"]["local_degree"]
    assert comp["same_decoder_blocks_as_sieve"]
    assert comp["mean_sieve_minus_selector"] == pytest.approx(0)
    assert comp["holm_p_value"] == 1
    assert report["n_distinct_decoder_block_selectors"] == 3
