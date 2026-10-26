"""Parade's categorical 2x2 frame square reuses existing PHB capture machinery."""
import json
import math
from collections import defaultdict

import pytest

from persona_holonomy.parade import export_frame_cases, plan_pilot


def test_parade_frame_export_is_complete_and_pair_matched(tmp_path):
    root = tmp_path / "pilot"
    plan_pilot(root)
    path = tmp_path / "frame-cases.jsonl"
    summary = export_frame_cases(root / "plan.json", path)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert summary["n_cases"] == 32
    assert summary["n_independent_seeds"] == 4
    assert summary["phase_axis_is_temporal"] is False
    assert len(rows) == 32
    grouped = defaultdict(set)
    for row in rows:
        assert row["model_inference"] is False
        assert row["schema_version"] == 1
        assert row["messages"][0]["role"] == "system"
        assert row["messages"][-1]["role"] == "user"
        assert "untrusted" in row["messages"][-1]["content"]
        grouped[(row["seed"], row["probe_id"])].add((row["condition"], row["phase"]))
    expected = {(a, c) for a in ("neutral", "enacted") for c in ("plain", "figurative")}
    assert len(grouped) == 8
    assert all(cells == expected for cells in grouped.values())
    with pytest.raises(FileExistsError):
        export_frame_cases(root / "plan.json", path)


def test_parade_frame_export_cli(tmp_path):
    from persona_holonomy.cli import main
    root = tmp_path / "pilot"
    main(["parade-plan", "--output-dir", str(root)])
    out = tmp_path / "frames.jsonl"
    main(["parade-frame-export", "--plan", str(root / "plan.json"), "--output", str(out)])
    assert len(out.read_text().splitlines()) == 32


def test_percinterp_can_analyze_cue_square_with_gauge_controls(tmp_path):
    pytest.importorskip("percinterp")
    from persona_holonomy.perc_holonomy import analyze_frames

    root = tmp_path / "pilot"
    plan_pilot(root)
    export = tmp_path / "frames.jsonl"
    export_frame_cases(root / "plan.json", export)
    cases = [json.loads(x) for x in export.read_text().splitlines()]
    rows = []
    for row in cases:
        c = int(row["condition"] == "enacted")
        t = int(row["phase"] == "figurative")
        s = row["seed"] - 30000
        d = int(row["probe_id"] == "data_release")
        theta = .09 * c + .05 * t + .03 * c * t
        phi = .06 * c - .03 * t
        a = math.sin(.83 * s + .41 * d) + .13 * s
        b = math.cos(.56 * s - .21 * d) + .17 * d
        vec = [a * math.cos(theta), b * math.cos(phi),
               a * math.sin(theta), b * math.sin(phi),
               .04 * (a + b)]
        rows.append({
            "schema_version": 1,
            "case_id": row["case_id"],
            "condition": row["condition"],
            "phase": row["phase"],
            "seed": row["seed"],
            "probe_id": row["probe_id"],
            "model": "SYNTHETIC-TOY",
            "tokenizer": "toy",
            "input_ids_sha256": "simulated-only",
            "vectors": {"hs_8": vec},
        })
    report = analyze_frames(rows, layer="hs_8",
                            conditions=("neutral", "enacted"),
                            phases=("plain", "figurative"),
                            phase_edge_kind="cue_adjacent",
                            rank=2, min_overlap=.2, null_shuffles=4)
    assert report["phase_edge_kind"] == "cue_adjacent"
    assert report["n_distinct_seeds"] == 4
    assert report["n_matched_seed_probe_samples"] == 8
    assert len(report["nodes"]) == 4
    assert len(report["edges"]) == 4
    assert all(e["transport_certified"] for e in report["edges"])
    assert sum(e["kind"] == "cue_adjacent" for e in report["edges"]) == 2
    assert report["gauge_invariance_max_error"] < 1e-5
    assert report["history_label_swap_null"]["permutation_unit"] == "seed_cluster"
    assert report["history_label_swap_null"]["n_permutation_units"] == 4


def test_phase_edge_kind_invalid_rejected():
    pytest.importorskip("percinterp")
    from persona_holonomy.perc_holonomy import analyze_frames
    with pytest.raises(ValueError, match="Unknown phase-axis"):
        analyze_frames([{"model": "m", "tokenizer": "t"}], layer="hs_8",
                       phase_edge_kind="invented")
