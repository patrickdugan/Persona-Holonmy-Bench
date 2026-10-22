"""The PCA-frame bridge is intentionally separate from direct attribution sieving."""
import copy
import math

import pytest

from persona_holonomy.perc_holonomy import analyze_frames


def synthetic_rows():
    rows = []
    for condition, c in (("ordered", 0), ("tail_matched", 1)):
        for phase, t in (("peak", 0), ("exit", 1), ("washout_2", 2)):
            theta = .07 * t + .06 * c
            phi = .08 * c + .03 * t
            for seed in range(8):
                a = math.sin(seed * .8) + seed * .09
                b = math.cos(seed * .57) + math.sin(seed * .21)
                vec = [a * math.cos(theta), b * math.cos(phi),
                       a * math.sin(theta), b * math.sin(phi),
                       .03 * (a + b)]
                rows.append({"schema_version": 1, "case_id": f"{condition}-{phase}-{seed}",
                             "condition": condition, "phase": phase, "seed": seed,
                             "probe_id": "flaky_ci", "model": "toy", "tokenizer": "toy",
                             "input_ids_sha256": f"digest-{seed}", "vectors": {"hs_8": vec}})
    return rows


def test_gauge_invariance_and_cycle_birth_for_matching_activation_frames():
    pytest.importorskip("percinterp")
    result = analyze_frames(synthetic_rows(), layer="hs_8", rank=2, min_overlap=.2, null_shuffles=4)
    assert result["schema"].endswith("frame_holonomy.v1")
    assert result["n_distinct_seeds"] == 8
    assert len(result["history_label_swap_null"]["null_max_loop_frobenius"]) == 4
    assert 0 <= result["history_label_swap_null"]["one_sided_permutation_p_value"] <= 1
    assert len(result["nodes"]) == 6
    assert len(result["edges"]) == 7
    assert result["holonomy"]["cycles_recorded"] == 2
    assert result["gauge_invariance_max_error"] < 1e-5
    assert all(e["transport_certified"] for e in result["edges"])
    assert all(0 <= e["rank_occupation"] <= 1 for e in result["edges"])
    assert all(0 <= n["rank_energy_fraction"] <= 1 for n in result["nodes"])


def test_misaligned_samples_and_rank_deficient_data_are_rejected():
    pytest.importorskip("percinterp")
    bad = synthetic_rows()[:-1]
    with pytest.raises(ValueError, match="same seed/probe"):
        analyze_frames(bad, layer="hs_8", rank=2)
    flat = synthetic_rows()
    for row in flat:
        row["vectors"]["hs_8"] = [1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="Rank-deficient"):
        analyze_frames(flat, layer="hs_8", rank=2)
    with pytest.raises(ValueError, match="two different conditions"):
        analyze_frames(synthetic_rows(), layer="hs_8", conditions=("ordered", "ordered"))
