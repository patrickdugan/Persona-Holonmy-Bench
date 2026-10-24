import pytest

from persona_holonomy.decomposition import ESTIMANDS, decompose, seed_estimates

LEVEL = {"ordered": 1.0, "neutral": 0.5, "lexical_match": 3.0, "lexical_neutral": 2.0}


def _rows(*, variants=("default",), genres=("gothic",), seeds=range(6), shift=None):
    rows = []
    for genre in genres:
        for variant in variants:
            for seed in seeds:
                for probe in ("a", "b"):
                    for condition, level in LEVEL.items():
                        value = level + 0.01 * seed + (0.1 if probe == "b" else 0)
                        if shift and condition == "ordered":
                            value += shift.get(variant, 0)
                        rows.append({"genre": genre, "system_variant": variant, "phase": "exit",
                                     "seed": seed, "probe_id": probe, "condition": condition,
                                     "copy_free_margin": value, "margin": value + 0.25 *
                                     (condition in ("ordered", "lexical_match"))})
    return rows


def test_estimands_match_definitions():
    est = seed_estimates(_rows(), metric="copy_free_margin")[("gothic", "default", "exit")]
    for seed in range(6):
        assert est["R"][seed] == pytest.approx(1.0 - 0.5)
        assert est["L"][seed] == pytest.approx(3.0 - 2.0)
        assert est["F"][seed] == pytest.approx(2.0 - 0.5)
        assert est["P"][seed] == pytest.approx(est["R"][seed] - est["L"][seed])
    assert set(ESTIMANDS) == {"R", "L", "F", "P"}


def test_pooling_system_effect_and_copying_share():
    rows = _rows(variants=("default", "plain"), genres=("gothic", "noir"),
                 shift={"plain": 0.4})
    report = decompose(rows)
    assert report["groups"]["gothic/default/exit"]["P"]["mean"] == pytest.approx(-0.5)
    assert report["pooled_over_genres"]["plain/exit"]["P"]["mean"] == pytest.approx(-0.1)
    delta = report["system_plain_minus_default"]["noir/exit"]
    assert delta["mean"] == pytest.approx(0.4)
    assert delta["positive_seeds"] == 6
    # Copyable readout adds 0.25 to ordered and lexical_match: R and L shift by +0.25, P by 0.
    copying = report["copying"]["gothic/default/exit"]
    assert copying["R"]["mean"] == pytest.approx(0.25)
    assert copying["P"]["mean"] == pytest.approx(0.0)


def test_duplicate_cells_are_rejected():
    rows = _rows()
    with pytest.raises(ValueError, match="Duplicate"):
        seed_estimates(rows + rows[:1], metric="copy_free_margin")
