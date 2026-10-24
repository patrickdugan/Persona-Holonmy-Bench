"""Orthogonal-null causal sweep tests on an actual tiny torch decoder."""
from __future__ import annotations

import copy
import json

import pytest

from persona_holonomy.protocol import PROBES, build_cases
from persona_holonomy.sweep import run_sweep, sweep_markdown, sweep_report, split_and_guard
from persona_holonomy.interventions import prepare_pairs, _patch_hook
from test_interventions import toy_model


def cases(repetitions=4, variation="seeded_v1"):
    return [x.to_dict() for x in build_cases(
        repetitions=repetitions, seed=22, variation=variation,
        conditions=("ordered", "tail_matched", "neutral"),
        phases=("exit",), probes=(PROBES[0],),
    )]


def test_erase_projection_is_removed_without_mutating_original_tensor():
    torch = pytest.importorskip("torch")
    block = torch.nn.Identity()
    original = torch.tensor([[[1., 0., 0.], [4., 3., 1.]]])
    direction = torch.tensor([2., 0., 0.])
    center = torch.tensor([1., 1., 1.])
    state = {}
    with _patch_hook(block, position=1, vector=direction, alpha=1,
                     mode="erase", reference=center, state=state):
        out = block(original)
    assert torch.equal(out[0, 1], torch.tensor([1., 3., 1.]))
    assert torch.equal(original[0, 1], torch.tensor([4., 3., 1.]))
    assert state["calls"] == 1
    with _patch_hook(block, position=1, vector=direction, alpha=0,
                     mode="erase", reference=center, state={}):
        assert torch.equal(block(original), original)
    with pytest.raises(ValueError, match="reference"):
        with _patch_hook(block, position=1, vector=direction, alpha=1,
                         mode="erase", state={}):
            block(original)


def test_split_rejects_identical_histories_and_holds_seeds():
    pairs = prepare_pairs(cases(), source_conditions=("ordered",),
                          target_condition="tail_matched", phases=("exit",))
    train, test = split_and_guard(pairs, holdout_fraction=.5, split_seed=33)
    assert set(train).isdisjoint(test)
    assert len(train) == len(test) == 2
    static_pairs = prepare_pairs(cases(variation="static"),
                                 source_conditions=("ordered",),
                                 target_condition="tail_matched", phases=("exit",))
    with pytest.raises(ValueError, match="leakage"):
        split_and_guard(static_pairs, holdout_fraction=.5, split_seed=33)


@pytest.mark.parametrize("options", [None, {"enable_thinking": False}])
def test_sweep_control_pairing_self_patch_and_token_diagnostic(options):
    model, tokenizer = toy_model()
    rows = run_sweep(cases=cases(), model=model, tokenizer=tokenizer,
                     layers=("0", "last"), alphas=(0., 1.),
                     source_conditions=("ordered",), phases=("exit",),
                     max_total_tokens=256, model_name="toy", split_seed=33,
                     chat_template_kwargs=options)
    for row in rows:
        assert ("chat_template_kwargs" in row) is (options is not None)
        if options is not None:
            assert row["chat_template_kwargs"] == options
    # 2 held-out seeds * 1 source * 2 modes * 2 controls * 2 alphas * 2 layers
    assert len(rows) == 32
    assert {row["layer"] for row in rows} == {0, 1}
    assert {row["mode"] for row in rows} == {"inject", "erase"}
    assert all(row["recipient_condition"] == ("ordered" if row["mode"] == "erase"
                                                   else "tail_matched") for row in rows)
    assert all(abs(row["effect"]) < 1e-8 for row in rows if row["alpha"] == 0)
    assert all(abs(row["self_patch_delta"]) < 1e-6 for row in rows)
    assert all(row["patch_l2_norm"] >= 0 for row in rows)
    assert all(row["first_token_effect"] is not None for row in rows)
    # Last-block output is patched after the block has processed all positions.
    # It only changes the logits for the *first* continuation token.
    assert max(abs(r["later_token_effect"]) for r in rows if r["layer"] == 1) < 1e-6
    assert any(abs(r["later_token_effect"]) > 1e-6 for r in rows
               if r["layer"] == 0 and r["alpha"] == 1)
    report = sweep_report(rows)
    assert report["identity_check_passed_1e_3"]
    assert report["max_abs_alpha_zero_effect"] == 0
    assert len(report["summaries"]) == 8
    assert any(s["alpha"] == 1. for s in report["summaries"])
    assert "specificity" in sweep_markdown(report).lower()
    with pytest.raises(ValueError, match="Duplicate"):
        sweep_report(rows + [rows[0]])
    with pytest.raises(ValueError, match="Unpaired"):
        sweep_report([r for r in rows if r is not rows[1]])


def test_sweep_validation_before_expensive_model_passes():
    model, tokenizer = toy_model()
    settings = dict(cases=cases(), model=model, tokenizer=tokenizer,
                    source_conditions=("ordered",), phases=("exit",))
    with pytest.raises(ValueError, match="repeated decoder block"):
        run_sweep(**settings, layers=("1", "last"))
    with pytest.raises(ValueError, match="Alphas"):
        run_sweep(**settings, alphas=(-.1,))
    with pytest.raises(ValueError, match="Alphas"):
        run_sweep(**settings, alphas=(.5, .5))
    with pytest.raises(ValueError, match="leakage"):
        run_sweep(**{**settings, "cases": cases(variation="static")})


def test_sweep_svg_render_and_missing_combination():
    from persona_holonomy.sweep_plot import render_sweep_svg
    demo = {"model": "toy<literal>", "summaries": [
        {"layer": k, "mode": mode, "direction_source": "ordered",
         "phase": "exit", "alpha": 1., "mean_paired_specificity": value,
         "seed_bootstrap_95pct_ci": [-.02, .1]}
        for k in (0, 1) for mode, value in (("inject", .05), ("erase", -.01))
    ]}
    xml = render_sweep_svg(demo)
    assert "<svg" in xml and "toy&lt;literal&gt;" in xml
    assert "inject" in xml and "erase" in xml
    with pytest.raises(ValueError, match="No sweep rows"):
        render_sweep_svg(demo, phase="washout_2")


def test_unseen_probe_holdout_changes_training_and_evaluation_sets():
    model, tokenizer = toy_model()
    # Each seed has two distinct technical questions; evaluation sees parser_validator only.
    source = [x.to_dict() for x in build_cases(
        repetitions=6, seed=71, variation="seeded_v1",
        conditions=("ordered", "tail_matched"), phases=("exit",),
        probes=(PROBES[0], PROBES[1]),
    )]
    rows = run_sweep(cases=source, model=model, tokenizer=tokenizer,
                     layers=("0",), alphas=(0., 1.),
                     source_conditions=("ordered",),
                     heldout_probe_ids=("parser_validator",),
                     phases=("exit",), max_total_tokens=256,
                     model_name="toy", split_seed=33,
                     self_check=False)
    assert {r["probe_id"] for r in rows} == {"parser_validator"}
    assert all(r["training_probe_ids"] == ["flaky_ci"] for r in rows)
    assert all(r["evaluation_probe_ids"] == ["parser_validator"] for r in rows)
    assert all(r["probe_holdout"] for r in rows)
    report = sweep_report(rows)
    assert report["training_probe_ids"] == ["flaky_ci"]
    assert report["evaluation_probe_ids"] == ["parser_validator"]
    assert report["probe_holdout"] is True
    assert report["hypothesis_family_size"] == 2  # Inject and erase; alpha=0 excluded.
    for s in report["summaries"]:
        assert s["paired_signflip_method"] == "exact_sign_flip"
        assert 0 <= s["paired_signflip_p_value"] <= 1
        assert (s["holm_familywise_p_value"] is None) == (s["alpha"] == 0)
    assert "Train probes" in sweep_markdown(report)


def test_probe_holdout_validation():
    model, tokenizer = toy_model()
    kwargs = dict(cases=cases(), model=model, tokenizer=tokenizer,
                  layers=("0",), alphas=(0., 1.),
                  source_conditions=("ordered",), phases=("exit",),
                  max_total_tokens=256)
    with pytest.raises(ValueError, match="at least one training"):
        run_sweep(**kwargs, heldout_probe_ids=("flaky_ci",))
    with pytest.raises(ValueError, match="must occur"):
        run_sweep(**kwargs, heldout_probe_ids=("missing_probe",))
    with pytest.raises(ValueError, match="unique"):
        run_sweep(**kwargs, heldout_probe_ids=("flaky_ci", "flaky_ci"))


def test_sweep_plan_matches_actual_forward_invocations_without_model():
    from persona_holonomy.sweep import sweep_plan
    c = [x.to_dict() for x in build_cases(
        repetitions=6, seed=71, variation="seeded_v1",
        conditions=("ordered", "tail_matched"), phases=("exit",),
        probes=(PROBES[0], PROBES[1]),
    )]
    plan = sweep_plan(cases=c, layers=("0", "last"), alphas=(0., 1.),
                      source_conditions=("ordered",), phases=("exit",),
                      split_seed=33, heldout_probe_ids=("parser_validator",))
    assert plan["n_training_pairs"] == 4
    assert plan["n_test_pairs"] == 2
    # 4 train pair captures + 2 test donors + 2 test references (self-check).
    assert plan["n_captures_per_layer"] == 12
    assert plan["n_self_patch_scoring_forwards_per_layer"] == 8
    assert plan["n_intervention_scoring_forwards_per_layer"] == 16
    assert plan["n_baseline_scoring_forwards"] == 8
    assert plan["estimated_total_model_forwards"] == 80
    assert plan["probe_holdout"]
    without_check = sweep_plan(cases=c, layers=("0", "last"),
                               alphas=(0., 1.), source_conditions=("ordered",),
                               phases=("exit",), split_seed=33,
                               heldout_probe_ids=("parser_validator",),
                               self_check=False)
    assert without_check["estimated_total_model_forwards"] < 80
