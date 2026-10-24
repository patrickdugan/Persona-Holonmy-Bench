"""Orthogonal-null causal sweep tests on an actual tiny torch decoder."""
from __future__ import annotations

import copy
import json
import statistics

import pytest

from persona_holonomy.protocol import PROBES, build_cases
from persona_holonomy.sweep import run_sweep, sweep_markdown, sweep_report, split_and_guard
from persona_holonomy.interventions import CausalProbe, prepare_pairs, _patch_hook
from test_interventions import toy_model


def cases(repetitions=4, variation="seeded_v1",
          conditions=("ordered", "tail_matched", "neutral")):
    return [x.to_dict() for x in build_cases(
        repetitions=repetitions, seed=22, variation=variation,
        conditions=conditions,
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
    with pytest.raises(ValueError, match="readout"):
        run_sweep(**settings, readout="unknown")
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="n_nulls"):
            run_sweep(**settings, n_nulls=invalid)


def test_sweep_defaults_are_byte_identical_to_explicit_style_single_null():
    model, tokenizer = toy_model()
    kwargs = dict(cases=cases(), model=model, tokenizer=tokenizer,
                  layers=("0",), alphas=(0., 1.), source_conditions=("ordered",),
                  max_total_tokens=256, split_seed=33)
    defaults = run_sweep(**kwargs)
    explicit = run_sweep(**kwargs, readout="style", n_nulls=1)
    assert json.dumps(defaults) == json.dumps(explicit)
    assert all(not {"readout", "n_nulls", "null_effects"} & row.keys()
               for row in defaults)


def test_sweep_copy_free_maps_baseline_patch_and_self_check(monkeypatch):
    torch = pytest.importorskip("torch")
    model, tokenizer = toy_model()
    source = cases(conditions=("ordered", "lexical_match"))

    def reject_style(*args, **kwargs):
        raise AssertionError("Copy-free sweeps must use the copy-free scorer throughout")

    monkeypatch.setattr(CausalProbe, "style_margin", reject_style)
    rows = run_sweep(cases=source, model=model, tokenizer=tokenizer,
                     layers=("0",), alphas=(0., 1.), source_conditions=("ordered",),
                     reference_condition="lexical_match", max_total_tokens=512,
                     split_seed=33, readout="copy_free")
    by_id = {case["case_id"]: case for case in source}
    runner = CausalProbe(model, tokenizer, layer="0", max_total_tokens=512)
    training = [(donor, reference) for donor, reference in prepare_pairs(
        source, source_conditions=("ordered",), target_condition="lexical_match",
        phases=("exit",)) if donor["seed"] in rows[0]["training_seeds"]]
    direction = torch.stack([
        runner.capture_vector(d["messages"]) - runner.capture_vector(r["messages"])
        for d, r in training]).float().mean(0)
    center = torch.stack([
        runner.capture_vector(r["messages"]) for _, r in training]).float().mean(0)
    baselines = {}
    self_scores = {}
    for row in rows:
        assert row["readout"] == "copy_free"
        case = by_id[row["recipient_case_id"]]
        assert case["probe_id"] == "flaky_ci"
        assert case["condition"] == ("ordered" if row["mode"] == "erase" else "lexical_match")
        if case["case_id"] not in baselines:
            baselines[case["case_id"]] = runner.copy_free_margin(
                case["messages"], probe_id=case["probe_id"])
            own = runner.capture_vector(case["messages"])
            self_scores[case["case_id"]] = runner.copy_free_margin(
                case["messages"], probe_id=case["probe_id"], vector=own,
                mode="transplant", alpha=1.)
        baseline = baselines[case["case_id"]]
        assert row["baseline_margin"] == baseline["copy_free_margin"]
        self_score = self_scores[case["case_id"]]
        assert row["self_patch_delta"] == pytest.approx(
            self_score["copy_free_margin"] - baseline["copy_free_margin"])
        assert abs(row["self_patch_delta"]) < 1e-6
        if row["alpha"] == 0:
            expected = baseline
        elif row["control"] == "candidate":
            patch = {"vector": direction, "mode": "add"}
            if row["mode"] == "erase":
                patch.update(mode="erase", reference=center)
            expected = runner.copy_free_margin(
                case["messages"], probe_id=case["probe_id"], alpha=row["alpha"], **patch)
        else:
            continue  # Null aggregation and reproducibility are checked below.
        assert row["patched_margin"] == expected["copy_free_margin"]
        for row_field, score_field in (
                ("effect", "copy_free_margin"),
                ("first_token_effect", "copy_free_first_token_margin"),
                ("later_token_effect", "copy_free_later_token_margin")):
            assert row[row_field] == pytest.approx(expected[score_field] - baseline[score_field])
    report = sweep_report(rows)
    assert report["readout"] == "copy_free"
    assert "copy_free" in sweep_markdown(report)
    with pytest.raises(ValueError, match="mix readouts"):
        sweep_report([*rows, {**rows[0], "readout": "style"}])


@pytest.mark.parametrize("readout", ["style", "copy_free"])
def test_multiple_nulls_average_effects_and_preserve_first_null(readout, monkeypatch):
    from persona_holonomy import sweep

    torch = pytest.importorskip("torch")
    model, tokenizer = toy_model()
    kwargs = dict(cases=cases(conditions=("ordered", "lexical_match")),
                  model=model, tokenizer=tokenizer, layers=("0",), alphas=(0., .5, 1.),
                  source_conditions=("ordered",), reference_condition="lexical_match",
                  max_total_tokens=512, split_seed=33, self_check=False, readout=readout)
    single = run_sweep(**kwargs, n_nulls=1)
    vectors = []
    original = sweep._orthogonal_random

    def record_null(direction, *, seed, torch):
        orth = original(direction, seed=seed, torch=torch)
        assert abs(float(torch.dot(orth, direction))) < 1e-4 * float(
            torch.linalg.vector_norm(direction)) ** 2
        assert float(torch.linalg.vector_norm(orth)) == pytest.approx(
            float(torch.linalg.vector_norm(direction)), rel=1e-6)
        vectors.append((seed, orth.clone()))
        return orth

    monkeypatch.setattr(sweep, "_orthogonal_random", record_null)
    scored = []
    method_name = "copy_free_margin" if readout == "copy_free" else "style_margin"
    original_margin = getattr(CausalProbe, method_name)

    def record_margin(runner, messages, **patch):
        result = original_margin(runner, messages, **patch)
        scored.append((json.dumps(messages), patch, result))
        return result

    monkeypatch.setattr(CausalProbe, method_name, record_margin)
    multiple = run_sweep(**kwargs, n_nulls=3)
    assert len(multiple) == len(single)
    assert all(len({seed for seed, _ in vectors[i:i + 3]}) == 3
               for i in range(0, len(vectors), 3))
    by_id = {case["case_id"]: case for case in kwargs["cases"]}
    for one, many in zip(single, multiple):
        if one["control"] == "candidate":
            assert one == many
            continue
        assert many["n_nulls"] == len(many["null_effects"]) == 3
        assert many["null_effects"][0] == one["effect"]
        assert many["effect"] == statistics.mean(many["null_effects"])
        assert many["patched_margin"] == pytest.approx(many["baseline_margin"] + many["effect"])
        assert many["patch_l2_norm"] == one["patch_l2_norm"]
        if many["alpha"] == 0:
            assert many["null_effects"] == [0., 0., 0.]
            assert many["first_token_effect"] == many["later_token_effect"] == 0
        else:
            messages = json.dumps(by_id[many["recipient_case_id"]]["messages"])
            base_score = next(s for m, patch, s in scored if m == messages and "vector" not in patch)
            # Each nonzero dose scores the candidate followed by three nulls.
            patches = [(patch, s) for m, patch, s in scored
                       if m == messages and patch.get("alpha") == many["alpha"]]
            assert len(patches) == 4
            for patch, _ in patches[1:]:
                assert many["alpha"] * float(torch.linalg.vector_norm(patch["vector"])) == pytest.approx(
                    many["patch_l2_norm"], rel=1e-6, abs=1e-7)
            prefix = "copy_free_" if readout == "copy_free" else ""
            assert many["patched_margin"] == statistics.mean(
                s[prefix + "margin"] for _, s in patches[1:])
            for row_field, score_field in (("first_token_effect", "first_token_margin"),
                                           ("later_token_effect", "later_token_margin")):
                field = prefix + score_field
                assert many[row_field] == statistics.mean(
                    s[field] - base_score[field] for _, s in patches[1:])
    # Repeated runs reproduce indexed null seeds, scores, and row ordering.
    assert json.dumps(multiple) == json.dumps(run_sweep(**kwargs, n_nulls=3))
    assert sweep_report(multiple)["n_records"] == len(multiple)


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


@pytest.mark.parametrize("readout,continuations", [("style", 2), ("copy_free", 5)])
@pytest.mark.parametrize("n_nulls", [1, 3])
@pytest.mark.parametrize("self_check", [False, True])
def test_sweep_plan_scales_and_matches_actual_forwards(readout, continuations, n_nulls, self_check):
    from persona_holonomy.sweep import sweep_plan

    source = cases(conditions=("ordered", "lexical_match"))
    settings = dict(cases=source, layers=("0", "last"), alphas=(0., .5, 1.),
                    source_conditions=("ordered",), reference_condition="lexical_match",
                    split_seed=33, readout=readout, n_nulls=n_nulls, self_check=self_check)
    plan = sweep_plan(**settings)
    assert plan["readout"] == readout
    assert plan["n_nulls"] == n_nulls
    assert plan["n_intervention_scoring_forwards_per_layer"] == (
        plan["n_test_pairs"] * 2 * (1 + n_nulls) * continuations * 2)
    assert plan["n_baseline_scoring_forwards"] == plan["n_distinct_baseline_cases"] * continuations
    assert plan["n_self_patch_scoring_forwards_per_layer"] == (
        plan["n_distinct_baseline_cases"] * continuations if self_check else 0)
    model, tokenizer = toy_model()
    forwards = []
    with model.register_forward_hook(lambda *args: forwards.append(1)):
        run_sweep(**settings, model=model, tokenizer=tokenizer, max_total_tokens=512)
    assert len(forwards) == plan["estimated_total_model_forwards"]


@pytest.mark.parametrize("settings", [{"readout": "unknown"}, {"n_nulls": 0}, {"n_nulls": 1.5}])
def test_sweep_plan_validates_readout_and_nulls(settings):
    from persona_holonomy.sweep import sweep_plan

    with pytest.raises(ValueError, match="readout|n_nulls"):
        sweep_plan(cases=cases(), layers=("0",), alphas=(0., 1.),
                   source_conditions=("ordered",), **settings)


@pytest.mark.parametrize("command", ["sweep", "sweep-plan"])
def test_sweep_cli_parses_and_forwards_readout_and_nulls(command, monkeypatch, tmp_path):
    from persona_holonomy import cli

    required = ["--input", "cases.jsonl"]
    if command == "sweep":
        required += ["--model", "cached-toy"]
    parser = cli.make_parser()
    defaults = parser.parse_args([command, *required])
    assert (defaults.readout, defaults.n_nulls) == ("style", 1)
    args = parser.parse_args([command, *required, "--readout", "copy_free", "--n-nulls", "3"])
    assert (args.readout, args.n_nulls) == ("copy_free", 3)
    with pytest.raises(SystemExit):
        parser.parse_args([command, *required, "--readout", "unknown"])
    forwarded = {}

    def fake_run(**kwargs):
        forwarded.update(kwargs)
        return 0

    def fake_plan(**kwargs):
        forwarded.update(kwargs)
        return {"estimated_total_model_forwards": 0, "layers_requested": ["0"], "n_test_pairs": 0}

    monkeypatch.setattr(cli, "sweep_hf", fake_run)
    monkeypatch.setattr(cli, "sweep_plan", fake_plan)
    input_path = tmp_path / "cases.jsonl"
    input_path.write_text("", encoding="utf-8")
    args.input = str(input_path)
    args.output = str(tmp_path / "output.json")
    args.func(args)
    assert forwarded["readout"] == "copy_free"
    assert forwarded["n_nulls"] == 3
