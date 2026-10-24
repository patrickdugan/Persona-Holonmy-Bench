import json
from pathlib import Path

import pytest

from persona_holonomy.cli import main


def test_end_to_end_mock(tmp_path):
    gen_path = tmp_path / "cases.jsonl"
    run_path = tmp_path / "responses.jsonl"
    report_prefix = tmp_path / "report"
    args = ["--repetitions", "1", "--phases", "peak,exit,washout_2",
            "--probes", "flaky_ci,parser_validator"]
    main(["generate", *args, "--output", str(gen_path)])
    cases = [json.loads(line) for line in gen_path.read_text().splitlines()]
    assert len(cases) == 4 * 3 * 2
    main(["run", *args, "--backend", "mock", "--output", str(run_path)])
    rows = [json.loads(line) for line in run_path.read_text().splitlines()]
    assert len(rows) == len(cases)
    assert {r["backend"] for r in rows} == {"mock"}
    main(["analyze", "--input", str(run_path), "--output-prefix", str(report_prefix)])
    report = json.loads(Path(str(report_prefix) + ".json").read_text())
    assert report["synthetic"]
    assert len(report["comparisons"]) == 9
    assert Path(str(report_prefix) + ".md").exists()


def test_plot(tmp_path):
    run_path = tmp_path / "responses.jsonl"
    prefix = tmp_path / "summary"
    svg_path = tmp_path / "trajectory.svg"
    main(["run", "--phases", "peak,exit,washout_2", "--output", str(run_path)])
    main(["analyze", "--input", str(run_path), "--output-prefix", str(prefix)])
    main(["plot", "--report", str(prefix) + ".json", "--output", str(svg_path)])
    svg = svg_path.read_text()
    assert svg.startswith("<svg")
    assert "SYNTHETIC DEMO" in svg
    assert "ordered" in svg


def test_run_execution_order_is_seeded_and_counterbalanced(tmp_path):
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    three = tmp_path / "three.jsonl"
    common = ["run", "--backend", "mock", "--repetitions", "2",
              "--phases", "baseline,peak,exit", "--probes", "flaky_ci"]
    main([*common, "--execution-seed", "9", "--output", str(one)])
    main([*common, "--execution-seed", "9", "--output", str(two)])
    main([*common, "--execution-seed", "10", "--output", str(three)])
    a = [json.loads(line)["case_id"] for line in one.read_text().splitlines()]
    b = [json.loads(line)["case_id"] for line in two.read_text().splitlines()]
    c = [json.loads(line)["case_id"] for line in three.read_text().splitlines()]
    assert a == b
    assert a != c
    assert sorted(a) == sorted(c)


def test_intervention_report_cli(tmp_path):
    from persona_holonomy.cli import main
    record = {
        "schema_version": 1, "experiment": "transplant", "model": "synthetic-toy",
        "tokenizer": "toy", "layer": 1, "target_condition": "tail_matched",
        "alpha": 1., "source_condition": "ordered", "seed": 3,
        "probe_id": "flaky_ci", "phase": "exit", "effect": .15,
        "self_patch_delta": 0.0, "target_case_id": "test-id",
        "negative_control": False,
    }
    raw = tmp_path / "interventions.jsonl"
    raw.write_text(json.dumps(record) + "\n")
    output = tmp_path / "summary"
    main(["intervention-report", "--input", str(raw), "--output-prefix", str(output)])
    report = json.loads(Path(str(output) + ".json").read_text())
    assert report["n_records"] == 1
    assert report["groups"][0]["mean_effect"] == .15
    assert Path(str(output) + ".md").exists()


def test_sweep_report_and_svg_cli(tmp_path):
    input_path = tmp_path / "sweep.jsonl"
    rows = []
    for control, effect in (("candidate", .05), ("orthogonal", .01)):
        rows.append({
            "schema_version": 2, "experiment": "direction_sweep",
            "model": "synthetic-fixture", "tokenizer": "toy",
            "reference_condition": "tail_matched", "layer": 1,
            "alpha": 1., "mode": "inject", "control": control,
            "direction_source": "ordered", "phase": "exit", "seed": 3,
            "probe_id": "flaky_ci", "training_seeds": [1, 2],
            "recipient_case_id": "test-id", "self_patch_delta": 0.,
            "effect": effect, "oriented_effect": effect,
            "first_token_effect": effect, "later_token_effect": 0.,
        })
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "sweep-report"
    main(["sweep-report", "--input", str(input_path),
          "--output-prefix", str(output)])
    report_path = Path(str(output) + ".json")
    report = json.loads(report_path.read_text())
    assert report["summaries"][0]["mean_paired_specificity"] == .04
    svg = tmp_path / "sweep.svg"
    main(["sweep-plot", "--report", str(report_path),
          "--output", str(svg)])
    assert "<svg" in svg.read_text()
    assert "synthetic-fixture" in svg.read_text()


def test_sweep_plan_cli_without_model(tmp_path):
    from persona_holonomy.cli import main
    data = tmp_path / "cases.jsonl"
    main(["generate", "--variation", "seeded_v1", "--repetitions", "6",
          "--conditions", "ordered,tail_matched", "--phases", "exit",
          "--probes", "flaky_ci,parser_validator", "--output", str(data)])
    report = tmp_path / "plan.json"
    main(["sweep-plan", "--input", str(data), "--layers", "0,last",
          "--alphas", "0,1", "--sources", "ordered",
          "--heldout-probes", "parser_validator", "--output", str(report)])
    obj = json.loads(report.read_text())
    import hashlib
    assert obj["input_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert obj["n_test_pairs"] > 0
    assert obj["train_probe_ids"] == ["flaky_ci"]
    assert obj["test_probe_ids"] == ["parser_validator"]
    assert obj["estimated_total_model_forwards"] > 0


def test_cli_version_matches_package_version():
    from persona_holonomy import __version__
    from persona_holonomy.cli import make_parser
    assert __version__ in make_parser().description


@pytest.mark.parametrize("no_thinking", [False, True])
@pytest.mark.parametrize("command,required,adapter,result", [
    ("capture", [], "capture", 1),
    ("intervene", [], "intervene_hf", 1),
    ("sweep", [], "sweep_hf", 1),
    ("freegen", [], "freegen_hf", 1),
    ("perc-components", ["--manifest", "selection.json", "--feature-map", "map.json"],
     "component_ablation_hf", 1),
    ("sae-capture", ["--revision", "f" * 40, "--output-dir", "capture"],
     "collect_hf", {"blocks": {"0": {}}}),
    ("sae-attribution", ["--registry", "registry.json"],
     "graph_hf", {"edges": [], "diagnostics": {"total_model_forwards": 0}}),
])
def test_hf_cli_parses_and_forwards_no_thinking(
        monkeypatch, command, required, adapter, result, no_thinking):
    from persona_holonomy import cli

    calls = []

    def fake_adapter(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(cli, adapter, fake_adapter)
    argv = [command, "--model", "local-model", "--input", "cases.jsonl", *required]
    if no_thinking:
        argv.append("--no-thinking")
    assert cli.make_parser().parse_args(argv).no_thinking is no_thinking
    cli.main(argv)
    assert len(calls) == 1
    assert calls[0]["chat_template_kwargs"] == (
        {"enable_thinking": False} if no_thinking else None)


@pytest.mark.parametrize("no_thinking", [False, True])
def test_pilot_cli_forwards_no_thinking(monkeypatch, no_thinking):
    from persona_holonomy import cli

    calls = []

    def fake_plan(**kwargs):
        calls.append(kwargs)
        return {"estimated_graph_forwards_upper_bound": 50}

    monkeypatch.setattr(cli, "plan_pilot", fake_plan)
    argv = ["pilot-plan", "--model", "local-model", "--revision", "f" * 40,
            "--output-dir", "pilot"]
    if no_thinking:
        argv.append("--no-thinking")
    assert cli.make_parser().parse_args(argv).no_thinking is no_thinking
    cli.main(argv)
    assert calls[0]["no_thinking"] is no_thinking
