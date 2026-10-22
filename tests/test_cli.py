import json
from pathlib import Path

from persona_holonomy.cli import main


def test_end_to_end_mock(tmp_path):
    gen_path = tmp_path / "cases.jsonl"
    run_path = tmp_path / "responses.jsonl"
    report_prefix = tmp_path / "report"
    args = ["--repetitions", "1", "--phases", "peak,exit,washout_2",
            "--probes", "flaky_ci,parser_validator"]
    main(["generate", *args, "--output", str(gen_path)])
    cases = [json.loads(line) for line in gen_path.read_text().splitlines()]
    assert len(cases) == 3 * 3 * 2
    main(["run", *args, "--backend", "mock", "--output", str(run_path)])
    rows = [json.loads(line) for line in run_path.read_text().splitlines()]
    assert len(rows) == len(cases)
    assert {r["backend"] for r in rows} == {"mock"}
    main(["analyze", "--input", str(run_path), "--output-prefix", str(report_prefix)])
    report = json.loads(Path(str(report_prefix) + ".json").read_text())
    assert report["synthetic"]
    assert len(report["comparisons"]) == 6
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
