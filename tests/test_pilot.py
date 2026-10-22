"""Offline pilot orchestration guards; all fake GPU phases use a stub runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from persona_holonomy.cli import make_parser
from persona_holonomy.pilot import (
    SCHEMA, STAGES, _commands, _load_plan, file_sha256, plan_pilot, run_stage,
)

REVISION = "f" * 40


def pilot(tmp_path, **overrides):
    kwargs = dict(output_dir=tmp_path / "pilot", model=str(tmp_path / "HF-Qwen"),
                  checkpoint_revision=REVISION)
    kwargs.update(overrides)
    return plan_pilot(**kwargs)


def local_model(tmp_path):
    model = tmp_path / "HF-Qwen"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2"}')
    (model / "tokenizer.json").write_text('{}')
    (model / "model.safetensors").write_bytes(b"synthetic weights, do not load")
    return model


def test_offline_plan_creates_immutable_cases_without_gpu(tmp_path):
    report = pilot(tmp_path)
    root = tmp_path / "pilot"
    assert report["schema"] == SCHEMA
    assert report["estimated_graph_forwards_upper_bound"] == 50
    assert report["capture_split"]["phb_train_seeds"] == [100, 101, 102, 104, 105]
    assert report["capture_split"]["phb_test_seeds_excluded"] == [103, 106, 107]
    assert report["capture_split"]["phb_training_probes"] == ["flaky_ci"]
    assert report["capture_split"]["phb_test_probes_excluded"] == ["parser_validator"]
    assert file_sha256(root / "cases.jsonl") == report["case_file_sha256"]
    assert len((root / "cases.jsonl").read_text().splitlines()) == 32
    assert json.loads((root / "progress.json").read_text())["completed"] == {}
    assert _load_plan(root / "plan.json")[1] == report
    with pytest.raises(FileExistsError, match="populated"):
        pilot(tmp_path)


def test_plan_rejects_mutable_revision_and_unbounded_cost(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        pilot(tmp_path, checkpoint_revision="main")
    with pytest.raises(ValueError, match="estimates"):
        pilot(tmp_path, max_model_forwards=49)
    with pytest.raises(ValueError, match="distinct"):
        pilot(tmp_path, layers="mid,mid")
    with pytest.raises(ValueError, match="seeds"):
        pilot(tmp_path, n_seeds=5)


def test_commands_are_local_and_never_download_or_trust_remote_code(tmp_path):
    plan = pilot(tmp_path)
    root = tmp_path / "pilot"
    for stage in STAGES:
        commands = _commands(stage, root, plan, "cuda", "float16")
        assert commands
        for argv in commands:
            assert argv[1:3] == ["-m", "persona_holonomy"]
            assert "--allow-download" not in argv
            assert "--trust-remote-code" not in argv
    assert "--max-input-tokens" in _commands("capture", root, plan, "cuda", "float16")[0]
    assert "--max-total-tokens" in _commands("evaluate", root, plan, "cuda", "float16")[1]


def test_stage_is_opt_in_and_enforces_model_and_artifact_integrity(tmp_path):
    plan = pilot(tmp_path)
    root = tmp_path / "pilot"
    with pytest.raises(ValueError, match="Finish stage capture"):
        run_stage(plan_path=root / "plan.json", stage="fit")
    with pytest.raises(ValueError, match="local HF"):
        run_stage(plan_path=root / "plan.json", stage="capture")
    local_model(tmp_path)
    executed = []

    def simulate(argv, *, check, cwd):
        executed.append(argv)
        assert check and cwd == str(root)
        (root / "capture").mkdir(exist_ok=True)
        (root / "capture/capture-manifest.json").write_text('{"schema":"synthetic"}')

    result = run_stage(plan_path=root / "plan.json", stage="capture", runner=simulate)
    assert result["status"] == "completed" and len(executed) == 1
    assert run_stage(plan_path=root / "plan.json", stage="capture", runner=simulate)["status"] == "previously_completed_verified"
    assert len(executed) == 1
    (root / "capture/capture-manifest.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="artifact changed"):
        run_stage(plan_path=root / "plan.json", stage="fit", runner=simulate)


def test_plan_and_cases_tampering_are_rejected_before_execution(tmp_path):
    pilot(tmp_path)
    root = tmp_path / "pilot"
    local_model(tmp_path)
    (root / "plan.json").write_text((root / "plan.json").read_text().replace('"top_features": 4', '"top_features": 5'))
    with pytest.raises(ValueError, match="plan changed"):
        run_stage(plan_path=root / "plan.json", stage="capture")
    # Fresh plan: now tamper with the case file rather than its metadata.
    pilot2 = plan_pilot(output_dir=tmp_path / "fresh", model="/fake/model", checkpoint_revision=REVISION)
    assert pilot2["schema"] == SCHEMA
    with (tmp_path / "fresh/cases.jsonl").open("a") as f:
        f.write('{}\n')
    with pytest.raises(ValueError, match="Frozen cases"):
        run_stage(plan_path=tmp_path / "fresh/plan.json", stage="capture")


def test_graph_and_evaluation_forward_budget_gate(tmp_path):
    pilot(tmp_path)
    root = tmp_path / "pilot"
    local_model(tmp_path)
    # Verify the entire preceding chain is frozen without executing GPU work.
    state = json.loads((root / "progress.json").read_text())
    for stage in ("capture", "fit"):
        stage_output = ("capture/capture-manifest.json" if stage == "capture"
                        else "saes/sae-registry.json")
        target = root / stage_output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"synthetic":true}')
        state["completed"][stage] = {stage_output: file_sha256(target)}
    (root / "progress.json").write_text(json.dumps(state))
    called = []

    def stub(argv, *, check, cwd):
        called.append(argv)
        (root / "attribution-plan.json").write_text(json.dumps(
            {"estimated_model_forwards_upper_bound": 9999}))

    with pytest.raises(ValueError, match="budget"):
        run_stage(plan_path=root / "plan.json", stage="graph", runner=stub)
    assert len(called) == 1, "The GPU graph estimation must not execute after over-budget preflight"


def test_cli_exposes_both_pilot_commands():
    parser = make_parser()
    a = parser.parse_args(["pilot-plan", "--model", "C:/cached/qwen", "--revision", REVISION,
                           "--output-dir", "runs/pilot"])
    assert a.top_features == 4 and a.max_model_forwards == 250
    b = parser.parse_args(["pilot-stage", "--plan", "runs/pilot/plan.json", "--stage", "graph"])
    assert b.stage == "graph" and b.device == "cuda"


def test_all_stage_transitions_on_stubbed_model_and_null_path_guard(tmp_path):
    pilot(tmp_path)
    root = tmp_path / "pilot"
    local_model(tmp_path)
    executed = []

    def fake_command(argv, *, check, cwd):
        assert check and cwd == str(root)
        command = argv[3]
        executed.append(command)
        if command == "sae-capture":
            dest = root / "capture/capture-manifest.json"
        elif command == "sae-fit":
            dest = root / "saes/sae-registry.json"
        else:
            dest = Path(argv[argv.index("--output") + 1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = {"synthetic_stub": True}
        if command == "sae-attribution-plan":
            data["estimated_model_forwards_upper_bound"] = 50
        if command == "sae-attribution":
            data["diagnostics"] = {"has_measured_source_to_sink_path": True}
        if command == "perc-components-plan":
            data["estimated_model_forwards"] = 70
        dest.write_text(json.dumps(data) + "\n")

    for stage in STAGES:
        result = run_stage(plan_path=root / "plan.json", stage=stage, runner=fake_command)
        assert result["status"] == "completed", stage
        assert len(result["artifacts"]) >= 1
    state = json.loads((root / "progress.json").read_text())
    assert list(state["completed"]) == list(STAGES)
    assert executed == [
        "sae-capture", "sae-fit", "sae-attribution-plan", "sae-attribution",
        "perc-select", "sae-feature-map", "perc-components-plan", "perc-components",
        "perc-components-report",
    ]
    assert run_stage(plan_path=root / "plan.json", stage="evaluate", runner=fake_command)["status"] == "previously_completed_verified"
    assert len(executed) == 9


def test_graph_without_spanning_path_blocks_percolation(tmp_path):
    pilot(tmp_path)
    root = tmp_path / "pilot"
    state = json.loads((root / "progress.json").read_text())
    for stage in STAGES[:3]:
        stage_outputs = {}
        from persona_holonomy.pilot import OUTPUTS
        for relative in OUTPUTS[stage]:
            file = root / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps({"diagnostics": {"has_measured_source_to_sink_path": False}}))
            stage_outputs[relative] = file_sha256(file)
        state["completed"][stage] = stage_outputs
    (root / "progress.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="does not span"):
        run_stage(plan_path=root / "plan.json", stage="select", runner=lambda *_, **__: pytest.fail("no process allowed"))
