import json

import pytest

from persona_holonomy.analysis import (
    analyze, paired_effect, read_jsonl, to_markdown, closure_effect, recovery_did,
)
from persona_holonomy.scoring import score_response


def _row(condition, seed, response, phase="exit", probe="flaky_ci"):
    return {
        "case_id": f"{condition}.{phase}.{probe}.s{seed}",
        "condition": condition,
        "phase": phase,
        "probe_id": probe,
        "seed": seed,
        "response": response,
        "history_chars": 300,
        "backend": "ollama",
        "generation_meta": {},
    }


def test_paired_effect_on_seed_cluster():
    rows = []
    for seed in range(5):
        rows.extend([_row("ordered", seed, "ACT IV: THE CATHEDRAL!", probe="flaky_ci"),
                     _row("ordered", seed, "ACT IV: THE CATHEDRAL!", probe="parser_validator"),
                     _row("permuted", seed, "A test may rely on timing."),
                     _row("permuted", seed, "A parser reads input.", probe="parser_validator")])
    for row in rows:
        row["scores"] = score_response(row["response"])
    effect = paired_effect(rows, phase="exit", left="ordered", right="permuted")
    assert effect["n_paired_probes"] == 10
    assert effect["n_seeds"] == 5
    assert effect["mean_difference"] > 0
    assert effect["bootstrap_95pct_ci"] is not None


def test_report_and_duplicates():
    rows = [_row("ordered", 1, "ACT IV: THE CATHEDRAL!"),
            _row("permuted", 1, "A normal response."),
            _row("neutral", 1, "A normal response.")]
    report = analyze(rows)
    assert not report["synthetic"]
    assert report["n_records"] == 3
    assert len(report["comparisons"]) == 2
    assert "Seed-bootstrap" in to_markdown(report)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(rows + [rows[0]])


def test_read_jsonl(tmp_path):
    path = tmp_path / "run.jsonl"
    payload = _row("ordered", 41, "hello")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert read_jsonl(path)[0] == payload
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path)


def test_recovery_with_paired_baseline_and_did():
    rows = []
    for seed in range(5):
        for condition in ("ordered", "permuted", "neutral"):
            for phase in ("baseline", "exit", "washout_2"):
                reply = "A parser checks a format."
                if condition == "ordered" and phase == "exit":
                    reply = "ACT IV: THE CATHEDRAL! Quest unlocked!"
                rows.append(_row(condition, seed, reply, phase=phase))
    for row in rows:
        row["scores"] = score_response(row["response"])
    closure = closure_effect(rows, condition="ordered", phase="exit")
    assert closure["n_seeds"] == 5
    assert closure["mean_change_from_baseline"] > 0
    assert closure["bootstrap_95pct_ci"][0] > 0
    did = recovery_did(rows, phase="exit")
    assert did["n_seeds"] == 5
    assert did["mean_difference_in_differences"] > 0
    assert recovery_did(rows, phase="washout_2")["mean_difference_in_differences"] == 0
    assert closure_effect(rows, condition="ordered", phase="peak") is None


def test_missing_baseline_yields_no_closure_and_no_did():
    rows = [_row("ordered", 1, "ACT IV: THE CATHEDRAL!"),
            _row("permuted", 1, "A normal response.")]
    for row in rows:
        row["scores"] = score_response(row["response"])
    assert closure_effect(rows, condition="ordered", phase="exit") is None
    assert recovery_did(rows, phase="exit") is None
