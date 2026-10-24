"""Free-generation activation patch and masked-ratings regression tests."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from persona_holonomy.freegen import (
    blind_export, blind_report, read_blind_report, run_freegen, write_blind_export,
)
from persona_holonomy.interventions import CausalProbe
from persona_holonomy.protocol import PROBES, build_cases
from test_interventions import toy_model


def generation_toy():
    torch = pytest.importorskip("torch")
    model, tok = toy_model()
    tok.pad_token_id = 0

    def fake_generate(*, input_ids, attention_mask=None, max_new_tokens=3,
                      do_sample=False, num_beams=1, use_cache=True, **_):
        # Mimic the two relevant HF forward shapes: prefix prefill and
        # subsequent one-token calls with KV caching.
        sequence = input_ids.clone()
        for step in range(max_new_tokens):
            current = sequence if step == 0 else sequence[:, -1:]
            scores = model(input_ids=current, use_cache=use_cache).logits
            next_token = torch.argmax(scores[:, -1:, :], dim=-1)
            sequence = torch.cat((sequence, next_token), dim=-1)
        return sequence

    model.generate = fake_generate
    tok.decode = lambda ids, skip_special_tokens=True: " ".join(f"T{i}" for i in ids)
    return model, tok


def probe_cases(reps=5):
    return [c.to_dict() for c in build_cases(
        repetitions=reps, seed=100, variation="seeded_v1",
        conditions=("ordered", "tail_matched"), phases=("exit",),
        probes=(PROBES[0], PROBES[1]),
    )]


def test_generate_prefill_only_hook_and_unmodified_suffix():
    torch = pytest.importorskip("torch")
    model, tok = generation_toy()
    case = probe_cases(1)[0]
    runner = CausalProbe(model, tok, layer="last", max_total_tokens=256)
    baseline = runner.generate(case["messages"], max_new_tokens=4)
    assert baseline["generated_tokens"] == 4
    direction = runner.capture_vector(case["messages"])
    patched = runner.generate(case["messages"], vector=direction,
                              mode="add", alpha=1, max_new_tokens=4)
    assert patched["prefix_sha256"] == baseline["prefix_sha256"]
    assert patched["generated_tokens"] == baseline["generated_tokens"]
    # alpha zero must reproduce the baseline, with or without a direction.
    sham = runner.generate(case["messages"], vector=direction,
                           mode="add", alpha=0, max_new_tokens=4)
    assert sham == baseline
    with pytest.raises(ValueError, match="total token budget"):
        runner.generate(case["messages"], max_new_tokens=255)
    with pytest.raises(ValueError, match="positive"):
        runner.generate(case["messages"], max_new_tokens=0)


@pytest.mark.parametrize("options", [None, {"enable_thinking": False}])
def test_freegen_trials_are_paired_and_seed_disjoint(options):
    model, tok = generation_toy()
    rows = run_freegen(cases=probe_cases(), model=model, tokenizer=tok,
                       layer="0", max_new_tokens=2, max_total_tokens=256,
                       chat_template_kwargs=options)
    for row in rows:
        assert ("chat_template_kwargs" in row) is (options is not None)
        if options is not None:
            assert row["chat_template_kwargs"] == options
    assert len(rows) > 0 and len(rows) % 3 == 0
    assert {r["control"] for r in rows} == {"baseline", "candidate", "orthogonal"}
    assert {r["probe_id"] for r in rows} == {"parser_validator"}
    assert all(r["seed"] in r["test_seeds"] and r["seed"] not in r["training_seeds"]
               for r in rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["trial_id"], []).append(row)
    assert all({r["control"] for r in group} ==
               {"baseline", "candidate", "orthogonal"} for group in grouped.values())
    assert all(len({r["prefix_sha256"] for r in group}) == 1
               for group in grouped.values())
    for group in grouped.values():
        assert group[0]["patch_l2_norm"] == 0
        assert group[1]["patch_l2_norm"] == pytest.approx(group[2]["patch_l2_norm"])
    with pytest.raises(ValueError, match="unseen heldout probes"):
        run_freegen(cases=probe_cases(), model=model, tokenizer=tok,
                    heldout_probe_ids=(), max_new_tokens=2)
    with pytest.raises(ValueError, match="alphas"):
        run_freegen(cases=probe_cases(), model=model, tokenizer=tok,
                    alphas=(0.,), max_new_tokens=2)


def _mock_rows(n_seeds=5):
    rows = []
    for seed in range(n_seeds):
        for mode in ("inject", "erase"):
            trial = f"0|ordered|{mode}|{seed}|parser_validator|exit|1"
            for control in ("baseline", "candidate", "orthogonal"):
                rows.append({"experiment": "heldout_freegen", "trial_id": trial,
                             "alpha": 1, "control": control, "seed": seed,
                             "probe_id": "parser_validator", "direction_source": "ordered",
                             "phase": "exit", "layer": 0, "mode": mode,
                             "probe_text": "Explain API validation.",
                             "response": "A valid technical answer."})
    return rows


def test_blind_export_randomization_and_complete_ratings():
    rows = _mock_rows()
    sheet, key = blind_export(rows, seed=32)
    assert len(sheet) == len(rows)
    assert len({s["sample_id"] for s in sheet}) == len(sheet)
    assert all("control" not in s and "mode" not in s for s in sheet)
    assert sheet != blind_export(rows, seed=33)[0]
    for sample in sheet:
        control = key["lookup"][sample["sample_id"]]["control"]
        mode = key["lookup"][sample["sample_id"]]["mode"]
        # A consistent effect with a neutral same-norm control.
        sample["style_rating_0_to_4"] = (
            3 if control == "candidate" and mode == "inject" else
            1 if control == "candidate" and mode == "erase" else 2)
        sample["answer_quality_0_to_4"] = 3
    report = blind_report(sheet, key)
    assert report["n_trials"] == len(rows) // 3
    assert len(report["summaries"]) == 2
    assert all(s["mean_style_specificity"] == 1 for s in report["summaries"])
    assert all(s["mean_answer_quality_specificity"] == 0 for s in report["summaries"])
    assert all(s["paired_signflip_p_value"] == pytest.approx(.0625)
               for s in report["summaries"])
    with pytest.raises(ValueError, match="complete"):
        blind_report(sheet[:-1], key)
    broken = copy.deepcopy(sheet)
    broken[0]["style_rating_0_to_4"] = ""
    with pytest.raises(ValueError, match="complete"):
        blind_report(broken, key)
    broken = copy.deepcopy(sheet)
    broken[0]["style_rating_0_to_4"] = 6
    with pytest.raises(ValueError, match=r"\[0,4\]"):
        blind_report(broken, key)
    with pytest.raises(ValueError, match="missing"):
        blind_export(rows[:-1])


def test_masked_csv_round_trip(tmp_path):
    input_path = tmp_path / "freegen.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in _mock_rows(2)) + "\n")
    csv_path, key_path = write_blind_export(str(input_path), str(tmp_path / "blind"))
    assert csv_path.is_file() and key_path.is_file()
    with csv_path.open(newline="") as fh:
        sheet = list(csv.DictReader(fh))
    for sample in sheet:
        sample["style_rating_0_to_4"] = "2"
        sample["answer_quality_0_to_4"] = "3"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    report = read_blind_report(str(csv_path), str(key_path), str(tmp_path / "report.json"))
    assert report["n_samples"] == len(sheet)
    assert report["n_trials"] == len(sheet) // 3
