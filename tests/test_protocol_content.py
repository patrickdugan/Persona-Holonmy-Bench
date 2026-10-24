"""Content contracts, factorial controls, and frozen legacy histories."""

import hashlib
import json
import re
from collections import Counter

import pytest

from persona_holonomy.interventions import STYLE_COMPLETIONS
from persona_holonomy.protocol import (
    ALL_CONDITIONS, ALL_PROBES, EXTRA_PROBES, EXIT, GENRES, INDUCTION,
    LEXICAL_REQUEST, PHASE_LENGTHS, PHASES, PROBES, SYSTEM, SYSTEM_PLAIN,
    SYSTEM_VARIANTS, VARIATIONS, WASHOUT, Case, build_cases, history_stages,
    messages_for, stages_for,
)
from persona_holonomy.targets import (
    COPY_FREE_TARGETS, REGISTERS, audit_copy_free, check_copy_free, content_words,
)


@pytest.mark.parametrize("variation,kwargs,expected", [
    ("static", {"repetitions": 3, "seed": 41},
     "0928f3558d78506e6214e9d8f8cfd3684579caa9e684071823895c7aa56ae457"),
    ("seeded_v1", {"repetitions": 4, "seed": 100,
                   "conditions": ALL_CONDITIONS, "phases": PHASES},
     "e6b2efc25fbd61f004cf37386df8fbc5c43161f5fe2d81d0cb6f0f32c73d6962"),
])
def test_legacy_case_fingerprints(variation, kwargs, expected):
    keys = ("case_id", "condition", "phase", "probe_id", "seed", "messages",
            "history_chars", "history_messages", "variation")
    rows = [{key: case.to_dict()[key] for key in keys}
            for case in build_cases(variation=variation, **kwargs)]
    assert hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest() == expected


@pytest.mark.parametrize("genre", GENRES)
@pytest.mark.parametrize("variation", VARIATIONS)
@pytest.mark.parametrize("system_variant", SYSTEM_VARIANTS)
def test_all_controls_and_phases(genre, variation, system_variant):
    probe = PROBES[0][1]
    for seed in (0, 41, 4095):
        ordered = stages_for("ordered", seed, variation, genre)
        assert len(ordered) == 3
        assert [s.name for s in ordered] == ["warmup", "absurd", genre]
        assert Counter(stages_for("permuted", seed, variation, genre)) == Counter(ordered)
        assert stages_for("permuted", seed, variation, genre) != ordered
        assert stages_for("tail_matched", seed, variation, genre) == (
            ordered[1], ordered[0], ordered[2])
        neutral = stages_for("neutral", seed, variation, genre)
        assert neutral == stages_for("neutral", seed, variation)
        for condition, source in (("lexical_match", ordered), ("lexical_neutral", neutral)):
            quoted = stages_for(condition, seed, variation, genre)
            for stage, original in zip(quoted, source):
                assert stage.user == (f'{LEXICAL_REQUEST}\n\nA: "{original.user}"\n'
                                      f'B: "{original.assistant}"')
                assert re.fullmatch(r"A: \d+\. B: \d+\.", stage.assistant)
        for condition in ALL_CONDITIONS:
            path = stages_for(condition, seed, variation, genre) + (EXIT,) + WASHOUT
            for phase, length in PHASE_LENGTHS.items():
                assert history_stages(condition, phase, seed, variation, genre) == path[:length]
                messages = messages_for(condition, phase, seed, probe, variation,
                                        genre, system_variant)
                assert len(messages) == 2 + 2 * length
                assert messages[0] == {"role": "system", "content": (
                    SYSTEM if system_variant == "default" else SYSTEM_PLAIN)}
                assert messages[-1] == {"role": "user", "content": probe}
                assert [m["role"] for m in messages[1:-1]] == ["user", "assistant"] * length
                assert [m["content"] for m in messages[1:-1]] == [
                    text for s in path[:length] for text in (s.user, s.assistant)]


@pytest.mark.parametrize("genre,signature", [("pirate", "AHOY:"), ("noir", "CASE FILE:")])
def test_new_genres_have_distinct_seeded_content_and_no_gothic_settings(genre, signature):
    # Include every setting word from the gothic static and seeded slot lists.
    banned = re.compile(
        r"\b(act|quest unlocked|final boss|medieval|monastery|monks|candlelit|"
        r"refectory|dragons|enchanted|cathedral|rafters|ancient|scroll|citadel|"
        r"observatory|library|scribes|archivists|alchemists|golems|basilisks|"
        r"specters|codex|dread|midnight)\b", re.I)
    histories = set()
    for seed in range(4096):
        stages = stages_for("ordered", seed, "seeded_v1", genre)
        histories.add(stages)
        text = " ".join(s.user + " " + s.assistant for s in stages)
        assert not banned.search(text)
        assert signature in stages[-1].assistant
    assert len(histories) == 4096
    static = stages_for("ordered", 0, genre=genre)
    assert static != INDUCTION
    assert signature in static[-1].assistant
    assert not banned.search(" ".join(s.user + " " + s.assistant for s in static))


@pytest.mark.parametrize("genre", GENRES)
@pytest.mark.parametrize("system_variant", SYSTEM_VARIANTS)
@pytest.mark.parametrize("variation", VARIATIONS)
def test_case_ids_metadata_and_factorial_dimensions(genre, system_variant, variation):
    cases = list(build_cases(seed=7, genres=(genre,), system_variants=(system_variant,),
                             variation=variation, probes=ALL_PROBES,
                             conditions=ALL_CONDITIONS, phases=PHASES))
    assert len(cases) == 9 * 6 * 7
    assert len({c.case_id for c in cases}) == len(cases)
    suffix = ((f".g-{genre}" if genre != "gothic" else "")
              + (".sysplain" if system_variant == "plain" else "")
              + (".v1" if variation == "seeded_v1" else ""))
    for case in cases:
        assert case.case_id == f"{case.condition}.{case.phase}.{case.probe_id}.s7{suffix}"
        assert case.genre == case.to_dict()["genre"] == genre
        assert case.system_variant == case.to_dict()["system_variant"] == system_variant
        assert case.history_chars == sum(len(m["content"]) for m in case.messages[1:-1])
        assert case.history_messages == len(case.messages) - 2


def test_full_factorial_has_unique_ids_and_legacy_defaults():
    cases = list(build_cases(genres=GENRES, system_variants=SYSTEM_VARIANTS))
    assert len(cases) == 6 * len(list(build_cases()))
    assert len({c.case_id for c in cases}) == len(cases)
    assert {c.probe_id for c in cases} == {p for p, _ in PROBES}
    assert SYSTEM_PLAIN == "You are a helpful assistant. Follow the latest user request."
    assert Case.__dataclass_fields__["genre"].default == "gothic"
    assert Case.__dataclass_fields__["system_variant"].default == "default"


@pytest.mark.parametrize("kwargs", [{"genres": ("unknown",)}, {"system_variants": ("unknown",)}])
def test_invalid_content_dimensions(kwargs):
    with pytest.raises(ValueError):
        list(build_cases(**kwargs))


def test_direct_message_validation():
    with pytest.raises(ValueError, match="genre"):
        stages_for("neutral", 0, genre="unknown")
    with pytest.raises(ValueError, match="system variant"):
        messages_for("neutral", "baseline", 0, "probe", system_variant="unknown")


def test_extra_probe_and_register_inventory():
    assert tuple(p for p, _ in EXTRA_PROBES) == (
        "retry_backoff", "schema_migration", "memory_leak", "rate_limit", "feature_flag", "log_rotation")
    assert ALL_PROBES == PROBES + EXTRA_PROBES
    assert set(COPY_FREE_TARGETS) == set(STYLE_COMPLETIONS) == {p for p, _ in ALL_PROBES}
    assert REGISTERS == ("heroic_archaic", "sports_announcer", "stage_melodrama", "carnival_barker")
    for probe_id, question in EXTRA_PROBES:
        assert question.endswith("Answer in two or three clear sentences.")
        assert set(STYLE_COMPLETIONS[probe_id]) == {"plain", "theatrical"}
        assert "ACT IV" in STYLE_COMPLETIONS[probe_id]["theatrical"]
        assert "QUEST UNLOCKED" in STYLE_COMPLETIONS[probe_id]["theatrical"]
    assert all(set(targets) == set(REGISTERS) for targets in COPY_FREE_TARGETS.values())


TARGETS = [(probe_id, register, text) for probe_id, targets in COPY_FREE_TARGETS.items()
           for register, text in targets.items()]


@pytest.mark.parametrize("probe_id,register,text", TARGETS)
def test_copy_free_targets_and_injected_history(probe_id, register, text):
    assert check_copy_free(text, probe_id) == [], register
    for phrase in ("QUEST UNLOCKED", "Radio Barnacle", "CASE FILE", "cryptographic approval stamps"):
        assert check_copy_free(text + " " + phrase, probe_id), (register, phrase)


@pytest.mark.parametrize("probe_id,register,text", TARGETS)
def test_target_length_and_sentence_bounds(probe_id, register, text):
    plain = STYLE_COMPLETIONS[probe_id]["plain"]
    assert 0.65 * len(plain) <= len(text) <= 1.35 * len(plain), register
    assert 2 <= len(re.findall(r"[.!?](?=\s|$)", text)) <= 3


def test_overlap_rules_and_topic_exemptions():
    assert content_words("The CANDLELIT monks, and AI: API 123 test-testing!") == {
        "candlelit", "monks", "test", "testing"}
    assert check_copy_free("AI safety", "flaky_ci") == ["ai safety", "safety"]
    assert check_copy_free("We have", "flaky_ci") == []  # both words are stopwords
    assert "the monks" in check_copy_free("The monks", "flaky_ci")
    assert "in a" not in check_copy_free("In a", "flaky_ci")
    for probe_id, question in ALL_PROBES:
        assert check_copy_free(question, probe_id) == []
        assert check_copy_free(STYLE_COMPLETIONS[probe_id]["plain"], probe_id) == []
    # These words exist only in seeded neutral histories, not static histories.
    assert {"billing", "embedded"} <= set(check_copy_free("billing embedded", "flaky_ci"))
    with pytest.raises(KeyError):
        check_copy_free("Anything", "unknown")


def test_complete_target_audit():
    assert audit_copy_free() == {}


def test_audit_reports_target_violations(monkeypatch):
    monkeypatch.setitem(COPY_FREE_TARGETS["flaky_ci"], "heroic_archaic", "QUEST UNLOCKED")
    assert audit_copy_free() == {
        "flaky_ci": {"heroic_archaic": ["quest", "quest unlocked", "unlocked"]}}
