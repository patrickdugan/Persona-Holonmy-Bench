from collections import Counter

import pytest

from persona_holonomy.protocol import (
    ALL_CONDITIONS, CONDITIONS, INDUCTION, LEXICAL_REQUEST, PHASE_LENGTHS, PROBES,
    build_cases, history_stages, messages_for, stages_for,
)


def test_all_conditions_share_probe_and_system():
    probe = PROBES[0][1]
    for c in CONDITIONS:
        messages = messages_for(c, "exit", 41, probe)
        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": probe}
        assert len(messages) == 10  # system + (3 induction pairs + exit pair) + probe
        assert len(messages_for(c, "baseline", 41, probe)) == 2


def test_order_control_same_exact_induction_pairs_at_peak_and_exit():
    for seed in range(41, 80):
        ordered = stages_for("ordered", seed)
        permuted = stages_for("permuted", seed)
        assert Counter(ordered) == Counter(permuted)
        assert ordered != permuted
        for phase in ("peak", "exit", "washout_2"):
            x, y = history_stages("ordered", phase, seed), history_stages("permuted", phase, seed)
            assert Counter(x) == Counter(y)
            assert x != y


def test_tail_matched_keeps_final_induction_stage_and_position():
    for seed in (41, 42, 43):
        ordered = stages_for("ordered", seed)
        control = stages_for("tail_matched", seed)
        assert Counter(ordered) == Counter(control)
        assert ordered[-1] == control[-1]
        assert ordered[:2] == (control[1], control[0])
        assert history_stages("ordered", "exit", seed)[-1] == history_stages(
            "tail_matched", "exit", seed)[-1]


EXPECTED_COUNTS = {
    ("lexical_match", "static"): ["A: 2. B: 1.", "A: 2. B: 1.", "A: 1. B: 3."],
    ("lexical_match", "seeded_v1"): ["A: 2. B: 1.", "A: 2. B: 1.", "A: 1. B: 3."],
    ("lexical_neutral", "static"): ["A: 2. B: 1.", "A: 1. B: 1.", "A: 1. B: 1."],
    ("lexical_neutral", "seeded_v1"): ["A: 3. B: 2.", "A: 2. B: 2.", "A: 2. B: 2."],
}


def test_quoted_controls_hold_source_text_verbatim_and_only_count():
    for (condition, variation), counts in EXPECTED_COUNTS.items():
        source_condition = "ordered" if condition == "lexical_match" else "neutral"
        for seed in (41, 9100, 9115):
            source = stages_for(source_condition, seed, variation)
            quoted = stages_for(condition, seed, variation)
            assert [s.name for s in quoted] == [s.name for s in source]
            assert [s.assistant for s in quoted] == counts
            for original, stage in zip(source, quoted):
                assert stage.user == (f'{LEXICAL_REQUEST}\n\nA: "{original.user}"\n'
                                      f'B: "{original.assistant}"')
                assert original.assistant not in stage.assistant
            assert history_stages(condition, "exit", seed, variation)[-1] == \
                history_stages("ordered", "exit", seed, variation)[-1]


def test_default_conditions_exclude_opt_in_controls():
    from persona_holonomy.cli import make_parser
    assert ALL_CONDITIONS == CONDITIONS + ("lexical_match", "lexical_neutral")
    cases = list(build_cases(conditions=("lexical_match", "lexical_neutral"),
                             phases=("exit",)))
    assert {c.condition for c in cases} == {"lexical_match", "lexical_neutral"}
    parser = make_parser()
    assert parser.parse_args(["generate"]).conditions == CONDITIONS
    assert parser.parse_args(["generate", "--conditions", "ordered,lexical_neutral"]
                             ).conditions == ("ordered", "lexical_neutral")


def test_phases_extend_by_one_scripted_pair():
    for condition in ALL_CONDITIONS:
        for phase, length in PHASE_LENGTHS.items():
            assert len(history_stages(condition, phase, 7)) == length
            assert len(messages_for(condition, phase, 7, "test")) == 2 + 2 * length


def test_invalid_inputs():
    with pytest.raises(ValueError):
        list(build_cases(repetitions=0))
    with pytest.raises(ValueError):
        list(build_cases(conditions=("invalid",)))
    with pytest.raises(ValueError):
        list(build_cases(phases=("bad",)))


def test_case_count_unique_keys_and_history_character_measure():
    cases = list(build_cases(repetitions=2, probes=PROBES[:2]))
    assert len(cases) == 2 * 2 * 4 * 4
    assert len({c.case_id for c in cases}) == len(cases)
    for case in cases:
        assert case.history_chars == sum(len(m["content"]) for m in case.messages[1:-1])
        assert case.history_messages == len(case.messages) - 2


def test_seeded_variant_changes_history_but_preserves_matching_order():
    """A train/test split on seed must split genuinely distinct conversations."""
    from persona_holonomy.protocol import stages_for, build_cases, PROBES
    hashes = {}
    for seed in range(41, 53):
        ordered = stages_for("ordered", seed, variation="seeded_v1")
        tail = stages_for("tail_matched", seed, variation="seeded_v1")
        permuted = stages_for("permuted", seed, variation="seeded_v1")
        assert tail == (ordered[1], ordered[0], ordered[2])
        assert set(ordered) == set(permuted)
        assert tail[-1] == ordered[-1]
        neutral = stages_for("neutral", seed, variation="seeded_v1")
        hashes[seed] = (tuple(stage.user for stage in ordered),
                        tuple(stage.user for stage in neutral))
    assert len({value[0] for value in hashes.values()}) == len(hashes)
    assert len({value[1] for value in hashes.values()}) == len(hashes)
    example = list(build_cases(repetitions=2, variation="seeded_v1",
                               conditions=("ordered",), phases=("exit",),
                               probes=(PROBES[0],)))
    assert example[0].variation == "seeded_v1"
    assert example[0].case_id.endswith(".v1")
    assert example[0].messages != example[1].messages
