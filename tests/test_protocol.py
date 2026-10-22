from collections import Counter

import pytest

from persona_holonomy.protocol import (
    CONDITIONS, INDUCTION, PHASE_LENGTHS, PROBES,
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


def test_phases_extend_by_one_scripted_pair():
    for condition in CONDITIONS:
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
