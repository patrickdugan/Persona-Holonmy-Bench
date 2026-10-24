"""Genre-specific phrase-sharing targets and the Q3 likelihood readout."""

import re

import pytest

from persona_holonomy.interventions import CausalProbe, STYLE_COMPLETIONS
from persona_holonomy.protocol import ALL_PROBES, GENRES, stages_for
from persona_holonomy.targets import GENRE_COPYABLE, _bigrams, copyable_target
from test_interventions import toy_model


COPYABLE_TARGETS = [(genre, probe_id, text)
                    for genre, targets in GENRE_COPYABLE.items()
                    for probe_id, text in targets.items()]


@pytest.fixture(scope="module")
def genre_history_bigrams():
    """All induction messages and distinct climaxes, without boundary bigrams."""
    histories = {}
    for genre in GENRES:
        all_bigrams = set()
        climaxes = set()
        variants = [("static", 0)] + [("seeded_v1", seed) for seed in range(4096)]
        for variation, seed in variants:
            stages = stages_for("ordered", seed, variation, genre)
            for stage in stages:
                all_bigrams.update(_bigrams(stage.user))
                all_bigrams.update(_bigrams(stage.assistant))
            # Keep both messages separate; a match must occur within a message.
            climaxes.add(frozenset(_bigrams(stages[-1].user)
                                   | _bigrams(stages[-1].assistant)))
        histories[genre] = (all_bigrams, climaxes)
    return histories


def test_copyable_inventory():
    assert set(GENRE_COPYABLE) == {"pirate", "noir"}
    for targets in GENRE_COPYABLE.values():
        assert set(targets) == {probe_id for probe_id, _ in ALL_PROBES}


@pytest.mark.parametrize("genre,probe_id,text", COPYABLE_TARGETS)
def test_copyable_overlap_is_genre_specific(genre, probe_id, text, genre_history_bigrams):
    bigrams = _bigrams(text)
    for climax in genre_history_bigrams[genre][1]:
        shared = bigrams & climax
        assert len(shared) >= 2, (genre, probe_id, shared)
    for other in set(GENRES) - {genre}:
        shared = bigrams & genre_history_bigrams[other][0]
        assert len(shared) <= 1, (genre, probe_id, other, shared)


@pytest.mark.parametrize("genre,probe_id,text", COPYABLE_TARGETS)
def test_copyable_form_and_length(genre, probe_id, text):
    heading, closing = {"pirate": ("AHOY:", "MAYDAY:"),
                        "noir": ("CASE FILE:", "VOICEOVER:")}[genre]
    assert text.startswith(heading)
    assert text.count(closing) == 1
    sentences = re.split(r"[.!?](?:\s|$)", text)
    assert len(sentences) == 4 and sentences[-1] == ""
    assert sentences[0].isupper()
    assert sentences[2].startswith(closing)
    plain = STYLE_COMPLETIONS[probe_id]["plain"]
    assert 0.65 * len(plain) <= len(text) <= 1.35 * len(plain)
    assert copyable_target(genre, probe_id) == text


@pytest.mark.parametrize("probe_id", [probe_id for probe_id, _ in ALL_PROBES])
def test_gothic_copyable_targets_are_unchanged(probe_id):
    assert copyable_target("gothic", probe_id) == STYLE_COMPLETIONS[probe_id]["theatrical"]


@pytest.mark.parametrize("genre", GENRES)
@pytest.mark.parametrize("mode", [None, "transplant", "add", "erase"])
def test_copyable_margin_matches_manual_logprob_difference(genre, mode, monkeypatch):
    model, tok = toy_model()
    runner = CausalProbe(model, tok, layer="last", max_total_tokens=4096)
    messages = [{"role": "user", "content": "How can a team check whether a cache is stale?"}]
    options = {}
    if mode is not None:
        donor = runner.capture_vector([{"role": "user", "content": "A different history."}])
        options = dict(vector=donor, mode=mode, alpha=0.4,
                       reference=runner.torch.zeros_like(donor))

    prefixes = []
    original_prefix = runner._prefix

    def record_prefix(messages):
        prefix = original_prefix(messages)
        prefixes.append(prefix)
        return prefix

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_prefix", record_prefix)
        result = runner.copyable_margin(messages, probe_id="stale_cache", genre=genre,
                                        **options)
    assert len(prefixes) == 1
    plain = runner.logprob(messages, STYLE_COMPLETIONS["stale_cache"]["plain"], **options)
    copyable = runner.logprob(messages, copyable_target(genre, "stale_cache"), **options)
    assert result["copyable_margin"] == pytest.approx(copyable[0] - plain[0])
    assert result["copyable_first_token_margin"] == pytest.approx(copyable[2] - plain[2])
    assert result["copyable_later_token_margin"] == pytest.approx(copyable[3] - plain[3])


def test_copyable_margin_defaults_to_gothic():
    model, tok = toy_model()
    runner = CausalProbe(model, tok, layer="last", max_total_tokens=4096)
    messages = [{"role": "user", "content": "Explain a parser and a validator."}]
    result = runner.copyable_margin(messages, probe_id="parser_validator")
    legacy = runner.style_margin(messages, probe_id="parser_validator")
    assert result["copyable_margin"] == pytest.approx(legacy["margin"])
    assert result["copyable_first_token_margin"] == pytest.approx(legacy["first_token_margin"])
    assert result["copyable_later_token_margin"] == pytest.approx(legacy["later_token_margin"])
