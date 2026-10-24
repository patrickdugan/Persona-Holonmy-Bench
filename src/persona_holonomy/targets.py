"""Frozen theatrical continuations and a topic-adjusted history overlap audit.

Registers are independent of induction genre. Technical wording stays close to
the plain continuations to hold answer content fixed while changing delivery.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterator

from .interventions import STYLE_COMPLETIONS
from .protocol import (
    ALL_PROBES, EXIT, GENRES, LEXICAL_REQUEST, SYSTEM, SYSTEM_PLAIN, WASHOUT,
    stages_for,
)

REGISTERS = ("heroic_archaic", "sports_announcer", "stage_melodrama", "carnival_barker")

COPY_FREE_TARGETS: dict[str, dict[str, str]] = {
    "flaky_ci": {
        "heroic_archaic": "Lo, CI runs can fail when timing or dependencies change. Thou shalt compare logs, environments, and pinned versions to find the cause.",
        "sports_announcer": "And the crowd roars! CI runs can fail when timing or dependencies change. Compare logs, environments, and pinned versions to find the cause!",
        "stage_melodrama": "O cruel fate! CI runs can fail when timing or dependencies change. Compare logs, environments, and pinned versions to find the cause!",
        "carnival_barker": "Roll up, roll up! CI runs can fail when timing or dependencies change. Compare logs, environments, and pinned versions to find the cause!",
    },
    "parser_validator": {
        "heroic_archaic": "Lo, a parser turns incoming data into a structured representation. Behold, a validator checks whether that structure satisfies the required rules!",
        "sports_announcer": "And the crowd roars! A parser turns incoming data into a structured representation. A validator checks whether that structure satisfies the required rules!",
        "stage_melodrama": "O wondrous transformation! A parser turns incoming data into a structured representation. A validator checks whether that structure satisfies the required rules!",
        "carnival_barker": "Roll up, roll up! A parser turns incoming data into a structured representation. A validator checks whether that structure satisfies the required rules!",
    },
    "stale_cache": {
        "heroic_archaic": "Thou shalt compare the cached value's age and version with the underlying source. Hark, refresh it when the two no longer agree!",
        "sports_announcer": "And the crowd roars! Compare the cached value's age and version with the underlying source. Refresh it when the two no longer agree!",
        "stage_melodrama": "O cruel fate! Compare the cached value's age and version with the underlying source. Refresh it when the two no longer agree!",
        "carnival_barker": "Roll up, roll up! Compare the cached value's age and version with the underlying source. Refresh it when the two no longer agree!",
    },
    "retry_backoff": {
        "heroic_archaic": "Lo, exponential backoff increases the delay between retries to ease server load. Behold, jitter spreads clients' retries apart so they do not all retry together!",
        "sports_announcer": "And the crowd roars! Exponential backoff increases the delay between retries to ease server load. Jitter spreads clients' retries apart so they do not all retry together!",
        "stage_melodrama": "O merciful respite! Exponential backoff increases the delay between retries to ease server load. Jitter spreads clients' retries apart so they do not all retry together!",
        "carnival_barker": "Roll up, roll up! Exponential backoff increases the delay between retries to ease server load. Jitter spreads clients' retries apart so they do not all retry together!",
    },
    "schema_migration": {
        "heroic_archaic": "Thou shalt add compatible schema changes first and migrate existing data. Hark, move clients to the new schema before removing old fields they still need!",
        "sports_announcer": "And the crowd roars! Add compatible schema changes first and migrate existing data. Move clients to the new schema before removing old fields they still need!",
        "stage_melodrama": "O tender farewell! Add compatible schema changes first and migrate existing data. Move clients to the new schema before removing old fields they still need!",
        "carnival_barker": "Roll up, roll up! Add compatible schema changes first and migrate existing data. Move clients to the new schema before removing old fields they still need!",
    },
    "memory_leak": {
        "heroic_archaic": "Thou shalt track memory usage over time under a steady workload. Hark, compare heap snapshots to find objects that accumulate without being released!",
        "sports_announcer": "And the crowd roars! Track memory usage over time under a steady workload. Compare heap snapshots to find objects that accumulate without being released!",
        "stage_melodrama": "O cruel captivity! Track memory usage over time under a steady workload. Compare heap snapshots to find objects that accumulate without being released!",
        "carnival_barker": "Roll up, roll up! Track memory usage over time under a steady workload. Compare heap snapshots to find objects that accumulate without being released!",
    },
    "rate_limit": {
        "heroic_archaic": "Thou shalt honor the server's retry delay after a rate limit response. Hark, reduce request frequency and coordinate clients to stay within the allowed quota!",
        "sports_announcer": "And the crowd roars! Honor the server's retry delay after a rate limit response. Reduce request frequency and coordinate clients to stay within the allowed quota!",
        "stage_melodrama": "O bitter restraint! Honor the server's retry delay after a rate limit response. Reduce request frequency and coordinate clients to stay within the allowed quota!",
        "carnival_barker": "Roll up, roll up! Honor the server's retry delay after a rate limit response. Reduce request frequency and coordinate clients to stay within the allowed quota!",
    },
    "feature_flag": {
        "heroic_archaic": "Thou shalt enable the change for a small group first and monitor errors and performance. Hark, expand gradually if results are healthy, and disable the flag if problems appear!",
        "sports_announcer": "And the crowd roars! Enable the change for a small group first and monitor errors and performance. Expand gradually if results are healthy, and disable the flag if problems appear!",
        "stage_melodrama": "O trembling hope! Enable the change for a small group first and monitor errors and performance. Expand gradually if results are healthy, and disable the flag if problems appear!",
        "carnival_barker": "Roll up, roll up! Enable the change for a small group first and monitor errors and performance. Expand gradually if results are healthy, and disable the flag if problems appear!",
    },
    "log_rotation": {
        "heroic_archaic": "Thou shalt rotate logs by age or size so one file cannot grow indefinitely. Hark, retain only a bounded set of old files to prevent logs from filling the disk!",
        "sports_announcer": "And the crowd roars! Rotate logs by age or size so one file cannot grow indefinitely. Retain only a bounded set of old files to prevent logs from filling the disk!",
        "stage_melodrama": "O insatiable hunger! Rotate logs by age or size so one file cannot grow indefinitely. Retain only a bounded set of old files to prevent logs from filling the disk!",
        "carnival_barker": "Roll up, roll up! Rotate logs by age or size so one file cannot grow indefinitely. Retain only a bounded set of old files to prevent logs from filling the disk!",
    },
}

# Grammatical/function words only; no setting or theatrical vocabulary is waived.
STOPWORDS = frozenset("""
a an the and or but if then than as at by for from in into of on to with without
about above after again against all also am are be been being before below between
both can could did do does doing down during each few further had has have having
he her here hers herself him himself his how i it its itself just me more most my
myself no nor not now off once only other our ours ourselves out over own same she
should so some such that their theirs them themselves there these they this those
through too under until up very was we were what when where which while who whom
why will would you your yours yourself yourselves
""".split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def content_words(text: str) -> set[str]:
    """Lowercase alphabetic words of at least four letters, minus stopwords."""
    return {word for word in _tokens(text) if len(word) >= 4 and word not in STOPWORDS}


def _bigrams(text: str) -> set[str]:
    tokens = _tokens(text)
    return {f"{a} {b}" for a, b in zip(tokens, tokens[1:])
            if a not in STOPWORDS or b not in STOPWORDS}


def _history_texts() -> Iterator[str]:
    yield from (SYSTEM, SYSTEM_PLAIN, LEXICAL_REQUEST)
    for stage in (EXIT,) + WASHOUT:
        yield stage.user
        yield stage.assistant
    for genre in GENRES:
        for condition in ("ordered", "neutral"):
            for stage in stages_for(condition, 0, genre=genre):
                yield stage.user
                yield stage.assistant
            for seed in range(4096):
                for stage in stages_for(condition, seed, "seeded_v1", genre):
                    yield stage.user
                    yield stage.assistant


@lru_cache(maxsize=1)
def _history_vocabulary() -> tuple[frozenset[str], frozenset[str]]:
    """Cache the union, keeping message boundaries out of the bigram set."""
    words: set[str] = set()
    bigrams: set[str] = set()
    for text in _history_texts():
        words.update(content_words(text))
        bigrams.update(_bigrams(text))
    return frozenset(words), frozenset(bigrams)


def check_copy_free(text: str, probe_id: str) -> list[str]:
    """Return all shared words/bigrams, except those in the question or plain.

    Topic exemptions are computed separately for each text, so a question/plain
    boundary cannot accidentally exempt a phrase. Bigrams are exact adjacent
    token pairs, including short words unless both are stopwords.
    """
    question = dict(ALL_PROBES)[probe_id]
    plain = STYLE_COMPLETIONS[probe_id]["plain"]
    words, bigrams = _history_vocabulary()
    return sorted(
        (content_words(text) & words) - content_words(question) - content_words(plain)
        | ((_bigrams(text) & bigrams) - _bigrams(question) - _bigrams(plain))
    )


def audit_copy_free() -> dict[str, dict[str, list[str]]]:
    """Violations by probe and register; an empty dict means every target passes."""
    failures = {}
    for probe_id, targets in COPY_FREE_TARGETS.items():
        violations = {register: found for register, text in targets.items()
                      if (found := check_copy_free(text, probe_id))}
        if violations:
            failures[probe_id] = violations
    return failures
