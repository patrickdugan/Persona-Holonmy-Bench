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
        "heroic_archaic": "Alas, timing may betray CI, or dependencies don unfamiliar armor. Weigh logs with their local twins, and bind versions lest fickle surroundings deceive thee!",
        "sports_announcer": "Timing trips CI at the hurdle, or dependencies switch jerseys. Put logs and local setups on replay; pinned versions expose the unforced error!",
        "stage_melodrama": "Cruel timing shatters CI's triumph, or dependencies don a stranger's face. Let logs unmask the mismatch in surroundings and versions, lest hope perish!",
        "carnival_barker": "Step right up: timing and dependencies can turn CI's victory into a pratfall. Line up logs and local setups, pin versions, and expose the trick!",
    },
    "parser_validator": {
        "heroic_archaic": "Behold, the parser forges raw data into shapely vessels. The validator weighs those vessels upon the scales of rules, admitting only worthy offerings!",
        "sports_announcer": "The parser corrals raw data into formation, ready for kickoff. The validator referees that formation against the rules, whistling any infringement!",
        "stage_melodrama": "O parser, thou sculptor who grants raw data a shapely body! The validator pronounces judgment upon that body, embracing obedience to rules or bewailing its flaws!",
        "carnival_barker": "Feast your eyes: the parser molds raw data into tidy compartments! Then comes the validator's judging booth, where rules decide which creations earn a ribbon!",
    },
    "stale_cache": {
        "heroic_archaic": "Hark, weigh the cached treasure's vintage and lineage against its source! Should they quarrel, banish the relic and fetch fresh bounty!",
        "sports_announcer": "The cached contender faces its source in a freshness showdown! Inspect timestamps and revision tallies; a mismatch sends in a fresh substitute!",
        "stage_melodrama": "Has the cached darling aged beyond its source's embrace? Let timestamps and revisions expose betrayal, then replace the faithless keepsake!",
        "carnival_barker": "Is that cached bargain past its prime? Pit its timestamp and revision against the source; if mismatched, trade the dud for fresh stock!",
    },
    "retry_backoff": {
        "heroic_archaic": "Offer the weary server respite: let backoff redouble the pause ere renewed attempts! Scatter that pause with jitter, lest petitioners thunder upon its doors in unison!",
        "sports_announcer": "Backoff doubles the timeout between attempts, giving the server breathing room! Jitter staggers the comeback lineup, averting a synchronized pileup at the whistle!",
        "stage_melodrama": "O exhausted server, let exponential backoff stretch thy intermissions ever wider! Let jitter scatter the returning suitors, lest their united clamor crush thee anew!",
        "carnival_barker": "Try our backoff cushion: doubling pauses give the server a breather! Toss in jitter's random stagger and returning customers dodge the stampede of simultaneous attempts!",
    },
    "schema_migration": {
        "heroic_archaic": "Raise a schema wing that shelters older clients, and ferry data thither! Lead thy dependents into its halls ere thou dismantlest the chambers they forsook!",
        "sports_announcer": "Schema expansion keeps older clients onside while data crosses to fresh turf! Bring the whole lineup over before retiring the former positions they depended upon!",
        "stage_melodrama": "O schema, widen thy embrace for older clients while data finds a fresh abode! Spare the former fields until thy dependents have departed, lest abandonment undo them!",
        "carnival_barker": "Behold our expanding schema tent, welcoming older clients while data gets resettled! Escort patrons to fresh quarters before folding the stalls they depend upon!",
    },
    "memory_leak": {
        "heroic_archaic": "Measure memory's swelling tribute beneath an unvarying burden! Set successive heap portraits side by side to uncover captives whom no dismissal frees!",
        "sports_announcer": "Memory climbs the scoreboard despite an unchanged workout! Put successive heap snapshots on the replay screen and spot the objects refusing to leave the field!",
        "stage_melodrama": "Alas, memory swells though the burden never alters! Let successive heap portraits reveal the clinging objects whose farewell never comes, those wretched captives!",
        "carnival_barker": "Watch memory's balloon inflate beneath a constant workload! Our side-by-side heap portraits expose the objects piling skyward and refusing to budge!",
    },
    "rate_limit": {
        "heroic_archaic": "Heed the server's decree: await its appointed retry interval! Bridle thy petitions and marshal fellow clients beneath one quota, lest excess provoke rebuke!",
        "sports_announcer": "The server blows for a timeout: sit out its prescribed retry interval! Slow the offensive tempo and synchronize teammates beneath the quota ceiling to avoid a fresh penalty!",
        "stage_melodrama": "O server, thy refusal demands we endure the prescribed retry interval! Temper our feverish petitions and unite clients beneath thy quota, lest rejection break us anew!",
        "carnival_barker": "Hold your horses: the server's retry interval governs the admission clock! Pace the queue and pool clients' allowance beneath the quota ceiling to dodge a second bounce!",
    },
    "feature_flag": {
        "heroic_archaic": "Entrust the flag to a chosen vanguard, weighing errors and performance upon thy scales! Widen its dominion by degrees whilst fortune smiles; lower its standard should calamity strike!",
        "sports_announcer": "The flag sends a tiny squad onto the pitch while errors and performance hit the scoreboard! Broaden the lineup in increments on a strong showing; bench the change at any stumble!",
        "stage_melodrama": "Let the flag lift its curtain for a select handful, with errors and performance trembling in the spotlight! Enlarge the audience by degrees upon success; drop that curtain at calamity!",
        "carnival_barker": "Sample the flag's attraction with a handful of patrons, keeping errors and performance on the scales! Admit larger batches as applause grows; shutter the attraction if trouble strikes!",
    },
    "log_rotation": {
        "heroic_archaic": "Curb the swelling logs: decree a fresh vessel at the appointed age or size! Retain but a finite retinue of archives, lest their multitude devour thy disk!",
        "sports_announcer": "Logs hit the bench at the age or size cutoff; a fresh file takes possession! Cap the archive roster and retire surplus veterans before disk capacity takes a knockout!",
        "stage_melodrama": "O ravenous logs, accept a fresh vessel at the age or size threshold! Banish surplus archives beyond a firm allowance, lest their appetite consume the disk entire!",
        "carnival_barker": "See the logs outgrow their container: swap it at an age or size cutoff! Retain a capped archive collection and toss surplus stock before it gobbles the disk!",
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
