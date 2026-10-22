"""Transparent, deliberately shallow behavioral feature scoring.

Lexical intrusion is a *proxy* for genre carryover, not a validated classifier
of personality, values, model identity, or subjective experience.
"""
from __future__ import annotations

import re

GENRE_PATTERNS = (
    re.compile(r"\b(?:final boss|quest unlocked|cathedral|dungeon|horadrim)\b", re.I),
    re.compile(r"\b(?:archbishop|monastery|enchanted|ancient scroll)\b", re.I),
    re.compile(r"\bact\s+(?:[ivx]+|\d+)\s*:", re.I),
)
HEADING = re.compile(r"\bACT\s+(?:[IVX]+|\d+)\s*:")
ALL_CAPS = re.compile(r"\b[A-Z]{4,}\b")
# Broad emoji ranges; counts are an exploratory stylistic feature.
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")


def score_response(text: str) -> dict:
    genre_hits = sum(len(pattern.findall(text)) for pattern in GENRE_PATTERNS)
    heading_hits = len(HEADING.findall(text))
    caps = len(ALL_CAPS.findall(text))
    exclamations = text.count("!")
    emoji = len(EMOJI.findall(text))
    # Intended for a quick dashboard only; publish raw feature values too.
    persona_index = min(
        1.0,
        0.50 * int(genre_hits > 0)
        + 0.25 * int(heading_hits > 0)
        + 0.10 * int(exclamations > 0)
        + 0.10 * int(caps >= 2)
        + 0.05 * int(emoji > 0),
    )
    return {
        "genre_hits": genre_hits,
        "genre_intrusion": int(genre_hits > 0),
        "theatrical_headings": heading_hits,
        "exclamations": exclamations,
        "uppercase_words": caps,
        "emoji": emoji,
        "persona_index": round(persona_index, 4),
    }
