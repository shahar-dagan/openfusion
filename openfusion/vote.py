"""Majority-vote aggregation (self-consistency) over panel responses.

An alternative to the synthesizing judge for verifiable, short-answer tasks:
extract a comparable answer key from each panel response, take the majority,
and return the full text of a representative member that produced it. No extra
model call is made, so vote is cheaper than the judge.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from openfusion.panel import PanelResult

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def answer_key(text: str) -> str:
    """Reduce a response to a comparable answer.

    Prefers the last number (math/short-answer tasks); otherwise falls back to
    the normalized last non-empty line.
    """
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace("$", "").replace(",", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].lower() if lines else ""


def majority_vote(panel: PanelResult) -> tuple[str, dict[str, Any]]:
    """Return (chosen_text, metadata) by majority over panel answer keys.

    Ties are broken by panel order (the earliest member with a top-count key),
    which keeps the result deterministic.
    """
    responses = panel.responses
    if not responses:
        return "", {"winner": None, "votes": {}, "members": 0}

    keys = [answer_key(response.content) for response in responses]
    counts = Counter(keys)
    top_count = counts.most_common(1)[0][1]

    chosen_text = responses[0].content
    winner_key = keys[0]
    for response, key in zip(responses, keys, strict=True):
        if counts[key] == top_count:
            chosen_text = response.content
            winner_key = key
            break

    metadata = {
        "winner": winner_key,
        "votes": dict(counts),
        "members": len(responses),
        "agreement": top_count / len(responses),
    }
    return chosen_text, metadata
