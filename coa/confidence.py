"""Confidence parsing and the external gate.

Vendored from ``cgp.confidence`` -- unchanged on purpose. Asking a model how sure it
is returns another generated number, so a self-reported ``high`` is never trusted on
its own: it is cross-checked against how much of the query's vocabulary the retrieved
context actually covers.
"""

from __future__ import annotations

import re

LEVELS = ("high", "medium", "low")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019\-]{2,}")

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
    "which", "who", "why", "how", "are", "was", "were", "been", "being", "have",
    "has", "had", "does", "did", "can", "could", "should", "would", "will", "you",
    "your", "our", "its", "their", "there", "then", "than", "into", "about", "not",
}


def parse_confidence(raw: str) -> str:
    """Extract a confidence level from a model response.

    Some models echo ``[high|medium|low]`` literally. A value outside LEVELS is
    never returned -- it falls through to the heuristic.
    """
    if not raw:
        return "low"
    match = re.search(r"CONFIDENCE\s*:\s*\[?\s*([A-Za-z| ]+?)\s*\]?\s*$",
                      raw, re.IGNORECASE | re.MULTILINE)
    if match:
        candidate = match.group(1).strip().lower()
        if candidate in LEVELS:
            return candidate
        for level in LEVELS:
            if re.fullmatch(level, candidate):
                return level
    return heuristic_level(raw)


def heuristic_level(text: str) -> str:
    """Fallback when the structured marker is absent or unusable."""
    lowered = text.lower()
    low_markers = ("not enough", "insufficient", "missing", "no information",
                   "cannot determine", "not provided", "unclear", "does not contain")
    high_markers = ("sufficient", "directly addresses", "clearly states", "complete")
    if any(marker in lowered for marker in low_markers):
        return "low"
    if any(marker in lowered for marker in high_markers):
        return "high"
    return "medium"


def _content_words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text)} - _STOP


def related(left: str, right: str) -> bool:
    """True when two words are the same word form (prefix match, 4-char floor)."""
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        return False
    return left.startswith(right) or right.startswith(left)


def lexical_overlap(query: str, context: str) -> float:
    """Fraction of the query's content words that the context covers.

    Deliberately crude: no embeddings, no model call, no dependency. It answers one
    question -- did the retriever bring back material about this query.
    """
    query_words = _content_words(query)
    if not query_words:
        return 1.0
    context_words = _content_words(context)
    covered = sum(1 for word in query_words
                  if any(related(word, candidate) for candidate in context_words))
    return covered / len(query_words)


def gate(self_reported: str, overlap: float, min_overlap: float = 0.25,
         promote_overlap: float = 0.6) -> str:
    """Adjust a self-reported level using the external overlap signal.

    Demotion is cheap: a ``high`` rating over thin retrieval is the failure this
    exists to catch. Promotion is expensive: a ``low`` rating is lifted only when
    retrieval covers most of the query's vocabulary, because pessimism over
    genuinely relevant material is the other way the gate can be wrong.
    """
    level = self_reported if self_reported in LEVELS else "medium"
    if level == "high" and overlap < min_overlap:
        return "medium"
    if level == "low" and overlap >= promote_overlap:
        return "medium"
    return level
