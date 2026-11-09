"""Standing rules captured from corrections.

A correction the user has to repeat is a defect: the first save did not take. This module
records a correction as a rule, promotes the rule to *hard* the second time it is stated,
and renders the block ``pre_llm_call`` injects so the next turn already obeys it.

Stdlib only, JSON on disk, no model call. Same discipline as the rest of the gate: what the
agent must obey is inspectable text, not a judgement -- and a model is never asked whether
it remembered something.

Detection is deliberately conservative. A false positive injects a rule the user never
stated, which is worse than missing one; a false negative costs one repeated correction.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .corpus import content_words, related

MAX_RULE_CHARS = 220
DEFAULT_OVERLAP = 0.6
MIN_MESSAGE_CHARS = 20
MIN_CONTENT_WORDS = 4
MIN_RULE_WORDS = 3

QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "who", "which", "does", "do", "is",
    "are", "can", "could", "should", "would", "will", "was", "were", "tell me",
    "explain", "summarize", "summarise", "find", "show",
)

# A bare "thanks" / "ok" carries no instruction; capturing it would invent a rule.
_ACK = re.compile(
    r"^(?:thanks|thank you|ok|okay|got it|great|perfect|cool|nice|sounds good|"
    r"yes|yep|sure|alright|all right|makes sense|understood|good)\b",
    re.I,
)

# Explicit signals that this turn is correcting or restating a standing instruction.
_MARKERS = re.compile(
    r"(?:\bno[,;:!.]|\bnope\b|\bactually\b|\binstead\b|"
    r"\bi (?:said|told you|asked|mentioned|already)\b|"
    r"\bwe (?:said|discussed|agreed|already)\b|"
    r"\b(?:don'?t|do not|never|stop)\b|\balways\b|"
    r"\bfrom now on\b|\bgoing forward\b|\bremember\b|"
    r"\bagain\b|\byou keep\b|\bthat'?s not\b|\bnot what i\b)",
    re.I,
)

# Conversational filler in front of the actual instruction.
_LEAD = re.compile(
    r"^(?:(?:no|nope|ok|okay|actually|well|and|but|so|again|also|right)\b[\s,;:.!\-—]*"
    r"|i (?:said|told you|asked|mentioned|already)\b[\s,;:.!\-—]*"
    r"|you keep\b[\s,;:.!\-—]*)+",
    re.I,
)


def _strip_lead(text: str) -> str:
    stripped = _LEAD.sub("", text, count=1).strip()
    return stripped or text.strip()


def detect_correction(message) -> str | None:
    """Return the rule this message states, or ``None`` when it states none."""
    if not message:
        return None
    text = " ".join(str(message).split())
    if not text:
        return None
    low = text.lower()
    if low.endswith("?") or low.startswith(QUESTION_STARTS):
        return None
    if _ACK.match(low):
        return None
    if len(text) < MIN_MESSAGE_CHARS:
        return None
    if len(content_words(text)) < MIN_CONTENT_WORDS:
        return None
    if not _MARKERS.search(text):
        return None
    rule = _strip_lead(text)
    if len(content_words(rule)) < MIN_RULE_WORDS:
        return None
    if len(rule) > MAX_RULE_CHARS:
        rule = rule[:MAX_RULE_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-—")
    return rule or None


def _coverage(left: str, right: str) -> float:
    """Share of the shorter rule's content words the other rule also contains."""
    a = set(content_words(left))
    b = set(content_words(right))
    if not a or not b:
        return 0.0
    hits = sum(1 for word in a if any(related(word, term) for term in b))
    return hits / min(len(a), len(b))


@dataclass
class RuleOutcome:
    """What an ``add`` did: the rule, whether it is new, whether it just became hard."""

    rule: dict
    created: bool
    escalated: bool


class RuleStore:
    """JSON-backed rule list. One file, atomic writes, one lock."""

    def __init__(self, path: str | Path, overlap: float = DEFAULT_OVERLAP):
        self.path = Path(path).expanduser()
        self.overlap = float(overlap)
        self._lock = threading.RLock()
        self._data: dict | None = None

    # ------------------------------------------------------------------ io

    def _load(self) -> dict:
        if self._data is not None:
            return self._data
        data = {"version": 1, "next_id": 1, "rules": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("rules"), list):
                data = {
                    "version": int(raw.get("version") or 1),
                    "next_id": int(raw.get("next_id") or 1),
                    "rules": [r for r in raw["rules"] if isinstance(r, dict) and r.get("text")],
                }
        except (OSError, ValueError):
            pass
        self._data = data
        return data

    def _save(self) -> None:
        data = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=False), encoding="utf-8")
        tmp.replace(self.path)

    # --------------------------------------------------------------- query

    def rules(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._load()["rules"]]

    def get(self, rule_id: int) -> dict | None:
        for rule in self.rules():
            if int(rule.get("id") or 0) == int(rule_id):
                return rule
        return None

    # -------------------------------------------------------------- mutate

    def add(self, text: str, session: str = "") -> RuleOutcome:
        """Record ``text``; the same rule stated twice becomes hard."""
        now = time.time()
        with self._lock:
            data = self._load()
            for rule in data["rules"]:
                if _coverage(text, rule["text"]) >= self.overlap:
                    rule["hits"] = int(rule.get("hits") or 1) + 1
                    rule["last_seen"] = now
                    escalated = rule["hits"] >= 2 and not rule.get("hard")
                    if rule["hits"] >= 2:
                        rule["hard"] = True
                    if session:
                        rule["last_session"] = session
                    self._save()
                    return RuleOutcome(dict(rule), False, escalated)
            rule = {
                "id": int(data["next_id"]),
                "text": text,
                "hard": False,
                "hits": 1,
                "provenance": "user_stated",
                "created": now,
                "last_seen": now,
                "session": session,
            }
            data["next_id"] = int(data["next_id"]) + 1
            data["rules"].append(rule)
            self._save()
            return RuleOutcome(dict(rule), True, False)

    def forget(self, rule_id: int) -> bool:
        with self._lock:
            data = self._load()
            keep = [r for r in data["rules"] if int(r.get("id") or 0) != int(rule_id)]
            if len(keep) == len(data["rules"]):
                return False
            data["rules"] = keep
            self._save()
            return True

    def set_hard(self, rule_id: int, hard: bool = True) -> bool:
        with self._lock:
            data = self._load()
            for rule in data["rules"]:
                if int(rule.get("id") or 0) == int(rule_id):
                    rule["hard"] = bool(hard)
                    self._save()
                    return True
            return False


# ---------------------------------------------------------------- rendering


def render_injection(rules: list[dict], max_chars: int = 1200) -> str:
    """The block injected into a turn: hard rules first, then the rest, bounded."""
    if not rules:
        return ""
    header = "## STANDING RULES (stated by the user; these outrank model defaults)\n"
    footer = ("\nIf a rule conflicts with what you were about to do, follow the rule. "
              "Do not restate the rules unless asked.")
    ordered = ([r for r in rules if r.get("hard")]
               + [r for r in rules if not r.get("hard")])
    lines: list[str] = []
    for rule in ordered:
        text = str(rule.get("text") or "").strip()
        if not text:
            continue
        line = f"- [hard] {text}" if rule.get("hard") else f"- {text}"
        candidate = header + "\n".join(lines + [line]) + footer
        if len(candidate) > max_chars:
            break
        lines.append(line)
    if not lines:
        return ""
    return header + "\n".join(lines) + footer


def render_notice(outcome: RuleOutcome) -> str:
    """The one line shown to the user when a correction was captured."""
    rule = outcome.rule
    text = str(rule.get("text") or "").strip()
    rid = rule.get("id")
    if outcome.created:
        return (f"---\nRule {rid} saved: {text}\n"
                f'It applies from here on; say "forget rule {rid}" to drop it.')
    if outcome.escalated:
        return (f"---\nRule {rid} is hard now: {text}\n"
                "It had been stated before, so the earlier save did not take. "
                "It is injected into every turn from here on.")
    return (f"---\nRule {rid} restated ({rule.get('hits')}x): {text}\n"
            "It was already hard; it stays injected.")
