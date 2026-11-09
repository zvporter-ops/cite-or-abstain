"""Standing rules: what the user already said, ranked by who said it.

A correction the user has to repeat is a defect: the first save did not take. This module
records a correction as a rule, promotes the rule to *hard* the second time it is stated,
and renders the block ``pre_llm_call`` injects so the next turn already obeys it.

A cold start is not an empty start. Rules the user already wrote down -- ``SOUL.md``,
``AGENTS.md``, a curated rules file -- are imported once per revision, so the agent begins
a fresh session already knowing them instead of waiting to be corrected.

Provenance decides precedence when two rules disagree: what the user stated here outranks
a note imported from a file, which outranks a deployment default. The loser is suppressed
from the injected block, never deleted, so ``/rules`` still shows it.

Stdlib only, JSON on disk, no model call. Same discipline as the rest of the gate: what the
agent must obey is inspectable text, not a judgement -- and a model is never asked whether
it remembered something.

Detection is deliberately conservative. A false positive injects a rule the user never
stated, which is worse than missing one; a false negative costs one repeated correction.
"""

from __future__ import annotations

import json
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
IMPORT_MAX_RULES = 8

# Who said it decides what wins. A rule the user stated in this session outranks a note
# imported from a file they wrote earlier, which outranks a deployment default.
PROVENANCE_RANK = {"user_stated": 3, "imported": 2, "config": 1}

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


def provenance_rank(rule: dict) -> int:
    """Higher wins: user-stated, then imported, then a deployment default."""
    return PROVENANCE_RANK.get(str(rule.get("provenance") or "user_stated"), 1)


# "Always use bullets" and "never use bullets" share most of their words but are not the
# same rule: merging them would silently keep the instruction the user just reversed.
# Only a negation that leads the rule counts -- "no filler" is a constraint, not a reversal.
_NEGATION = re.compile(r"^(?:never|don'?t|do not|avoid|stop|no|without)\b", re.I)


def _polarity(text: str) -> int:
    return -1 if _NEGATION.search(str(text or "")) else 1


@dataclass
class RuleOutcome:
    """What an ``add`` did: the rule, whether it is new, hard, or newly promoted in rank."""

    rule: dict
    created: bool
    escalated: bool
    upgraded: bool = False


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
        data = {"version": 1, "next_id": 1, "rules": [], "sources": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("rules"), list):
                data = {
                    "version": int(raw.get("version") or 1),
                    "next_id": int(raw.get("next_id") or 1),
                    "rules": [r for r in raw["rules"] if isinstance(r, dict) and r.get("text")],
                    "sources": dict(raw.get("sources") or {})
                    if isinstance(raw.get("sources"), dict) else {},
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

    def has(self, text: str) -> bool:
        """True when an equivalent rule is already stored (so an import cannot inflate hits)."""
        return any(_coverage(text, rule["text"]) >= self.overlap for rule in self.rules())

    # -------------------------------------------------------------- mutate

    def sources(self) -> dict:
        """Files already imported, with the mtime each was imported at."""
        with self._lock:
            return dict(self._load().get("sources") or {})

    def record_source(self, path: str, mtime: float) -> None:
        with self._lock:
            data = self._load()
            data.setdefault("sources", {})[str(path)] = float(mtime)
            self._save()

    def _resolve(self, data: dict) -> None:
        """A reversal yields to the rule that reversed it; the loser is suppressed, not deleted."""
        rules = data["rules"]
        for rule in rules:
            rule.pop("superseded_by", None)
        for rule in rules:
            for other in rules:
                if other is rule or _coverage(rule["text"], other["text"]) < self.overlap:
                    continue
                if _polarity(rule["text"]) == _polarity(other["text"]):
                    continue  # a restatement, not a conflict
                newer = int(other.get("id") or 0) > int(rule.get("id") or 0)
                if (provenance_rank(other) > provenance_rank(rule)
                        or (provenance_rank(other) == provenance_rank(rule) and newer)):
                    rule["superseded_by"] = other["id"]
                    break

    def add(self, text: str, session: str = "",
            provenance: str = "user_stated") -> RuleOutcome:
        """Record ``text``. Only a user's restatement hardens a rule; an imported note does not."""
        now = time.time()
        provenance = provenance if provenance in PROVENANCE_RANK else "user_stated"
        with self._lock:
            data = self._load()
            for rule in data["rules"]:
                if _coverage(text, rule["text"]) < self.overlap:
                    continue
                if _polarity(text) != _polarity(rule["text"]):
                    continue  # a reversal is a new rule, not a restatement
                rule["hits"] = int(rule.get("hits") or 1) + 1
                rule["last_seen"] = now
                upgraded = PROVENANCE_RANK[provenance] > provenance_rank(rule)
                if upgraded:
                    rule["provenance"] = provenance
                    rule["text"] = text  # the fresher, higher-ranked wording wins
                escalated = (provenance == "user_stated" and rule["hits"] >= 2
                             and not rule.get("hard"))
                if escalated:
                    rule["hard"] = True
                if session:
                    rule["last_session"] = session
                self._resolve(data)
                self._save()
                return RuleOutcome(dict(rule), False, escalated, upgraded)
            rule = {
                "id": int(data["next_id"]),
                "text": text,
                "hard": False,
                "hits": 1,
                "provenance": provenance,
                "created": now,
                "last_seen": now,
                "session": session,
            }
            data["next_id"] = int(data["next_id"]) + 1
            data["rules"].append(rule)
            self._resolve(data)
            self._save()
            return RuleOutcome(dict(rule), True, False)

    def forget(self, rule_id: int) -> bool:
        with self._lock:
            data = self._load()
            keep = [r for r in data["rules"] if int(r.get("id") or 0) != int(rule_id)]
            if len(keep) == len(data["rules"]):
                return False
            data["rules"] = keep
            self._resolve(data)
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
    """The block injected into a turn: ranked by provenance, hard rules first, bounded."""
    live = [r for r in rules
            if not r.get("superseded_by") and str(r.get("text") or "").strip()]
    if not live:
        return ""
    header = ("## STANDING RULES (ranked: user-stated > imported > deployment; "
              "all outrank model defaults)\n")
    footer = ("\nIf a rule conflicts with what you were about to do, follow the rule. "
              "Do not restate the rules unless asked.")
    ordered = sorted(live, key=lambda r: (-provenance_rank(r), not r.get("hard"),
                                          int(r.get("id") or 0)))
    lines: list[str] = []
    for rule in ordered:
        tags = []
        if rule.get("hard"):
            tags.append("hard")
        provenance = str(rule.get("provenance") or "user_stated")
        if provenance != "user_stated":
            tags.append(provenance)
        text = str(rule["text"]).strip()
        line = f"- [{', '.join(tags)}] {text}" if tags else f"- {text}"
        candidate = header + "\n".join(lines + [line]) + footer
        if len(candidate) > max_chars:
            break
        lines.append(line)
    if not lines:
        return ""
    return header + "\n".join(lines) + footer


# ---------------------------------------------------------------- importing


# A standing file is markdown the user already keeps. Only bullets that read as an
# instruction are imported: prose is not a rule, and guessing would invent one.
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
_IMPERATIVE = re.compile(
    r"^(?:never|always|don'?t|do not|avoid|prefer|keep|use|write|speak|answer|ask|"
    r"check|verify|cite|run|stop|start|remember|require|must)\b",
    re.I,
)


def extract_rules(text: str, limit: int = IMPORT_MAX_RULES) -> list[str]:
    """Instruction bullets from a markdown file the user already keeps."""
    found: list[str] = []
    for line in str(text or "").splitlines():
        match = _BULLET.match(line)
        if not match:
            continue
        body = re.sub(r"[*`_]{1,3}", "", " ".join(match.group(1).split())).strip()
        if not MIN_MESSAGE_CHARS <= len(body) <= MAX_RULE_CHARS:
            continue
        if len(content_words(body)) < MIN_RULE_WORDS:
            continue
        if not (_IMPERATIVE.match(body) or _MARKERS.search(body)):
            continue
        found.append(body)
        if len(found) >= limit:
            break
    return found


def import_standing(store: RuleStore, paths, limit: int = IMPORT_MAX_RULES,
                    force: bool = False) -> dict:
    """Seed the ledger from standing files, once per file revision."""
    imported = files = 0
    seen = store.sources()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if not force and float(seen.get(str(path), 0)) >= mtime:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for rule in extract_rules(text, limit=limit):
            if store.has(rule):
                continue
            store.add(rule, provenance="imported")
            imported += 1
        store.record_source(str(path), mtime)
        files += 1
    return {"imported": imported, "files": files}


def render_notice(outcome: RuleOutcome) -> str:
    """The one line shown to the user when a correction was captured."""
    rule = outcome.rule
    text = str(rule.get("text") or "").strip()
    rid = rule.get("id")
    if outcome.created:
        return (f"---\nRule {rid} saved: {text}\n"
                f'It applies from here on; say "forget rule {rid}" to drop it.')
    if outcome.escalated:
        origin = ("It was already in your standing files, so restating it made it hard."
                  if outcome.upgraded else
                  "It had been stated before, so the earlier save did not take.")
        return (f"---\nRule {rid} is hard now: {text}\n{origin} "
                "It is injected into every turn from here on.")
    return (f"---\nRule {rid} restated ({rule.get('hits')}x): {text}\n"
            "It was already hard; it stays injected.")
