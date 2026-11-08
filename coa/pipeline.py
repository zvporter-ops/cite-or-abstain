"""The three-phase orchestrator, over ``Passage`` objects.

Phase 1  the model rates its confidence against retrieved passages
Phase 2  if the rating is not "high", the model names what it needs and the index
         retrieves those targets (the gap -> retrieval loop)
Phase 3  the model answers over the combined evidence, with limitations

The decisive detail is the re-gate: confidence is scored again on everything retrieval
actually produced, not on the initial context. Scoring only the initial context would
make the gap -> retrieval loop decorative, because the decision could never change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .confidence import gate, lexical_overlap, parse_confidence
from .corpus import INTERNAL, PUBLIC, Passage, relevance
from .domains import build_prompts

LLMCall = Callable[[str, str], str]

MAX_CONTEXT_CHARS = 6_000
MAX_PASSAGE_CHARS = 1_200
INITIAL_K = 4


@dataclass
class RunResult:
    query: str
    domain: str
    confidence: str
    self_reported: str
    overlap: float
    answer: str = ""
    limitations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    context_requests: list[str] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)
    context_chars: int = 0
    abstained: bool = False
    passages: list[Passage] = field(default_factory=list)
    tier: str = "none"          # highest tier that contributed evidence
    internal_count: int = 0
    internal_withheld: int = 0  # internal passages that matched but were not sent

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "domain": self.domain,
            "confidence": self.confidence,
            "self_reported": self.self_reported,
            "abstained": self.abstained,
            "tier": self.tier,
            "internal_passages": self.internal_count,
            "internal_withheld": self.internal_withheld,
            "overlap": round(self.overlap, 3),
            "context_chars": self.context_chars,
            "answer": self.answer,
            "limitations": self.limitations,
            "recommendations": self.recommendations,
            "context_requests": self.context_requests,
            "citations": [p.cite() for p in self.passages],
        }


def _bullets(block: str) -> list[str]:
    """Split a block into items. Bullet markers are optional; plain lines count too."""
    items = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        stripped = re.sub(r"^(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if stripped:
            items.append(stripped)
    return items


def parse_context_needed(raw: str) -> list[str]:
    match = re.search(r"CONTEXT_NEEDED\s*:\s*(.*)$", raw, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return [item for item in _bullets(match.group(1)) if len(item) > 3][:5]


def parse_answer(raw: str) -> tuple[str, list[str], list[str]]:
    """Split a phase-3 response into answer, limitations, recommendations."""
    answer = limitations = recommendations = ""
    match = re.search(r"ANSWER\s*:\s*(.*?)(?=LIMITATIONS\s*:|RECOMMENDATIONS\s*:|$)",
                      raw, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
    match = re.search(r"LIMITATIONS\s*:\s*(.*?)(?=RECOMMENDATIONS\s*:|$)",
                      raw, re.IGNORECASE | re.DOTALL)
    if match:
        limitations = match.group(1)
    match = re.search(r"RECOMMENDATIONS\s*:\s*(.*)$", raw, re.IGNORECASE | re.DOTALL)
    if match:
        recommendations = match.group(1)
    if not answer:
        answer = raw.strip()
    return answer, _bullets(limitations), _bullets(recommendations)


def render(passages: list[Passage], max_chars: int = MAX_CONTEXT_CHARS,
           per_passage: int = MAX_PASSAGE_CHARS) -> str:
    """Render passages as numbered, citable evidence blocks within a char budget."""
    blocks: list[str] = []
    total = 0
    for idx, passage in enumerate(passages, start=1):
        block = f"[{idx}] [{passage.tier}] {passage.cite()}\n{passage.text[:per_passage]}"
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total += len(block) + 2
    return "\n\n".join(blocks)


def orchestrator(query: str, index, llm: LLMCall, domain: str = "research",
                 prompts: dict | None = None, max_retrievals: int = 2,
                 min_overlap: float = 0.25, k: int = INITIAL_K,
                 relevance_floor: float = 0.0,
                 internal_allowed: bool = True) -> RunResult:
    """Run the three phases. ``index`` needs ``retrieve(query, k, tier) -> list[Passage]``.

    ``relevance_floor`` drops internal passages that only share an incidental word with
    the question, so a launch memo does not count as covering "what is a rollback window".

    ``internal_allowed=False`` keeps the internal tier out of every prompt while still
    counting what matched, so the caller can report what was withheld. The clearance rule
    belongs here rather than at the call sites: three callers exist, and a rule enforced in
    only two of them is not a rule.
    """
    prompts = prompts or build_prompts(domain)
    run = RunResult(query=query, domain=domain, confidence="low",
                    self_reported="low", overlap=0.0)

    # Internal outranks public: retrieve it first, then fall back. Both go into the
    # context in that order, so the model reads the user's own material before
    # anything public.
    matched_internal = [p for p in index.retrieve(query, k=k, tier=INTERNAL)
                        if relevance(query, p) >= relevance_floor]
    if internal_allowed:
        internal = matched_internal
    else:
        internal = []
        run.internal_withheld = len(matched_internal)
    public = list(index.retrieve(query, k=k, tier=PUBLIC))
    passages = internal + public
    initial_context = render(passages)
    run.passages = passages
    run.internal_count = len(internal)
    run.tier = INTERNAL if internal else (PUBLIC if public else "none")
    run.context_chars = len(initial_context)

    # Phase 1 -- confidence, cross-checked against an external signal.
    raw_1 = llm(prompts["phase1"],
                f"## CONTEXT\n{initial_context or '(no context retrieved)'}\n\n"
                f"## QUESTION\n{query}\n\nEvaluate your confidence in answering this question.")
    self_reported = parse_confidence(raw_1)
    overlap = lexical_overlap(query, initial_context)
    run.self_reported = self_reported
    run.overlap = overlap
    run.confidence = gate(self_reported, overlap, min_overlap=min_overlap)
    run.phases.append({"phase": 1, "self_reported": self_reported,
                       "gated": run.confidence, "overlap": round(overlap, 3)})

    full_context = initial_context
    if run.confidence != "high":
        # Phase 2 -- the model names what it needs; the index fetches it.
        raw_2 = llm(prompts["phase2"],
                    f"## QUESTION\n{query}\n\n"
                    f"## PREVIOUS ASSESSMENT\n{raw_1}\n\nName what you need to look up.")
        requests = parse_context_needed(raw_2)
        run.context_requests = requests
        run.phases.append({"phase": 2, "requests": requests})

        seen = {(p.path, p.start_line) for p in passages}
        extra: list[Passage] = []
        tiers = (INTERNAL, PUBLIC) if internal_allowed else (PUBLIC,)
        for request in requests[:max_retrievals]:
            for tier in tiers:
                for passage in index.retrieve(request, k=2, tier=tier):
                    key = (passage.path, passage.start_line)
                    if key not in seen:
                        seen.add(key)
                        extra.append(passage)
        if extra:
            passages = passages + extra
            run.passages = passages
            run.internal_count = sum(1 for p in passages if p.tier == INTERNAL)
            full_context = render(passages)
            run.context_chars = len(full_context)

    # Re-gate on everything retrieval actually produced.
    final_overlap = lexical_overlap(query, full_context)
    run.overlap = final_overlap
    run.confidence = gate(self_reported, final_overlap, min_overlap=min_overlap)
    run.phases.append({"phase": "gate", "self_reported": self_reported,
                       "initial_overlap": round(overlap, 3),
                       "final_overlap": round(final_overlap, 3),
                       "gated": run.confidence})

    # Phase 3 -- answer over everything retrieved, unless the gate said "low".
    # At that level no generation call is made: a low-confidence answer would be
    # exactly the confabulation this pipeline exists to prevent.
    if run.confidence == "low":
        run.abstained = True
        run.answer = (
            f"Abstained: the retrieved context does not support an answer "
            f"(confidence={run.confidence}, corpus overlap={run.overlap:.2f}). "
            f"No answer was generated."
        )
        run.limitations = [item for item in _bullets(raw_1)
                           if not item.lower().startswith(("confidence", "reasons"))]
        run.recommendations = list(run.context_requests)
        return run

    raw_3 = llm(prompts["phase3"],
                f"## CONTEXT\n{full_context or '(no context retrieved)'}\n\n"
                f"## CONFIDENCE ASSESSMENT\n{raw_1}\n\n"
                f"## ADDITIONAL MATERIAL REQUESTED\n"
                f"{chr(10).join(run.context_requests) or '(none)'}\n\n"
                f"## QUESTION\n{query}\n\n"
                + ("Answer from the internal material first. If the internal material does "
                   "not settle the question and you rely on public material, say plainly "
                   "that the answer comes from public sources rather than the user's own "
                   "documents.\n\n"
                   if internal_allowed else
                   "Internal material is not available to this model. Answer only from the "
                   "PUBLIC material above and say plainly that the answer comes from public "
                   "sources rather than the user's own documents.\n\n")
                + "Answer the question.")
    answer, limitations, recommendations = parse_answer(raw_3)
    run.answer = answer
    run.limitations = limitations
    run.recommendations = recommendations
    run.phases.append({"phase": 3})
    return run


def evaluate(query: str, index, llm: LLMCall, domain: str = "research", **kwargs) -> dict:
    """Convenience wrapper returning a plain dict."""
    return orchestrator(query, index, llm, domain=domain, **kwargs).to_dict()
