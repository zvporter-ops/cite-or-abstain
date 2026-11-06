"""The trust scorecard.

A score is a measurement of *this deployment* -- this corpus, this model, this config --
not a claim about the plugin in general. Two halves:

``gate probes``      deterministic. An invented quote and an invented figure must be caught;
                     a verbatim quote and a paraphrase must not be.
``pipeline probes``  real model calls. The pipeline must abstain on questions the corpus
                     cannot answer, and must answer one it can.

Both are run here so the agent-side command, the tool and the desktop pane all report the
same number from the same code.
"""

from __future__ import annotations

from typing import Any

from .claimgate import SENTENCE, check as gate_check
from .corpus import INTERNAL, PUBLIC, CorpusIndex, content_words
from .pipeline import orchestrator

UNANSWERABLE_PROBES = (
    "What is the best soil mix for growing tomatoes?",
    "Who won the 1998 FIFA World Cup final?",
)

INVENTED_QUOTE = "The board approved a 47% increase in Q3."
INVENTED_FIGURE = "The reconciliation total was 918,442 units."


def build_index(settings: dict, db_path) -> CorpusIndex | None:
    """Index from plain plugin settings, with no dependency on a live plugin ctx."""
    sources: list[dict] = []
    for entry in settings.get("corpus_roots") or []:
        sources.append(entry if isinstance(entry, dict) else {"path": entry, "tier": INTERNAL})
    for path in settings.get("public_roots") or []:
        sources.append({"path": path, "tier": PUBLIC})
    sources = [s for s in sources if str(s.get("path", "")).strip()]
    if not sources:
        return None
    index = CorpusIndex(sources, db_path)
    if index.stats()["passages"] == 0:
        index.build()
    return index


def _first_sentence(text: str) -> str:
    for sentence in SENTENCE.findall(text):
        sentence = sentence.strip()
        if len(sentence) >= 25:
            return sentence
    return ""


def probe_question(passage) -> str | None:
    """A question this passage can answer, phrased from its own vocabulary.

    Built from content words rather than the raw first line: a sentence lifted verbatim
    makes a question that no retriever scores well ("...about Project Falcon budget is
    250000. The launch window is March...?").
    """
    words = content_words(passage.text)[:6]
    if not words:
        return None
    return "What does the corpus say about " + " ".join(words) + "?"


def gate_probes(index: Any) -> tuple[dict[str, bool | None], str | None]:
    """Deterministic checks. Returns ``(checks, answerable_question)``."""
    checks: dict[str, bool | None] = {}
    sample = index.sample_passages(INTERNAL, 3)
    if not sample:
        return checks, None
    passage = sample[0]
    real = _first_sentence(passage.text)
    if real:
        checks["verbatim_quote_allowed"] = gate_check(f'"{real}"', [passage]).ok
        checks["paraphrase_allowed"] = gate_check("In short, " + real, [passage]).ok
    checks["invented_quote_caught"] = not gate_check(f'"{INVENTED_QUOTE}"', [passage]).ok
    checks["invented_figure_caught"] = not gate_check(INVENTED_FIGURE, [passage]).ok
    return checks, probe_question(passage)


def run_audit(index: Any, llm, *, domain: str = "research",
              min_overlap: float = 0.25, relevance_floor: float = 0.6,
              local: bool = True) -> dict:
    """Run both halves and return the scorecard."""
    stats = index.stats()
    checks, answerable = gate_probes(index)

    model_error: str | None = None
    abstained = 0
    answered = 0
    try:
        for probe in UNANSWERABLE_PROBES:
            run = orchestrator(probe, index, llm, domain=domain,
                               min_overlap=min_overlap, relevance_floor=relevance_floor)
            abstained += int(run.abstained)
        if answerable:
            run = orchestrator(answerable, index, llm, domain=domain,
                               min_overlap=min_overlap, relevance_floor=relevance_floor)
            answered += int(not run.abstained)
    except Exception as exc:  # a broken endpoint must not lose the gate results
        model_error = f"{type(exc).__name__}: {exc}"

    passed = [v for v in checks.values() if v is not None]
    gate_score = (sum(1 for v in passed if v) / len(passed)) if passed else 0.0
    abstain_score = abstained / len(UNANSWERABLE_PROBES)
    answer_score = (answered / 1) if answerable else 0.0
    score = round(100 * (0.5 * gate_score + 0.3 * abstain_score + 0.2 * answer_score))
    verdict = "trusted" if score >= 85 else "partial" if score >= 60 else "untrusted"

    return {
        "verdict": verdict,
        "score": score,
        "corpus": {
            "documents": stats["documents"],
            "passages": stats["passages"],
            "by_tier": stats["by_tier"],
            "sources": stats["sources"],
        },
        "gate": checks,
        "pipeline": {
            "abstained_on_unanswerable": f"{abstained}/{len(UNANSWERABLE_PROBES)}",
            "answered_known_question": f"{answered}/{1 if answerable else 0}",
            "model_error": model_error,
        },
        "clearance": {
            "session_model": "local" if local else "remote",
            "internal_material": "shareable with this model" if local
            else "withheld from this model",
        },
    }


def render(report: dict) -> str:
    """Markdown form, for the ``/trust`` command."""
    if report.get("error"):
        return f"Trust report unavailable: {report['error']}"
    lines = [f"**Trust report — {report['verdict']} ({report['score']}/100)**", ""]
    corpus = report["corpus"]
    tiers = corpus["by_tier"]
    lines.append(
        f"Corpus: {corpus['documents']} documents, {corpus['passages']} passages "
        f"({tiers.get(INTERNAL, 0)} internal, {tiers.get(PUBLIC, 0)} public).")
    lines.append("")
    lines.append("Gate:")
    for name, ok in report["gate"].items():
        if ok is None:
            continue
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {name.replace('_', ' ')}")
    pipeline = report["pipeline"]
    lines.append("")
    lines.append("Pipeline:")
    lines.append(f"- abstained on unanswerable: {pipeline['abstained_on_unanswerable']}")
    lines.append(f"- answered a known question: {pipeline['answered_known_question']}")
    if pipeline["model_error"]:
        lines.append(f"- model error: {pipeline['model_error']}")
    lines.append("")
    lines.append(f"Clearance: {report['clearance']['session_model']} model, "
                 f"internal material {report['clearance']['internal_material']}.")
    return "\n".join(lines)
