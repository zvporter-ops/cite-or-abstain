"""Cite or Abstain -- evidence-gated answering with a provenance hierarchy.

Two gates and one clearance rule:

``pre_llm_call``         retrieves internal material first, then public; injects the
                         evidence with its tier, or a clearance notice when internal
                         material exists but the active model is remote.
``transform_llm_output`` checks the finished reply against that evidence. Quoted spans
                         must be verbatim; figures, dates, URLs and identifiers must
                         appear in a retrieved passage, and claims that rest only on
                         public material are disclosed as such.
``/trust``               runs a battery against the configured deployment and reports a
                         score: does it abstain when it should, catch what it should,
                         and leave legitimate paraphrase alone.

The model is never asked to grade its own answer. A model asked to police itself
defends itself; the gate is mechanical, so it cannot be talked out of a violation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from .coa import leakguard
from .coa.claimgate import SENTENCE, check as gate_check, render_report
from .coa.corpus import (
    INTERNAL, PUBLIC, CorpusIndex, normalize_roots, relevance,
)
from .coa.llm import EndpointError, ModelError, ctx_llm, endpoint_llm
from .coa.pipeline import orchestrator, render

logger = logging.getLogger(__name__)

_QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "who", "which", "does", "do", "is",
    "are", "can", "could", "should", "would", "will", "was", "were", "tell me",
    "explain", "summarize", "summarise", "find", "show",
)

UNANSWERABLE_PROBES = (
    "What is the best soil mix for growing tomatoes?",
    "Who won the 1998 FIFA World Cup final?",
)

_LOCK = threading.RLock()
_INDEX: CorpusIndex | None = None
_LAST: dict[str, dict] = {}
_STRICTNESS: dict[str, str] = {}
_CTX = None


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _data_dir() -> Path:
    path = _hermes_home() / "plugin-data" / "cite-or-abstain"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg() -> dict:
    if _CTX is None:
        return {}
    return {
        "roots": _CTX.get_config("corpus_roots", []) or [],
        "public_roots": _CTX.get_config("public_roots", []) or [],
        "strictness": _CTX.get_config("strictness", "flag") or "flag",
        "endpoint": _CTX.get_config("endpoint", "") or "",
        "endpoint_model": _CTX.get_config("endpoint_model", "") or "",
        "domain": _CTX.get_config("domain", "research") or "research",
        "max_evidence_chars": int(_CTX.get_config("max_evidence_chars", 6000) or 6000),
        "min_overlap": float(_CTX.get_config("min_overlap", 0.25) or 0.25),
        "internal_relevance": float(_CTX.get_config("internal_relevance", 0.6) or 0.6),
    }


def _sources() -> list[dict]:
    """Internal roots plus any public reference roots, as tiered entries."""
    cfg = _cfg()
    entries = [e if isinstance(e, dict) else {"path": e, "tier": INTERNAL}
               for e in cfg.get("roots", [])]
    entries += [{"path": p, "tier": PUBLIC} for p in cfg.get("public_roots", [])]
    return [e for e in entries if str(e.get("path", "")).strip()]


def _index() -> CorpusIndex | None:
    """Lazy singleton index; rebuilt when the configured sources change."""
    global _INDEX
    sources = _sources()
    if not sources:
        return None
    signature = [(str(s["path"]), s.get("tier", INTERNAL)) for s in sources]
    with _LOCK:
        current = ([(str(p), t) for p, t in _INDEX.sources] if _INDEX else None)
        changed = current != signature
        if changed:
            _INDEX = CorpusIndex(sources, _data_dir() / "index.db")
        assert _INDEX is not None
        if changed or _INDEX.stats()["passages"] == 0:
            built = _INDEX.build()
            logger.info("cite-or-abstain indexed %s documents, %s passages (%s), %s removed",
                        built.documents, built.passages, built.by_tier, built.removed)
        return _INDEX


def _llm():
    cfg = _cfg()
    if cfg.get("endpoint"):
        key = ""
        try:
            from agent.secret_scope import get_secret_str
            key = get_secret_str("COA_API_KEY", "")
        except Exception:
            key = os.environ.get("COA_API_KEY", "")
        return endpoint_llm(cfg["endpoint"], key, cfg.get("endpoint_model", ""))
    return ctx_llm(_CTX)


def _session_is_local() -> bool:
    """Where does the active model live? Unclassified counts as remote (fail closed)."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        base = ((cfg.get("model") or {}).get("base_url")) or ""
        return leakguard.classify(base) == leakguard.LOCAL
    except Exception:
        return False


def _strictness(session_id: str) -> str:
    return _STRICTNESS.get(session_id) or _cfg().get("strictness", "flag")


def _looks_like_question(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return text.endswith("?") or text.startswith(_QUESTION_STARTS)


# --------------------------------------------------------------------- hooks


def on_pre_llm_call(session_id: str = "", user_message: str = "", **kwargs):
    """Retrieve tiered evidence and inject it; never raise into the turn."""
    try:
        if not _looks_like_question(user_message):
            return None
        index = _index()
        if index is None:
            return None

        internal_all = index.retrieve(user_message, k=4, tier=INTERNAL)
        floor = _cfg().get("internal_relevance", 0.6)
        internal = [p for p in internal_all if relevance(user_message, p) >= floor]
        public = index.retrieve(user_message, k=4, tier=PUBLIC)
        local = _session_is_local()

        # Clearance rule: internal material never travels to a remote model.
        if internal and not local:
            _LAST[session_id] = {
                "passages": public,
                "query": user_message,
                "internal_withheld": len(internal),
                "allow_public": True,
            }
            return {"context": leakguard.redaction_notice(len(internal))}

        passages = internal + public
        _LAST[session_id] = {
            "passages": passages,
            "query": user_message,
            "internal_withheld": 0,
            "allow_public": not internal,
        }
        if not passages:
            return {"context": (
                "## EVIDENCE GATE\n"
                "No internal or public material covers this question. Do not answer it "
                "from general knowledge, and do not include figures, dates, quotations "
                "or citations. Reply in one short paragraph: the corpus is silent on "
                "this, and name the specific material that would answer it."
            )}
        cfg = _cfg()
        block = render(passages, max_chars=cfg.get("max_evidence_chars", 6000))
        precedence = (
            "Answer from INTERNAL material first. If internal material does not settle "
            "the question and you rely on PUBLIC material, say plainly that the answer "
            "comes from public sources rather than the user's own documents."
            if internal else
            "No internal material covers this question. If you answer, say clearly that "
            "it rests on public material."
        )
        return {"context": (
            "## EVIDENCE (tiered: internal outranks public)\n"
            f"{block}\n\n{precedence} Cite the passage ids you used."
        )}
    except Exception as exc:  # a broken corpus must not break the turn
        logger.warning("cite-or-abstain pre_llm_call failed: %s", exc)
        return None


def on_transform_llm_output(response_text: str = "", session_id: str = "", **kwargs):
    """Gate the finished reply against the turn's tiered evidence."""
    try:
        mode = _strictness(session_id)
        if mode == "off" or not response_text:
            return None
        entry = _LAST.get(session_id)
        if not entry:
            return None

        withheld = int(entry.get("internal_withheld") or 0)
        if withheld and mode == "strict":
            return (
                "I can't answer that here. Internal material matches this question, and "
                "the active model is not running locally, so the material was withheld "
                "rather than sent to a remote provider. Use `corpus_ask`, which answers "
                "it against the local endpoint."
            )

        passages = entry.get("passages") or []
        report = gate_check(response_text, passages,
                            allow_public=bool(entry.get("allow_public")))

        if mode == "strict" and not report.ok:
            kinds = sorted({f.kind for f in report.violations})
            what = " and ".join(kinds)
            return (
                f"I can't answer that from your corpus. The draft reply relied on {what} "
                "that the retrieved evidence does not contain, so it is withheld rather "
                "than passed on unverified. "
                + ("Retrieved sources were: " + ", ".join(p.cite() for p in passages[:4])
                   if passages else
                   "No passage covers this question, so nothing in the reply could be "
                   "verified against it.")
            )
        if not passages and not report.findings:
            return None
        return response_text + render_report(report)
    except Exception as exc:
        logger.warning("cite-or-abstain transform_llm_output failed: %s", exc)
        return None


# --------------------------------------------------------------------- tools


def _tool_corpus_ask(args: dict, **kwargs) -> str:
    question = (args.get("question") or "").strip()
    if not question:
        return json.dumps({"error": "question is required"})
    index = _index()
    if index is None:
        return json.dumps({"error": "no corpus configured; set corpus_roots in the plugin settings"})
    cfg = _cfg()
    try:
        run = orchestrator(question, index, _llm(),
                           domain=args.get("domain") or cfg["domain"],
                           min_overlap=cfg["min_overlap"],
                           relevance_floor=cfg["internal_relevance"])
    except (EndpointError, ModelError) as exc:
        return json.dumps({"error": f"model unavailable: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    payload = run.to_dict()
    if not run.abstained:
        payload["claim_gate"] = gate_check(
            run.answer, run.passages,
            allow_public=(run.tier != INTERNAL)).to_dict()
    return json.dumps(payload)


def _tool_evidence_report(args: dict, **kwargs) -> str:
    session_id = kwargs.get("session_id") or ""
    entry = _LAST.get(session_id)
    answer = (args.get("answer") or "").strip()
    if not entry:
        return json.dumps({"error": "no evidence recorded for this session yet"})
    if not answer:
        return json.dumps({"error": "answer is required when no turn evidence is cached"})
    return json.dumps(gate_check(answer, entry.get("passages") or [],
                                 allow_public=bool(entry.get("allow_public"))).to_dict())


def _first_sentence(text: str) -> str:
    for sentence in SENTENCE.findall(text):
        sentence = sentence.strip()
        if len(sentence) >= 25:
            return sentence
    return ""


def _trust_report() -> dict:
    """Run the battery: gate probes (deterministic) + pipeline probes (model calls)."""
    index = _index()
    if index is None:
        return {"error": "no corpus configured"}
    cfg = _cfg()
    stats = index.stats()
    sample = index.sample_passages(INTERNAL, 3)

    gate: dict[str, bool | None] = {}
    answerable = None
    if sample:
        passage = sample[0]
        real = _first_sentence(passage.text)
        title = passage.text.strip().splitlines()[0].lstrip("# ").strip()[:120]
        if real:
            gate["verbatim_quote_allowed"] = gate_check(f'"{real}"', [passage]).ok
            gate["paraphrase_allowed"] = gate_check("In short, " + real, [passage]).ok
        gate["invented_quote_caught"] = not gate_check(
            '"The board approved a 47% increase in Q3."', [passage]).ok
        gate["invented_figure_caught"] = not gate_check(
            "The reconciliation total was 918,442 units.", [passage]).ok
        if title:
            answerable = f"What does the corpus say about {title}?"

    model_error = None
    abstained = 0
    answered = 0
    try:
        llm = _llm()
        for probe in UNANSWERABLE_PROBES:
            run = orchestrator(probe, index, llm, domain=cfg["domain"],
                               min_overlap=cfg["min_overlap"],
                               relevance_floor=cfg["internal_relevance"])
            abstained += int(run.abstained)
        if answerable:
            run = orchestrator(answerable, index, llm, domain=cfg["domain"],
                               min_overlap=cfg["min_overlap"],
                               relevance_floor=cfg["internal_relevance"])
            answered += int(not run.abstained)
    except (EndpointError, ModelError) as exc:
        model_error = str(exc)
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"

    checks = [v for v in gate.values() if v is not None]
    gate_score = (sum(1 for v in checks if v) / len(checks)) if checks else 0.0
    abstain_score = abstained / len(UNANSWERABLE_PROBES)
    answer_score = answered / 1 if answerable else 0.0
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
        "gate": gate,
        "pipeline": {
            "abstained_on_unanswerable": f"{abstained}/{len(UNANSWERABLE_PROBES)}",
            "answered_known_question": f"{answered}/{1 if answerable else 0}",
            "model_error": model_error,
        },
        "clearance": {
            "session_model": "local" if _session_is_local() else "remote",
            "internal_material": "shareable with this model"
            if _session_is_local() else "withheld from this model",
        },
    }


def _render_trust(report: dict) -> str:
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
    for name, passed in report["gate"].items():
        if passed is None:
            continue
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}")
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


CORPUS_ASK_SCHEMA = {
    "name": "corpus_ask",
    "description": (
        "Answer a question from the user's corpus using the evidence-gated pipeline, "
        "internal material first. Returns confidence, tier, answer, limitations, "
        "citations, whether the run abstained, and the claim gate report. Use this "
        "instead of answering from memory when the question concerns the user's material."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to answer from the corpus."},
            "domain": {
                "type": "string",
                "enum": ["research", "code_review", "debugging", "system_ops", "content_creation"],
                "description": "Optional lens; defaults to the plugin setting.",
            },
        },
        "required": ["question"],
    },
}

EVIDENCE_REPORT_SCHEMA = {
    "name": "evidence_report",
    "description": (
        "Check a reply against the evidence retrieved for this session. Reports quoted "
        "spans that are not verbatim, figures, dates, URLs or identifiers that no "
        "retrieved passage contains, which claims rest only on public material, and the "
        "per-sentence support map."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The reply text to verify."},
        },
        "required": ["answer"],
    },
}

AUDIT_SCHEMA = {
    "name": "corpus_audit",
    "description": (
        "Audit the configured deployment and return a trust score: whether the claim "
        "gate catches invented quotes and figures, whether legitimate paraphrase passes, "
        "and whether the pipeline abstains on unanswerable questions. Use when the user "
        "asks whether they can trust this setup."
    ),
    "parameters": {"type": "object", "properties": {}},
}


# ------------------------------------------------------------------- register


def _cmd_ground(raw_args: str, **kwargs) -> str:
    mode = (raw_args or "").strip().lower()
    if mode not in ("off", "flag", "strict"):
        return f"Evidence gate is {_cfg().get('strictness', 'flag')}. Use /ground off|flag|strict."
    _CTX.set_config("strictness", mode)
    _STRICTNESS.clear()
    return {
        "off": "Evidence gate disabled.",
        "flag": "Evidence gate set to flag: replies carry an evidence report.",
        "strict": "Evidence gate set to strict: unsupported claims withhold the reply.",
    }[mode]


def _cmd_trust(raw_args: str, **kwargs) -> str:
    return _render_trust(_trust_report())


def register(ctx) -> None:
    global _CTX
    _CTX = ctx

    available = lambda: bool(_sources())  # noqa: E731

    ctx.register_tool("corpus_ask", "cite_or_abstain", CORPUS_ASK_SCHEMA,
                      _tool_corpus_ask, check_fn=available,
                      description="Answer from the corpus, internal tier first.")
    ctx.register_tool("evidence_report", "cite_or_abstain", EVIDENCE_REPORT_SCHEMA,
                      _tool_evidence_report, check_fn=available,
                      description="Verify a reply against retrieved evidence.")
    ctx.register_tool("corpus_audit", "cite_or_abstain", AUDIT_SCHEMA,
                      lambda args, **kw: json.dumps(_trust_report()),
                      check_fn=available,
                      description="Score the deployment's trustworthiness.")

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    # hold_stream: the reply is withheld until the gate has run. Streaming straight to the
    # user would mean showing text that has not been checked -- the opposite of the point --
    # and a tool-using turn otherwise finishes on a path the transform never sees.
    ctx.register_hook("transform_llm_output", on_transform_llm_output, hold_stream=True)

    ctx.register_command("ground", _cmd_ground,
                         description="Set the evidence gate mode: off | flag | strict",
                         args_hint="<off|flag|strict>")
    ctx.register_command("trust", _cmd_trust,
                         description="Run the trust battery and score this deployment")

    skill = Path(__file__).parent / "skills" / "confidence-gated-pipeline" / "SKILL.md"
    if skill.exists():
        ctx.register_skill(
            "confidence-gated-pipeline", skill,
            description="Separate evaluation from generation: rate confidence, name the gap, "
                        "retrieve it, answer over evidence, or abstain.",
        )
