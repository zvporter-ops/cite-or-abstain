"""Coldstart -- evidence-gated answering with a provenance hierarchy.

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

Standing rules ride on the same two hooks. A correction is recorded as a rule by
``transform_llm_output``, promoted to *hard* when it is stated again, and injected at the
start of every turn by ``pre_llm_call`` -- because a correction the user has to repeat is
a save that did not take, not a personality quirk. ``/rules`` lists them.

The model is never asked to grade its own answer. A model asked to police itself
defends itself; the gate is mechanical, so it cannot be talked out of a violation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from .coa import audit as audit_mod
from .coa import leakguard
from .coa import rules as rules_mod
from .coa.claimgate import check as gate_check, render_report
from .coa.corpus import (
    INTERNAL, PUBLIC, CorpusIndex, normalize_roots, relevance,
)
from .coa.llm import EndpointError, ModelError, ctx_llm, endpoint_llm
from .coa.pipeline import orchestrator, render

logger = logging.getLogger(__name__)

# One list, shared with the correction ledger, so "is this a question?" cannot drift
# between the evidence gate and the rule detector.
_QUESTION_STARTS = rules_mod.QUESTION_STARTS

_LOCK = threading.RLock()
_INDEX: CorpusIndex | None = None
_LAST: dict[str, dict] = {}
_STRICTNESS: dict[str, str] = {}
# The user's message, kept separately from ``_LAST`` so the correction ledger sees every
# turn while the evidence gate keeps its current meaning (an entry means "this turn
# retrieved evidence", and non-question turns must not start getting evidence reports).
_LAST_MSG: dict[str, str] = {}
# Turn bookkeeping for the ledger: a turn's reply may pass through the transform more than
# once, but the same correction stated in a *later* turn must be counted again -- that is
# exactly the repetition the ledger exists to detect.
_TURN: dict[str, int] = {}
_TURN_DONE: dict[str, int] = {}
_RULES: rules_mod.RuleStore | None = None
_CTX = None


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _data_dir() -> Path:
    path = _hermes_home() / "plugin-data" / "coldstart"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg() -> dict:
    if _CTX is None:
        return {}
    return {
        "roots": _CTX.get_config("corpus_roots", []) or [],
        "public_roots": _CTX.get_config("public_roots", []) or [],
        # No ``or "flag"`` here: a config value of False means "off", and collapsing it
        # to the default would leave the gate running after the user turned it off.
        "strictness": _CTX.get_config("strictness", "flag"),
        "endpoint": _CTX.get_config("endpoint", "") or "",
        "endpoint_model": _CTX.get_config("endpoint_model", "") or "",
        "domain": _CTX.get_config("domain", "research") or "research",
        "max_evidence_chars": int(_CTX.get_config("max_evidence_chars", 6000) or 6000),
        "min_overlap": float(_CTX.get_config("min_overlap", 0.25) or 0.25),
        "internal_relevance": float(_CTX.get_config("internal_relevance", 0.6) or 0.6),
        "rules_enabled": _CTX.get_config("rules_enabled", True),
        "rules_max_chars": int(_num(_CTX.get_config("rules_max_chars", 1200), 1200)),
        "rule_overlap": _num(_CTX.get_config("rule_overlap", 0.6), 0.6),
    }


def _num(value, default: float) -> float:
    """Config can hand back a string, a bool or None; a bad value must not break a turn."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _rules_enabled() -> bool:
    """``hermes config set ... rules_enabled false`` writes a YAML boolean, not "false"."""
    value = _cfg().get("rules_enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("off", "false", "no", "0", "")
    return bool(value)


def _rules_store() -> rules_mod.RuleStore:
    global _RULES
    with _LOCK:
        if _RULES is None:
            _RULES = rules_mod.RuleStore(_data_dir() / "rules.json",
                                         overlap=_cfg().get("rule_overlap", 0.6))
        return _RULES


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
            logger.info("coldstart indexed %s documents, %s passages (%s), %s removed",
                        built.documents, built.passages, built.by_tier, built.removed)
        return _INDEX


def _llm():
    cfg = _cfg()
    if cfg.get("endpoint"):
        key = ""
        try:
            from agent.secret_scope import get_secret_str
            key = get_secret_str("COA_API_KEY", "") or ""
        except Exception as exc:
            # No environment fallback. A stray ``COA_API_KEY`` in the environment would
            # become a bearer token this plugin sent on the user's behalf, from a file
            # nobody configured it to read -- the secret scope is the only sanctioned path.
            logger.debug("secret scope unavailable; sending no API key: %s", exc)
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


def _model_is_local() -> bool:
    """Is the model this plugin will actually call local?

    With an ``endpoint`` configured the plugin calls that endpoint, so the endpoint decides;
    otherwise it calls Hermes' active model. This is the one question the clearance rule
    needs, and every path that touches internal material asks it here rather than guessing.
    """
    endpoint = _cfg().get("endpoint") or ""
    if endpoint:
        return leakguard.classify(endpoint) == leakguard.LOCAL
    return _session_is_local()


def _normalize_mode(value) -> str:
    """Coerce whatever the config holds into one of off|flag|strict.

    ``hermes config set ... strictness off`` writes a YAML boolean, so the value can
    arrive as False rather than "off"; treating that as unknown would silently leave
    the gate running after the user turned it off.
    """
    if isinstance(value, bool):
        return "flag" if value else "off"
    if isinstance(value, int):
        return "off" if value == 0 else "flag"
    text = str(value or "").strip().lower()
    return text if text in ("off", "flag", "strict") else "flag"


def _strictness(session_id: str) -> str:
    raw = _STRICTNESS.get(session_id) or _cfg().get("strictness", "flag")
    return _normalize_mode(raw)


def _record_gate(session_id: str, report, mode: str, withheld: int = 0) -> None:
    """Persist the last gate outcome so the desktop pane can show what just happened."""
    try:
        payload = {
            "ts": time.time(),
            "session_id": session_id,
            "mode": mode,
            "withheld_internal": withheld,
            "ok": report.ok,
            "coverage_internal": round(report.coverage_internal, 3),
            "coverage_any": round(report.coverage_any, 3),
            "internal_passages": report.internal_count,
            "public_passages": report.public_count,
            "violations": [{"kind": f.kind, "span": f.span[:120]} for f in report.violations],
            "public_only": [{"kind": f.kind, "span": f.span[:120]} for f in report.public_only],
            "soft": sum(1 for f in report.findings if f.severity == "soft"),
        }
        path = _data_dir() / "last_gate.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("could not record the gate outcome: %s", exc)


def _looks_like_question(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    return text.endswith("?") or text.startswith(_QUESTION_STARTS)


def _append_notice(text: str, notice: str) -> str:
    """Put the rule notice on its own block instead of gluing it to the last line."""
    if not notice:
        return text
    return text.rstrip() + "\n\n" + notice


def _capture_rule(session_id: str) -> str:
    """Record a correction as a standing rule; return the one-line notice, if any."""
    if not _rules_enabled():
        return ""
    message = _LAST_MSG.get(session_id) or ""
    if not message:
        return ""
    turn = _TURN.get(session_id, 0)
    if _TURN_DONE.get(session_id) == turn:
        return ""
    rule_text = rules_mod.detect_correction(message)
    if not rule_text:
        _TURN_DONE[session_id] = turn
        return ""
    _TURN_DONE[session_id] = turn
    outcome = _rules_store().add(rule_text, session=session_id)
    logger.info("coldstart rule %s %s: %s", outcome.rule.get("id"),
                "created" if outcome.created else
                ("escalated to hard" if outcome.escalated else "restated"),
                outcome.rule.get("text"))
    return rules_mod.render_notice(outcome)


# --------------------------------------------------------------------- hooks


def on_pre_llm_call(session_id: str = "", user_message: str = "", **kwargs):
    """Inject standing rules, then tiered evidence; never raise into the turn."""
    try:
        # Kept even when this turn is not a question: the correction ledger runs on the
        # finished reply and needs the message that produced it.
        _LAST_MSG[session_id] = user_message or ""
        _TURN[session_id] = _TURN.get(session_id, 0) + 1

        rules_block = ""
        try:
            if _rules_enabled():
                rules_block = rules_mod.render_injection(
                    _rules_store().rules(),
                    max_chars=_cfg().get("rules_max_chars", 1200))
        except Exception as exc:  # a broken rule store must not break the turn
            logger.debug("coldstart rules injection skipped: %s", exc)

        def with_rules(block: str) -> str:
            return f"{rules_block}\n\n{block}" if rules_block else block

        if not _looks_like_question(user_message):
            return {"context": rules_block} if rules_block else None
        index = _index()
        if index is None:
            return {"context": rules_block} if rules_block else None

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
            return {"context": with_rules(leakguard.redaction_notice(
                len(internal), corpus_ask_local=_model_is_local()))}

        passages = internal + public
        _LAST[session_id] = {
            "passages": passages,
            "query": user_message,
            "internal_withheld": 0,
            "allow_public": not internal,
        }
        if not passages:
            return {"context": with_rules(
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
        return {"context": with_rules(
            "## EVIDENCE (tiered: internal outranks public)\n"
            f"{block}\n\n{precedence} Cite the passage ids you used."
        )}
    except Exception as exc:  # a broken corpus must not break the turn
        logger.warning("coldstart pre_llm_call failed: %s", exc)
        return None


def on_transform_llm_output(response_text: str = "", session_id: str = "", **kwargs):
    """Record any correction, then gate the finished reply against the turn's evidence."""
    try:
        notice = ""
        try:
            notice = _capture_rule(session_id)
        except Exception as exc:  # the ledger must not break the reply
            logger.warning("coldstart rule capture failed: %s", exc)

        mode = _strictness(session_id)
        if not response_text:
            return None
        if mode == "off":
            return _append_notice(response_text, notice) if notice else None
        entry = _LAST.get(session_id)
        if not entry:
            return _append_notice(response_text, notice) if notice else None

        withheld = int(entry.get("internal_withheld") or 0)
        passages = entry.get("passages") or []
        report = gate_check(response_text, passages,
                            allow_public=bool(entry.get("allow_public")))
        _record_gate(session_id, report, mode, withheld)

        if withheld and mode == "strict":
            hint = ("Use `corpus_ask`, which answers it against the local endpoint."
                    if _model_is_local() else
                    "To answer it locally, set a local endpoint in the plugin settings "
                    "(`plugins.entries.coldstart.settings.endpoint`); without one, "
                    "`corpus_ask` would call this same remote model.")
            return _append_notice(
                "I can't answer that here. Internal material matches this question, and "
                "the active model is not running locally, so the material was withheld "
                "rather than sent to a remote provider. " + hint, notice)

        if mode == "strict" and not report.ok:
            kinds = sorted({f.kind for f in report.violations})
            what = " and ".join(kinds)
            reply = (
                f"I can't answer that from your corpus. The draft reply relied on {what} "
                "that the retrieved evidence does not contain, so it is withheld rather "
                "than passed on unverified. "
                + ("Retrieved sources were: " + ", ".join(p.cite() for p in passages[:4])
                   if passages else
                   "No passage covers this question, so nothing in the reply could be "
                   "verified against it.")
            )
            return _append_notice(reply, notice)
        if not passages and not report.findings:
            return notice or None
        return _append_notice(response_text + render_report(report), notice)
    except Exception as exc:
        logger.warning("coldstart transform_llm_output failed: %s", exc)
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
    # This tool calls the same model as the turn unless an endpoint is configured, so the
    # clearance rule applies here too: private passages stay out of a non-local model's
    # context, and the run falls back to the public tier.
    internal_allowed = _model_is_local()
    try:
        run = orchestrator(question, index, _llm(),
                           domain=args.get("domain") or cfg["domain"],
                           min_overlap=cfg["min_overlap"],
                           relevance_floor=cfg["internal_relevance"],
                           internal_allowed=internal_allowed)
    except (EndpointError, ModelError) as exc:
        return json.dumps({"error": f"model unavailable: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    payload = run.to_dict()
    if run.internal_withheld:
        payload["notice"] = (
            f"{run.internal_withheld} internal passage(s) matched but were withheld: the "
            "active model is not local. This answer uses public material only. Set a local "
            "`endpoint` in the plugin settings to answer from internal material."
        )
    if not run.passages:
        # Nothing retrieved: abstain here rather than let a model answer over an empty
        # context, which is the exact confabulation this plugin exists to stop.
        payload["abstained"] = True
        payload["answer"] = (
            "Abstained: nothing in the corpus covers this question. "
            + (f"{run.internal_withheld} internal passage(s) matched but cannot be sent to "
               "a non-local model; set a local endpoint to answer from them."
               if run.internal_withheld else "No answer was generated.")
        )
        return json.dumps(payload)
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


def _trust_report() -> dict:
    """Run the scorecard against this deployment."""
    index = _index()
    if index is None:
        return {"error": "no corpus configured"}
    cfg = _cfg()
    internal_allowed = _model_is_local()
    # The gate probes sample internal passages, so an audit against a non-local model would
    # ship private text to it. Refuse instead of quietly scoring the public tier, which
    # would report a number for a deployment this plugin will not actually run.
    if not internal_allowed and index.stats()["by_tier"].get(INTERNAL, 0):
        return {"error": (
            "internal material is configured and the active model is not local, so the "
            "audit would send private passages to it. Set a local endpoint in the plugin "
            "settings (`plugins.entries.coldstart.settings.endpoint`) and run the "
            "audit again."
        )}
    return audit_mod.run_audit(
        index, _llm(),
        domain=cfg["domain"],
        min_overlap=cfg["min_overlap"],
        relevance_floor=cfg["internal_relevance"],
        local=internal_allowed,
        internal_allowed=internal_allowed,
    )


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
        return f"Evidence gate is {_normalize_mode(_cfg().get('strictness', 'flag'))}. Use /ground off|flag|strict."
    _CTX.set_config("strictness", mode)
    _STRICTNESS.clear()
    return {
        "off": "Evidence gate disabled.",
        "flag": "Evidence gate set to flag: replies carry an evidence report.",
        "strict": "Evidence gate set to strict: unsupported claims withhold the reply.",
    }[mode]


def _cmd_trust(raw_args: str, **kwargs) -> str:
    return audit_mod.render(_trust_report())


def _cmd_rules(raw_args: str, **kwargs) -> str:
    """List the captured rules, or forget / promote one by id."""
    store = _rules_store()
    args = (raw_args or "").strip().split()
    if not args:
        rules = store.rules()
        if not rules:
            return ("No rules captured yet. Correct me once and it becomes a rule; state "
                    "the same correction again and it becomes hard and is injected into "
                    "every turn.")
        lines = ["Captured rules (hard rules are injected into every turn):"]
        for rule in rules:
            mark = "[hard] " if rule.get("hard") else ""
            lines.append(f"  {rule.get('id')}. {mark}{rule.get('text')} "
                         f"(stated {rule.get('hits', 1)}x)")
        lines.append("Use /rules forget <id> to drop one, /rules hard <id> to promote one.")
        return "\n".join(lines)
    verb = args[0].lower()
    if verb in ("forget", "drop", "remove"):
        try:
            rule_id = int(args[1])
        except (IndexError, ValueError):
            return "Usage: /rules forget <id>"
        return (f"Rule {rule_id} forgotten." if store.forget(rule_id)
                else f"No rule {rule_id}.")
    if verb in ("hard", "promote"):
        try:
            rule_id = int(args[1])
        except (IndexError, ValueError):
            return "Usage: /rules hard <id>"
        return (f"Rule {rule_id} is hard now: it is injected into every turn."
                if store.set_hard(rule_id) else f"No rule {rule_id}.")
    return "Usage: /rules | /rules forget <id> | /rules hard <id>"


def register(ctx) -> None:
    global _CTX
    _CTX = ctx

    available = lambda: bool(_sources())  # noqa: E731

    ctx.register_tool("corpus_ask", "coldstart", CORPUS_ASK_SCHEMA,
                      _tool_corpus_ask, check_fn=available,
                      description="Answer from the corpus, internal tier first.")
    ctx.register_tool("evidence_report", "coldstart", EVIDENCE_REPORT_SCHEMA,
                      _tool_evidence_report, check_fn=available,
                      description="Verify a reply against retrieved evidence.")
    ctx.register_tool("corpus_audit", "coldstart", AUDIT_SCHEMA,
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
    ctx.register_command("rules", _cmd_rules,
                         description="List captured standing rules; forget or promote one",
                         args_hint="[forget|hard <id>]")

    skill = Path(__file__).parent / "skills" / "confidence-gated-pipeline" / "SKILL.md"
    if skill.exists():
        ctx.register_skill(
            "confidence-gated-pipeline", skill,
            description="Separate evaluation from generation: rate confidence, name the gap, "
                        "retrieve it, answer over evidence, or abstain.",
        )
