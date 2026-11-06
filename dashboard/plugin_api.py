"""Backend for the Cite or Abstain desktop pane.

Mounted at ``/api/plugins/cite-or-abstain/`` by Hermes's dashboard router, so the pane can
read the corpus state, flip the gate mode, and run the trust battery without shelling out.

Nothing here is a second implementation: the scorecard is ``coa.audit``, the same module the
``/trust`` command runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coa import audit as audit_mod  # noqa: E402
from coa import leakguard  # noqa: E402
from coa.corpus import CorpusIndex  # noqa: E402
from coa.llm import endpoint_llm  # noqa: E402

PLUGIN_ID = "cite-or-abstain"
MODES = ("off", "flag", "strict")

router = APIRouter()

_LOCK = Lock()
_INDEX_CACHE: dict[tuple, CorpusIndex] = {}


def _home() -> Path:
    import os
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def _settings() -> dict:
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly() or {}
    entries = ((cfg.get("plugins") or {}).get("entries") or {})
    return (entries.get(PLUGIN_ID) or {}).get("settings") or {}


def _model_is_local() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        base = ((cfg.get("model") or {}).get("base_url")) or ""
        return leakguard.classify(base) == leakguard.LOCAL
    except Exception:
        return False


def _index(settings: dict) -> CorpusIndex | None:
    key = (
        tuple(str(e) for e in settings.get("corpus_roots") or []),
        tuple(str(p) for p in settings.get("public_roots") or []),
    )
    with _LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        index = audit_mod.build_index(settings, _home() / "plugin-data" / PLUGIN_ID / "index.db")
        if index is not None:
            _INDEX_CACHE.clear()
            _INDEX_CACHE[key] = index
        return index


def _last_gate() -> dict | None:
    path = _home() / "plugin-data" / PLUGIN_ID / "last_gate.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _llm(settings: dict):
    endpoint = (settings.get("endpoint") or "").strip()
    if not endpoint:
        return None
    key = ""
    try:
        from agent.secret_scope import get_secret_str
        key = get_secret_str("COA_API_KEY", "")
    except Exception:
        key = ""
    return endpoint_llm(endpoint, key, settings.get("endpoint_model") or "")


@router.get("/status")
def status() -> dict:
    """Everything the pane shows on load."""
    settings = _settings()
    index = _index(settings)
    corpus = {"documents": 0, "passages": 0, "by_tier": {}, "sources": []}
    if index is not None:
        stats = index.stats()
        corpus = {
            "documents": stats["documents"],
            "passages": stats["passages"],
            "by_tier": stats["by_tier"],
            "sources": stats["sources"],
        }
    return {
        "mode": settings.get("strictness") or "flag",
        "corpus": corpus,
        "model": "local" if _model_is_local() else "remote",
        "last_gate": _last_gate(),
        "endpoint": bool((settings.get("endpoint") or "").strip()),
    }


@router.post("/mode")
def set_mode(payload: dict) -> dict:
    mode = str((payload or {}).get("mode") or "").strip().lower()
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {', '.join(MODES)}")
    from hermes_cli.config import set_config_value
    set_config_value(f"plugins.entries.{PLUGIN_ID}.settings.strictness", mode, force=True)
    return {"mode": mode}


@router.post("/audit")
def run_audit() -> dict:
    """Run the trust battery. Slow by nature: it makes real model calls."""
    settings = _settings()
    index = _index(settings)
    if index is None:
        return {"error": "no corpus configured"}
    llm = _llm(settings)
    if llm is None:
        return {"error": "set an endpoint in the plugin settings to run the audit"}
    return audit_mod.run_audit(
        index, llm,
        domain=settings.get("domain") or "research",
        min_overlap=float(settings.get("min_overlap") or 0.25),
        relevance_floor=float(settings.get("internal_relevance") or 0.6),
        local=_model_is_local(),
    )
