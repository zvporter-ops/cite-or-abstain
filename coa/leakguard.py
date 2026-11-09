"""Leak guard: decide whether private material may leave the machine.

The rule is fail-closed. A plugin that sends trade secrets to a cloud provider because
it could not work out where the model lives has already done the damage, so an
unclassified endpoint is treated as remote.

Local means: loopback, an RFC1918 address, a Tailscale CGNAT address (100.64.0.0/10 --
``ipaddress`` does NOT report those as private), or an mDNS/``.lan``/``.internal`` host.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
LOCAL_SUFFIXES = (".local", ".lan", ".internal", ".home", ".localhost")
TAILSCALE = ipaddress.ip_network("100.64.0.0/10")

LOCAL = "local"
CLOUD = "cloud"


def is_local_url(url: str | None) -> bool:
    """True when ``url`` points at something on this machine or this network."""
    if not url:
        return False
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in LOOPBACK_HOSTS:
        return True
    if host.endswith(LOCAL_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip in TAILSCALE


def classify(url: str | None) -> str:
    """``local`` or ``cloud``. Anything unparseable is ``cloud``."""
    return LOCAL if is_local_url(url) else CLOUD


def provider_base_url(cfg: dict | None, provider: str | None) -> str:
    """The base_url config registers for a provider, or "" when it does not say."""
    name = str(provider or "").split(":", 1)[-1]
    for table in ("providers", "custom_providers"):
        entry = ((cfg or {}).get(table) or {}).get(name)
        if isinstance(entry, dict):
            base = str(entry.get("base_url") or "").strip()
            if base:
                return base
    return ""


def remote_fallback(cfg: dict | None) -> bool:
    """True when a configured fallback model is not local.

    A fallback takes over mid-session -- typically while the local server is down -- and the
    same turn hook then sends internal passages to it. Unclassifiable counts as remote, the
    way an unparseable endpoint does.
    """
    entries: list = []
    for key in ("fallback_model", "fallback_providers"):
        value = (cfg or {}).get(key) or []
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            entries.extend(value)
    for entry in entries:
        if not isinstance(entry, dict):
            return True
        base = (str(entry.get("base_url") or "").strip()
                or provider_base_url(cfg, entry.get("provider")))
        if not base or classify(base) != LOCAL:
            return True
    return False


def same_model(left: str, right: str) -> bool:
    """Compare model names across provider prefixes: ``custom:spark/m`` is ``m``."""
    def leaf(value: str) -> str:
        return str(value or "").strip().lower().split("/")[-1].split(":")[-1]

    a, b = leaf(left), leaf(right)
    return bool(a) and a == b


def redaction_notice(count: int) -> str:
    """The directive injected when private evidence must not be sent.

    A tool call answers into this same turn, so ``corpus_ask`` is not a way around a remote
    model: that model reads the tool's output. The only remedy is a local model for the turn.
    """
    return (
        "## CLEARANCE\n"
        f"{count} passage(s) of internal material match this question, and the model this "
        "turn reaches is not running locally, so they have NOT been included here: sending "
        "them to a remote provider would disclose private material.\n"
        "Do not answer this question from general knowledge. Tell the user that internal "
        "material exists but cannot be sent to this model, and that answering it needs a "
        "local model for the turn (`corpus_ask` would only hand the material to the same "
        "remote model)."
    )
