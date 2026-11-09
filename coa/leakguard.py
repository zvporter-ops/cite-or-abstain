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


def redaction_notice(count: int, corpus_ask_local: bool = True) -> str:
    """The directive injected when private evidence must not be sent.

    ``corpus_ask_local`` says whether the plugin has a local endpoint to answer through.
    With no endpoint configured ``corpus_ask`` calls Hermes' active model -- the same
    remote model this notice is protecting against -- so recommending it here would
    hand the material over by another door.
    """
    head = (
        "## CLEARANCE\n"
        f"{count} passage(s) of internal material match this question, and the active "
        "model is not running locally, so they have NOT been included here: sending them "
        "to a remote provider would disclose private material.\n"
        "Do not answer this question from general knowledge. Tell the user that internal "
        "material exists but cannot be sent to this model"
    )
    if corpus_ask_local:
        return head + ", and that `corpus_ask` answers it locally instead."
    return head + (
        ", and that answering it needs a local endpoint set in the plugin settings "
        "(`plugins.entries.coldstart.settings.endpoint`)."
    )
