"""Model adapters.

Two ways to reach a model, both ``Callable[[str, str], str]``:

``endpoint_llm``  an OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama, a hosted API)
``ctx_llm``       Hermes' own host-owned LLM facade, so the plugin runs on the user's
                  active model with no second configuration

The endpoint client keeps the two vLLM settings that fail silently: ``chat_template_kwargs``
belongs at the top level of the request body (nesting it in ``extra_body`` is accepted and
ignored), and ``max_tokens`` must cover a reasoning model's thinking trace as well as the
answer. Both are handled here, once.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

DEFAULT_MAX_TOKENS = 8_000
DEFAULT_TIMEOUT = 120


class EndpointError(RuntimeError):
    """The endpoint is not configured or not reachable."""


class ModelError(RuntimeError):
    """The endpoint answered but returned no usable content."""


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": "Bearer " + key} if key else {}


def discover_model(base: str, key: str = "", timeout: int = 20) -> str:
    """Return the first model id served at ``base``."""
    request = urllib.request.Request(base.rstrip("/") + "/models", headers=_auth(key))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise EndpointError(f"{base}/models returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise EndpointError(f"{base}/models unreachable: {exc.reason}") from exc
    ids = [item.get("id") for item in payload.get("data", []) if item.get("id")]
    if not ids:
        raise ModelError(f"{base} served no models")
    return ids[0]


def chat(base: str, key: str, model: str, system: str, user: str,
         max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.1,
         timeout: int = DEFAULT_TIMEOUT) -> tuple[str, dict]:
    """One chat completion. Returns ``(content, metadata)``."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **_auth(key)},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ModelError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise EndpointError(f"{base}/chat/completions unreachable: {exc.reason}") from exc

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("content") or ""), {
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage", {}),
        "model": payload.get("model", model),
    }


def endpoint_llm(base: str, key: str = "", model: str = "",
                 max_tokens: int = DEFAULT_MAX_TOKENS, timeout: int = DEFAULT_TIMEOUT
                 ) -> Callable[[str, str], str]:
    """Adapter over an OpenAI-compatible endpoint."""
    resolved = model or discover_model(base, key)
    state = {"model": resolved}

    def call(system: str, user: str) -> str:
        content, meta = chat(base, key, state["model"], system, user,
                             max_tokens=max_tokens, timeout=timeout)
        if not content and meta.get("finish_reason") == "length":
            # A reasoning model spent the whole budget on its trace. Retry once with
            # more room rather than silently returning nothing.
            content, _ = chat(base, key, state["model"], system, user,
                              max_tokens=max_tokens * 2, timeout=timeout)
        return content

    return call


def ctx_llm(ctx, max_tokens: int = 2_000, temperature: float = 0.1
            ) -> Callable[[str, str], str]:
    """Adapter over Hermes' host-owned LLM facade (``ctx.llm``)."""
    def call(system: str, user: str) -> str:
        result = ctx.llm.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return getattr(result, "text", "") or ""

    return call
