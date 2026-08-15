"""Chat client for any OpenAI-compatible endpoint.

The system was developed against OpenRouter, but nothing in it is specific to
that provider: point ``MAVE_BASE_URL`` at a local vLLM, Ollama, llama.cpp or any
other OpenAI-compatible server and the pipeline runs unchanged.

Reproducing the published numbers needs no endpoint and no API key at all — the
draws are replayed from the shipped cache, and the client below is only ever
constructed on an actual cache miss.

Two hard-won details are encoded here rather than left to the caller:

- ``n > 1`` is ignored by several providers, so sampled draws are issued as
  separate single requests (see :func:`mave.pipeline.draws`).
- Reasoning/thinking must be switched off, or it eats the ``max_tokens`` budget
  and the reply comes back empty. OpenRouter takes a unified flag; endpoints that
  refuse it outright are detected on the first refusal and retried without it.
"""

import os
import random
import threading
import time
from collections import Counter, defaultdict

from . import cache

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Retries for transient upstream failures (rate limits, dead fallback providers).
TRANSIENT_RETRIES = 10
TRANSIENT_STATUS = (404, 408, 409, 429, 500, 502, 503, 529)

_LOCK = threading.Lock()
_USAGE: dict[str, dict] = {}
_DEGRADED: Counter = Counter()

# One lock per request key, so concurrent identical calls wait for the first
# instead of stampeding the endpoint.
_INFLIGHT: dict[str, threading.Lock] = defaultdict(threading.Lock)
_INFLIGHT_GUARD = threading.Lock()


def base_url() -> str:
    return os.environ.get("MAVE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL


def api_key() -> str | None:
    for name in ("MAVE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    return None


def note_degraded(kind: str) -> None:
    """Record a tolerated-but-weakening event, so a degraded run cannot look clean.

    An empty provider response is a lost draw and a failed jury lens counts as a
    No. Neither should kill a run, but a run that hits many of them is not the
    run it reports being.
    """
    with _LOCK:
        _DEGRADED[kind] += 1


def degraded() -> dict:
    with _LOCK:
        return dict(_DEGRADED)


def usage() -> dict:
    with _LOCK:
        per_model = {m: dict(u) for m, u in _USAGE.items()}
    total = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    for entry in per_model.values():
        for field in total:
            total[field] += entry[field]
    total["cost_usd"] = round(total["cost_usd"], 6)
    return {"per_model": per_model, **total}


def _track(model: str, response) -> None:
    stats = getattr(response, "usage", None)
    if not stats:
        return
    with _LOCK:
        entry = _USAGE.setdefault(model, {"requests": 0, "prompt_tokens": 0,
                                          "completion_tokens": 0, "cost_usd": 0.0})
        entry["requests"] += 1
        entry["prompt_tokens"] += stats.prompt_tokens or 0
        entry["completion_tokens"] += stats.completion_tokens or 0
        entry["cost_usd"] += getattr(stats, "cost", None) or 0.0


class LLM:
    """One model endpoint, safe to share across threads.

    ``model`` is the name sent over the wire; ``cache_slug`` is the cache identity
    and must stay fixed, so that serving the same weights under a different name
    still replays the shipped draws.
    """

    def __init__(self, model: str, cache_slug: str | None = None, offline: bool = False,
                 timeout: float = 120.0):
        self.model = model
        self.cache_slug = cache_slug or model
        self.offline = offline
        self.timeout = timeout
        self._client = None
        self._no_reasoning_flag = False

    def _endpoint(self):
        """Built on the first cache miss, so a pure replay needs no key."""
        if self._client is None:
            from openai import OpenAI

            key = api_key()
            if not key:
                raise SystemExit(
                    "A model call is needed but no API key is set. Either drop --live so the "
                    "run replays the shipped cache, or set MAVE_API_KEY and MAVE_BASE_URL "
                    "(see .env.example)."
                )
            self._client = OpenAI(base_url=base_url(), api_key=key,
                                  timeout=self.timeout, max_retries=3)
        return self._client

    def chat(self, system: str, user: str, temperature: float = 0.0,
             max_tokens: int = 4096, seed: int | None = None) -> str:
        """One chat completion, cached. ``seed`` keeps repeated sampled draws distinct."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        key = cache.request_key(self.cache_slug, messages, temperature, max_tokens, seed)
        with _INFLIGHT_GUARD:
            lock = _INFLIGHT[key]
        with lock:
            hit = cache.get(self.cache_slug, messages, temperature, max_tokens, seed)
            if hit:
                return hit
            if self.offline:
                raise cache.CacheMiss(
                    f"{self.cache_slug}: this response is not in the shipped cache."
                )
            text = self._request(messages, temperature, max_tokens, seed)
            if text:
                cache.put(self.cache_slug, messages, temperature, max_tokens, seed, text)
            return text

    def _request(self, messages, temperature, max_tokens, seed) -> str:
        extra_body = {} if self._no_reasoning_flag else {"reasoning": {"enabled": False}}
        for attempt in range(TRANSIENT_RETRIES + 1):
            try:
                response = self._endpoint().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    # Mistral's first-party endpoints reject temperature 0 unless top_p
                    # is exactly 1 ("top_p must be 1 when using greedy sampling").
                    # Third-party hosts of the same weights accept the default, which is
                    # why this only surfaces on some servers.
                    **({"top_p": 1.0} if temperature == 0 else {}),
                    **({} if seed is None else {"seed": seed}),
                )
                break
            except Exception as exc:
                message = str(exc).lower()
                if "reasoning" in message and not self._no_reasoning_flag:
                    # Some endpoints refuse the disable flag outright and 400 on it.
                    # Drop it and let the model think — the reply is read the same way.
                    self._no_reasoning_flag = True
                    extra_body = {}
                    continue
                status = getattr(exc, "status_code", None)
                if attempt == TRANSIENT_RETRIES or status not in TRANSIENT_STATUS:
                    raise
                time.sleep(min(2 ** attempt, 60) * (1.0 + random.random()))
        _track(self.model, response)
        # A provider can return HTTP 200 with no choices (overload mid-request).
        # Treat it as an empty draw: not cached, so it is retried next run.
        choices = getattr(response, "choices", None)
        if not choices:
            note_degraded("empty_response")
            return ""
        message = choices[0].message
        return (message.content or getattr(message, "reasoning", None) or "").strip()
