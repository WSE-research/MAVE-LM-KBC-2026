"""Response cache — the reason this repository reproduces exactly.

A model response is keyed by everything that produced it: the model's cache
identity, the two message turns, the temperature, the token limit and the seed.
Re-running the shipped system therefore replays the shipped draws instead of
sampling new ones, which costs nothing and — more importantly — gives the same
numbers on every machine.

That last part is not pedantry. One model slug is served by several endpoints at
different quantizations, and an unpinned run is a different mixture of them every
time; the same configuration drawn fresh scored 0.02 below the file it shipped.
A response cache is therefore **not** a regenerable artifact, which is why the
draw set behind the published predictions is committed here rather than ignored.

Two layers, looked up in this order:

``cache/<slug>.jsonl.gz``   shipped, read-only — the published draw set
``cache/<slug>.jsonl``      local, appended to — anything drawn since

Running with ``offline=True`` turns a miss into an error instead of an API call,
so a reproduction run either replays the shipped draws or fails loudly.
"""

import gzip
import hashlib
import json
import threading

from . import CACHE_DIR


class CacheMiss(RuntimeError):
    """Raised instead of calling the API when running offline."""


_LOCK = threading.Lock()
_LOADED: dict[str, dict[str, str]] = {}   # slug -> key -> response text
_STATS = {"hits": 0, "misses": 0}


def _digest(messages: list[dict], temperature: float, max_tokens: int, seed: int | None) -> str:
    payload = json.dumps({"m": messages, "t": temperature, "mt": max_tokens, "s": seed},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request_key(slug: str, messages, temperature, max_tokens, seed) -> str:
    """Globally unique key for one request (used to de-duplicate in-flight calls)."""
    return f"{slug}:{_digest(messages, temperature, max_tokens, seed)}"


def _file(slug: str, suffix: str = ".jsonl"):
    return CACHE_DIR / (slug.replace("/", "_") + suffix)


def _read(path, entries: dict[str, str], opener) -> None:
    if not path.exists():
        return
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                entries[record["key"]] = record["text"]


def _entries(slug: str) -> dict[str, str]:
    if slug not in _LOADED:
        entries: dict[str, str] = {}
        _read(_file(slug, ".jsonl.gz"), entries, gzip.open)   # shipped draws first…
        _read(_file(slug), entries, open)                     # …local draws win on a clash
        _LOADED[slug] = entries
    return _LOADED[slug]


def get(slug, messages, temperature, max_tokens, seed) -> str | None:
    with _LOCK:
        # An empty entry counts as a miss: empty responses are transient failures,
        # and replaying one would silently weaken the run.
        text = _entries(slug).get(_digest(messages, temperature, max_tokens, seed)) or None
        _STATS["hits" if text is not None else "misses"] += 1
        return text


def put(slug, messages, temperature, max_tokens, seed, text: str) -> None:
    if not text:
        return
    key = _digest(messages, temperature, max_tokens, seed)
    with _LOCK:
        entries = _entries(slug)
        if key in entries:
            return
        entries[key] = text
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_file(slug), "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")


def stats() -> dict:
    with _LOCK:
        return dict(_STATS)
