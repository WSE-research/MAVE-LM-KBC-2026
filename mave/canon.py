"""The neural canonicaliser: one name per stock exchange, decided by the model.

Bucketing, not cosmetics. Per-string normalization cannot do this job: asked
separately, the model answers "HKEX", "Hong Kong Stock Exchange" and "SEHK" with
three different strings, the vote splits three ways, and a candidate that a
majority of draws actually named falls under the agreement bar.

This replaced 88 hand-written rewrite rules. It degrades to plain
parenthetical-stripping when a call fails, so a failure costs normalization and
never the candidate itself.
"""

import json
import re

from .config import load_prompt, render

PROMPT = "company_canon_string"


def _strip_parentheses(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()


def _json_name(raw: str) -> str:
    try:
        obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError, AttributeError):
        return ""
    name = obj.get("name") if isinstance(obj, dict) else None
    return name.strip() if isinstance(name, str) else ""


def exchange_names(llm, names) -> dict[str, str]:
    """Raw name -> canonical name, for the names of ONE draw."""
    canonical = {str(n): _strip_parentheses(str(n)) for n in names if _strip_parentheses(str(n))}
    if not canonical:
        return canonical
    prompt = load_prompt(PROMPT)
    for raw in list(canonical):
        try:
            name = _json_name(llm.chat(prompt["system"], render(prompt["user"], raw=raw),
                                       temperature=0.0, max_tokens=60))
        except Exception:
            name = ""
        if name:
            canonical[raw] = name
    return canonical
