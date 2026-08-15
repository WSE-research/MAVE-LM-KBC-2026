"""Reading a list of names out of a model response.

Strict JSON first, then a bracket scanner that tolerates prose around the list,
then recovery of the complete items from a list truncated by ``max_tokens`` (the
award relation routinely emits lists long enough to hit the limit), then a
bare-string fallback.
"""

import json
import re

_QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _items(value) -> list[str] | None:
    return [str(x).strip() for x in value if x] if isinstance(value, list) else None


def parse(raw: str) -> list[str]:
    raw = (raw or "").strip()
    try:
        items = _items(json.loads(raw))
        if items is not None:
            return items
    except (json.JSONDecodeError, ValueError):
        pass

    start = raw.find("[")
    if start < 0:
        cleaned = raw.strip("\"'")
        return [cleaned] if cleaned and cleaned != "[]" else []

    depth, in_string, escaped = 0, False, False
    for i, char in enumerate(raw[start:], start):
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    items = _items(json.loads(raw[start:i + 1]))
                    if items is not None:
                        return items
                except (json.JSONDecodeError, ValueError):
                    pass
                break

    # Truncated list: keep the items that came through whole.
    salvaged = [m.group(1).strip() for m in _QUOTED_RE.finditer(raw[start:]) if m.group(1).strip()]
    if len(salvaged) >= 2:
        return salvaged
    cleaned = raw.strip("\"'")
    return [cleaned] if cleaned and cleaned != "[]" else []
