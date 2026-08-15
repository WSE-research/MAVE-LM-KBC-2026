"""Bridge to the official evaluator's own string normalization.

Votes and de-duplication inside the pipeline must use the exact normalization the
scorer counts with, or two spellings it would score as one answer are counted as
two. So it is imported from the upstream-owned ``evaluate.py`` rather than copied.
"""

import importlib.util
import threading

from . import ROOT

_LOCK = threading.Lock()
_normalize = None


def normalize_string(text: str) -> str:
    global _normalize
    if _normalize is None:
        with _LOCK:
            if _normalize is None:
                spec = importlib.util.spec_from_file_location("mave_official_evaluate",
                                                              ROOT / "evaluate.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _normalize = module.normalize_string
    return _normalize(text)
