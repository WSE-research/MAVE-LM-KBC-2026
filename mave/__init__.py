"""MAVE — a closed-book knowledge-base construction system for LM-KBC 2026.

Given a (subject, relation) pair, predict every correct object from what an
open-weight language model already knows: no retrieval, no fine-tuning, no
external lookups at inference time.

The whole system is one pipeline (:mod:`mave.pipeline`) whose per-relation
differences are declared in ``config/system.yaml`` rather than written in code.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"

RELATIONS = [
    "awardWonBy",
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
    "hasArea",
    "hasCapacity",
    "personHasCityOfDeath",
]

# The official scorer counts a numeric answer correct iff |pred - gold| / gold
# <= 0.05 (``evaluate.numeric_true_positives``). Everything in the pipeline that
# reasons about that band reads it from here, so a change to the scorer cannot
# leave a stale copy behind. It is a property of the task, not a tuning knob.
SCORER_TOLERANCE = 0.05
