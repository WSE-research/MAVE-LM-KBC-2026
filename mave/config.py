"""The system definition: ``config/system.yaml`` plus ``config/prompts/*.yaml``.

One file describes the whole system. Per relation it declares which elicitations
to run (``sources``), what shape the answer has (``answer``) and how much
agreement it takes to emit a candidate (``quorum``) — and nothing else. Every
prompt is a versioned YAML file, so a prompt change is a readable diff.

An unknown parameter is an error rather than a silent no-op: a config that
declares something the pipeline does not implement would otherwise look like it
runs and quietly score something else.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import CONFIG_DIR, RELATIONS

# Every per-relation parameter the pipeline implements.
KNOWN_PARAMS = {
    "answer",     # point | set | singleton — the shape of the relation's answer
    "sources",    # the elicitations, in declaration order
    "quorum",     # any | corroborated | majority — how many sources must name a candidate
    "panel",      # settles — emit the leading candidate only if the pool agreed on it
    "stance",     # which words in a status field assert, and which deny
    "veto",       # deny — fact testimony may empty a prediction, never invent one
    "normalise",  # exchange — the neural canonicaliser groups spellings of one venue
    "collapse",   # entity | person — drop later names of an entity already emitted
    "jury",       # the three-lens verifier
    "admit",      # jury_majority — admit a single-source candidate the jury affirms
    "emit",       # expected_score — let the metric decide how many numbers to emit
    "align",      # say it in the form the scorer can see
}


@dataclass
class Plan:
    relation: str
    prompt_name: str
    prompt: dict[str, str]
    params: dict


@dataclass
class System:
    name: str
    description: str
    model: str          # name sent over the wire
    cache_slug: str     # cache identity — fixed, so renaming the endpoint still replays
    params_b: float     # published parameter count, in billions
    plans: dict[str, Plan]
    raw: dict = field(repr=False, default_factory=dict)


@lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, str]:
    """A prompt file is a flat mapping of named string fields; sources pick fields by name."""
    path = CONFIG_DIR / "prompts" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (CONFIG_DIR / "prompts").glob("*.yaml"))
        raise FileNotFoundError(f"Unknown prompt {name!r}. Available: {available}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, str)}


def render(template: str, **placeholders: str) -> str:
    """Replace ``{name}`` placeholders, leaving other braces (JSON examples) intact."""
    for key, value in placeholders.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def load_system(path: str | Path = None) -> System:
    path = Path(path) if path else CONFIG_DIR / "system.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = raw["model"]

    unknown = set(raw["relations"]) - set(RELATIONS)
    missing = set(RELATIONS) - set(raw["relations"])
    if unknown or missing:
        raise SystemExit(f"{path.name}: unknown relations {sorted(unknown)}, "
                         f"missing {sorted(missing)}")

    plans = {}
    for relation, spec in raw["relations"].items():
        stale = set(spec["params"]) - KNOWN_PARAMS
        if stale:
            raise SystemExit(f"{path.name}: {relation} declares {sorted(stale)}, which this "
                             f"pipeline does not implement. Known: {sorted(KNOWN_PARAMS)}")
        plans[relation] = Plan(relation=relation, prompt_name=spec["prompt"],
                               prompt=load_prompt(spec["prompt"]), params=spec["params"])

    return System(name=raw["name"], description=raw.get("description", ""),
                  model=model["name"], cache_slug=model["cache"],
                  params_b=float(model["params_b"]), plans=plans, raw=raw)
