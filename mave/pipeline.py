"""One pipeline, six relations.

Every relation runs the same flow. Only the declarations in ``config/system.yaml``
differ: which questions to ask, how many draws each is worth, and what shape of
answer the task defines.

    for each declared source
        draws   = elicit(source)      # greedy, a dense sample, a sweep, or two-step recall
        names   = read(draw)          # number | json | list | marker
        names   = normalise(names)    # group the spellings of one entity
        voice   = the source's draws, normalised to ONE voice
    support(c)  = mean over the sources of the fraction of that source's draws asserting c
    emit        = by the declared answer shape
        point       -> the number with the best expected score under the metric
        singleton   -> the leading candidate, iff the pool settled on it
        set         -> every candidate clearing the quorum, else the leading one
                       iff the pool settled on it

**Nothing here is fitted.** Grep the decision layer and the only numbers that
decide anything are ``> ½`` (the meaning of "the draws settle rather than
scatter"), a count of agreeing sources, the jury's own 2-of-3, and the official
scorer's 5 % tolerance — which is imported, not chosen. What a relation declares
is its prompts, its answer shape (read off the task definition) and its agreement
quorum, and the quorum follows from set-F1 arithmetic: include a candidate whose
probability of being right exceeds roughly half the achievable F1.

Two rules run through all of it.

**Silence is not denial.** A source that recalls nothing is ignorant, not
negative, and never counts as evidence against. Confusing the two cost 0.03 F1 on
the company relation.

**One source, one voice.** A source's draws are normalised to total 1, so asking
one framing nine times cannot out-shout another asked once.
"""

import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import SCORER_TOLERANCE, numbers
from .canon import exchange_names
from .client import note_degraded
from .config import render
from .metric import normalize_string
from .names import collapse_person_names, collapse_same_entity
from .parsing import parse


@dataclass
class Context:
    """Everything one relation's pipeline run needs."""

    relation: str
    prompt: dict[str, str]
    params: dict
    llm: object


@dataclass
class Voice:
    """One source's testimony, its own draws normalised to a single voice."""

    share: dict[str, float] = field(default_factory=dict)   # candidate -> fraction of draws
    yes: float = 0.0    # fraction of draws asserting the subject has an object
    no: float = 0.0     # fraction explicitly denying it


# ------------------------------------------------------------------------- elicit

def _draws(ctx: Context, system: str, user: str, spec: dict, max_tokens: int) -> list[str]:
    """One greedy draw plus ``draws - 1`` sampled ones; ``greedy: false`` samples all.

    Two sources sharing a prompt must declare distinct ``seed`` bases, or their
    sampled draws collapse onto the same cached responses.
    """
    seed, temperature = spec.get("seed", 1000), spec.get("temperature", 0.7)
    out = []
    for d in range(spec.get("draws", 1)):
        if d == 0 and spec.get("greedy", True):
            out.append(ctx.llm.chat(system, user, temperature=0.0, max_tokens=max_tokens))
        else:
            out.append(ctx.llm.chat(system, user, temperature=temperature,
                                    max_tokens=max_tokens, seed=seed + d))
    return out


def _sweep(ctx: Context, subject: str, source: dict) -> list[dict] | None:
    """The per-call placeholder bindings of a sweep source, or None for a plain one.

    ``sweep`` is a literal list — decades, disciplines — asking the question the
    way the answer list is actually organised, which is how a long recall stops
    collapsing onto its most famous members. ``years`` is the same idea over a span
    the model has to name first.
    """
    if "sweep" in source:
        return source["sweep"]
    if "years" not in source:
        return None
    spec = source["years"]
    span = parse(ctx.llm.chat(ctx.prompt[spec["system"]],
                              render(ctx.prompt[spec["ask"]], subject=subject),
                              max_tokens=spec["max_tokens"]))
    try:
        first, last = int(span[0]), int(span[1])
    except (IndexError, ValueError):
        first = last = 0
    return [{spec["var"]: str(year)}
            for year in range(last, max(first, last - spec["span"]) - 1, -1)]


def _elicit(ctx: Context, subject: str, source: dict) -> list[str]:
    """Every raw response one source produces."""
    system = ctx.prompt[source.get("system", "system")]
    max_tokens = source.get("max_tokens", 512)
    ask = source.get("ask")

    if "route" in source:
        # Ask in the language the subject is documented in. On companies the
        # home-market question reached the best recall of any source at the lowest
        # false-listing rate — better on both axes than anything asked in English.
        spec = source["route"]
        code = re.sub(r"[^a-z]", "", ctx.llm.chat(
            ctx.prompt[spec["system"]], render(ctx.prompt[spec["ask"]], subject=subject),
            max_tokens=spec["max_tokens"]).strip().lower())[:2]
        ask = spec["prefix"] + code
        if ask not in ctx.prompt:
            ask = spec["prefix"] + spec["fallback"]
        if ask not in ctx.prompt:
            return []

    if "context" in source:
        # Two-step: recall a free-text account, then read the answer off it. The
        # diversity belongs in the ACCOUNT — each draw re-samples it, so every draw
        # is a fresh attempt at remembering rather than a reworded one, and the
        # extraction stays greedy. Sampling the extraction instead would only
        # reword a single recall attempt.
        spec = source["context"]
        return [ctx.llm.chat(system, render(ctx.prompt[ask], subject=subject, context=account),
                             temperature=0.0, max_tokens=max_tokens)
                for account in _draws(ctx, ctx.prompt[spec["system"]],
                                      render(ctx.prompt[spec["ask"]], subject=subject),
                                      spec, spec["max_tokens"])]

    sweep = _sweep(ctx, subject, source)
    if sweep is None:
        return _draws(ctx, system, render(ctx.prompt[ask], subject=subject), source, max_tokens)

    # A sweep's calls are independent by construction — each binds a different
    # decade, discipline or year — so they run concurrently. ``map`` preserves order,
    # which matters because the award relation sweeps 130 years inside a single item
    # and the runner's own parallelism is across items, not within one.
    def one(call: dict) -> str:
        bindings = {k: str(v) for k, v in call.items() if k != "ask"}
        return ctx.llm.chat(system, render(ctx.prompt[call.get("ask", ask)],
                                           subject=subject, **bindings),
                            temperature=0.0, max_tokens=max_tokens)

    if len(sweep) < 4:
        return [one(call) for call in sweep]
    with ThreadPoolExecutor(max_workers=min(8, len(sweep))) as pool:
        return list(pool.map(one, sweep))


# --------------------------------------------------------------------------- read

def json_object(raw: str) -> dict | None:
    """Forgiving ``{...}`` extraction. None means the draw did not parse — which is
    ignorance, not a denial."""
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    for text in (raw[start:end + 1], raw[start:end + 1].replace("'", '"')):
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _stance(value, vocabulary: dict) -> int:
    """What a draw's status field says: +1 asserts, -1 denies, 0 nothing usable.

    Denial words are tested first: "delisted" and "no longer listed" both contain
    "listed", and reading either as an assertion costs a whole item.
    """
    text = ("" if value is None else str(value)).strip().lower()
    if not text:
        return 0
    if any(word in text for word in vocabulary.get("deny", ())):
        return -1
    if any(word in text for word in vocabulary.get("assert", ())) \
            and "not" not in text and not text.startswith("un"):
        return 1
    return 0


def _fields(obj: dict, keys: list[str], split: bool) -> list[str]:
    """The first of the declared fields that holds anything."""
    for key in keys:
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = re.split(r"[;,]", value) if split else [value]
        elif not isinstance(value, list):
            value = [value]
        found = [str(v).strip() for v in value if str(v).strip()]
        if found:
            return found
    return []


def _read(ctx: Context, text: str, source: dict) -> tuple[list[str], int | None]:
    """One draw -> (names, the stance it states, if it states one)."""
    mode = source.get("read", "list")

    if mode == "number":
        value = numbers.read(text, source["from"])
        return ([] if value is None else [repr(value)]), None

    if mode == "marker":
        # Free-text recall closed by an explicit "<MARKER>: a, b" line. No marker
        # line at all means the source said nothing usable, so it abstains.
        match = re.search(source["marker"] + r"\s*:\s*(.+)", text or "", re.I)
        if not match:
            return [], None
        tail = match.group(1).strip()
        if tail.lower().startswith(("none", "n/a")):
            return [], -1       # the one place a marker line states a denial
        return [part.strip(" .") for part in tail.split(",") if part.strip(" .")], None

    if mode == "json":
        obj = json_object(text)
        if obj is None:
            return [], None
        found = _fields(obj, source["field"], ctx.params["answer"] == "set") \
            if "field" in source else []
        stance = _stance(obj.get(source["status"]), ctx.params.get("stance", {})) \
            if "status" in source else None
        return found, stance

    return parse(text), None


# ------------------------------------------------------------------------ the voice

def _normalise(ctx: Context, names: list[str]) -> list[str]:
    """Group the spellings of one entity before the vote, so agreement is not split
    across wordings of the same answer."""
    if ctx.params.get("normalise") == "exchange":
        return sorted(set(exchange_names(ctx.llm, names).values()))
    return names


def _vote_key(ctx: Context, name: str) -> str:
    """The official evaluator's own normalization, so two spellings it would score as
    one answer are counted once. Numbers vote as themselves."""
    return name if ctx.params["answer"] == "point" else normalize_string(name)


def _voice(ctx: Context, subject: str, source: dict, spellings: dict[str, Counter]) -> Voice:
    denials = ctx.params.get("stance", {}).get("deny", ())
    share: Counter = Counter()
    yes = no = 0
    texts = _elicit(ctx, subject, source)
    for text in texts:
        found, stance = _read(ctx, text, source)
        found = _normalise(ctx, [n for n in found if n and n.lower() not in denials])
        if stance is None:
            # Naming something is the assertion; naming nothing is not a denial.
            stance = 1 if found else 0
        yes += stance > 0
        no += stance < 0
        for name in dict.fromkeys(found):
            key = _vote_key(ctx, name)
            if key:
                share[key] += 1
                spellings[key][name] += 1
    total = max(len(texts), 1)
    return Voice({c: n / total for c, n in share.items()}, yes / total, no / total)


# ------------------------------------------------------------------------- verify

QUORUM = {
    "any": lambda support, sources: sources >= 1,
    "corroborated": lambda support, sources: sources >= 2,
    "majority": lambda support, sources: support > 0.5,
}

# Three lenses on one concrete claim. Each is a different QUESTION, not a
# rewording: the claim asked straight, the role the entity plays (a place of death
# against birth, career or burial), and the question asked open and only then
# compared. Verifying one concrete pair is a different question from asking the
# pool again — which is the point, because where the pool does not settle, the
# support of claims that turn out right and claims that turn out wrong is
# indistinguishable.
_LENSES = (("jury_direct_system", "jury_direct_user", "yes"),
           ("jury_role_system", "jury_role_user", "role"),
           ("jury_open_system", "jury_open_user", "same"))


def _jury(ctx: Context, subject: str, claim: str) -> int:
    """How many of the three lenses affirm the concrete claim."""
    spec = ctx.params["jury"]
    votes = 0
    for system_field, user_field, kind in _LENSES:
        try:
            text = ctx.llm.chat(ctx.prompt[system_field],
                                render(ctx.prompt[user_field], subject=subject,
                                       **{spec["var"]: claim}),
                                temperature=0.0, max_tokens=80)
            if kind == "yes":
                votes += (text or "").strip().lower().startswith("y")
            elif kind == "role":
                votes += str((json_object(text) or {}).get("role", "")).lower() == spec["role"]
            else:
                votes += (json_object(text) or {}).get("same") is True
        except Exception:
            # A lens that fails is not really a No, but the jury has no third state.
            # Tolerated so one bad call cannot kill an item; recorded so a run
            # degraded by rate limiting cannot look clean.
            note_degraded("jury_lens_failed")
    return votes


# --------------------------------------------------------------------------- emit

_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*$")
_HONORIFIC_RE = re.compile(
    r"^(?:sir|dame|dr|prof|professor|mr|mrs|ms|lord|lady|rev|hon)\.?\s+", re.IGNORECASE)


def _align_to_metric(names: list[str]) -> list[str]:
    """Say the same thing in the form the scorer can see, then drop what it reads as a repeat.

    Not a judgement about any relation — a correction for two places where our
    spelling and ``evaluate.normalize_string`` disagree, both found by reading the
    evaluator rather than by measuring the model, and both costing an item twice
    over: once as a false positive, once as a gold group left unmatched.

    - A trailing parenthetical. The scorer turns punctuation into a space instead
      of dropping the bracket, so "Russia (Kaliningrad Oblast)" and "Russia"
      normalise apart.
    - A leading honorific. Gold records the person, so "Sir William Henry Bragg"
      normalises away from "william henry bragg". Guarded: what remains must keep
      two tokens, or the match was part of the name and not a title — without that
      the rule eats the "Lady" of Lady Gaga.

    Then de-duplicate under the scorer's own normalization: precision divides by
    the number of predictions while true positives count deduplicated normal forms,
    so two spellings of one entity can never both score.
    """
    kept, seen = [], set()
    for name in names:
        base = _QUALIFIER_RE.sub("", name).strip() or name
        stripped = _HONORIFIC_RE.sub("", base).strip()
        if len(stripped.split()) >= 2:
            base = stripped
        key = normalize_string(base)
        if key not in seen:
            seen.add(key)
            kept.append(base)
    return kept


def answer(ctx: Context, subject: str) -> list[str]:
    """Predict the objects of one (subject, relation) item."""
    params = ctx.params
    spellings: dict[str, Counter] = defaultdict(Counter)
    heard = [(source, _voice(ctx, subject, source, spellings)) for source in params["sources"]]

    # A source may be declared ``pool: false``: it testifies about the FACT (does
    # this subject have an object at all) without putting candidates in the pool.
    pool = [voice for source, voice in heard if source.get("pool", True)]
    support: dict[str, float] = defaultdict(float)
    sources_naming: Counter = Counter()
    for voice in pool:
        for candidate, share in voice.share.items():
            support[candidate] += share / max(len(pool), 1)
            sources_naming[candidate] += 1

    if params["answer"] == "point":
        # An empty answer scores the same zero as a wrong one and the gold spans
        # three orders of magnitude, so there is deliberately nothing to fall back
        # on. One number by default, for an arithmetic reason rather than because
        # the relation is single-valued: gold holds one value, so k predictions
        # score 2/(k+1) when one lands, and a second value pays only if it lands
        # more than F1/2 of the time.
        candidates = [(float(c), w) for c, w in support.items()]
        if params.get("emit") == "expected_score":
            return [numbers.text(v)
                    for v in numbers.expected_score_vote(candidates, SCORER_TOLERANCE)]
        value = numbers.max_coverage_vote(candidates, SCORER_TOLERANCE)
        return [] if value is None else [numbers.text(value)]

    if not support:
        return []
    ranked = sorted(support, key=lambda c: -support[c])   # stable: ties keep first-seen order
    leading = ranked[0]

    def settles() -> bool:
        """The pool itself agreed on the leading candidate — a plain majority of the
        source voices, so it adds no fitted number."""
        if params["panel"] != "settles":
            raise ValueError(f"unknown panel rule {params['panel']!r}")
        return support[leading] > 0.5

    testimony = [voice for source, voice in heard if source.get("fact", True)]
    if params.get("veto") == "deny" \
            and sum(v.no for v in testimony) > sum(v.yes for v in testimony):
        # An explicit denial outranks the pool. Not a violation of "silence is not
        # denial": ``no`` counts only draws that SAID so, and a source that recalls
        # nothing contributes to neither side.
        return []

    if params["answer"] == "singleton":
        winners = [leading] if settles() else []
    else:
        winners = [c for c in ranked if QUORUM[params["quorum"]](support[c], sources_naming[c])]
        if params.get("admit") == "jury_majority":
            # The quorum drops single-source candidates as a CLASS. ``admit`` asks
            # whether that class is separable, and applies the same arithmetic one
            # level down: emit a candidate whose chance of being right exceeds F1/2.
            # On the border relation, one-source candidates score 0.17 / 0.33 / 0.67
            # / 0.78 by jury votes against a bar of 0.47, so a majority of the jury
            # clears it with room. The same construction loses on companies, which
            # is why it is declared per relation rather than built into the quorum.
            winners += [c for c in ranked
                        if c not in winners and sources_naming[c] == 1
                        and _jury(ctx, subject, spellings[c].most_common(1)[0][0]) >= 2]
        if not winners and params.get("panel") and settles():
            winners = [leading]

    out = [spellings[c].most_common(1)[0][0] for c in winners]
    if params.get("align"):
        out = _align_to_metric(out)
    # Emitting one entity twice wins at most one bipartite match, so the second name
    # is a false positive for nothing. Best-supported first, so the better-supported
    # spelling survives.
    if params.get("collapse") == "entity":
        out = collapse_same_entity(out)
    elif params.get("collapse") == "person":
        out = collapse_person_names(out)
    return out
