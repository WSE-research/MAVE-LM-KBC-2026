"""Same-entity collapse for the set-valued relations (non-neural).

The evaluator matches predictions to gold by alias set with bipartite matching, so
emitting two *names of the same entity* wins at most one match and the second name
is a false positive for nothing. Collapsing them is a pure precision gain that can
never cost recall.

Two shapes of the same defect, because they need different instruments:

- **Countries** have a small, enumerable set of exonym/endonym and rename pairs,
  so a list is exactly right. This is general knowledge, not something fitted to
  the labels.
- **People** do not. Name variants are generative — ``Robert E. Tarjan`` against
  ``Robert Tarjan``, ``Stuart J. Russell`` against ``Stuart Russell`` — so the
  rule is over name *structure* instead, and deliberately the conservative form:
  merge only when the surname is identical and every shared given-name position is
  equal or an initial of the other. The tempting nickname clause (``fred`` ~
  ``frederick``) measured very slightly better and is the one clause that can join
  two genuinely different people, so it is left out.

Companies are handled elsewhere — venue names need a model to group them, which is
:mod:`mave.canon`.
"""

import re
import unicodedata

# Each set is one real-world entity under several common English names.
ALIAS_GROUPS: list[set[str]] = [
    {"cotedivoire", "ivorycoast", "republicofcotedivoire", "theivorycoast"},
    {"netherlands", "holland", "thenetherlands"},
    {"myanmar", "burma"},
    {"czechia", "czechrepublic"},
    {"easttimor", "timorleste"},
    {"eswatini", "swaziland"},
    {"caboverde", "capeverde"},
    {"northmacedonia", "macedonia", "republicofnorthmacedonia"},
    {"drcongo", "democraticrepublicofthecongo", "democraticrepublicofcongo",
     "congokinshasa", "zaire"},
    {"republicofthecongo", "republicofcongo", "congobrazzaville"},
    {"unitedstates", "unitedstatesofamerica", "usa", "us"},
    {"unitedkingdom", "uk", "greatbritain"},
    {"vatican", "vaticancity", "holysee"},
    {"türkiye", "turkiye", "turkey"},
    {"southkorea", "republicofkorea"},
    {"northkorea", "democraticpeoplesrepublicofkorea"},
    {"russia", "russianfederation"},
    {"iran", "islamicrepublicofiran"},
    {"syria", "syrianarabrepublic"},
    {"laos", "laopeoplesdemocraticrepublic"},
    {"uae", "unitedarabemirates"},
    {"china", "peoplesrepublicofchina", "prchina", "mainlandchina"},
]

HONORIFICS = {"dr", "prof", "professor", "sir", "dame", "mr", "mrs", "ms", "rev",
              "archbishop", "bishop", "president", "senator", "judge", "gen", "general",
              "lord", "lady"}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "kbe", "obe", "cbe", "frs"}
# Surname particles, so "Frederik Willem de Klerk" keeps "de klerk" as the surname.
PARTICLES = {"de", "van", "von", "del", "della", "di", "da", "du", "la", "le", "el", "al",
             "bin", "ibn", "den", "der", "ten", "ter", "af", "av", "dos", "das"}


def _fold(name: str) -> str:
    """Accent-stripped, lowercase, punctuation-free."""
    text = unicodedata.normalize("NFKD", name.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", _fold(name))


_GROUP_OF = {_key(name): index for index, group in enumerate(ALIAS_GROUPS) for name in group}


def collapse_same_entity(candidates: list[str]) -> list[str]:
    """Drop later names denoting an entity an earlier name already covers.

    Order is preserved and the first spelling wins, so the pipeline's own ranking
    decides which surface form is emitted.
    """
    seen_groups: set[int] = set()
    seen_keys: set[str] = set()
    kept: list[str] = []
    for candidate in candidates:
        key = _key(candidate)
        group = _GROUP_OF.get(key)
        if group is not None:
            if group in seen_groups:
                continue
            seen_groups.add(group)
        elif key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(candidate)
    return kept


def _name_parts(name: str) -> tuple[list[str], str]:
    """(given-name tokens, surname), after stripping honorifics and suffixes."""
    parts = [w for w in re.sub(r"[^a-z ]", " ", _fold(name)).split() if w and w not in HONORIFICS]
    while parts and parts[-1] in NAME_SUFFIXES:
        parts.pop()
    if len(parts) < 2:
        return [], (parts[0] if parts else "")
    i = len(parts) - 1
    while i > 1 and parts[i - 1] in PARTICLES:
        i -= 1
    return parts[:i], " ".join(parts[i:])


def same_person(a: str, b: str) -> bool:
    """Identical surname, given names equal or initials of one another.

    Rejects rather than guesses: two different given names on one surname are two
    people, and a missing middle name is tolerated only as a shorter sequence.
    """
    given_a, surname_a = _name_parts(a)
    given_b, surname_b = _name_parts(b)
    if not surname_a or surname_a != surname_b or not given_a or not given_b:
        return False
    for x, y in zip(given_a, given_b):
        if x == y or (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)):
            continue
        return False
    return True


def collapse_person_names(candidates: list[str]) -> list[str]:
    """Drop later spellings of a person an earlier name already covers (order preserved)."""
    kept: list[str] = []
    for candidate in candidates:
        if not any(same_person(candidate, k) for k in kept):
            kept.append(candidate)
    return kept
