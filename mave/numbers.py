"""Numbers: reading one out of recalled text, and emitting the one the metric rewards.

Both jobs belong to the two numeric relations, ``hasArea`` and ``hasCapacity``.

Reading is not "take the first number". The recalls these prompts elicit carry
decoys — a population, a year, the areas of the comparison entities, a trailing
``km2`` whose exponent parses as a 2 — so every source declares *where* its
number sits: after its own answer marker, or under a named field of the record it
was asked to write out.

Emitting is not "take the most common value" either, and this is where the metric
does the deciding. A numeric answer counts as correct iff it lands within 5 % of
gold, so each candidate ``v`` defines an interval of winning predictions
``[v(1-tol), v(1+tol)]`` — and because that band scales with the hypothesised
*gold* rather than with the prediction, the best point routinely lies *between*
candidates. 60 000 and 65 000 are both won only on ``[61 750, 63 000]``; picking
either candidate, or the heaviest cluster's median, scores strictly worse.
"""

import itertools
import re
import statistics

# A squared-unit token ("km2", "mi²") would otherwise contribute its exponent.
_SQUARED_UNIT_RE = re.compile(r"\b(?:km|mi|m|ft)\s*[2²]", re.I)
_LOOSE_NUM_RE = re.compile(r"[-+]?\d[\d,\.]*")
ANSWER_MARKERS = ("final answer", "answer", "area", "capacity")


def _loose_numbers(text: str) -> list[float]:
    values = []
    for match in _LOOSE_NUM_RE.finditer(text or ""):
        try:
            values.append(float(match.group(0).rstrip(".").replace(",", "")))
        except ValueError:
            continue
    return values


def read(text: str, where: str) -> float | None:
    """Pull this source's number out of one draw.

    ``where`` is either ``marker`` — the first number after the source's last
    answer line — or the name of a field in the record the source was asked to
    write out. Both fall back to the last number in the draw.
    """
    text = _SQUARED_UNIT_RE.sub(" ", text or "")
    if where == "marker":
        pattern = re.compile("(?:%s)\\s*[:\\-]?" % "|".join(map(re.escape, ANSWER_MARKERS)), re.I)
    else:
        pattern = re.compile(re.escape(where) + r"\"?\s*[:=]\s*", re.I)
    for match in reversed(list(pattern.finditer(text))):
        # A named field is read from a short window: a sibling field (a population,
        # a coordinate) can sit right after the one we asked for.
        window = text[match.end():] if where == "marker" else text[match.end():match.end() + 40]
        values = _loose_numbers(window)
        if values:
            return values[0]
    values = _loose_numbers(text)
    return values[-1] if values else None


def text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


# ------------------------------------------------------------------ the emitters

def max_coverage_vote(values: list[tuple[float, float]], tol: float) -> float | None:
    """The single point covered by the most candidate weight — used by ``hasArea``.

    The optimum is always attained at some interval's left endpoint, so the
    continuous search collapses to a scan over those. Ties resolve to the widest
    feasible band, then to the smaller value; the emitted number is the centre of
    that band, which is not cosmetic — on ``hasArea`` the choice of point inside
    the winning band spans 13 F1 points.
    """
    values = [(v, w) for v, w in values if v and v > 0 and w > 0]
    if not values:
        return None
    intervals = [(v * (1 - tol), v * (1 + tol), w) for v, w in values]
    best = None
    for left, _, _ in intervals:
        stabbed = [(lo, hi, w) for lo, hi, w in intervals if lo <= left <= hi]
        lo, hi = max(s[0] for s in stabbed), min(s[1] for s in stabbed)
        if lo > hi:
            continue
        key = (-sum(s[2] for s in stabbed), -(hi - lo), (lo + hi) / 2)
        if best is None or key < best[0]:
            best = (key, (lo + hi) / 2)
    return best[1] if best else statistics.median(v for v, _ in values)


def _won(values: list[tuple[float, float]], tol: float) -> dict[float, frozenset]:
    """Point -> the candidates a prediction at that point would score.

    Only window endpoints and the candidates themselves can be optimal, exactly as
    in :func:`max_coverage_vote`.
    """
    points = sorted({v * (1 - tol) for v, _ in values} | {v * (1 + tol) for v, _ in values}
                    | {v for v, _ in values})
    return {p: frozenset(v for v, _ in values if abs(p - v) / v <= tol) for p in points}


def _centre(won: frozenset, tol: float) -> float | None:
    """The point that scores exactly ``won``, placed in the middle of the band doing so."""
    lo = max(v * (1 - tol) for v in won)
    hi = min(v * (1 + tol) for v in won)
    return (lo + hi) / 2 if lo <= hi else None


def expected_score_vote(values: list[tuple[float, float]], tol: float,
                        kmax: int = 3) -> list[float]:
    """How *many* numbers to emit, decided by the metric — used by ``hasCapacity``.

    Gold holds one number, so a prediction set ``P`` of which one member lands
    scores ``2/(|P|+1)``, and the expected score of ``P`` is that factor times the
    candidate weight covered by the union of its windows. :func:`max_coverage_vote`
    maximises exactly this at ``|P| = 1``; running the same maximisation over the
    size as well introduces no constant. Where the candidates agree, a second
    window covers almost no new weight and the ``2/3`` factor rejects it; where
    they scatter it does, and ``2/3`` is worth paying. The size is therefore a
    per-item comparison of two expected scores, not a relation-wide bet.
    """
    values = [(v, w) for v, w in values if v and v > 0 and w > 0]
    if not values:
        return []
    weight = {v: w for v, w in values}
    won = _won(values, tol)

    # A point whose winnings are contained in another's can never be in an optimal set.
    kept: list[float] = []
    for point in sorted(won, key=lambda p: (-len(won[p]), p)):
        if not any(won[point] < won[other] for other in kept):
            kept.append(point)

    best: tuple[float, list[float]] = (-1.0, [])
    for size in range(1, min(kmax, len(kept)) + 1):
        for combo in itertools.combinations(kept, size):
            covered = frozenset().union(*(won[p] for p in combo))
            score = sum(weight[v] for v in covered) * 2.0 / (size + 1)
            if score > best[0]:
                best = (score, list(combo))
    points = [c for c in (_centre(won[p], tol) for p in best[1]) if c is not None]
    return points or [max_coverage_vote(values, tol)]
