"""Three-valued probabilistic truth.

PREDICATES.md section 1 says every predicate returns a probability, that `all`
is min and `any` is max, and that occluded evidence returns null which
propagates to UNVERIFIED. Taken literally, "null propagates" has a pathology:

    all: [ state_is(latch, closed) = 0.02,      # confidently FALSE
           in_zone(vial, glovebox)  = null ]    # occluded

Strict null propagation makes that whole clause null, so a step we can prove is
not satisfied gets reported UNVERIFIED instead. Over a run that inflates the
UNVERIFIED rate, and PLAN.md section 14 lists UNVERIFIED rate as a metric that
is useless if it is too high.

So a truth value is an INTERVAL [lo, hi] over the probability:

    a known probability p  ->  (p, p)
    unknown (occluded)     ->  (0, 1)     - it could be anything

`all` is still min, `any` still max; they just apply to both ends. On fully
known input this is exactly the documented semantics - lo == hi == min - so it
is a strict generalisation, not a redefinition. On partly unknown input it
keeps the information that survives:

    all[(0.02,0.02), (0,1)] -> (0.0, 0.02)   confidently NOT satisfied
    all[(0.91,0.91), (0,1)] -> (0.0, 0.91)   genuinely unknown -> UNVERIFIED

The engine reads the result against one threshold:

    lo >= t  ->  satisfied
    hi <  t  ->  not satisfied
    else     ->  UNVERIFIED

An occluded camera still cannot manufacture a deviation, because widening the
interval can only ever move a value out of "satisfied" and into UNVERIFIED -
never into "not satisfied". That is the property the whole design rests on and
`test_truth.py` asserts it directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: Interval width below which a truth value counts as a point estimate.
_TIGHT = 1e-9


@dataclass(frozen=True, slots=True)
class Truth:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lo <= self.hi <= 1.0):
            raise ValueError(f"Truth bounds out of order or outside [0,1]: ({self.lo}, {self.hi})")

    # -- constructors ----------------------------------------------------
    @staticmethod
    def known(p: float) -> Truth:
        p = min(1.0, max(0.0, float(p)))
        return Truth(p, p)

    @staticmethod
    def certain(b: bool) -> Truth:
        return TRUE if b else FALSE

    # -- queries ---------------------------------------------------------
    @property
    def is_known(self) -> bool:
        return (self.hi - self.lo) < _TIGHT

    @property
    def is_unknown(self) -> bool:
        """Fully uninformative. Distinct from merely imprecise."""
        return self.lo <= _TIGHT and self.hi >= 1.0 - _TIGHT

    @property
    def point(self) -> float | None:
        """The probability, when there is a single one. None when unknown.

        This is the `null` of PREDICATES.md, recovered for logging and display.
        """
        return self.lo if self.is_known else None

    def satisfied(self, threshold: float) -> bool:
        """True only when the whole interval clears the bar."""
        return self.lo >= threshold

    def refuted(self, threshold: float) -> bool:
        """True only when no part of the interval reaches the bar."""
        return self.hi < threshold

    def undecided(self, threshold: float) -> bool:
        return not self.satisfied(threshold) and not self.refuted(threshold)

    def __str__(self) -> str:
        if self.is_unknown:
            return "unknown"
        if self.is_known:
            return f"{self.lo:.2f}"
        return f"[{self.lo:.2f}, {self.hi:.2f}]"


TRUE = Truth(1.0, 1.0)
FALSE = Truth(0.0, 0.0)
UNKNOWN = Truth(0.0, 1.0)


# --------------------------------------------------------------------------
# combinators
# --------------------------------------------------------------------------
def t_all(items: Iterable[Truth]) -> Truth:
    """Conjunction. min, per PREDICATES.md.

    Chosen over a product because a product collapses toward zero with clause
    count, which would penalise a six-clause step for being specific. An empty
    conjunction is vacuously true.
    """
    items = list(items)
    if not items:
        return TRUE
    return Truth(min(t.lo for t in items), min(t.hi for t in items))


def t_any(items: Iterable[Truth]) -> Truth:
    """Disjunction. max, per PREDICATES.md.

    Chosen over noisy-or because noisy-or over-counts correlated evidence, and
    two clauses supported by the same camera are about as correlated as
    evidence gets. An empty disjunction is false.
    """
    items = list(items)
    if not items:
        return FALSE
    return Truth(max(t.lo for t in items), max(t.hi for t in items))


def t_not(t: Truth) -> Truth:
    """Negation. Reflects the interval, so unknown stays unknown."""
    return Truth(1.0 - t.hi, 1.0 - t.lo)
