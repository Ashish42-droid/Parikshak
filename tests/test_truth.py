"""Three-valued truth, and the one property the whole design rests on."""

from __future__ import annotations

import pytest

from parikshak.engine.truth import FALSE, TRUE, UNKNOWN, Truth, t_all, t_any, t_not


def test_known_is_a_point():
    t = Truth.known(0.9)
    assert t.is_known and t.point == pytest.approx(0.9)


def test_unknown_is_the_whole_interval():
    assert UNKNOWN.is_unknown
    assert UNKNOWN.point is None


def test_bounds_must_be_ordered():
    with pytest.raises(ValueError):
        Truth(0.8, 0.2)


# -- combinators, on fully known input, are exactly the documented semantics
def test_all_is_min_on_known_input():
    assert t_all([Truth.known(0.9), Truth.known(0.4)]).point == pytest.approx(0.4)


def test_any_is_max_on_known_input():
    assert t_any([Truth.known(0.9), Truth.known(0.4)]).point == pytest.approx(0.9)


def test_not_is_one_minus():
    assert t_not(Truth.known(0.3)).point == pytest.approx(0.7)


def test_all_does_not_penalise_clause_count():
    """PREDICATES.md picked min over a product for exactly this reason: a
    six-clause step should not score lower than a one-clause step for being
    specific."""
    six = t_all([Truth.known(0.9)] * 6)
    one = t_all([Truth.known(0.9)])
    assert six.point == pytest.approx(one.point)


def test_empty_conjunction_is_vacuously_true():
    assert t_all([]) == TRUE


def test_empty_disjunction_is_false():
    assert t_any([]) == FALSE


# -- three-valued behaviour ------------------------------------------------
def test_unknown_does_not_mask_a_refutation():
    """The pathology strict null-propagation has: a clause we can prove false
    should stay false even when a sibling is occluded, or the UNVERIFIED rate
    inflates with cases we actually decided."""
    t = t_all([Truth.known(0.02), UNKNOWN])
    assert t.refuted(0.85)
    assert not t.undecided(0.85)


def test_unknown_does_block_a_confirmation():
    """The other direction must NOT hold: a satisfied clause plus an occluded
    sibling is not a satisfied conjunction."""
    t = t_all([Truth.known(0.91), UNKNOWN])
    assert not t.satisfied(0.85)
    assert t.undecided(0.85)


def test_not_of_unknown_is_unknown():
    assert t_not(UNKNOWN).is_unknown


def test_the_three_verdicts_are_exclusive_and_total():
    for t in (TRUE, FALSE, UNKNOWN, Truth(0.2, 0.9), Truth.known(0.85)):
        verdicts = [t.satisfied(0.85), t.refuted(0.85), t.undecided(0.85)]
        assert sum(verdicts) == 1, f"{t} gave {verdicts}"


# -- the load-bearing property --------------------------------------------
@pytest.mark.parametrize("lo,hi", [(0.9, 0.9), (0.86, 0.99), (0.4, 0.4), (0.0, 1.0)])
def test_losing_information_never_creates_a_deviation(lo, hi):
    """Occlusion widens intervals. Widening must be able to move a value OUT of
    "satisfied" and into UNVERIFIED, but must never move it INTO "refuted" -
    that is precisely the guarantee that an occluded camera cannot manufacture
    an alert. Everything else in the system leans on this.
    """
    exact = Truth(lo, hi)
    widened = Truth(min(lo, 0.0), max(hi, 1.0))
    if not exact.refuted(0.85):
        assert not widened.refuted(0.85)


def test_widening_a_refuted_value_only_ever_softens_it():
    exact = Truth.known(0.1)
    assert exact.refuted(0.85)
    widened = t_all([exact, UNKNOWN])
    assert widened.hi <= exact.hi  # min keeps the refutation
