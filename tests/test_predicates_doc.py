"""schema/PREDICATES.md is the contract. This keeps it true.

The document says it is the contract between the learned layer and the symbolic
layer, and for several phases it was quietly wrong: `occurred` and `assert_true`
existed in the validator and nowhere in the doc, `crew_confirmed` was documented
as needing no capability while the code required `crew_confirm`, and
`hand_in_zone` was documented without the rack-frame dependency it obviously
has. None of that failed anything, because nothing compared the two.

Now something does. A predicate, combinator or capability dependency that the
validator knows and the document does not is a failing test, so the contract a
judge reads is the contract the code enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from parikshak.pdl.validator import COMBINATORS, VOCAB

DOC = Path(__file__).resolve().parent.parent / "schema" / "PREDICATES.md"
TEXT = DOC.read_text(encoding="utf-8")


def section(number: int) -> str:
    match = re.search(rf"^## {number}\. [^\n]*\n(.*?)(?=^## \d+\. |\Z)", TEXT, re.S | re.M)
    assert match, f"PREDICATES.md has no section {number}"
    return match.group(1)


@pytest.mark.parametrize("name", sorted(VOCAB))
def test_every_predicate_the_validator_accepts_is_documented(name):
    assert f"| `{name}` |" in section(3), (
        f"`{name}` is accepted by the validator but has no row in PREDICATES.md section 3")


def test_the_document_describes_no_predicate_the_validator_rejects():
    documented = set(re.findall(r"^\| `([a-z_]+)` \| `\(", section(3), re.M))
    assert documented == set(VOCAB), (
        f"documented but not accepted: {sorted(documented - set(VOCAB))}; "
        f"accepted but not documented: {sorted(set(VOCAB) - documented)}")


@pytest.mark.parametrize("combinator", sorted(COMBINATORS))
def test_every_combinator_is_documented(combinator):
    assert f"`{combinator}:" in section(1), (
        f"combinator `{combinator}` is accepted but not in PREDICATES.md section 1")


def test_capability_table_matches_what_the_validator_enforces():
    """Union across rows, because a predicate may appear in more than one
    (`aligned` needs the rack frame AND 6-DoF pose)."""
    implied: dict[str, set[str]] = {}
    for line in section(4).splitlines():
        if not line.startswith("| `"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        preds = re.findall(r"`([a-z0-9_]+)`", cols[0])
        caps = re.findall(r"`([a-z0-9_]+)`", cols[1])
        for p in preds:
            implied.setdefault(p, set()).update(caps)
    actual = {name: set(spec["caps"]) for name, spec in VOCAB.items()}
    mismatched = {
        name: {"doc": sorted(implied.get(name, set())), "code": sorted(actual[name])}
        for name in actual if implied.get(name, set()) != actual[name]
    }
    assert not mismatched, f"capability table disagrees with the validator: {mismatched}"
    assert set(implied) == set(actual), f"table lists unknown predicates: {set(implied) - set(actual)}"


def test_crew_tokens_are_documented():
    assert "crew_tokens" in section(6)


def test_the_deployed_build_is_documented():
    body = section(7)
    assert "builds/" in body and "--build" in body
