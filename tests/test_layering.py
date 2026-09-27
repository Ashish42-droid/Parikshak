"""The hard rule from PLAN.md section 5, enforced instead of remembered.

    perception/ may not import engine/
    engine/     may not import perception/
    both import belief/

Stated as a rule in a document, this gets broken in week 6 by someone who needs
one function and is in a hurry. Stated as a failing test, it does not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "parikshak"

FORBIDDEN = [
    ("perception", "engine"),
    ("engine", "perception"),
]


def imported_modules(path: Path) -> set[str]:
    """Every dotted module name this file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize(("subpkg", "banned"), FORBIDDEN)
def test_layer_isolation(subpkg: str, banned: str) -> None:
    banned_prefix = f"parikshak.{banned}"
    offences: list[str] = []
    for src in sorted((PKG / subpkg).rglob("*.py")):
        for mod in imported_modules(src):
            if mod == banned_prefix or mod.startswith(banned_prefix + "."):
                offences.append(f"{src.relative_to(PKG.parent)} imports {mod}")
    assert not offences, (
        f"parikshak/{subpkg}/ must not import parikshak/{banned}/:\n  "
        + "\n  ".join(offences)
    )


def test_belief_depends_on_nothing_in_parikshak_but_itself() -> None:
    """belief/ is the contract. If it grows a dependency on a layer above it,
    the layers can no longer be built in parallel."""
    offences: list[str] = []
    for src in sorted((PKG / "belief").rglob("*.py")):
        for mod in imported_modules(src):
            if mod.startswith("parikshak.") and not mod.startswith("parikshak.belief"):
                offences.append(f"{src.relative_to(PKG.parent)} imports {mod}")
    assert not offences, "belief/ must not import other parikshak layers:\n  " + "\n  ".join(offences)
