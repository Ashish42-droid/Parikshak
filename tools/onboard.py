#!/usr/bin/env python3
"""Onboard a procedure: will it run on the deployed build with zero retraining?

    python tools/onboard.py procedures/crx2_colloid_resuspension.yaml

This is the printout behind the demo's closing line. It loads the procedure
through the same gate the engine uses, checks every requirement against the
build manifest the procedure names, and says - in one word - whether onboarding
it needs a data campaign.

Exit code 0 means the file loads and fits the build: it can run tonight.
Exit code 1 means it does not, and every reason is named.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parikshak.io.asr import CommandGrammar  # noqa: E402
from parikshak.pdl.build import BuildError, PerceptionBuild, onboard  # noqa: E402
from parikshak.pdl.loader import Procedure, ProcedureError  # noqa: E402
from parikshak.pdl.validator import Report, walk_predicates  # noqa: E402


def predicates_used(doc: dict) -> list[str]:
    """Every predicate the procedure asks the build to answer."""
    rep = Report()
    names: set[str] = set()
    for step in (doc.get("flow") or {}).get("steps") or []:
        trees = [step.get("preconditions"), step.get("verification")]
        trees += [inv.get("expr") for inv in step.get("invariants") or []]
        trees += [br.get("when") for br in step.get("branch") or []]
        for tree in trees:
            for name, _args, _path in walk_predicates(tree, step.get("id", "?"), rep):
                names.add(name)
    return sorted(names)


def report(path: Path, build_path: Path | None) -> bool:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta = doc.get("procedure") or {}
    target = (doc.get("requires") or {}).get("min_model_version", "")

    try:
        build = PerceptionBuild.load(build_path) if build_path else PerceptionBuild.named(target)
    except BuildError as exc:
        print(f"\n{path.name}: cannot find the build it targets - {exc}")
        return False

    print(f"\nOnboarding {meta.get('id')}  \"{meta.get('title')}\"")
    print(f"  against {build.name}   (motion registry {build.motion_registry})")
    print("-" * 74)

    load_error = ""
    procedure = None
    try:
        procedure = Procedure.load(path, strict=True, build=build)
    except ProcedureError as exc:
        load_error = str(exc)

    result = onboard(doc, build)
    for label, detail, ok in result.rows:
        print(f"  {label:<18} {'OK     ' if ok else 'MISSING'}  {detail}")

    used = predicates_used(doc)
    print(f"  {'predicates':<18} {'OK     '}  {len(used)} used: {', '.join(used)}")

    if procedure is not None:
        grammar = CommandGrammar.for_procedure(procedure)
        tokens = ", ".join(f'"{t}"' for t in procedure.crew_tokens) or "none"
        print(f"  {'crew tokens':<18} {'OK     '}  {tokens}")
        print(f"  {'voice grammar':<18} {'OK     '}  {len(grammar.vocabulary)} words "
              f"(generic commands + this procedure's tokens)")

    print("-" * 74)
    blockers = list(result.errors)
    if load_error and not blockers:
        blockers = [load_error]

    if blockers:
        print("  RETRAINING REQUIRED: YES" if result.retraining_required
              else "  DOES NOT LOAD (authoring errors, no retraining needed)")
        for b in blockers:
            for line in str(b).splitlines():
                print(f"    - {line}")
        return False

    print("  RETRAINING REQUIRED: NO")
    print("  This procedure runs on the deployed build as it stands. No new data,")
    print("  no new weights, no code change - the file is the whole onboarding.")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Will this procedure run with zero retraining?")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--build", type=Path, default=None,
                    help="build manifest (default: builds/<requires.min_model_version>.json)")
    args = ap.parse_args(argv)
    ok = all([report(p, args.build) for p in args.files])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
