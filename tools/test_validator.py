#!/usr/bin/env python3
"""
Negative tests for the PDL validator.

A validator that has only ever returned PASS is not evidence of anything. Each
case below mutates the known-good procedure in one specific way and asserts the
validator rejects it with the expected message. These are the authoring mistakes
a payload engineer will actually make at 2am.

    python tools/test_validator.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parikshak.pdl.validator import Report, check  # noqa: E402

GOOD = Path(__file__).parent.parent / "procedures" / "csp1_colloid_sample_processing.yaml"


def step(doc, sid):
    return next(s for s in doc["flow"]["steps"] if s["id"] == sid)


# Each case: (name, mutation, substring expected in some error)
CASES = [
    (
        "unknown predicate is rejected",
        lambda d: step(d, "S05").__setitem__(
            "verification", {"all": [{"latch_is_shut": {"entity": "latch"}}]}),
        "unknown predicate",
    ),
    (
        "undeclared entity reference is rejected",
        lambda d: step(d, "S05").__setitem__(
            "verification", {"all": [{"visible": {"entity": "hatch_lever"}}]}),
        "undeclared entity 'hatch_lever'",
    ),
    (
        "detector class outside the trained vocabulary is rejected",
        lambda d: d["entities"]["latch"].__setitem__("detector_class", "torque_wrench"),
        "needs a NEW detection dataset",
    ),
    (
        "state outside the entity's declared states is rejected",
        lambda d: step(d, "S05").__setitem__(
            "verification", {"all": [{"state_is": {"entity": "latch", "state": "ajar"}}]}),
        "state 'ajar' not in",
    ),
    (
        "using a predicate whose capability is undeclared is rejected",
        lambda d: d["requires"]["capabilities"].remove("object_state"),
        "object_state",
    ),
    (
        "motion class the deployed TCN does not emit is rejected",
        # S06's motion clause is wrapped in `occurred`, which scopes it to the
        # whole step. The validator must still see through the combinator.
        lambda d: step(d, "S06")["verification"]["all"][0]["occurred"]["motion_is"]
        .__setitem__("motion_class", "unscrew"),
        "not emitted by the deployed TCN",
    ),
    (
        "unknown predicate nested inside `occurred` is rejected",
        lambda d: step(d, "S06")["verification"]["all"].__setitem__(
            0, {"occurred": {"levitates": {"entity": "vial_a"}}}),
        "unknown predicate",
    ),
    (
        "dangling next target is rejected",
        lambda d: step(d, "S05").__setitem__("next", ["S99"]),
        "unreachable",
    ),
    (
        "duration with min > max is rejected",
        lambda d: step(d, "S05").__setitem__(
            "duration", {"min_s": 90, "nominal_s": 8, "sigma_s": 4, "max_s": 1}),
        "min_s <= nominal_s <= max_s",
    ),
    (
        "unordered group member with the wrong successor is rejected",
        lambda d: step(d, "S02").__setitem__("next", ["S04"]),
        "must declare next: [G_GATHER]",
    ),
    (
        "flag never established by any effect is rejected",
        lambda d: step(d, "S02").__setitem__(
            "preconditions", {"all": [{"assert_true": "hatch_sealed"}]}),
        "never established by any step",
    ),
    (
        "header duration that contradicts the arithmetic is rejected",
        lambda d: d["procedure"].__setitem__("estimated_duration_s", 4200),
        "one of the two is wrong",
    ),
    (
        "cycle in the procedure graph is rejected",
        lambda d: step(d, "S12").__setitem__("next", ["S11"]),
        "cycle in procedure graph",
    ),
    (
        "terminal step declaring a successor is rejected",
        lambda d: step(d, "S14").__setitem__("next", ["S01"]),
        "terminal step must have an empty next",
    ),
    (
        "degenerate zone volume is rejected",
        lambda d: d["zones"].__setitem__(
            "tray_t1", {"type": "box", "min": [0.5, 0.5, 0.5], "max": [0.1, 0.1, 0.1]}),
        "degenerate box",
    ),
    (
        "step with no verification is rejected",
        lambda d: step(d, "S05").pop("verification"),
        "missing verification",
    ),
    (
        "crew_confirmed token that is not declared in crew_tokens is rejected",
        # The one-letter typo that would otherwise be discovered at the rack,
        # when the crew says the right words and nothing happens.
        lambda d: step(d, "FAULT_END").__setitem__(
            "verification", {"all": [{"crew_confirmed": {"token": "acknowleged"}}]}),
        "not declared in crew_tokens",
    ),
    (
        "crew token that is also an override token is rejected",
        lambda d: d["crew_tokens"].append("continue anyway"),
        "collides with alert_policy.crew_override",
    ),
    (
        "two crew tokens that sound identical are rejected",
        lambda d: d["crew_tokens"].append("Label, Verified"),
        "same spoken phrase",
    ),
    (
        "crew token that YAML parsed as a boolean is rejected",
        lambda d: d["crew_tokens"].append(True),
        "YAML 1.1 treats bare",
    ),
]


# Warning-level cases. These are advice about the deployed build rather than
# malformed input, so they must NOT become errors - a procedure that is merely
# optimistic about detector precision still has to load.
WARNING_CASES = [
    (
        "a single-frame margin tighter than the detector's noise warns",
        # S13's `inside` margin is 0.02 m; single-frame checks keep 5 sigma.
        lambda d: d["requires"].__setitem__("position_sigma_m", 0.010),
        "under 5x the declared position_sigma_m",
    ),
    (
        "a stability tolerance under its window's noise floor warns",
        # S06 holds stable for 1.5 s at 0.03 m: 8 samples at 5 Hz, floor
        # ~1.98 sigma, so 20 mm of noise (floor 0.040 m) must warn.
        lambda d: d["requires"].__setitem__("position_sigma_m", 0.020),
        "under the noise floor for a",
    ),
    (
        "a tolerance comfortably above the noise floor does not warn",
        lambda d: d["requires"].__setitem__("position_sigma_m", 0.002),
        None,
    ),
    (
        "no declared sigma means the precision check stays dormant",
        lambda d: None,
        None,
    ),
    (
        "an unlatched motion_count in a verification warns",
        lambda d: step(d, "S09")["verification"]["all"].append(
            {"motion_count": {"motion_class": "press", "n": 2, "window_s": 30}}),
        "motion_count in a verification is not latched with occurred",
    ),
    (
        "a motion_count latched with occurred does not warn",
        lambda d: step(d, "S09")["verification"]["all"].append(
            {"occurred": {"motion_count": {"motion_class": "press", "n": 2, "window_s": 30}}}),
        None,
    ),
    (
        "a declared crew token nothing waits for warns",
        lambda d: d["crew_tokens"].append("sample photographed"),
        "no crew_confirmed waits for it",
    ),
]

def main() -> int:
    base = yaml.safe_load(GOOD.read_text(encoding="utf-8"))

    rep = Report()
    check(copy.deepcopy(base), rep)
    if rep.errors:
        print("FAIL  baseline procedure does not validate:")
        for e in rep.errors:
            print("     ", e)
        return 1
    print(f"ok    baseline validates clean ({len(rep.warnings)} warnings)")

    failures = 0
    for name, mutate, expected in CASES:
        doc = copy.deepcopy(base)
        mutate(doc)
        rep = Report()
        check(doc, rep)
        blob = " | ".join(rep.errors)
        if expected in blob:
            print(f"ok    {name}")
        else:
            failures += 1
            print(f"FAIL  {name}")
            print(f"        expected substring: {expected!r}")
            print(f"        got errors: {rep.errors or '(none - mutation was NOT caught)'}")

    for name, mutate, expected in WARNING_CASES:
        doc = copy.deepcopy(base)
        mutate(doc)
        rep = Report()
        check(doc, rep)
        blob = " | ".join(rep.warnings)
        hit = expected is None or expected in blob
        clean = expected is not None or not blob
        if rep.errors:
            failures += 1
            print(f"FAIL  {name} (raised errors, should only warn): {rep.errors[:2]}")
        elif hit and clean:
            print(f"ok    {name}")
        else:
            failures += 1
            print(f"FAIL  {name}")
            print(f"        expected warning: {expected!r}")
            print(f"        got warnings: {rep.warnings or '(none)'}")

    total = len(CASES) + len(WARNING_CASES)
    print(f"\n{total - failures}/{total} negative cases caught")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
