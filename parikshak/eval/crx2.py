"""CRX-2 world model, step scripts, and golden scenarios.

Fixtures, not runtime. This is the ONLY place outside procedures/ that knows
CRX-2 exists - which is the point: the engine, perception, GUI and I/O layers
run it without a single line that names it, and tests/test_second_procedure.py
asserts exactly that.

Same rack as CSP-1, different starting arrangement: vial B begins in the cold
locker as a returned sample, and there is no cartridge or sample bag on the
bench. The synthesiser takes the starting world as data for the same reason the
engine takes the procedure as data.
"""

from __future__ import annotations

from parikshak.eval import synth
from parikshak.eval.corpus import Injected, Scenario, session_tail_s
from parikshak.eval.synth import RunBuilder

HOME = {
    "vial_a": "tray_t1",
    "vial_b": "locker_l1_cold",
    "latch": "glovebox_interior",
    "process_unit": "glovebox_interior",
    "locker_l1": "locker_l1_cold",
    "restraint": "restraint_zone",
    "crew_hand": "workspace",
}
STATES = {"latch": "closed", "process_unit": "idle"}
#: The vials start in different zones, so no sibling nudge is needed.
OFFSETS: dict[str, float] = {}

AGITATIONS = 10


# ------------------------------------------------------------------ scripts
# Read as a crew member performing the procedure, not as fixture assembly.

def s01_restrain(b: RunBuilder):
    b.motion("reach").hold(3)
    b.restrain().motion("idle").hold(4)


def s02_get_vial_b(b: RunBuilder):
    b.motion("reach").hold(2)
    b.grasp("vial_b", "left").move("vial_b", "workspace").motion("idle").hold(4)


def s03_confirm_idle(b: RunBuilder):
    # Looks at the unit, then says it. The pause is real: a confirmation the
    # instant the step is reached would be below min_s and read as too fast.
    b.motion("idle").hold(2)
    b.confirm("unit idle").hold(1)


def s04_agitate(b: RunBuilder):
    for _ in range(AGITATIONS):
        b.motion("agitate").hold(0.4)
        b.motion("idle").hold(0.4)


def s05_settle(b: RunBuilder):
    b.motion("reach").hold(2)
    b.move("vial_b", "tray_t1").release("vial_b").motion("idle").hold(7)


def s06_confirm_uniform(b: RunBuilder):
    b.hold(2)
    b.confirm("suspension uniform").hold(1)


def s07_return(b: RunBuilder):
    b.motion("reach").hold(1.5)
    b.grasp("vial_b", "left").hold(1)
    b.move("vial_b", "locker_l1_cold").release("vial_b").motion("idle").hold(6)


def s08_confirm_complete(b: RunBuilder):
    b.hold(1)
    b.confirm("run complete").hold(1)


NOMINAL = [
    ("S01", s01_restrain), ("S02", s02_get_vial_b), ("S03", s03_confirm_idle),
    ("S04", s04_agitate), ("S05", s05_settle), ("S06", s06_confirm_uniform),
    ("S07", s07_return), ("S08", s08_confirm_complete),
]
ALL_STEPS = [sid for sid, _ in NOMINAL]


# ---------------------------------------------------------- deviant scripts
def s02_take_vial_a(b: RunBuilder):
    # Reaches into the tray instead of the cold locker. Vial A is a vial,
    # exactly as vial B is - only the declared confusable pair says it is
    # the wrong one.
    b.motion("reach").hold(2)
    b.grasp("vial_a", "right").move("vial_a", "workspace").motion("idle").hold(6)


def s04_near_unit(b: RunBuilder):
    # Agitates right beside the powered processing unit. Every step still
    # completes, so sequence checking alone would never notice.
    b.at("vial_b", tuple(v + 0.03 for v in b.pos["process_unit"]))
    s04_agitate(b)
    b.move("vial_b", "workspace")


def occlude_s05(b: RunBuilder, sid: str):
    # mutate() runs after step `sid`: arm on S04, lift after S05.
    if sid == "S04":
        b.occlude("vial_b", zone="workspace")
    if sid == "S05":
        b.clear_occlusion()


def builder(proc: dict) -> RunBuilder:
    return RunBuilder(proc, home=HOME, states=STATES, offsets=OFFSETS)


def run(proc: dict, **kw) -> RunBuilder:
    return synth.run(proc, nominal=NOMINAL, home=HOME, states=STATES, offsets=OFFSETS, **kw)


# ------------------------------------------------------------------ golden
def build_golden(proc: dict) -> list[dict]:
    """The CRX-2 golden corpus. Same shape as CSP-1's, so the same tool writes it
    and the same replay harness checks it - which is itself part of the claim."""
    specs: list[dict] = []

    def add(name, b, expect, notes):
        specs.append({"name": name, "b": b, "expect": expect, "notes": notes})

    add("nominal", run(proc),
        {"complete": ALL_STEPS, "no_alert": True},
        "All 8 steps performed correctly and in order.")

    add("legal_reorder_S03_S02",
        run(proc, order=["S01", "S03", "S02"] + ALL_STEPS[3:]),
        {"complete": ALL_STEPS, "no_alert": True},
        "Unit confirmed idle before the vial is retrieved. S02 and S03 are an "
        "unordered group, so this is legal and must be silent.")

    add("skip_S04_agitate", run(proc, skip={"S04"}),
        {"skipped": ["S04"]},
        "Vial placed to settle without being agitated. The sample is never "
        "resuspended, and only motion_count can tell.")

    add("skip_S07_return", run(proc, skip={"S07"}),
        {"skipped": ["S07"]},
        "Run declared complete with the vial still on the bench. Critical step.")

    b = builder(proc)
    b.hold(2)
    s01_restrain(b)
    s02_take_vial_a(b)
    s03_confirm_idle(b)
    b.hold(4)
    add("wrong_object_vial_a", b, {"wrong_object": ["S02"]},
        "Vial A taken instead of vial B. CSP-1's wrong-object case, inverted, "
        "with no change anywhere but the procedure file.")

    add("hazard_vial_near_unit", run(proc, replace={"S04": s04_near_unit}),
        {"hazard": ["S04"]},
        "Vial agitated within 12 cm of the powered unit. A geometric invariant, "
        "built on `near`, which CSP-1 does not use.")

    add("occlusion_S05", run(proc, mutate=occlude_s05),
        {"unverified_at_least": ["S05"], "no_alert": True},
        "Crew torso blocks the tray while the vial settles. UNVERIFIED, and "
        "nothing spoken - the step was performed correctly, just unseen.")

    return specs


# ------------------------------------------------------------------ degraded
def corpus_scenarios() -> list[Scenario]:
    """CRX-2's scenarios for the degraded corpus, with injected ground truth.

    The golden cases, scripted as whole sessions: every run - the wrong-object
    one included - goes to the end, and keeps recording for session_tail_s
    after the last action, derived from CRX-2's own final step. Degradation and
    scoring are shared with CSP-1 (corpus.build_corpus, eval.harness), so a
    number here means the same thing as the number there.
    """
    def scripted(**kw):
        def build(proc: dict) -> RunBuilder:
            return run(proc, tail_s=session_tail_s(proc, NOMINAL), **kw)
        return build

    return [
        Scenario("nominal", scripted(), (), {"complete": ALL_STEPS},
                 "Correct run. Must be silent."),
        Scenario("legal_reorder_S03_S02",
                 scripted(order=["S01", "S03", "S02"] + ALL_STEPS[3:]), (),
                 {"complete": ALL_STEPS, "no_alert": True},
                 "S02 and S03 are an unordered group. Either order is correct."),
        Scenario("skip_S04_agitate", scripted(skip={"S04"}),
                 (Injected("SKIP", "S04"),), {"skipped": ["S04"]},
                 "Vial settled without being agitated."),
        Scenario("skip_S07_return", scripted(skip={"S07"}),
                 (Injected("SKIP", "S07"),), {"skipped": ["S07"]},
                 "Run declared complete with the vial still on the bench."),
        Scenario("wrong_object_vial_a", scripted(replace={"S02": s02_take_vial_a}),
                 (Injected("WRONG_OBJECT", "S02"),), {"wrong_object": ["S02"]},
                 "Vial A taken instead of vial B."),
        Scenario("hazard_vial_near_unit", scripted(replace={"S04": s04_near_unit}),
                 (Injected("HAZARD", "S04"),), {"hazard": ["S04"]},
                 "Vial agitated within 12 cm of the powered unit."),
        Scenario("occlusion_S05", scripted(mutate=occlude_s05), (),
                 {"unverified_at_least": ["S05"]},
                 "Tray blocked while the vial settles. UNVERIFIED, never SKIPPED."),
    ]
