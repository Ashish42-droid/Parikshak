"""Predicate evaluation against BeliefFrames.

The tests that matter here are the ones about NOT knowing: occlusion, lost rack
lock, and windows that have not filled yet. Getting "satisfied" right is easy;
the false-alarm number lives entirely in the unknown cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parikshak.belief.frame import (
    BeliefFrame,
    BodyBelief,
    HandBelief,
    MotionBelief,
    ObjectBelief,
    contact_key,
)
from parikshak.belief.trace import read_trace
from parikshak.engine.predicates import PredicateEvaluator, RunFacts
from parikshak.pdl import load_procedure

ROOT = Path(__file__).resolve().parent.parent
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
NOMINAL = ROOT / "traces" / "golden" / "nominal.jsonl"

THR = 0.85


@pytest.fixture(scope="module")
def proc():
    return load_procedure(CSP1)


def obj(pos=(0.0, 0.0, 0.0), *, conf=0.93, visible=True, occluded=False,
        state=None, state_conf=0.91, held_by=None, quat=None, track_id=1):
    return ObjectBelief(visible=visible, conf=conf, pos_rack=pos, quat_rack=quat,
                        track_id=track_id, state=state,
                        state_conf=state_conf if state else 0.0,
                        held_by=held_by, occluded=occluded)


def hand(contact=None, grasp="none", *, present=True, wrist=(0.0, 0.0, 0.0), conf=0.9):
    return HandBelief(present=present, conf=conf, wrist_rack=wrist,
                      contact_with=contact, grasp_type=grasp,
                      grasp_conf=conf if grasp != "none" else 0.0)


def frame(t=0.0, *, objects=None, hands=None, body=None, motion=("idle", 0.88),
          lock=True, occlusion=None, contacts=None, confirmations=()):
    return BeliefFrame(
        t_mono=t, t_utc="2026-12-11T04:30:00Z", frame_lock=lock,
        objects=objects or {}, hands=hands or {"left": hand(), "right": hand()},
        body=body, motion=MotionBelief(cls=motion[0], conf=motion[1]),
        occlusion=occlusion or {}, contacts=contacts or {},
        confirmations=tuple(confirmations))


def ask(proc, frames, tree, facts=None):
    """Replay `frames`, evaluating `tree` on each one. Returns the last value.

    Evaluating every frame is not a test convenience, it is the contract:
    `hold_for` accumulates its window per evaluate() call, so a tree that is
    only checked occasionally has a window full of holes. The engine evaluates
    the active step's tree once per frame for exactly this reason.
    """
    facts = facts or RunFacts()
    ev = PredicateEvaluator(proc)
    result = None
    for f in frames:
        ev.ingest(f)
        result = ev.evaluate(tree, facts)
    assert result is not None, "ask() needs at least one frame"
    return result


# -- rule 1: occluded is not false ----------------------------------------
def test_occluded_object_is_unknown_not_absent(proc):
    f = frame(objects={"vial_a": obj(None, visible=False, occluded=True)})
    assert ask(proc, [f], {"absent": {"entity": "vial_a"}}).is_unknown


def test_genuinely_missing_object_is_absent(proc):
    """Not visible AND not occluded is a real absence. The distinction is the
    whole reason ObjectBelief carries both flags."""
    f = frame(objects={"vial_a": obj(None, visible=False, occluded=False, conf=0.0)})
    assert ask(proc, [f], {"absent": {"entity": "vial_a"}}).satisfied(THR)


def test_occluded_object_is_not_out_of_its_zone(proc):
    """If occlusion read as "outside", `not in_zone(...)` would fire as a
    confident deviation every time a torso crossed the camera."""
    f = frame(objects={"vial_a": obj(None, visible=False, occluded=True)})
    t = ask(proc, [f], {"in_zone": {"entity": "vial_a", "zone": "tray_t1"}})
    assert t.is_unknown
    assert not t.refuted(THR)


def test_unreported_entity_is_unknown(proc):
    """Perception not mentioning an object is not evidence about it."""
    assert ask(proc, [frame()], {"visible": {"entity": "vial_a"}}).is_unknown


def test_occluded_state_is_unknown(proc):
    f = frame(objects={"latch": obj(occluded=True, visible=False, state="closed")})
    assert ask(proc, [f], {"state_is": {"entity": "latch", "state": "closed"}}).is_unknown


# -- rule 2: no rack lock, no geometry ------------------------------------
@pytest.mark.parametrize("tree", [
    {"in_zone": {"entity": "vial_a", "zone": "tray_t1"}},
    {"near": {"a": "vial_a", "b": "vial_b", "dist_m": 0.5}},
    {"inside": {"a": "vial_a", "container": "tray_t1"}},
    {"stable": {"entity": "vial_a", "seconds": 1.0}},
    {"hand_in_zone": {"hand": "any", "zone": "workspace"}},
    {"body_in_zone": {"zone": "workspace"}},
])
def test_geometry_is_unknown_without_rack_lock(proc, tree):
    """AprilTag lock lost means the extrinsics are stale. Evaluating against a
    stale transform is how a rotated rack produces confident nonsense."""
    f = frame(lock=False,
              objects={"vial_a": obj((0, 0, 0)), "vial_b": obj((0.1, 0, 0))},
              body=BodyBelief(conf=0.9, joints_rack={"pelvis": (0, 0, 0)}))
    assert ask(proc, [f], tree).is_unknown


def test_non_geometric_predicates_survive_frame_loss(proc):
    """Losing the rack frame must not blind the whole system - object state is
    still readable, which is what keeps a hazard invariant alive."""
    f = frame(lock=False, objects={"latch": obj(state="closed")})
    t = ask(proc, [f], {"state_is": {"entity": "latch", "state": "closed"}})
    assert t.satisfied(THR)


# -- manipulation ----------------------------------------------------------
def test_grasped_reads_the_contact_head(proc):
    f = frame(objects={"vial_a": obj()},
              hands={"left": hand("vial_a", "grasp"), "right": hand()})
    assert ask(proc, [f], {"grasped": {"entity": "vial_a"}}).satisfied(THR)


def test_reaching_is_not_grasping(proc):
    """`reach` is the hand on its way. Counting it as a grasp is what makes a
    naive detector fire before the crew has touched anything."""
    f = frame(objects={"vial_a": obj()},
              hands={"left": hand("vial_a", "reach"), "right": hand()})
    assert ask(proc, [f], {"grasped": {"entity": "vial_a"}}).refuted(THR)


def test_grasped_honours_the_hand_argument(proc):
    f = frame(objects={"vial_a": obj()},
              hands={"left": hand("vial_a", "grasp"), "right": hand()})
    assert ask(proc, [f], {"grasped": {"entity": "vial_a", "hand": "left"}}).satisfied(THR)
    assert ask(proc, [f], {"grasped": {"entity": "vial_a", "hand": "right"}}).refuted(THR)
    assert ask(proc, [f], {"grasped": {"entity": "vial_a", "hand": "both"}}).refuted(THR)


def test_released_requires_having_held_it_first(proc):
    """An object nobody ever picked up has not been released. Without this,
    every untouched object on the rack reads as released from frame one."""
    f = frame(objects={"vial_a": obj()})
    assert ask(proc, [f], {"released": {"entity": "vial_a"}}).refuted(THR)


def test_released_after_a_grasp_and_a_settle(proc):
    """The settle window is RELEASE_SETTLE_S (2 s) - long enough for its noise
    floor to sit under the 3 cm tolerance - so 2.4 s of let-go frames."""
    held = frame(0.0, objects={"vial_a": obj((0, 0, 0), held_by="left")},
                 hands={"left": hand("vial_a", "grasp"), "right": hand()})
    letgo = [frame(0.2 * i, objects={"vial_a": obj((0, 0, 0))})
             for i in range(1, 13)]
    assert ask(proc, [held, *letgo], {"released": {"entity": "vial_a"}}).satisfied(THR)


def test_released_is_unknown_when_no_hand_was_ever_tracked(proc):
    """A camera-only run has no hand model. Never having seen the vial held is
    then not evidence it was never held, and must not refute a stowing step."""
    no_hands = {"left": hand(present=False), "right": hand(present=False)}
    frames = [frame(0.2 * i, objects={"vial_a": obj((0, 0, 0))}, hands=no_hands)
              for i in range(15)]
    t = ask(proc, frames, {"released": {"entity": "vial_a"}})
    assert not t.refuted(THR) and not t.satisfied(THR)


def test_released_is_undecided_while_the_settle_window_fills(proc):
    """Let go 0.6 s ago: not yet provably staying put, and not evidence it was
    never released. Reading FALSE here got a correctly stowed CRX-2 vial called
    SKIPPED when the crew confirmed completion quickly."""
    held = [frame(0.2 * i, objects={"vial_a": obj((0.05 * i, 0, 0), held_by="left")},
                  hands={"left": hand("vial_a", "grasp"), "right": hand()})
            for i in range(10)]
    letgo = [frame(2.0 + 0.2 * i, objects={"vial_a": obj((0.45, 0, 0))})
             for i in range(1, 4)]
    t = ask(proc, [*held, *letgo], {"released": {"entity": "vial_a"}})
    assert not t.satisfied(THR) and not t.refuted(THR)


def test_released_is_false_while_still_drifting(proc):
    """In microgravity a released object that is still moving has not been
    stowed, it has been launched."""
    held = frame(0.0, objects={"vial_a": obj((0, 0, 0), held_by="left")},
                 hands={"left": hand("vial_a", "grasp"), "right": hand()})
    # As long as the settle test, so this fails on drift, not on a short window.
    drift = [frame(0.2 * i, objects={"vial_a": obj((0.05 * i, 0, 0))})
             for i in range(1, 13)]
    assert not ask(proc, [held, *drift], {"released": {"entity": "vial_a"}}).satisfied(THR)


def test_contacting_reads_the_frame_contact_map(proc):
    f = frame(objects={"vial_a": obj(), "cartridge_sc_a": obj()},
              contacts={contact_key("vial_a", "cartridge_sc_a"): 0.89})
    assert ask(proc, [f], {"contacting": {"a": "vial_a", "b": "cartridge_sc_a"}}).satisfied(THR)
    assert ask(proc, [f], {"contacting": {"a": "cartridge_sc_a", "b": "vial_a"}}).satisfied(THR)


# -- alignment -------------------------------------------------------------
def test_aligned_needs_orientation_for_both(proc):
    """Most classes never get 6-DoF pose. Missing orientation is UNKNOWN, not a
    claim that two things line up."""
    f = frame(objects={"vial_a": obj(quat=(1, 0, 0, 0)), "cartridge_sc_a": obj(quat=None)})
    tree = {"aligned": {"a": "vial_a", "b": "cartridge_sc_a", "tol_deg": 12}}
    assert ask(proc, [f], tree).is_unknown


def test_aligned_within_tolerance(proc):
    f = frame(objects={"vial_a": obj(quat=(1, 0, 0, 0)),
                       "cartridge_sc_a": obj(quat=(1, 0, 0, 0))})
    tree = {"aligned": {"a": "vial_a", "b": "cartridge_sc_a", "tol_deg": 12}}
    assert ask(proc, [f], tree).satisfied(THR)


def test_aligned_outside_tolerance(proc):
    f = frame(objects={"vial_a": obj(quat=(0.906, 0.0, 0.423, 0.0)),
                       "cartridge_sc_a": obj(quat=(1, 0, 0, 0))})
    tree = {"aligned": {"a": "vial_a", "b": "cartridge_sc_a", "tol_deg": 12}}
    assert ask(proc, [f], tree).refuted(THR)


# -- containment -----------------------------------------------------------
def test_inside_an_entity_needs_a_declared_extent(proc):
    """The detector reports a centroid and cannot know how big the bag is. A
    guessed radius would silently decide whether a critical step passed."""
    f = frame(objects={"cartridge_sc_a": obj((0, 0, 0)), "latch": obj((0.01, 0, 0))})
    tree = {"inside": {"a": "cartridge_sc_a", "container": "latch"}}
    assert ask(proc, [f], tree).is_unknown  # latch declares no extent_m


def test_inside_a_declared_container(proc):
    bag = proc.entity("sample_bag_sb01")
    assert bag.extent_m is not None
    f = frame(objects={"cartridge_sc_a": obj((0.0, 0.0, 0.0)),
                       "sample_bag_sb01": obj((0.02, 0.0, 0.0))})
    tree = {"inside": {"a": "cartridge_sc_a", "container": "sample_bag_sb01"}}
    assert ask(proc, [f], tree).satisfied(THR)


# -- temporal --------------------------------------------------------------
def test_hold_for_is_not_satisfied_before_the_window_fills(proc):
    frames = [frame(0.2 * i, objects={"latch": obj(state="closed")}) for i in range(3)]
    tree = {"hold_for": {"seconds": 1.5,
                         "expr": {"state_is": {"entity": "latch", "state": "closed"}}}}
    assert not ask(proc, frames, tree).satisfied(THR)


def test_hold_for_is_satisfied_once_the_window_fills(proc):
    """Regression: the coverage check originally tested the FILTERED window's
    oldest entry, which is always inside the window by construction, so lo was
    pinned at 0 and no hold_for could ever be satisfied."""
    frames = [frame(0.2 * i, objects={"latch": obj(state="closed")}) for i in range(12)]
    tree = {"hold_for": {"seconds": 1.5,
                         "expr": {"state_is": {"entity": "latch", "state": "closed"}}}}
    assert ask(proc, frames, tree).satisfied(THR)


def test_hold_for_refutes_a_break_that_persists(proc):
    """A clause that has genuinely gone false is known false within a frame or
    two - one disagreeing frame is tolerated as noise, a sustained one is not."""
    frames = [frame(0.2 * i, objects={"latch": obj(state="closed")}) for i in range(12)]
    frames += [frame(2.4, objects={"latch": obj(state="open")}),
               frame(2.6, objects={"latch": obj(state="open")})]
    tree = {"hold_for": {"seconds": 1.5,
                         "expr": {"state_is": {"entity": "latch", "state": "closed"}}}}
    assert ask(proc, frames, tree).refuted(THR)


def test_stable_detects_a_settled_object(proc):
    frames = [frame(0.2 * i, objects={"vial_a": obj((0.0, 0.0, 0.0))}) for i in range(20)]
    assert ask(proc, frames, {"stable": {"entity": "vial_a", "seconds": 1.5,
                                         "tol_m": 0.01}}).satisfied(THR)


def _gapped(proc, positions, hidden):
    """Frames at 5 Hz for vial_a at `positions`, with the indices in `hidden`
    occluded - no position, as perception reports a hidden object."""
    return [frame(0.2 * i, objects={"vial_a": obj(None if i in hidden else p,
                                                  occluded=(i in hidden),
                                                  visible=(i not in hidden))})
            for i, p in enumerate(positions)]


def test_a_still_object_briefly_hidden_mid_window_is_stable(proc):
    """Positions before and after a short occlusion agree, so it did not move.
    Demanding an unbroken window left CRX-2's settle and stow steps UNVERIFIED
    under 15-30% occlusion, where an unbroken 5 s stretch is rare."""
    frames = _gapped(proc, [(0.0, 0.0, 0.0)] * 13, hidden={5, 6})
    tree = {"stable": {"entity": "vial_a", "seconds": 2.0, "tol_m": 0.03}}
    assert ask(proc, frames, tree).satisfied(THR)


def test_a_partly_hidden_moving_object_is_unknown_not_refuted(proc):
    """A broken window may confirm stability but never refute it: fewer samples
    mean more noise, and losing information must never create a deviation."""
    frames = _gapped(proc, [(0.05 * i, 0.0, 0.0) for i in range(13)], hidden={5, 6})
    t = ask(proc, frames, {"stable": {"entity": "vial_a", "seconds": 2.0, "tol_m": 0.03}})
    assert not t.satisfied(THR) and not t.refuted(THR)


def test_a_window_with_a_half_mostly_hidden_stays_unknown(proc):
    frames = _gapped(proc, [(0.0, 0.0, 0.0)] * 13, hidden={3, 4, 5, 6, 7})
    t = ask(proc, frames, {"stable": {"entity": "vial_a", "seconds": 2.0, "tol_m": 0.03}})
    assert not t.satisfied(THR) and not t.refuted(THR)


def test_stable_rejects_a_drifting_object(proc):
    frames = [frame(0.2 * i, objects={"vial_a": obj((0.02 * i, 0.0, 0.0))}) for i in range(20)]
    assert ask(proc, frames, {"stable": {"entity": "vial_a", "seconds": 1.5,
                                         "tol_m": 0.01}}).refuted(THR)


def test_stable_is_unknown_if_the_object_was_hidden_in_the_window(proc):
    frames = [frame(0.2 * i, objects={"vial_a": obj((0.0, 0.0, 0.0))}) for i in range(20)]
    frames.append(frame(4.0, objects={"vial_a": obj(None, visible=False, occluded=True)}))
    assert ask(proc, frames, {"stable": {"entity": "vial_a", "seconds": 1.5}}).is_unknown


def test_occurred_latches_a_momentary_action(proc):
    """By the time the unit is running, the press is over. Without `occurred`
    every momentary action would have to be caught in its single frame."""
    facts = RunFacts(current_step="S09")
    ev = PredicateEvaluator(proc)
    tree = {"occurred": {"motion_is": {"motion_class": "press", "min_conf": 0.55}}}
    ev.ingest(frame(0.0, motion=("press", 0.9)))
    assert ev.evaluate(tree, facts, "S09.v").satisfied(THR)
    for i in range(1, 10):
        ev.ingest(frame(0.2 * i, motion=("idle", 0.9)))
    assert ev.evaluate(tree, facts, "S09.v").satisfied(THR)


def test_occurred_does_not_leak_between_steps(proc):
    """A motion seen during step 4 must not satisfy step 9."""
    ev = PredicateEvaluator(proc)
    tree = {"occurred": {"motion_is": {"motion_class": "press", "min_conf": 0.55}}}
    ev.ingest(frame(0.0, motion=("press", 0.9)))
    assert ev.evaluate(tree, RunFacts(current_step="S04"), "v").satisfied(THR)
    ev.ingest(frame(0.2, motion=("idle", 0.9)))
    assert not ev.evaluate(tree, RunFacts(current_step="S09"), "v").satisfied(THR)


# -- engine-scoped ---------------------------------------------------------
def test_elapsed_since_needs_the_step_to_have_completed(proc):
    ev = PredicateEvaluator(proc)
    ev.ingest(frame(70.0))
    tree = {"elapsed_since": {"step_id": "S09", "seconds": 60}}
    assert ev.evaluate(tree, RunFacts()).refuted(THR)
    assert ev.evaluate(tree, RunFacts(step_completed_at={"S09": 5.0})).satisfied(THR)
    assert ev.evaluate(tree, RunFacts(step_completed_at={"S09": 20.0})).refuted(THR)


def test_crew_confirmed_reads_run_facts_not_perception(proc):
    """Perception must never be able to assert that the crew confirmed
    something."""
    tree = {"crew_confirmed": {"token": "label verified"}}
    assert ask(proc, [frame()], tree).refuted(THR)
    assert ask(proc, [frame()], tree,
               RunFacts(confirmations={"label verified"})).satisfied(THR)


def test_occluded_predicate_is_never_unknown(proc):
    """This is the predicate that gates the others. If it could itself be
    unknown there would be nothing to stand on."""
    assert ask(proc, [frame()], {"occluded": {"zone": "workspace"}}).is_known
    f = frame(occlusion={"workspace": 0.7})
    assert ask(proc, [f], {"occluded": {"zone": "workspace"}}).point == pytest.approx(0.7)


# -- integration against the real corpus ----------------------------------
def test_every_csp1_step_is_satisfiable_on_the_nominal_run(proc):
    """The check that a procedure and a perception contract actually fit each
    other. It is what surfaced that BeliefFrame v1.0 could not express S07's
    `aligned`/`contacting`, and that S13's container had no declared extent -
    both of which would have shown up in week 9 as "step never completes"."""
    _, frames = read_trace(NOMINAL)
    facts = RunFacts(
        confirmations={"label verified", "fault logged", "acknowledged"},
        step_completed_at=dict.fromkeys(proc.order, -1000.0),
        flags=dict.fromkeys(
            ["cartridge_retrieved", "vial_retrieved", "vial_attached", "label_ok",
             "cartridge_inserted", "run_started", "run_complete", "latch_open",
             "cartridge_removed", "bag_sealed", "restrained"], True),
    )
    ev = PredicateEvaluator(proc)
    best = dict.fromkeys(proc.order, 0.0)
    for f in frames:
        ev.ingest(f)
        for sid in proc.order:
            step = proc.step(sid)
            if step.verification is None:
                best[sid] = 1.0
                continue
            facts.current_step = sid
            best[sid] = max(best[sid],
                            ev.evaluate(step.verification, facts, f"{sid}.verification").lo)
    never = [s for s, v in best.items() if v < THR]
    assert not never, f"steps that can never be verified on a correct run: {never}"


# -- confidence gates: below the bar is unknown, not false -----------------
def test_a_sighting_below_the_authors_gate_is_unknown_not_false(proc):
    """CSP-1 S04 asks for the cartridge at 0.90. Seen at 0.82 it is neither the
    sighting the author accepts nor an absence. Returning FALSE made it
    accumulate as evidence that the crew skipped the label check."""
    f = frame(objects={"cartridge_sc_a": obj(conf=0.82)})
    t = ask(proc, [f], {"visible": {"entity": "cartridge_sc_a", "min_conf": 0.90}})
    assert t.is_unknown
    assert not t.refuted(THR)


def test_a_sighting_above_the_gate_is_evidence(proc):
    f = frame(objects={"cartridge_sc_a": obj(conf=0.93)})
    t = ask(proc, [f], {"visible": {"entity": "cartridge_sc_a", "min_conf": 0.90}})
    assert t.satisfied(THR)


def test_the_right_motion_below_the_gate_is_unknown(proc):
    f = frame(motion=("insert", 0.50))
    t = ask(proc, [f], {"motion_is": {"motion_class": "insert", "min_conf": 0.55}})
    assert t.is_unknown


def test_no_motion_model_claims_nothing(proc):
    """With no motion model the pipeline reports idle at confidence 0. That is
    not a claim that the crew did something else - a "press" clause must read
    unknown, or every momentary step in a camera-only run looks skipped."""
    t = ask(proc, [frame(motion=("idle", 0.0))],
            {"motion_is": {"motion_class": "press"}})
    assert not t.refuted(THR) and not t.satisfied(THR)


def test_a_motion_count_with_no_confident_frame_is_unknown(proc):
    frames = [frame(0.2 * i, motion=("idle", 0.0)) for i in range(20)]
    t = ask(proc, frames, {"motion_count": {"motion_class": "press", "n": 3, "window_s": 30}})
    assert not t.refuted(THR) and not t.satisfied(THR)


def test_a_motion_count_short_under_a_confident_classifier_is_false(proc):
    frames = [frame(0.2 * i, motion=("idle", 0.9)) for i in range(20)]
    t = ask(proc, frames, {"motion_count": {"motion_class": "press", "n": 3, "window_s": 30}})
    assert t.refuted(THR)


def test_a_different_motion_is_still_false(proc):
    """The TCN is saying something else is happening. That is evidence."""
    f = frame(motion=("reach", 0.95))
    t = ask(proc, [f], {"motion_is": {"motion_class": "insert", "min_conf": 0.55}})
    assert t.refuted(THR)


# -- stability under centroid noise ----------------------------------------
def _jittered(rng, centre, sigma):
    return tuple(c + rng.gauss(0.0, sigma) for c in centre)


def test_a_stationary_object_reads_stable_under_centroid_noise(proc):
    """The estimator measures drift, not the largest single-sample jitter. The
    old max-distance-from-first-sample statistic read a stowed, stationary
    object as drifting 70-100% of the time at 8-15 mm of per-axis noise, and
    the step waiting on it never verified."""
    import random
    rng = random.Random(7)
    tree = {"stable": {"entity": "vial_a", "seconds": 2.0, "tol_m": 0.03}}
    ev = PredicateEvaluator(proc)
    held = 0
    frames = [frame(0.2 * i, objects={"vial_a": obj(_jittered(rng, (0.1, 0.0, 0.1), 0.008))})
              for i in range(150)]
    for i, f in enumerate(frames):
        ev.ingest(f)
        t = ev.evaluate(tree, RunFacts())
        if i >= 10 and t.satisfied(THR):
            held += 1
    assert held / (len(frames) - 10) >= 0.95


def test_a_drifting_object_is_still_caught_under_the_same_noise(proc):
    import random
    rng = random.Random(7)
    frames = [frame(0.2 * i, objects={"vial_a": obj(_jittered(rng, (0.05 * 0.2 * i, 0.0, 0.1), 0.008))})
              for i in range(30)]
    t = ask(proc, frames, {"stable": {"entity": "vial_a", "seconds": 2.0, "tol_m": 0.03}})
    assert t.refuted(THR)


def test_hold_for_tolerates_an_isolated_disagreeing_frame(proc):
    """A strict minimum made hold_for(stable(...)) read FALSE at the moment the
    crew moved on in four of six false SKIPs on the degraded corpus."""
    frames = [frame(0.2 * i, objects={"latch": obj(state="open" if i == 9 else "closed")})
              for i in range(14)]
    tree = {"hold_for": {"seconds": 1.5,
                         "expr": {"state_is": {"entity": "latch", "state": "closed"}}}}
    assert ask(proc, frames, tree).satisfied(THR)
