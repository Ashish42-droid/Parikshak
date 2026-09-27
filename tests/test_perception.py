"""Perception: contact features, belief assembly, and the detector loop.

The tests that earn their place here are the ones about refusing to guess -
what the pipeline emits when the rack frame is lost, and when an object simply
was not seen. Those two cases are where a perception layer quietly invents the
evidence that makes an engine accuse the wrong person.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from parikshak.belief.frame import GRASP_TYPES, BeliefFrame
from parikshak.perception.backends import (
    ScriptedDetector,
    ScriptedHands,
    ScriptedMotion,
    ScriptedPose,
)
from parikshak.perception.contact import (
    LABELS,
    N_FEATURES,
    ContactHead,
    ContactMLP,
    ContactSmoother,
    GeometricContact,
    features,
)
from parikshak.perception.pipeline import EntityBinding, PerceptionPipeline, PipelineConfig
from parikshak.perception.rackframe import CameraIntrinsics, RackFrameEstimator, TagLayout
from parikshak.perception.types import BodyKeypoints, Detection, HandLandmarks, MotionState

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = ROOT / "racks" / "msg_a_fiducials.json"

BLANK = np.zeros((8, 8, 3), dtype=np.uint8)


def hand(side="left", wrist=(100.0, 100.0), *, aperture=40.0, span=60.0, score=0.9):
    """A synthetic hand: wrist at `wrist`, fingers extending +x, thumb offset by
    `aperture` from the index tip."""
    pts = np.tile(np.asarray(wrist, dtype=float), (21, 1))
    pts[12] = (wrist[0] + span, wrist[1])                    # middle tip sets the span
    pts[8] = (wrist[0] + span * 0.95, wrist[1])              # index tip
    pts[4] = (wrist[0] + span * 0.95, wrist[1] + aperture)   # thumb tip
    pts[16] = (wrist[0] + span * 0.9, wrist[1])
    pts[20] = (wrist[0] + span * 0.8, wrist[1])
    return HandLandmarks(side=side, score=score, points_px=pts)


def box_at(cx, cy, size=40.0, cls="vial", score=0.9, track_id=1, **kw):
    h = size / 2
    return Detection(cls=cls, score=score, box_px=(cx - h, cy - h, cx + h, cy + h),
                     track_id=track_id, **kw)


# -- the contract between perception and belief ----------------------------
def test_contact_labels_match_the_belief_vocabulary():
    """perception and belief must agree on what a hand can be doing. If they
    drift, a trace becomes unparseable and nobody finds out until the run."""
    assert set(LABELS) == set(GRASP_TYPES)


# -- contact features ------------------------------------------------------
def test_feature_vector_has_the_declared_width():
    x = features(hand(), box_at(160, 100))
    assert x.shape == (N_FEATURES,)
    assert np.all(np.isfinite(x))


def test_features_are_scale_free():
    """Everything is normalised by the hand's own span, so a model trained at
    one camera distance transfers to another. That is what makes adding a third
    camera a mounting job rather than a data campaign."""
    near = features(hand(wrist=(100, 100), aperture=40, span=60), box_at(160, 100, 40))
    far = features(hand(wrist=(200, 200), aperture=20, span=30), box_at(230, 200, 20))
    # ratios (indices 0-7) must agree; absolute scores and velocities need not
    assert np.allclose(near[:8], far[:8], atol=1e-9)


def test_closing_speed_is_signed():
    """Approaching and withdrawing must not look the same - it is the whole
    difference between `reach` and `release`."""
    h, d = hand(wrist=(100, 100)), box_at(200, 100)
    approaching = features(h, d, prev_gap=2.0, dt=0.1)
    withdrawing = features(h, d, prev_gap=0.1, dt=0.1)
    assert approaching[8] > 0 > withdrawing[8]


def test_distance_is_to_the_box_not_its_centre():
    """A hand on the rim of a large object is touching it. Centre distance
    would score that the same as a hand a box-width away."""
    # Fingertips reach x = 190. The small box ends at 160 (tips outside it);
    # the large box spans 90-210, so the same tips are already on it.
    h = hand(wrist=(130, 100), span=60)
    small = box_at(150, 100, size=20)
    large = box_at(150, 100, size=120)
    assert features(h, large)[1] == 0.0
    assert features(h, small)[1] > 0.0


# -- the geometric fallback ------------------------------------------------
def test_geometric_fallback_reports_a_closed_hand_on_an_object_as_grasp():
    model = GeometricContact()
    x = features(hand(wrist=(100, 100), aperture=10), box_at(158, 100, size=30))
    assert LABELS[int(np.argmax(model(x)))] == "grasp"


def test_geometric_fallback_calls_an_approaching_open_hand_reach_not_grasp():
    """`reach` is the moment a naive detector fires early. Counting it as a
    grasp makes `grasped(vial_a)` true before the crew has touched anything."""
    model = GeometricContact()
    # Open hand (aperture 55/60 = 0.92, well above CLOSED_APERTURE) closing on
    # an object still 0.6 hand-spans away: reaching, not yet holding.
    x = features(hand(wrist=(100, 100), aperture=55), box_at(210, 100, size=30),
                 prev_gap=3.0, dt=0.1)
    assert LABELS[int(np.argmax(model(x)))] == "reach"


def test_geometric_fallback_reports_nothing_for_a_distant_hand():
    model = GeometricContact()
    x = features(hand(wrist=(100, 100)), box_at(900, 700, size=30))
    assert LABELS[int(np.argmax(model(x)))] == "none"


def test_fallback_probabilities_are_a_distribution():
    model = GeometricContact()
    for gap in (140, 200, 400, 900):
        p = model(features(hand(), box_at(gap, 100)))
        assert p.min() >= 0.0
        assert np.isclose(p.sum(), 1.0)


# -- the learned head ------------------------------------------------------
def test_mlp_rejects_weights_that_disagree_with_the_feature_width():
    """A weights file from an older feature set must fail loudly at load, not
    silently produce confident nonsense."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="features"):
        ContactMLP(w1=rng.normal(size=(N_FEATURES - 3, 8)), b1=np.zeros(8),
                   w2=rng.normal(size=(8, 8)), b2=np.zeros(8),
                   w3=rng.normal(size=(8, len(LABELS))), b3=np.zeros(len(LABELS)))


def test_mlp_rejects_weights_with_the_wrong_class_count():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="classes"):
        ContactMLP(w1=rng.normal(size=(N_FEATURES, 8)), b1=np.zeros(8),
                   w2=rng.normal(size=(8, 8)), b2=np.zeros(8),
                   w3=rng.normal(size=(8, 3)), b3=np.zeros(3))


def test_mlp_forward_pass_is_a_distribution():
    rng = np.random.default_rng(1)
    mlp = ContactMLP(w1=rng.normal(size=(N_FEATURES, 16)), b1=np.zeros(16),
                     w2=rng.normal(size=(16, 16)), b2=np.zeros(16),
                     w3=rng.normal(size=(16, len(LABELS))), b3=np.zeros(len(LABELS)))
    p = mlp(features(hand(), box_at(160, 100)))
    assert p.shape == (len(LABELS),)
    assert np.isclose(p.sum(), 1.0)


def test_head_reports_which_weights_produced_a_run():
    """Goes into the log header. A record that cannot name its weights cannot
    be audited."""
    assert ContactHead().weights_sha256 == "geometric-fallback"
    assert not ContactHead().is_learned


def test_a_hand_is_assigned_at_most_one_object():
    """`grasped(entity)` has no way to say "partially". Forcing the choice here
    keeps the ambiguity where it can be measured."""
    head = ContactHead()
    states = head([hand(wrist=(100, 100), aperture=10)],
                  [box_at(158, 100, size=30, track_id=1),
                   box_at(162, 104, size=30, cls="sample_cartridge", track_id=2)])
    assert len(states) == 1
    assert states[0].track_id in (1, 2)


# -- smoothing -------------------------------------------------------------
def test_smoother_suppresses_a_single_frame_flicker():
    """Per-frame contact flickers, and a `grasped` clause that flickers turns
    into a step that will not verify."""
    from parikshak.perception.types import ContactState
    sm = ContactSmoother(window=5)
    grasp = ContactState("left", "grasp", 0.9, "vial", 1)
    blip = ContactState("left", "none", 0.9)
    out = [sm([grasp])[0] for _ in range(4)]
    out.append(sm([blip])[0])
    assert out[-1].label == "grasp"


# -- the pipeline ----------------------------------------------------------
@pytest.fixture
def rack():
    return RackFrameEstimator(TagLayout.load(LAYOUT),
                              CameraIntrinsics.from_fov(1600, 1200, 65.0))


BINDINGS = {
    "vial_a": EntityBinding("vial_a", "vial"),
    "cartridge_sc_a": EntityBinding("cartridge_sc_a", "sample_cartridge"),
    "latch": EntityBinding("latch", "glovebox_latch"),
}


def make_pipeline(rack, detections=None, hands=None, pose=None, motion=None):
    return PerceptionPipeline(
        BINDINGS, rack,
        detector=ScriptedDetector([detections or []]),
        hands=ScriptedHands([hands or []]),
        pose=ScriptedPose([pose]),
        motion=ScriptedMotion([motion or ("idle", 0.8)]),
        config=PipelineConfig(fps=10.0))


def test_pipeline_emits_a_valid_belief_frame(rack):
    pipe = make_pipeline(rack, detections=[box_at(300, 300, cls="vial", track_id=7)])
    frame = pipe.step(BLANK, 0.0)
    assert isinstance(frame, BeliefFrame)
    assert BeliefFrame.from_json(frame.to_json()) == frame


def test_every_declared_entity_appears_every_frame(rack):
    """The engine asks about entities the procedure declares, not about what
    happened to be detected. A missing key would read as "unreported" and become
    UNKNOWN rather than the intended "not visible"."""
    frame = make_pipeline(rack).step(BLANK, 0.0)
    assert set(frame.objects) == set(BINDINGS)


def test_without_a_rack_lock_there_are_no_coordinates(rack):
    """Rule two. A stale extrinsic reused as if current is how a rotated rack
    produces confident nonsense, so geometry is withheld entirely."""
    det = box_at(300, 300, cls="vial", track_id=7, pos_rack=(0.1, 0.2, 0.3))
    frame = make_pipeline(rack, detections=[det]).step(BLANK, 0.0)
    assert frame.frame_lock is False
    assert frame.objects["vial_a"].pos_rack is None
    assert frame.objects["vial_a"].visible is True     # seen, just not located


def test_an_undetected_entity_is_not_visible_but_also_not_declared_absent(rack):
    """Rule one. `absent` must never fire because a torso was in the way, so
    perception reports "not seen" and leaves occlusion to the mask."""
    frame = make_pipeline(rack).step(BLANK, 0.0)
    vial = frame.objects["vial_a"]
    assert vial.visible is False
    assert vial.occluded is False
    assert vial.conf == 0.0
    assert vial.pos_rack is None


def test_occlusion_mask_is_carried_through(rack):
    frame = make_pipeline(rack).step(BLANK, 0.0, occlusion={"glovebox_interior": 0.8})
    assert frame.zone_occlusion("glovebox_interior") == 0.8


def test_low_scoring_detections_are_not_bound(rack):
    weak = box_at(300, 300, cls="vial", score=0.05, track_id=7)
    frame = make_pipeline(rack, detections=[weak]).step(BLANK, 0.0)
    assert frame.objects["vial_a"].visible is False


def test_a_grasped_object_is_reported_as_held(rack):
    det = box_at(158, 100, size=30, cls="vial", track_id=7)
    pipe = make_pipeline(rack, detections=[det],
                         hands=[hand(side="left", wrist=(100, 100), aperture=10)])
    frame = pipe.step(BLANK, 0.0)
    assert frame.hands["left"].grasp_type in GRASP_TYPES
    assert frame.hands["left"].contact_with == "vial_a"
    assert frame.objects["vial_a"].held_by == "left"


def test_an_absent_hand_is_reported_as_not_present(rack):
    frame = make_pipeline(rack).step(BLANK, 0.0)
    assert frame.hands["left"].present is False
    assert frame.hands["left"].grasp_type == "none"


def test_body_is_withheld_without_a_rack_lock(rack):
    """Joints are only meaningful in rack coordinates - that is what makes
    `body_restrained` work with the crew inverted."""
    body = BodyKeypoints(score=0.9, points_px=np.zeros((3, 2)),
                         names=("pelvis",), points_rack={"pelvis": (0.0, 0.0, 0.6)})
    frame = make_pipeline(rack, pose=body).step(BLANK, 0.0)
    assert frame.body is None


def test_model_versions_are_reported_for_the_log(rack):
    versions = make_pipeline(rack).model_versions()
    assert versions["contact"] == "geometric-fallback"
    assert "detector" in versions


def test_timestamps_advance(rack):
    pipe = make_pipeline(rack)
    a, b = pipe.step(BLANK, 0.0), pipe.step(BLANK, 0.2)
    assert b.t_mono > a.t_mono
    assert b.t_utc > a.t_utc


# -- what the engine gets --------------------------------------------------
def test_the_engine_can_evaluate_a_pipeline_frame(rack):
    """The contract, exercised across the boundary. perception writes it, the
    engine reads it, and neither imports the other."""
    from parikshak.engine.predicates import PredicateEvaluator, RunFacts
    from parikshak.pdl import load_procedure

    proc = load_procedure(ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
    bindings = EntityBinding.from_procedure(proc)
    pipe = PerceptionPipeline(bindings, rack,
                              detector=ScriptedDetector([[]]),
                              hands=ScriptedHands([[]]),
                              motion=ScriptedMotion([("idle", 0.8)]))
    frame = pipe.step(BLANK, 0.0)

    ev = PredicateEvaluator(proc)
    ev.ingest(frame)
    facts = RunFacts(current_step="S06")
    for sid in proc.order:
        step = proc.step(sid)
        if step.verification is None:
            continue
        truth = ev.evaluate(step.verification, facts, f"{sid}.v")
        # Nothing is detected and there is no rack lock, so nothing may be
        # confidently satisfied - but it must not crash, and it must not be
        # confidently refuted either.
        assert not truth.satisfied(0.85)


def test_bindings_are_derived_from_the_procedure():
    from parikshak.pdl import load_procedure
    proc = load_procedure(ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
    bindings = EntityBinding.from_procedure(proc)
    assert set(bindings) == set(proc.entities)
    assert bindings["vial_a"].detector_class == "vial"
    # vial_a and vial_b share a detector class - the instance key separates them
    assert bindings["vial_b"].detector_class == "vial"
