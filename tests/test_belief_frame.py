"""BeliefFrame is the contract every other module is written against.

These tests are not about coverage. They are about making a breaking change to
the contract impossible to do by accident: if someone adds a field in week 9,
something here goes red before a 60-trace regression corpus silently rots.
"""

from __future__ import annotations

import pytest

from parikshak.belief.frame import (
    BELIEF_SCHEMA_VERSION,
    BeliefFrame,
    contact_key,
    BodyBelief,
    HandBelief,
    MotionBelief,
    ObjectBelief,
    TraceFormatError,
    empty_frame,
)


def a_frame() -> BeliefFrame:
    return BeliefFrame(
        t_mono=1.25,
        t_utc="2026-12-11T04:30:01Z",
        frame_lock=True,
        objects={
            "vial_a": ObjectBelief(
                visible=True, conf=0.93, pos_rack=(-0.17, -0.09, 0.06),
                quat_rack=(1.0, 0.0, 0.0, 0.0), track_id=318,
                state=None, state_conf=0.0, held_by="left", occluded=False),
            "latch": ObjectBelief(
                visible=True, conf=0.93, pos_rack=(0.185, 0.01, 0.18), quat_rack=None,
                track_id=206, state="closed", state_conf=0.91, held_by=None, occluded=False),
        },
        hands={
            "left": HandBelief(present=True, conf=0.9, wrist_rack=(-0.17, -0.09, 0.06),
                               contact_with="vial_a", grasp_type="grasp", grasp_conf=0.9),
            "right": HandBelief(present=True, conf=0.9, wrist_rack=(0.0, -0.075, 0.125),
                                contact_with=None, grasp_type="none", grasp_conf=0.0),
        },
        body=BodyBelief(conf=0.88, joints_rack={"pelvis": (0.0, -0.15, 0.625)}),
        motion=MotionBelief(cls="reach", conf=0.88),
        occlusion={"glovebox_interior": 0.4},
        contacts={"cartridge_sc_a|vial_a": 0.89},
        confirmations=("label_verified",),
    )


# -- round trip ------------------------------------------------------------
def test_round_trip_is_lossless():
    f = a_frame()
    assert BeliefFrame.from_json(f.to_json()) == f


def test_empty_frame_round_trips():
    f = empty_frame()
    assert BeliefFrame.from_json(f.to_json()) == f


def test_empty_frame_has_no_rack_lock():
    """Before AprilTag acquisition we do not have a rack frame, and saying we do
    is how stale extrinsics get used as if they were current."""
    assert empty_frame().frame_lock is False


def test_field_order_on_disk_is_stable():
    """Trace diffs are a debugging tool. Reordering fields makes every diff
    useless even when nothing changed."""
    assert list(a_frame().to_json()) == list(BeliefFrame.FIELDS)


def test_lists_deserialise_to_tuples():
    """JSON has no tuples. Coordinates must come back hashable and comparable,
    or frame equality silently fails between a written and a read frame."""
    f = BeliefFrame.from_json(a_frame().to_json())
    assert f.objects["vial_a"].pos_rack == (-0.17, -0.09, 0.06)
    assert isinstance(f.objects["vial_a"].pos_rack, tuple)


# -- the closed schema -----------------------------------------------------
def test_unknown_field_is_rejected():
    d = a_frame().to_json()
    d["confidence"] = 0.9  # plausible typo for a field that does not exist
    with pytest.raises(TraceFormatError, match="unknown field"):
        BeliefFrame.from_json(d)


def test_missing_field_is_rejected():
    d = a_frame().to_json()
    del d["frame_lock"]
    with pytest.raises(TraceFormatError, match="missing field"):
        BeliefFrame.from_json(d)


def test_unknown_nested_field_is_rejected():
    d = a_frame().to_json()
    d["objects"]["vial_a"]["velocity"] = [0, 0, 0]
    with pytest.raises(TraceFormatError, match="unknown field"):
        BeliefFrame.from_json(d)


# -- value ranges ----------------------------------------------------------
@pytest.mark.parametrize("bad", [1.5, -0.1])
def test_confidence_outside_unit_interval_is_rejected(bad):
    d = a_frame().to_json()
    d["objects"]["vial_a"]["conf"] = bad
    with pytest.raises(TraceFormatError, match="outside"):
        BeliefFrame.from_json(d)


def test_grasp_type_outside_contact_vocabulary_is_rejected():
    """The contact MLP emits exactly five classes. A sixth means perception and
    the engine disagree about what a hand can be doing."""
    d = a_frame().to_json()
    d["hands"]["left"]["grasp_type"] = "holding"
    with pytest.raises(TraceFormatError, match="contact-head vocabulary"):
        BeliefFrame.from_json(d)


def test_held_by_must_be_a_real_hand():
    d = a_frame().to_json()
    d["objects"]["vial_a"]["held_by"] = "gripper"
    with pytest.raises(TraceFormatError, match="held_by"):
        BeliefFrame.from_json(d)


def test_unknown_hand_side_is_rejected():
    d = a_frame().to_json()
    d["hands"]["third"] = d["hands"]["left"]
    with pytest.raises(TraceFormatError, match="unknown side"):
        BeliefFrame.from_json(d)


def test_boolean_state_is_rejected_with_the_yaml_hint():
    """PLAN.md section 6 defect 3: YAML 1.1 turns a bare `off` into False. If that
    ever reaches a trace, the error must name the cause - this failure is
    otherwise invisible until it happens on the rack."""
    d = a_frame().to_json()
    d["objects"]["latch"]["state"] = False
    with pytest.raises(TraceFormatError, match="booleans"):
        BeliefFrame.from_json(d)


def test_malformed_coordinate_is_rejected():
    d = a_frame().to_json()
    d["objects"]["vial_a"]["pos_rack"] = [0.1, 0.2]
    with pytest.raises(TraceFormatError, match="expected"):
        BeliefFrame.from_json(d)


# -- null semantics --------------------------------------------------------
def test_occluded_object_may_have_no_position():
    """Occlusion means unknown, not zero. A [0,0,0] centroid for a hidden object
    would place it at the rack origin and satisfy geometry predicates."""
    d = a_frame().to_json()
    d["objects"]["vial_a"]["pos_rack"] = None
    d["objects"]["vial_a"]["occluded"] = True
    d["objects"]["vial_a"]["visible"] = False
    assert BeliefFrame.from_json(d).objects["vial_a"].pos_rack is None


def test_body_may_be_absent_entirely():
    d = a_frame().to_json()
    d["body"] = None
    assert BeliefFrame.from_json(d).body is None


def test_unlisted_zone_reads_as_unoccluded():
    f = a_frame()
    assert f.zone_occlusion("glovebox_interior") == 0.4
    assert f.zone_occlusion("workspace") == 0.0


# -- versioning ------------------------------------------------------------
def test_schema_version_is_pinned():
    """Changing this constant is the deliberate act that invalidates the corpus.
    It should require editing a test, not just a source file."""
    assert BELIEF_SCHEMA_VERSION == "1.1"


# -- v1.1: what CSP-1 step S07 needs ---------------------------------------
def test_contact_keys_are_symmetric_and_have_one_spelling():
    """Contact is a symmetric relation. Two spellings means the engine can ask
    the question one way and get an answer the other."""
    assert contact_key("vial_a", "cartridge_sc_a") == contact_key("cartridge_sc_a", "vial_a")


def test_unsorted_contact_key_is_rejected():
    d = a_frame().to_json()
    d["contacts"] = {"vial_a|cartridge_sc_a": 0.9}
    with pytest.raises(TraceFormatError, match="sorted order"):
        BeliefFrame.from_json(d)


def test_malformed_contact_key_is_rejected():
    d = a_frame().to_json()
    d["contacts"] = {"vial_a": 0.9}
    with pytest.raises(TraceFormatError, match="entity names"):
        BeliefFrame.from_json(d)


def test_contact_lookup_is_order_independent():
    f = a_frame()
    assert f.contact("vial_a", "cartridge_sc_a") == 0.89
    assert f.contact("cartridge_sc_a", "vial_a") == 0.89
    assert f.contact("vial_a", "latch") == 0.0


def test_orientation_is_optional():
    """Most classes never get 6-DoF pose. `aligned` is the only predicate that
    needs it, and requiring it everywhere would make the contract lie about what
    the deployed build actually estimates."""
    f = a_frame()
    assert f.objects["latch"].quat_rack is None
    assert f.objects["vial_a"].quat_rack == (1.0, 0.0, 0.0, 0.0)


def test_malformed_quaternion_is_rejected():
    d = a_frame().to_json()
    d["objects"]["vial_a"]["quat_rack"] = [1.0, 0.0, 0.0]
    with pytest.raises(TraceFormatError, match=r"w, x, y, z"):
        BeliefFrame.from_json(d)
