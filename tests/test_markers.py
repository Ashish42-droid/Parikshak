"""Fiducial-marked props, on rendered pixels with known ground truth.

A live run without trained weights sees objects through AprilTags stuck to them.
These tests render the rack face and the props through a known camera pose and
check the whole path the live run uses - tag detection, rack lock, prop pose,
binding, state - against the millimetre truth the renderer knows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from parikshak.eval.scene import TagScene  # noqa: E402
from parikshak.perception.backends import AprilTagDetector  # noqa: E402
from parikshak.perception.markers import (  # noqa: E402
    MarkerDetector,
    PropMarkerError,
    PropMarkers,
    PropTag,
    SharedTags,
    quaternion_wxyz,
)
from parikshak.perception.pipeline import (  # noqa: E402
    EntityBinding,
    PerceptionPipeline,
    PipelineConfig,
)
from parikshak.perception.rackframe import RackFrameEstimator, TagLayout  # noqa: E402
from parikshak.pdl import load_procedure  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
CRX2 = ROOT / "procedures" / "crx2_colloid_resuspension.yaml"


@pytest.fixture(scope="module")
def csp1():
    return load_procedure(CSP1)


@pytest.fixture(scope="module")
def props():
    return PropMarkers.load(ROOT / "racks" / "props_csp1.json")


@pytest.fixture
def scene():
    return TagScene.facing_rack()


def rack_tags():
    return list(LAYOUT.corners.items())


def live(procedure, props, scene):
    """The perception half of a live run, as `parikshak run` assembles it."""
    rack = RackFrameEstimator(LAYOUT, scene.intrinsics)
    tags = SharedTags(AprilTagDetector())
    pipeline = PerceptionPipeline(
        EntityBinding.from_procedure(procedure), rack,
        detector=MarkerDetector(props, procedure, rack, tags), tags=tags,
        config=PipelineConfig(fps=10.0))
    return pipeline, rack


# -- the props files --------------------------------------------------------
@pytest.mark.parametrize("procedure, path", [(CSP1, "props_csp1.json"),
                                             (CRX2, "props_crx2.json")])
def test_the_shipped_props_files_fit_their_procedures(procedure, path):
    markers = PropMarkers.load(ROOT / "racks" / path)
    assert markers.problems(load_procedure(procedure), LAYOUT) == []


def test_props_that_would_misreport_are_rejected(csp1):
    markers = PropMarkers(0.06, {
        201: PropTag(201, "cartridge_sc_b"),           # not declared
        211: PropTag(211, "latch", "half_open"),       # not a declared state
        212: PropTag(212, "latch", "open"),
        213: PropTag(213, "latch", "open"),            # duplicate
        101: PropTag(101, "vial_a"),                   # a rack fiducial id
    })
    problems = " | ".join(markers.problems(csp1, LAYOUT))
    assert "cartridge_sc_b" in problems
    assert "half_open" in problems
    assert "duplicates tag 212" in problems
    assert "rack fiducial" in problems
    with pytest.raises(PropMarkerError):
        markers.check(csp1, LAYOUT)


# -- geometry on rendered frames -------------------------------------------
def test_the_rack_locks_from_a_rendered_frame(scene):
    rack = RackFrameEstimator(LAYOUT, scene.intrinsics)
    ext = rack.update(AprilTagDetector()(scene.render(rack_tags())), 0.0)
    assert ext is not None
    assert ext.reproj_rms_px < 1.0
    assert np.linalg.norm(ext.camera_position_rack - scene.camera_position_rack) < 0.005


#: Bounds from the 80-placement grid in perception/markers.py for a 60 mm tag:
#: lateral max 13.8 mm, total max 43.4 mm. Depth is the weak axis.
LATERAL_TOL_M = 0.015
TOTAL_TOL_M = 0.045


def lateral(pos, truth) -> float:
    return float(np.hypot(pos[0] - truth[0], pos[1] - truth[1]))


def test_a_prop_is_located_and_oriented_in_rack_coordinates(csp1, props, scene):
    truth = np.array([0.10, -0.10, 0.05])
    image = scene.render(rack_tags() + [(202, TagScene.square(truth, props.tag_size_m))])
    pipeline, _rack = live(csp1, props, scene)
    frame = pipeline.step(image, 0.0)

    vial = frame.objects["vial_a"]
    assert frame.frame_lock and vial.visible
    assert lateral(vial.pos_rack, truth) < LATERAL_TOL_M
    assert np.linalg.norm(np.asarray(vial.pos_rack) - truth) < TOTAL_TOL_M
    # Facing the crew, like the rack face: the identity rotation.
    w = vial.quat_rack[0]
    assert np.degrees(2 * np.arccos(min(1.0, abs(w)))) < 5.0


def test_vial_a_and_vial_b_are_told_apart_by_their_tags_not_by_order(csp1, props, scene):
    """Both are the detector class `vial`. Class-level binding would assign them
    in declaration order; a tag is the instance key, so it must not."""
    left, right = np.array([-0.17, -0.09, 0.06]), np.array([0.17, 0.01, 0.09])
    image = scene.render(rack_tags() + [(203, TagScene.square(left, props.tag_size_m)),
                                        (202, TagScene.square(right, props.tag_size_m))])
    frame = live(csp1, props, scene)[0].step(image, 0.0)
    a, b = frame.objects["vial_a"], frame.objects["vial_b"]
    # Identity, not precision - precision has its own test. Each vial must sit
    # where ITS tag is, not where the other one is.
    assert lateral(a.pos_rack, right) < LATERAL_TOL_M
    assert lateral(b.pos_rack, left) < LATERAL_TOL_M
    assert lateral(a.pos_rack, right) < lateral(a.pos_rack, left)
    assert lateral(b.pos_rack, left) < lateral(b.pos_rack, right)
    assert a.track_id != b.track_id


def test_a_state_tag_sets_the_state(csp1, props, scene):
    image = scene.render(rack_tags() + [(212, TagScene.square((0.1, 0.0, 0.1), 0.06))])
    latch = live(csp1, props, scene)[0].step(image, 0.0).objects["latch"]
    assert latch.state == "open" and latch.state_conf > 0.85


def test_contradictory_state_tags_claim_no_state(csp1, props, scene):
    """Both the open and the closed tag in view is a rig that cannot tell. An
    absent state reads UNKNOWN in the engine; a guessed one would not."""
    image = scene.render(rack_tags() + [(211, TagScene.square((0.0, 0.0, 0.1), 0.06)),
                                        (212, TagScene.square((0.2, 0.0, 0.1), 0.06))])
    latch = live(csp1, props, scene)[0].step(image, 0.0).objects["latch"]
    assert latch.visible and latch.state is None


def test_without_the_rack_in_view_a_prop_is_seen_but_not_located(csp1, props, scene):
    image = scene.render([(202, TagScene.square((0.1, -0.1, 0.05), 0.06))])
    frame = live(csp1, props, scene)[0].step(image, 0.0)
    assert not frame.frame_lock
    assert frame.objects["vial_a"].visible
    assert frame.objects["vial_a"].pos_rack is None


def test_tags_are_detected_once_per_image():
    calls = []

    def detector(image):
        calls.append(image)
        return []

    shared = SharedTags(detector)
    first, second = np.zeros((4, 4, 3)), np.zeros((4, 4, 3))
    shared(first)
    shared(first)
    shared(second)
    assert len(calls) == 2


@pytest.mark.parametrize("axis, degrees", [((0, 0, 1), 90.0), ((1, 0, 0), 180.0),
                                          ((0, 1, 0), 45.0), ((1, 1, 1), 120.0)])
def test_quaternions_come_out_in_w_x_y_z_order(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    rot, _ = cv2.Rodrigues(axis * np.radians(degrees))
    w, x, y, z = quaternion_wxyz(rot)
    half = np.radians(degrees) / 2
    assert w == pytest.approx(np.cos(half), abs=1e-9)
    assert np.allclose(np.array([x, y, z]), axis * np.sin(half), atol=1e-9)
