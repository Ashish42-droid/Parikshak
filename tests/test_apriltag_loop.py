"""Closed loop: render real AprilTags, detect them, recover the rack frame.

Everything else in the rack-frame tests feeds the solver corners computed from
the ground-truth pose. This feeds it corners found by an actual detector in an
actual image, which is the only way to catch the failure it is here to catch.

**Why this test exists.** Corner ORDER is a convention, and pupil-apriltags and
OpenCV's ArUco use opposite ones. Measured before the adapter was fixed, the
reversed order produced 71 px of reprojection error while the recovered
translation was still only 4 mm out - a pose that looks entirely plausible and
is wrong by a reflection. Every downstream coordinate then sits in the wrong
place and nothing anywhere reports a problem. On a rack, that is a week.

The assertion that catches it is sub-pixel reprojection. A pose built from
mis-ordered corners cannot reproject cleanly no matter how plausible its
translation looks.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from parikshak.perception.backends import AprilTagDetector
from parikshak.perception.rackframe import (
    CameraIntrinsics,
    RackFrameEstimator,
    TagLayout,
    solve_pose,
)

cv2 = pytest.importorskip("cv2", reason="rendering real tags needs OpenCV")

ROOT = Path(__file__).resolve().parent.parent
LAYOUT_PATH = ROOT / "racks" / "msg_a_fiducials.json"

WIDTH, HEIGHT, HFOV = 1600, 1200, 65.0
MARKER_PX, QUIET_PX = 240, 60

#: A pose built from mis-ordered corners still yields a plausible translation.
#: Reprojection is what exposes it, so that is what is asserted.
MAX_REPROJ_PX = 2.0


@pytest.fixture(scope="module")
def layout() -> TagLayout:
    return TagLayout.load(LAYOUT_PATH)


@pytest.fixture(scope="module")
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_fov(WIDTH, HEIGHT, HFOV)


@pytest.fixture(scope="module")
def dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def render(layout, intrinsics, dictionary, R, t) -> np.ndarray:
    """Paint each tag onto a blank image at where the given pose puts it."""
    img = np.full((HEIGHT, WIDTH), 255, np.uint8)
    for tag_id in layout.tag_ids:
        px = intrinsics.project((R @ layout.corners[tag_id].T).T + t)
        if np.any(np.isnan(px)):
            continue
        if px.min() < -40 or px[:, 0].max() > WIDTH + 40 or px[:, 1].max() > HEIGHT + 40:
            continue
        marker = np.pad(cv2.aruco.generateImageMarker(dictionary, tag_id, MARKER_PX),
                        QUIET_PX, constant_values=255)
        lo, hi = QUIET_PX, QUIET_PX + MARKER_PX
        # Map the MARKER's own corners - not the padded image's - so the quiet
        # zone falls outside the tag footprint where a detector expects it.
        src = np.float32([[lo, hi], [hi, hi], [hi, lo], [lo, lo]])
        h, _ = cv2.findHomography(src, px.astype(np.float32))
        img = np.minimum(img, cv2.warpPerspective(marker, h, (WIDTH, HEIGHT),
                                                  borderValue=255))
    return img


def rotation(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


BASE_R = np.diag([1.0, -1.0, -1.0])
BASE_T = np.array([0.0, 0.0, 1.05])


@pytest.fixture(scope="module")
def detector():
    d = AprilTagDetector()
    if not d.available:
        pytest.skip("no AprilTag backend available")
    return d


# -- the loop --------------------------------------------------------------
def test_rendered_tags_are_detected(layout, intrinsics, dictionary, detector):
    obs = detector(render(layout, intrinsics, dictionary, BASE_R, BASE_T))
    assert {o.tag_id for o in obs} == set(layout.tag_ids)


def test_corner_order_matches_the_layout_convention(
        layout, intrinsics, dictionary, detector):
    """The regression this file exists for.

    Corner index i as reported by the adapter must correspond to corner index i
    in the layout. If the adapter stops normalising order, this fails here
    rather than as an unexplained accuracy problem on the rack.
    """
    obs = {o.tag_id: o for o in detector(
        render(layout, intrinsics, dictionary, BASE_R, BASE_T))}
    tag_id = layout.tag_ids[0]
    expected = intrinsics.project((BASE_R @ layout.corners[tag_id].T).T + BASE_T)
    got = obs[tag_id].corners_px
    permutation = [int(np.argmin(np.linalg.norm(expected - p, axis=1))) for p in got]
    assert permutation == [0, 1, 2, 3], (
        f"detector corner i maps to layout corner {permutation}; the adapter is not "
        f"normalising corner order and every recovered pose will be wrong by a "
        f"reflection while still looking plausible")


def test_pose_from_detected_corners_reprojects_subpixel(
        layout, intrinsics, dictionary, detector):
    """The assertion that actually catches a corner-order mistake. Mis-ordered
    corners gave 71 px here while the translation error was only 4 mm."""
    obs = {o.tag_id: o for o in detector(
        render(layout, intrinsics, dictionary, BASE_R, BASE_T))}
    ids = sorted(obs)
    ext = solve_pose(layout.object_points(ids),
                     np.vstack([obs[i].corners_px for i in ids]), intrinsics)
    assert ext.reproj_rms_px < MAX_REPROJ_PX, f"{ext.reproj_rms_px:.2f} px"
    assert np.abs(ext.t - BASE_T).max() < 0.01


def test_canonicalisation_through_the_real_detector(
        layout, intrinsics, dictionary, detector):
    """A point bonded to the rack comes back at its rack coordinates, having
    gone through rendering, detection and PnP rather than through algebra."""
    target = np.array([[0.17, 0.01, 0.09]])
    est = RackFrameEstimator(layout, intrinsics, min_tags=2)
    ext = est.update(detector(render(layout, intrinsics, dictionary, BASE_R, BASE_T)), 0.0)
    assert ext is not None
    got = ext.to_rack((BASE_R @ target.T).T + BASE_T)[0]
    error = float(np.abs(got - target[0]).max())
    assert error < 0.005, f"{error * 1000:.1f} mm through the real detector"


@pytest.mark.parametrize("axis,deg", [("z", 15), ("z", 180), ("y", 20), ("x", 12)])
def test_rack_rotation_survives_the_real_detector(
        layout, intrinsics, dictionary, detector, axis, deg):
    """The orientation-agnostic claim, end to end.

    Rotations are kept modest enough that all four tags stay inside a 65-degree
    field of view - the 1.1 m rack does not fit the frame at 90 degrees. The
    algebraic proof in test_rackframe.py covers the full 0/90/180 sweep; this
    confirms the same result survives a real detector.
    """
    target = np.array([[0.17, 0.01, 0.09], [-0.48, -0.18, 0.03]])

    def canonicalised(world_rot):
        R = BASE_R @ world_rot.T
        obs = detector(render(layout, intrinsics, dictionary, R, BASE_T))
        if len(obs) < 2:
            pytest.skip(f"only {len(obs)} tags in frame at {axis} {deg} deg")
        est = RackFrameEstimator(layout, intrinsics, min_tags=2)
        ext = est.update(obs, 0.0)
        assert ext is not None and ext.reproj_rms_px < MAX_REPROJ_PX
        return ext.to_rack((R @ target.T).T + BASE_T)

    reference = canonicalised(np.eye(3))
    rotated = canonicalised(rotation(axis, deg))
    drift = float(np.abs(rotated - reference).max())
    assert drift < 0.03, f"{axis} {deg} deg drifted {drift * 1000:.1f} mm"


def test_an_empty_image_yields_no_lock(layout, intrinsics, detector):
    """No tags means no frame, means UNVERIFIED downstream - never a guess."""
    est = RackFrameEstimator(layout, intrinsics, min_tags=2)
    assert est.update(detector(np.full((HEIGHT, WIDTH), 255, np.uint8)), 0.0) is None
    assert not est.locked
