"""Rack-frame recovery, and the orientation-agnostic proof.

The claim this file has to support is the one the problem statement calls
optional and hard: the system works with the rack, the camera or the crew member
in any orientation. Our answer is that orientation-agnosticism is a property of
the COORDINATE SYSTEM, not of a learned model - so it is provable with geometry
rather than with a held-out set, and provable now rather than after a data
campaign.

`test_rotating_the_rack_changes_nothing` is that proof. Everything else here
exists to stop it from being accidentally true.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from parikshak.perception.rackframe import (
    CameraIntrinsics,
    Extrinsics,
    RackFrameError,
    RackFrameEstimator,
    TagLayout,
    TagObservation,
    solve_pose,
)

ROOT = Path(__file__).resolve().parent.parent
LAYOUT_PATH = ROOT / "racks" / "msg_a_fiducials.json"

#: PLAN.md week 5 exit gate: canonicalised coordinates must agree within 3 cm
#: at 0, 90 and 180 degrees of rack rotation.
ROTATION_GATE_M = 0.03

#: AprilTag corner localisation on a global-shutter sensor. 0.5 px is a routine
#: figure; 1 px is pessimistic for a bonded, well-lit tag.
REALISTIC_CORNER_NOISE_PX = 0.5


@pytest.fixture(scope="module")
def layout() -> TagLayout:
    return TagLayout.load(LAYOUT_PATH)


@pytest.fixture(scope="module")
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_fov(1920, 1080, 70.0)


def rotation(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


#: A camera mounted out from the rack face, looking back at it.
BASE_R = np.diag([1.0, -1.0, -1.0])
BASE_T = np.array([0.02, -0.03, 1.15])


def observe(layout, intrinsics, R, t, tag_ids=None, noise_px=0.0, rng=None):
    """What the camera would see, given a rack->camera pose."""
    tag_ids = list(tag_ids or layout.tag_ids)
    obj = layout.object_points(tag_ids)
    px = intrinsics.project((R @ obj.T).T + t)
    if noise_px:
        px = px + (rng or np.random.default_rng(0)).normal(0, noise_px, px.shape)
    return [TagObservation(tid, px[4 * i:4 * i + 4]) for i, tid in enumerate(tag_ids)]


# -- the layout ------------------------------------------------------------
def test_the_layout_csp1_references_exists(layout):
    """CSP-1 names racks/msg_a_fiducials.json. A procedure pointing at a missing
    layout file is a run that cannot start."""
    assert set(layout.tag_ids) == {101, 102, 103, 104}
    assert layout.family == "tag36h11"


def test_layout_round_trips(tmp_path, layout):
    p = tmp_path / "l.json"
    p.write_text(json.dumps(layout.to_json()), encoding="utf-8")
    again = TagLayout.load(p)
    assert again.tag_ids == layout.tag_ids
    for tid in layout.tag_ids:
        assert np.allclose(again.corners[tid], layout.corners[tid])


def test_layout_rejects_malformed_corners(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"tags": {"101": {"corners": [[0, 0, 0], [1, 0, 0]]}}}),
                 encoding="utf-8")
    with pytest.raises(RackFrameError, match="4 corners"):
        TagLayout.load(p)


def test_asking_for_an_unlaid_tag_is_an_error(layout):
    with pytest.raises(RackFrameError, match="not in the layout"):
        layout.object_points([999])


# -- pose recovery ---------------------------------------------------------
def test_pose_is_recovered_exactly_from_clean_corners(layout, intrinsics):
    obs = observe(layout, intrinsics, BASE_R, BASE_T)
    obj = layout.object_points([o.tag_id for o in obs])
    px = np.vstack([o.corners_px for o in obs])
    ext = solve_pose(obj, px, intrinsics)
    assert np.allclose(ext.R, BASE_R, atol=1e-9)
    assert np.allclose(ext.t, BASE_T, atol=1e-9)
    assert ext.reproj_rms_px < 1e-6


def test_coplanar_tags_are_solved_correctly(layout, intrinsics):
    """Rack fiducials are bonded to a FLAT face, so every real solve is the
    coplanar case - and the general 12-DOF DLT is degenerate for it: the third
    column of the projection matrix is unconstrained and the rotation comes back
    numerically fine and physically wrong. This asserts the homography path is
    what actually runs."""
    zs = {c[2] for tid in layout.tag_ids for c in layout.corners[tid]}
    assert len(zs) == 1, "layout is not planar; this test no longer covers the real case"
    obs = observe(layout, intrinsics, BASE_R, BASE_T)
    ext = solve_pose(layout.object_points([o.tag_id for o in obs]),
                     np.vstack([o.corners_px for o in obs]), intrinsics)
    assert ext.reproj_rms_px < 1e-6


def test_transform_inverts_itself(layout, intrinsics):
    obs = observe(layout, intrinsics, BASE_R, BASE_T)
    ext = solve_pose(layout.object_points([o.tag_id for o in obs]),
                     np.vstack([o.corners_px for o in obs]), intrinsics)
    pts = np.array([[0.17, 0.01, 0.09], [-0.48, -0.18, 0.03], [0.0, 0.0, 0.0]])
    assert np.allclose(ext.to_rack(ext.to_cam(pts)), pts, atol=1e-9)


def test_camera_position_is_reported_in_rack_coordinates(layout, intrinsics):
    obs = observe(layout, intrinsics, BASE_R, BASE_T)
    ext = solve_pose(layout.object_points([o.tag_id for o in obs]),
                     np.vstack([o.corners_px for o in obs]), intrinsics)
    expected = -BASE_R.T @ BASE_T
    assert np.allclose(ext.camera_position_rack, expected, atol=1e-9)


def test_too_few_correspondences_is_refused(intrinsics):
    with pytest.raises(RackFrameError, match="at least 4"):
        solve_pose(np.zeros((3, 3)), np.zeros((3, 2)), intrinsics)


def test_points_behind_the_camera_do_not_project(intrinsics):
    out = intrinsics.project(np.array([[0.1, 0.1, -1.0]]))
    assert np.all(np.isnan(out))


# -- THE PROOF -------------------------------------------------------------
@pytest.mark.parametrize("axis,deg", [
    ("z", 0), ("z", 90), ("z", 180), ("z", 270),
    ("x", 90), ("x", 180),           # crew working inverted
    ("y", 35),                        # rack tilted off-axis
])
def test_rotating_the_rack_changes_nothing(layout, intrinsics, axis, deg):
    """The orientation-agnostic proof, and the week-5 exit gate.

    Rotating the rack by R is the same as rotating the camera by R-inverse
    relative to it. A point bonded to the rack must come back at the same rack
    coordinates in every case, because "up" is the rack's +Y and gravity is
    never sampled. Includes 180 degrees about X - the crew member upside down.
    """
    target_rack = np.array([[0.17, 0.01, 0.09]])
    rng = np.random.default_rng(11)

    def canonicalised(world_rot):
        R = BASE_R @ world_rot.T
        obs = observe(layout, intrinsics, R, BASE_T,
                      noise_px=REALISTIC_CORNER_NOISE_PX, rng=rng)
        ext = RackFrameEstimator(layout, intrinsics).update(obs, 0.0)
        assert ext is not None
        return ext.to_rack((R @ target_rack.T).T + BASE_T)[0]

    reference = canonicalised(np.eye(3))
    rotated = canonicalised(rotation(axis, deg))
    error = float(np.abs(rotated - reference).max())
    assert error < ROTATION_GATE_M, (
        f"{axis}-axis {deg} deg: canonicalised coordinates moved {error * 1000:.1f} mm, "
        f"gate is {ROTATION_GATE_M * 1000:.0f} mm")


def test_the_rotation_gate_holds_with_all_four_tags_under_noise(layout, intrinsics):
    """The measured error budget, not a single lucky draw.

    At 0.5 px corner noise with four tags the p95 error is about 1 mm against a
    30 mm gate. With only two tags visible it is roughly 13 mm - still passing,
    but the margin is gone by 1-2 px, which is why `min_tags_for_lock: 2` is a
    floor to survive occlusion and not a target to design for.
    """
    rng = np.random.default_rng(3)
    targets = np.array([[0.17, 0.01, 0.09], [-0.48, -0.18, 0.03], [0.36, 0.14, 0.06]])
    errors = []
    for _ in range(60):
        got = {}
        for axis, deg in (("z", 0), ("z", 90), ("z", 180)):
            R = BASE_R @ rotation(axis, deg).T
            obs = observe(layout, intrinsics, R, BASE_T,
                          noise_px=REALISTIC_CORNER_NOISE_PX, rng=rng)
            ext = RackFrameEstimator(layout, intrinsics).update(obs, 0.0)
            got[deg] = ext.to_rack((R @ targets.T).T + BASE_T)
        for deg in (90, 180):
            errors.append(float(np.abs(got[deg] - got[0]).max()))
    p95 = float(np.percentile(errors, 95))
    assert p95 < ROTATION_GATE_M, f"p95 canonicalisation error {p95 * 1000:.1f} mm"


# -- lock behaviour --------------------------------------------------------
def test_lock_requires_the_declared_minimum_tags(layout, intrinsics):
    est = RackFrameEstimator(layout, intrinsics, min_tags=2)
    one = observe(layout, intrinsics, BASE_R, BASE_T, tag_ids=[101])
    assert est.update(one, 0.0) is None
    assert not est.locked
    assert "required for lock" in est.last_reject


def test_unknown_tags_are_ignored_not_trusted(layout, intrinsics):
    """A tag from another rack, or a printout on a laptop lid, must not
    contribute to the pose."""
    est = RackFrameEstimator(layout, intrinsics, min_tags=2)
    stray = TagObservation(777, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    obs = observe(layout, intrinsics, BASE_R, BASE_T, tag_ids=[101]) + [stray]
    assert est.update(obs, 0.0) is None


def test_a_pose_that_does_not_reproject_is_rejected(layout, intrinsics):
    """A pose that does not reproject is a pose that is wrong. Accepting it puts
    every downstream coordinate quietly out of place, which is worse than having
    no lock at all."""
    est = RackFrameEstimator(layout, intrinsics, max_reproj_px=1.0)
    obs = observe(layout, intrinsics, BASE_R, BASE_T,
                  noise_px=40.0, rng=np.random.default_rng(5))
    assert est.update(obs, 0.0) is None
    assert "reprojection" in est.last_reject


def test_last_known_pose_is_held_briefly_then_dropped(layout, intrinsics):
    """A rack bolted to a wall does not move in half a second, so a brief
    dropout may reuse the last extrinsic. Reusing it INDEFINITELY is how a
    rotated rack produces confident nonsense, so the hold expires."""
    est = RackFrameEstimator(layout, intrinsics, min_tags=2, max_hold_s=3.0)
    good = observe(layout, intrinsics, BASE_R, BASE_T)
    assert est.update(good, 0.0) is not None

    assert est.update([], 1.0) is not None, "should still hold at 1 s"
    assert est.update([], 2.9) is not None, "should still hold at 2.9 s"
    assert est.update([], 3.5) is None, "must drop the lock past max_hold_s"
    assert not est.locked
    assert "lock dropped" in est.last_reject


def test_no_lock_means_no_coordinates(layout, intrinsics):
    """The rule the engine depends on: without a rack frame, perception returns
    nothing rather than a stale guess. Downstream this becomes UNVERIFIED."""
    est = RackFrameEstimator(layout, intrinsics)
    assert est.to_rack(np.array([[0.0, 0.0, 1.0]])) is None


def test_regaining_tags_re_locks(layout, intrinsics):
    est = RackFrameEstimator(layout, intrinsics, min_tags=2, max_hold_s=1.0)
    good = observe(layout, intrinsics, BASE_R, BASE_T)
    est.update(good, 0.0)
    est.update([], 5.0)
    assert not est.locked
    assert est.update(good, 5.2) is not None
    assert est.locked
