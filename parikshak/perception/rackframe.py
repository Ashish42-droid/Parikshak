"""Rack-frame recovery: AprilTag 36h11 -> PnP -> canonicalisation.

This is the orientation-agnostic trick, and it is about two hundred lines rather
than a research project. Fiducials are bonded to the rack face, PnP recovers the
camera->rack transform every frame, and every joint, centroid and zone test is
then expressed in RACK coordinates.

Gravity never enters the pipeline. Rotate the rack, the camera, or the crew
member and every number downstream is unchanged, because "up" is the rack's +Y
and not the world's. Orientation-agnosticism becomes a property of the
coordinate system rather than something a model has to learn - which is why it
can be proved with a protractor instead of a held-out set.

Frame convention (schema/PREDICATES.md section 2):
    origin  centroid of the rack fiducial cluster
    +X      across the rack face, to the crew's right when facing it
    +Y      along the face, orthogonal to X ("up" FOR READABILITY ONLY)
    +Z      out of the face, toward the crew

The pose solver is pure numpy on purpose. Rack fiducials are coplanar - they are
bonded to a flat face - and the general 12-DOF DLT is *degenerate* for coplanar
points: the third column of the projection matrix is unconstrained and the
recovered rotation is garbage. So the planar case goes through a homography,
which is the correct estimator for it, and the general DLT is kept for a rack
whose tags span more than one face.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Below this, the object points are treated as lying on a single plane and the
#: homography path is used. Rack fiducials are bonded flat, so this is the
#: normal case, not the exception.
PLANARITY_TOL_M = 1e-4


class RackFrameError(ValueError):
    """The rack frame could not be recovered from what was observed."""


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels. Distortion is assumed already removed -
    undistort once at capture rather than in every geometric predicate."""

    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)

    def normalise(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels -> normalised image coordinates (z = 1 plane)."""
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        return np.column_stack([(pixels[:, 0] - self.cx) / self.fx,
                                (pixels[:, 1] - self.cy) / self.fy])

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Camera-frame 3D points -> pixels. Points behind the camera come back
        as NaN rather than mirrored in front of it."""
        pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        z = pts[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pts[:, 0] / z + self.cx
            v = self.fy * pts[:, 1] / z + self.cy
        bad = z <= 1e-9
        u = np.where(bad, np.nan, u)
        v = np.where(bad, np.nan, v)
        return np.column_stack([u, v])

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> CameraIntrinsics:
        """Approximate intrinsics from a horizontal field of view. For tests and
        for sizing a lens before the real calibration exists - never for a
        delivered measurement."""
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0)


# --------------------------------------------------------------------------
# fiducial layout
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TagLayout:
    """Where each tag's corners sit in RACK coordinates.

    Corner positions are stored explicitly rather than derived from a centre
    and a size. Corner ORDER is a convention that differs between detectors, and
    a silent mismatch produces a pose that is wrong by a rotation while looking
    perfectly plausible - the worst kind of failure. Writing the four points out
    makes the convention auditable in the file a person actually measured.
    """

    family: str
    tag_size_m: float
    corners: dict[int, np.ndarray]      # tag_id -> (4, 3)

    @property
    def tag_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.corners))

    def object_points(self, tag_ids: Sequence[int]) -> np.ndarray:
        missing = [t for t in tag_ids if t not in self.corners]
        if missing:
            raise RackFrameError(f"tag(s) {missing} are not in the layout {self.tag_ids}")
        return np.vstack([self.corners[t] for t in tag_ids])

    @classmethod
    def load(cls, path: str | Path) -> TagLayout:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        corners = {}
        for tag_id, entry in d["tags"].items():
            arr = np.asarray(entry["corners"], dtype=float)
            if arr.shape != (4, 3):
                raise RackFrameError(
                    f"tag {tag_id}: expected 4 corners of [x, y, z], got shape {arr.shape}")
            corners[int(tag_id)] = arr
        return cls(family=d.get("family", "tag36h11"),
                   tag_size_m=float(d.get("tag_size_m", 0.06)),
                   corners=corners)

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "tag_size_m": self.tag_size_m,
            "tags": {str(k): {"corners": v.tolist()} for k, v in sorted(self.corners.items())},
        }

    @classmethod
    def planar(cls, centres: dict[int, tuple[float, float]], tag_size_m: float,
               family: str = "tag36h11", z: float = 0.0) -> TagLayout:
        """Build a layout for tags bonded flat on the rack face at height `z`.

        Corner order is counter-clockwise from the bottom-left as seen by the
        crew, which is what pupil-apriltags reports. Change it here, in one
        place, if the deployed detector disagrees - and re-run the rotated-rack
        test, which is what catches a wrong convention.
        """
        h = tag_size_m / 2.0
        offsets = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])
        return cls(family=family, tag_size_m=tag_size_m,
                   corners={tid: offsets + np.array([cx, cy, z])
                            for tid, (cx, cy) in centres.items()})


@dataclass(frozen=True, slots=True)
class TagObservation:
    """One tag seen in one image. `corners_px` in the detector's order."""

    tag_id: int
    corners_px: np.ndarray              # (4, 2)

    def __post_init__(self) -> None:
        arr = np.asarray(self.corners_px, dtype=float)
        if arr.shape != (4, 2):
            raise RackFrameError(f"tag {self.tag_id}: expected (4, 2) corners, got {arr.shape}")


# --------------------------------------------------------------------------
# the transform
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Extrinsics:
    """Rigid transform taking RACK coordinates into CAMERA coordinates.

        p_cam = R @ p_rack + t

    Downstream almost always wants the inverse - perception measures in the
    camera and the engine reasons in the rack - so `to_rack` is the method that
    gets used and `R`/`t` are kept in the direction PnP natively solves.
    """

    R: np.ndarray                       # (3, 3), rack -> camera
    t: np.ndarray                       # (3,)
    n_tags: int
    reproj_rms_px: float

    @property
    def R_cam_to_rack(self) -> np.ndarray:
        return self.R.T

    @property
    def camera_position_rack(self) -> np.ndarray:
        """Where the camera is, in rack coordinates. Useful for the GUI overlay
        and for checking a rig was re-mounted where it was measured."""
        return -self.R.T @ self.t

    def to_rack(self, points_cam: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        return (self.R.T @ (pts - self.t).T).T

    def to_cam(self, points_rack: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_rack, dtype=float).reshape(-1, 3)
        return (self.R @ pts.T).T + self.t

    def project(self, points_rack: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        return intrinsics.project(self.to_cam(points_rack))


# --------------------------------------------------------------------------
# pose estimation
# --------------------------------------------------------------------------
def _orthonormalise(m: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix, with det = +1."""
    u, _s, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _is_planar(points: np.ndarray) -> tuple[bool, np.ndarray | None]:
    """Do these points lie on a plane? If so, return an orthonormal basis
    (origin, e1, e2, n) packed as a (4, 3) array."""
    centred = points - points.mean(axis=0)
    _u, s, vt = np.linalg.svd(centred)
    if s[2] > PLANARITY_TOL_M * max(1.0, s[0]):
        return False, None
    basis = np.vstack([points.mean(axis=0), vt[0], vt[1], vt[2]])
    return True, basis


def _pose_from_planar(obj: np.ndarray, norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose from coplanar correspondences, via a homography.

    The general DLT cannot do this: with all object points on a plane the third
    column of the projection matrix is unconstrained, so the solver returns a
    rotation that is numerically fine and physically wrong. Rack fiducials are
    bonded to a flat face, so this is the path that actually runs.
    """
    planar, basis = _is_planar(obj)
    assert planar and basis is not None
    origin, e1, e2, _n = basis
    local = np.column_stack([(obj - origin) @ e1, (obj - origin) @ e2])

    # Homography from plane coordinates to normalised image coordinates.
    n = len(local)
    a = np.zeros((2 * n, 9))
    for i, ((x, y), (u, v)) in enumerate(zip(local, norm)):
        a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
    _u, _s, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)

    h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]
    scale = (np.linalg.norm(h1) + np.linalg.norm(h2)) / 2.0
    if scale < 1e-12:
        raise RackFrameError("degenerate homography - tag corners are collinear")
    r1, r2, t_plane = h1 / scale, h2 / scale, h3 / scale
    r_plane = _orthonormalise(np.column_stack([r1, r2, np.cross(r1, r2)]))

    # Plane basis -> rack, then rack -> camera.
    basis_to_rack = np.column_stack([e1, e2, np.cross(e1, e2)])
    r = r_plane @ basis_to_rack.T
    t = t_plane - r @ origin

    # The homography is defined up to sign; pick the one that puts the rack in
    # front of the camera rather than behind it.
    if np.median((r @ obj.T).T[:, 2] + t[2]) < 0:
        r_plane = _orthonormalise(np.column_stack([-r1, -r2, np.cross(-r1, -r2)]))
        r = r_plane @ basis_to_rack.T
        t = -t_plane - r @ origin
    return r, t


def _pose_from_dlt(obj: np.ndarray, norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """General 12-DOF DLT, for a rack whose tags span more than one face."""
    n = len(obj)
    a = np.zeros((2 * n, 12))
    for i, (p, (u, v)) in enumerate(zip(obj, norm)):
        ph = np.array([p[0], p[1], p[2], 1.0])
        a[2 * i, 0:4] = -ph
        a[2 * i, 8:12] = u * ph
        a[2 * i + 1, 4:8] = -ph
        a[2 * i + 1, 8:12] = v * ph
    _u, _s, vt = np.linalg.svd(a)
    proj = vt[-1].reshape(3, 4)

    m, t = proj[:, :3], proj[:, 3]
    scale = np.linalg.norm(m, axis=1).mean()
    if scale < 1e-12:
        raise RackFrameError("degenerate DLT solution")
    r = _orthonormalise(m)
    t = t / scale
    if np.median((r @ obj.T).T[:, 2] + t[2]) < 0:
        r = _orthonormalise(-m)
        t = -t
    return r, t


def solve_pose(object_points: np.ndarray, image_points_px: np.ndarray,
               intrinsics: CameraIntrinsics, *, refine: bool = True) -> Extrinsics:
    """Recover the rack->camera transform from >= 4 point correspondences.

    Pure numpy by default. When OpenCV is present the closed-form answer is used
    to seed an iterative refinement, which is worth roughly an order of
    magnitude in reprojection error on real, noisy corners - but the result is
    never *worse*, because the refinement is rejected if it does not improve.
    """
    obj = np.asarray(object_points, dtype=float).reshape(-1, 3)
    px = np.asarray(image_points_px, dtype=float).reshape(-1, 2)
    if len(obj) != len(px):
        raise RackFrameError(f"{len(obj)} object points but {len(px)} image points")
    if len(obj) < 4:
        raise RackFrameError(f"need at least 4 correspondences, got {len(obj)}")

    norm = intrinsics.normalise(px)
    planar, _basis = _is_planar(obj)
    r, t = _pose_from_planar(obj, norm) if planar else _pose_from_dlt(obj, norm)

    if refine:
        r, t = _refine(obj, px, intrinsics, r, t)
    return Extrinsics(R=r, t=t, n_tags=len(obj) // 4,
                      reproj_rms_px=_reprojection_rms(obj, px, intrinsics, r, t))


def _reprojection_rms(obj, px, intrinsics, r, t) -> float:
    projected = intrinsics.project((r @ obj.T).T + t)
    if np.any(np.isnan(projected)):
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((projected - px) ** 2, axis=1))))


def _refine(obj, px, intrinsics, r, t):
    """Iterative refinement via OpenCV, if it is installed. Optional by design:
    the geometry must be testable on a machine with no vision stack."""
    try:
        import cv2
    except ImportError:
        return r, t
    try:
        rvec, _ = cv2.Rodrigues(r)
        ok, rvec, tvec = cv2.solvePnP(
            obj.astype(np.float64), px.astype(np.float64),
            intrinsics.matrix, None, rvec, t.reshape(3, 1).astype(np.float64),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return r, t
        r2, _ = cv2.Rodrigues(rvec)
        t2 = tvec.reshape(3)
    except Exception:
        return r, t
    before = _reprojection_rms(obj, px, intrinsics, r, t)
    after = _reprojection_rms(obj, px, intrinsics, r2, t2)
    return (r2, t2) if after <= before else (r, t)


# --------------------------------------------------------------------------
# per-frame estimator
# --------------------------------------------------------------------------
class RackFrameEstimator:
    """Tracks the rack frame across a run, including how it is allowed to fail.

    `hold_last_known` mirrors the procedure's `frame.on_frame_loss` policy: when
    the tags go out of view the last good extrinsic may be reused briefly,
    because a rack bolted to a wall does not move in half a second. After
    `max_hold_s` the lock is dropped and `frame_lock` goes False, which makes
    every geometric predicate UNVERIFIED downstream.

    That expiry is the whole point. A stale extrinsic reused indefinitely is how
    a rotated rack produces confident nonsense, and confident nonsense is worse
    than an admitted blind spot.
    """

    def __init__(self, layout: TagLayout, intrinsics: CameraIntrinsics, *,
                 min_tags: int = 2, max_hold_s: float = 3.0,
                 max_reproj_px: float = 6.0) -> None:
        self.layout = layout
        self.intrinsics = intrinsics
        self.min_tags = min_tags
        self.max_hold_s = max_hold_s
        self.max_reproj_px = max_reproj_px

        self.extrinsics: Extrinsics | None = None
        self.last_solved_t: float | None = None
        self.last_reject: str = ""

    @property
    def locked(self) -> bool:
        return self.extrinsics is not None

    def update(self, observations: Sequence[TagObservation], t: float) -> Extrinsics | None:
        """Fold in this frame's tag detections. Returns the current extrinsics,
        or None when the frame is not locked."""
        usable = [o for o in observations if o.tag_id in self.layout.corners]

        if len(usable) < self.min_tags:
            self.last_reject = (f"{len(usable)} known tag(s) visible, "
                                f"{self.min_tags} required for lock")
            return self._hold(t)

        tag_ids = [o.tag_id for o in usable]
        obj = self.layout.object_points(tag_ids)
        px = np.vstack([np.asarray(o.corners_px, dtype=float) for o in usable])
        try:
            candidate = solve_pose(obj, px, self.intrinsics)
        except (RackFrameError, np.linalg.LinAlgError) as exc:
            self.last_reject = f"pose solve failed: {exc}"
            return self._hold(t)

        if candidate.reproj_rms_px > self.max_reproj_px:
            # A pose that does not reproject is a pose that is wrong. Accepting
            # it would put every downstream coordinate quietly out of place.
            self.last_reject = (f"reprojection {candidate.reproj_rms_px:.1f}px exceeds "
                                f"{self.max_reproj_px:.1f}px")
            return self._hold(t)

        self.extrinsics = candidate
        self.last_solved_t = t
        self.last_reject = ""
        return candidate

    def _hold(self, t: float) -> Extrinsics | None:
        if self.extrinsics is None or self.last_solved_t is None:
            return None
        if (t - self.last_solved_t) > self.max_hold_s:
            self.extrinsics = None
            self.last_reject += f" (held past max_hold_s={self.max_hold_s}s, lock dropped)"
            return None
        return self.extrinsics

    def to_rack(self, points_cam: np.ndarray) -> np.ndarray | None:
        """Canonicalise camera-frame points. None when there is no lock - the
        caller must then report UNVERIFIED rather than guess."""
        if self.extrinsics is None:
            return None
        return self.extrinsics.to_rack(points_cam)
