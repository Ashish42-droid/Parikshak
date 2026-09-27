"""Overlay geometry: rack-frame things, drawn in image pixels.

The live view has to show zones, object centroids, hands, the crew skeleton and
the rack axes themselves, all of which are defined in RACK coordinates while the
screen is in pixels. This module does that projection and nothing else - it
returns primitives, not pixels, so the geometry is testable without a display
and the same scene can be drawn by Qt, by OpenCV, or into an image for a report.

The rack axes are not decoration. Demo beat three rotates the rack ninety
degrees and the overlay stays locked to it, which is the visible form of the
whole orientation-agnostic argument: what moves on screen is the camera's view,
and what stays put is every number the engine reasons about.

Two things matter and are easy to get wrong:

  **Near-plane clipping.** A line from behind the camera to in front of it
  projects to a segment that sweeps across the frame in the wrong direction.
  Left unclipped, a zone edge behind the lens draws a diagonal streak over the
  whole view. Segments are clipped to the near plane before projection.

  **No lock, no overlay.** Without rack extrinsics the scene is empty and says
  so. Drawing the last known zone positions over a rack that has since moved
  would be the visual equivalent of the stale-extrinsic bug the engine refuses
  to commit.

Projection is done in batches - every segment of a zone transformed, clipped
and projected in one array operation. Measured on the live path, projecting
segment by segment cost 29 ms a frame at p50, as much as detecting the tags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from parikshak.perception.rackframe import CameraIntrinsics, Extrinsics

#: Anything closer than this to the lens is behind it for drawing purposes.
NEAR_PLANE_M = 0.05

#: Length of the drawn rack axes, in metres.
AXIS_LEN_M = 0.15

#: Points used to draw a sphere zone as a ring facing the camera.
SPHERE_SEGMENTS = 24


@dataclass(frozen=True, slots=True)
class Polyline:
    """A drawable run of points, in image pixels."""

    points: tuple[tuple[float, float], ...]
    kind: str                      # zone | zone_active | body | axis | hand
    label: str = ""
    closed: bool = False

    @property
    def is_empty(self) -> bool:
        return len(self.points) < 2


@dataclass(frozen=True, slots=True)
class Marker:
    """A labelled point: an object centroid, a wrist, a joint."""

    x: float
    y: float
    kind: str                      # object | object_held | object_lost | hand | tag
    label: str = ""
    detail: str = ""


@dataclass
class OverlayScene:
    polylines: list[Polyline] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    #: Shown as text over the view when there is nothing trustworthy to draw.
    notice: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.polylines and not self.markers

    def kinds(self) -> set[str]:
        return {p.kind for p in self.polylines} | {m.kind for m in self.markers}


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
def _clip_near(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip one camera-frame segment to the near plane.

    An unclipped segment with one endpoint behind the lens projects to a line
    running the wrong way across the frame - a zone edge behind the camera draws
    a diagonal streak over the whole view. Cheap to fix, very visible when not.
    Kept for single segments; `_project_segments` does the same test in bulk.
    """
    az, bz = a[2], b[2]
    if az >= NEAR_PLANE_M and bz >= NEAR_PLANE_M:
        return a, b
    if az < NEAR_PLANE_M and bz < NEAR_PLANE_M:
        return None
    t = (NEAR_PLANE_M - az) / (bz - az)
    cut = a + t * (b - a)
    return (cut, b) if az < NEAR_PLANE_M else (a, cut)


def _project_segments(segments: Sequence[tuple[np.ndarray, np.ndarray]] | np.ndarray,
                      ext: Extrinsics, intr: CameraIntrinsics) -> list[tuple]:
    """Rack-frame segments -> clipped pixel segments, in one batch.

    Accepts a sequence of (a, b) endpoint pairs or an (N, 2, 3) array. Segments
    entirely behind the near plane are dropped; segments crossing it are cut at
    the plane - the same rule as `_clip_near`, applied to every segment at once.
    """
    seg = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if len(seg) == 0:
        return []
    cam = (ext.R @ seg.reshape(-1, 3).T).T + ext.t
    a, b = cam[0::2], cam[1::2]
    az, bz = a[:, 2], b[:, 2]
    keep = ~((az < NEAR_PLANE_M) & (bz < NEAR_PLANE_M))
    a, b, az, bz = a[keep], b[keep], az[keep], bz[keep]
    if len(a) == 0:
        return []
    # A segment parallel to the near plane divides 0 by 0 here. Its cut point is
    # never selected below - only an endpoint behind the plane takes the cut -
    # so the NaN is harmless and the warning is noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (NEAR_PLANE_M - az) / (bz - az)
        cut = a + t[:, None] * (b - a)
    a = np.where((az < NEAR_PLANE_M)[:, None], cut, a)
    b = np.where((bz < NEAR_PLANE_M)[:, None], cut, b)
    px = intr.project(np.vstack([a, b]))
    pa, pb = px[: len(a)], px[len(a):]
    good = ~(np.isnan(pa).any(axis=1) | np.isnan(pb).any(axis=1))
    return [((float(p[0]), float(p[1])), (float(q[0]), float(q[1])))
            for p, q in zip(pa[good], pb[good])]


def _project_point(p, ext: Extrinsics, intr: CameraIntrinsics) -> tuple[float, float] | None:
    cam = ext.to_cam(np.asarray(p, dtype=float).reshape(1, 3))[0]
    if cam[2] < NEAR_PLANE_M:
        return None
    px = intr.project(cam.reshape(1, 3))[0]
    if np.any(np.isnan(px)):
        return None
    return (float(px[0]), float(px[1]))


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
_BOX_CACHE: dict[tuple, np.ndarray] = {}


def _box_edges(lo, hi) -> np.ndarray:
    """The twelve edges of an axis-aligned box, in rack coordinates, as (12, 2, 3).

    Cached per box: a procedure's zones do not change during a run, and working
    out which corner pairs share an edge every frame was pure overhead.
    """
    key = (tuple(float(v) for v in lo), tuple(float(v) for v in hi))
    edges = _BOX_CACHE.get(key)
    if edges is None:
        lo_, hi_ = np.asarray(key[0]), np.asarray(key[1])
        corners = np.array([[x, y, z] for x in (lo_[0], hi_[0])
                            for y in (lo_[1], hi_[1]) for z in (lo_[2], hi_[2])])
        pairs = []
        for i, a in enumerate(corners):
            for b in corners[i + 1:]:
                # Two corners share an edge when they differ on exactly one axis.
                if int(np.sum(~np.isclose(a, b))) == 1:
                    pairs.append((a, b))
        edges = np.asarray(pairs, dtype=float)
        _BOX_CACHE[key] = edges
    return edges


def _sphere_ring(centre, radius: float, ext: Extrinsics) -> np.ndarray:
    """A ring around the sphere, facing the camera, as (SEGMENTS, 2, 3).

    Drawn facing the viewer rather than as a wireframe globe: the point on
    screen is "here is a region of this size", and three orthogonal circles are
    noise over a live camera feed.
    """
    centre = np.asarray(centre, dtype=float)
    # The camera's own axes, expressed in rack coordinates.
    right, up = ext.R_cam_to_rack[:, 0], ext.R_cam_to_rack[:, 1]
    angles = np.linspace(0, 2 * math.pi, SPHERE_SEGMENTS, endpoint=False)
    pts = centre + radius * (np.cos(angles)[:, None] * right + np.sin(angles)[:, None] * up)
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)


#: Which joints to connect, for the crew skeleton. Sparse because BodyBelief is
#: sparse - only the joints the posture predicates ask about are carried.
SKELETON = (
    ("ankle_l", "pelvis"), ("ankle_r", "pelvis"),
    ("pelvis", "neck"), ("neck", "head"),
    ("neck", "shoulder_l"), ("neck", "shoulder_r"),
    ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
)


# --------------------------------------------------------------------------
def build_overlay(procedure, frame, extrinsics: Extrinsics | None,
                  intrinsics: CameraIntrinsics, *,
                  active_zones: Sequence[str] = (),
                  active_objects: Sequence[str] = (),
                  show_zones: bool = True, show_axes: bool = True,
                  show_body: bool = True) -> OverlayScene:
    """Everything to draw over this frame.

    `active_zones` and `active_objects` come from the current step's declared
    objects, so the overlay highlights what the crew is being asked to work
    with rather than lighting up the whole rack at once.
    """
    scene = OverlayScene()

    if extrinsics is None or not frame.frame_lock:
        # Rule two, in visual form. Drawing the last known zone positions over a
        # rack that has since moved would be the stale-extrinsic bug, rendered.
        scene.notice = "NO RACK LOCK - overlay suppressed, geometry unverified"
        return scene

    if show_axes:
        axes = np.zeros((3, 2, 3))
        for axis in range(3):
            axes[axis, 1, axis] = AXIS_LEN_M
        for axis, label in enumerate(("+X", "+Y", "+Z")):
            for a, b in _project_segments(axes[axis:axis + 1], extrinsics, intrinsics):
                scene.polylines.append(Polyline((a, b), "axis", label))

    if show_zones:
        active = set(active_zones)
        for name, zone in procedure.zones.items():
            if zone.kind == "box":
                segments = _box_edges(zone.lo, zone.hi)
            else:
                segments = _sphere_ring(zone.centre, zone.radius, extrinsics)
            kind = "zone_active" if name in active else "zone"
            occluded = frame.zone_occlusion(name)
            label = f"{name} ({occluded:.0%} occluded)" if occluded > 0.2 else name
            for a, b in _project_segments(segments, extrinsics, intrinsics):
                scene.polylines.append(Polyline((a, b), kind, label))

    highlight = set(active_objects)
    for name, ob in frame.objects.items():
        if ob.pos_rack is None:
            continue
        px = _project_point(ob.pos_rack, extrinsics, intrinsics)
        if px is None:
            continue
        if ob.occluded or not ob.visible:
            kind = "object_lost"
        elif ob.held_by is not None:
            kind = "object_held"
        else:
            kind = "object"
        entity = procedure.entities.get(name)
        detail = f"{ob.conf:.2f}"
        if ob.state:
            detail += f" · {ob.state}"
        if name in highlight:
            detail += " · this step"
        scene.markers.append(
            Marker(px[0], px[1], kind, entity.spoken if entity else name, detail))

    for side, h in frame.hands.items():
        if not h.present or h.wrist_rack is None:
            continue
        px = _project_point(h.wrist_rack, extrinsics, intrinsics)
        if px is None:
            continue
        detail = h.grasp_type if h.grasp_type != "none" else ""
        if h.contact_with:
            detail = f"{h.grasp_type} {h.contact_with}"
        scene.markers.append(Marker(px[0], px[1], "hand", side, detail))

    if show_body and frame.body is not None:
        joints = frame.body.joints_rack
        segments = [(np.asarray(joints[a]), np.asarray(joints[b]))
                    for a, b in SKELETON if a in joints and b in joints]
        for a, b in _project_segments(segments, extrinsics, intrinsics):
            scene.polylines.append(Polyline((a, b), "body", "crew"))

    if scene.is_empty:
        scene.notice = "rack locked, nothing detected in view"
    return scene
