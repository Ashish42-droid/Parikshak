"""Fiducial-marked props: an object detector made of AprilTags.

    racks/props_<procedure>.json
        tag_size_m   printed edge of each prop tag's black square, metres
        tags         tag id -> {"entity": name, "state": optional state}

Until trained detector weights exist, a live run needs SOME way to see a vial.
This is the honest one. Each prop carries a small AprilTag, found by the same
detector that locks the rack frame; the tag's pose, mapped through the rack
extrinsics, is the object's rack-frame position and orientation. Everything
downstream - bindings, predicates, the engine, the display - is unchanged, and
the procedure file is not touched: the tag IS the instance key, so vial A and
vial B are told apart exactly rather than by declaration order.

A state is a tag too. A latch carries one tag visible only when closed and
another visible only when open; whichever is seen is the state. If both are seen
the state is not claimed at all, because two tags disagreeing is a rig that
cannot tell, and the engine turns an absent state into UNKNOWN.

What this cannot give, and does not pretend to: hand contact, motion classes and
crew pose. Predicates that need them read UNKNOWN in a camera-only run (see
PREDICATES.md), and `parikshak run` states which ones up front.

**Measured precision.** 80 placements across the rack face, 0 to 0.30 m out
from it, rendered at 1280x720 through a 70 degree lens from 1.5 m (the scene
tests/test_markers.py uses), with sub-pixel corner refinement:

    tag     lateral error p50 / p95 / max    depth error p50 / p95 / max
    60 mm    1.4 /  9.0 / 13.8 mm             5.5 / 34.0 / 41.2 mm
    40 mm    4.3 / 14.3 / 17.0 mm            22.3 / 48.0 / 69.1 mm

Lateral position is good; depth is the weak axis, because one small square seen
from far away carries little range information. Hence 60 mm is the default, and
hence a zone only a few centimetres deep is at the edge of what one camera and
one tag can resolve. These are renders: a real webcam, with blur and autofocus
hunting, will be worse, and its own numbers are the ones to publish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from parikshak.perception.rackframe import (
    Extrinsics,
    RackFrameError,
    RackFrameEstimator,
    TagLayout,
    TagObservation,
    solve_pose,
)
from parikshak.perception.types import Detection

#: Confidence reported for a decoded tag. A 36h11 decode is not a probabilistic
#: detection - the family's Hamming distance makes a false decode vanishingly
#: rare - but it is not certainty about the OBJECT either: a tag can be peeled
#: off, or stuck on the wrong vial. 0.9 clears the engine's 0.85 completion bar
#: without claiming more than a sticker can.
MARKER_SCORE = 0.9

#: A prop pose that does not reproject is a wrong pose, exactly as for the rack.
MAX_PROP_REPROJ_PX = 3.0


class PropMarkerError(ValueError):
    """A props file that would make the markers lie to the engine."""


@dataclass(frozen=True, slots=True)
class PropTag:
    tag_id: int
    entity: str
    state: str | None = None


@dataclass(frozen=True, slots=True)
class PropMarkers:
    tag_size_m: float
    tags: dict[int, PropTag]
    family: str = "tag36h11"

    @classmethod
    def load(cls, path: str | Path) -> PropMarkers:
        p = Path(path)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PropMarkerError(f"props file not found: {p}") from exc
        except json.JSONDecodeError as exc:
            raise PropMarkerError(f"{p}: not valid JSON - {exc.msg}") from exc
        size = d.get("tag_size_m")
        if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
            raise PropMarkerError(f"{p}: tag_size_m must be a positive number, got {size!r}")
        raw = d.get("tags")
        if not isinstance(raw, dict) or not raw:
            raise PropMarkerError(f"{p}: 'tags' must map tag ids to props")
        tags: dict[int, PropTag] = {}
        for key, entry in raw.items():
            try:
                tag_id = int(key)
            except ValueError as exc:
                raise PropMarkerError(f"{p}: tag id {key!r} is not an integer") from exc
            if not isinstance(entry, dict) or not isinstance(entry.get("entity"), str):
                raise PropMarkerError(f"{p}: tag {key} needs an 'entity' name")
            state = entry.get("state")
            if state is not None and not isinstance(state, str):
                raise PropMarkerError(f"{p}: tag {key} state must be a string")
            tags[tag_id] = PropTag(tag_id, entry["entity"], state)
        return cls(tag_size_m=float(size), tags=tags, family=d.get("family", "tag36h11"))

    def problems(self, procedure: Any, layout: TagLayout | None = None) -> list[str]:
        """Everything that would make these props misreport the procedure.

        Checked before a run starts, because each one fails silently at the rack:
        a typo'd entity is a prop the engine never sees, an undeclared state is
        a state no predicate can match, and a tag id shared with the rack layout
        would be read as both a fiducial and an object.
        """
        out: list[str] = []
        seen: dict[tuple[str, str | None], int] = {}
        for tag in self.tags.values():
            entity = procedure.entities.get(tag.entity)
            if entity is None:
                out.append(f"tag {tag.tag_id}: entity '{tag.entity}' is not declared by "
                           f"{procedure.id}")
                continue
            if tag.state is not None and tag.state not in (entity.states or ()):
                out.append(f"tag {tag.tag_id}: state '{tag.state}' is not a declared state of "
                           f"'{tag.entity}' ({list(entity.states or ())})")
            key = (tag.entity, tag.state)
            if key in seen:
                out.append(f"tag {tag.tag_id}: duplicates tag {seen[key]} for "
                           f"{tag.entity}{'/' + tag.state if tag.state else ''}")
            seen[key] = tag.tag_id
            if layout is not None and tag.tag_id in layout.corners:
                out.append(f"tag {tag.tag_id}: also a rack fiducial in the layout - it would be "
                           f"read as both")
        return out

    def check(self, procedure: Any, layout: TagLayout | None = None) -> None:
        problems = self.problems(procedure, layout)
        if problems:
            raise PropMarkerError("props file does not fit the procedure:\n  "
                                  + "\n  ".join(problems))


class SharedTags:
    """Detect tags once per image, for both the rack lock and the props.

    Holds a reference to the last image rather than keying on id(), so a new
    frame allocated at a freed frame's address can never be served the old
    detections.
    """

    def __init__(self, detector) -> None:
        self.detector = detector
        self._image = None
        self._obs: list[TagObservation] = []

    def __call__(self, image) -> list[TagObservation]:
        if image is not self._image:
            self._obs = list(self.detector(image))
            self._image = image
        return list(self._obs)


def quaternion_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (w, x, y, z), the BeliefFrame order."""
    m = np.asarray(r, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = (0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    arr = np.asarray(q, dtype=float)
    arr = arr / np.linalg.norm(arr)
    if arr[0] < 0:
        arr = -arr
    return tuple(float(v) for v in arr)  # type: ignore[return-value]


class MarkerDetector:
    """The ObjectDetector protocol, answered with fiducials.

    Must be called AFTER the rack estimator has seen the same image - the
    pipeline does that in `step` - because prop positions are only rack-frame
    positions once this frame's extrinsics exist. Without a lock a prop is still
    reported as seen, with no position: visible, and geometrically unknown.
    """

    weights_sha256 = "fiducial-markers"

    def __init__(self, markers: PropMarkers, procedure: Any, rack: RackFrameEstimator,
                 tags: SharedTags, *, score: float = MARKER_SCORE,
                 max_reproj_px: float = MAX_PROP_REPROJ_PX) -> None:
        self.markers = markers
        self.rack = rack
        self.tags = tags
        self.score = score
        self.max_reproj_px = max_reproj_px
        self.classes = {name: e.detector_class for name, e in procedure.entities.items()}
        h = markers.tag_size_m / 2.0
        # Same corner convention as TagLayout.planar: counter-clockwise from the
        # bottom-left, which is what AprilTagDetector reports for either backend.
        self._object = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])

    def __call__(self, image) -> list[Detection]:
        seen: dict[str, list[tuple[PropTag, TagObservation]]] = {}
        for obs in self.tags(image):
            prop = self.markers.tags.get(obs.tag_id)
            if prop is None or prop.entity not in self.classes:
                continue
            seen.setdefault(prop.entity, []).append((prop, obs))

        extrinsics = self.rack.extrinsics
        out: list[Detection] = []
        for entity, sightings in seen.items():
            states = {prop.state for prop, _ in sightings if prop.state is not None}
            state = next(iter(states)) if len(states) == 1 else None
            _prop, obs = sightings[0]
            xs, ys = obs.corners_px[:, 0], obs.corners_px[:, 1]
            pos, quat = self._pose(obs, extrinsics)
            out.append(Detection(
                cls=self.classes[entity], score=self.score,
                box_px=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                pos_rack=pos, quat_rack=quat,
                state=state, state_score=self.score if state else 0.0,
                entity=entity,
            ))
        return out

    def _pose(self, obs: TagObservation, extrinsics: Extrinsics | None):
        if extrinsics is None:
            return None, None
        try:
            tag = solve_pose(self._object, obs.corners_px, self.rack.intrinsics)
        except (RackFrameError, np.linalg.LinAlgError):
            return None, None
        if tag.reproj_rms_px > self.max_reproj_px:
            return None, None
        pos = extrinsics.to_rack(tag.t.reshape(1, 3))[0]
        return (tuple(float(v) for v in pos),
                quaternion_wxyz(extrinsics.R_cam_to_rack @ tag.R))
