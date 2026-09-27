"""Perception's internal vocabulary.

None of this crosses into `belief/`. The engine is handed a BeliefFrame and
never sees a bounding box, a knuckle keypoint or a track score - which is what
lets the detector, the hand model and the pose model each be swapped without
touching a procedure file or a line of engine code.

The boundary is worth stating precisely, because it is easy to erode: anything
here answers "what does the image contain", and anything in `belief/` answers
"what does the engine need to know". A box in pixels is the first; "the vial is
grasped, in rack coordinates, at 0.9 confidence" is the second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

#: MediaPipe hand landmark indices used downstream. The other fourteen are real
#: and unused: the contact head needs fingertips and the wrist, and carrying the
#: rest through the pipeline would invite someone to reason about knuckles.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detected in one image, before tracking."""

    cls: str                    # detector class, e.g. "vial" - NOT an entity name
    score: float
    box_px: tuple[float, float, float, float]      # x1, y1, x2, y2
    #: Rack-frame centroid, when the rack is locked and depth is available.
    #: None is the honest answer without a lock, and it propagates to UNVERIFIED.
    pos_rack: Vec3 | None = None
    quat_rack: tuple[float, float, float, float] | None = None
    state: str | None = None
    state_score: float = 0.0
    track_id: int | None = None
    #: The procedure entity, when the detector resolved the instance itself - a
    #: fiducial marker IS an instance key. None for a class-level detector, whose
    #: detections the pipeline binds by class.
    entity: str | None = None

    @property
    def centre_px(self) -> Vec2:
        x1, y1, x2, y2 = self.box_px
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area_px(self) -> float:
        x1, y1, x2, y2 = self.box_px
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def iou(self, other: Detection) -> float:
        ax1, ay1, ax2, ay2 = self.box_px
        bx1, by1, bx2, by2 = other.box_px
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = self.area_px + other.area_px - inter
        return inter / union if union > 1e-9 else 0.0

    def contains_px(self, point: Vec2) -> bool:
        x1, y1, x2, y2 = self.box_px
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

    def distance_to_px(self, point: Vec2) -> float:
        """Distance from a point to the box, zero inside it.

        Distance-to-box rather than distance-to-centre: a hand touching the rim
        of a large object is in contact with it, and centre distance would score
        that the same as a hand a box-width away.
        """
        x1, y1, x2, y2 = self.box_px
        dx = max(x1 - point[0], 0.0, point[0] - x2)
        dy = max(y1 - point[1], 0.0, point[1] - y2)
        return math.hypot(dx, dy)


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """21 keypoints for one hand, in pixels, plus the rack-frame wrist.

    The full 21 stay inside perception. Only the wrist reaches the BeliefFrame,
    because `hand_in_zone` is the only predicate that asks where a hand is.
    """

    side: str                   # "left" | "right"
    score: float
    points_px: np.ndarray       # (21, 2)
    wrist_rack: Vec3 | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.points_px, dtype=float)
        if arr.shape != (21, 2):
            raise ValueError(f"expected 21 hand landmarks, got shape {arr.shape}")

    @property
    def wrist_px(self) -> Vec2:
        return tuple(self.points_px[WRIST])       # type: ignore[return-value]

    @property
    def fingertips_px(self) -> np.ndarray:
        return np.asarray(self.points_px)[list(FINGERTIPS)]

    @property
    def span_px(self) -> float:
        """Wrist-to-middle-fingertip distance: the hand's apparent scale.

        Everything the contact head measures is normalised by this, so the same
        model works at any camera distance without retraining per rig.
        """
        pts = np.asarray(self.points_px)
        return float(np.linalg.norm(pts[MIDDLE_TIP] - pts[WRIST])) or 1.0

    @property
    def grip_aperture(self) -> float:
        """Thumb-to-index separation, normalised by hand span.

        Small means closed around something; large means open. It is the single
        most informative scalar for telling `grasp` from `reach`, and it is a
        ratio, so it does not care how far away the camera is.
        """
        pts = np.asarray(self.points_px)
        return float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])) / self.span_px


@dataclass(frozen=True, slots=True)
class BodyKeypoints:
    """Crew pose. Sparse on purpose - only the joints posture predicates ask
    about. A full 26-kpt Halpe set is welcome but not required, which is what
    makes descope ladder item 3 a config change."""

    score: float
    points_px: np.ndarray                       # (K, 2)
    names: tuple[str, ...]
    points_rack: dict[str, Vec3] = field(default_factory=dict)

    def rack_joints(self) -> dict[str, Vec3]:
        return dict(self.points_rack)


@dataclass(frozen=True, slots=True)
class ContactState:
    """What the contact head says one hand is doing.

    `reach` is deliberately distinct from `grasp`: the hand on its way to an
    object is the moment a naive detector fires early, and `grasped` must not be
    true until the object is actually held.
    """

    hand: str
    label: str                  # none | reach | grasp | manipulate | release
    score: float
    entity_cls: str | None = None
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class MotionState:
    """Active class of the temporal model over its trailing window."""

    label: str
    score: float
