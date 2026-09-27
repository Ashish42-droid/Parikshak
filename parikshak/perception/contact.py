"""Hand-object contact: {none, reach, grasp, manipulate, release}.

The small learned component that makes `grasped` mean something. It is a
three-layer MLP over ~50k parameters, not because a bigger model would not
score better, but because the input is a dozen geometric quantities and
anything larger would be memorising our rack.

Two things here are worth more than the architecture:

**The features are scale-free.** Every distance is divided by the hand's own
span, and every velocity by span-per-second. A model trained at one camera
distance therefore transfers to another without retraining, which is what makes
"add a third camera" a mounting job rather than a data campaign.

**There is a geometric fallback.** With no weights loaded, `GeometricContact`
answers the same question from aperture and proximity alone. It is worse - it
cannot tell `manipulate` from `grasp` - but it is honest, it is deterministic,
and it means the pipeline runs end to end before DexYCB has finished
downloading. PLAN.md risk 7 pre-commits to exactly this: if the contact head
underperforms, affected steps fall back to geometry-only verification.

Training is out of scope here. This file defines the feature vector, the
forward pass and the fallback, so that the day the weights exist they drop in
without anything downstream changing.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from parikshak.perception.types import ContactState, Detection, HandLandmarks

#: Output vocabulary. Must match belief.frame.GRASP_TYPES exactly - the whole
#: contract depends on perception and the engine agreeing on what a hand can be
#: doing, and there is a test asserting they do.
LABELS = ("none", "reach", "grasp", "manipulate", "release")

#: Number of features per (hand, candidate object) pair. See `features()`.
N_FEATURES = 12


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def features(hand: HandLandmarks, det: Detection, *,
             prev_gap: float | None = None,
             prev_obj_centre: tuple[float, float] | None = None,
             dt: float = 0.1) -> np.ndarray:
    """Geometric features for one (hand, object) pair.

    Every length is normalised by the hand's span, so the vector is invariant to
    how far the camera is from the rack. That property is why one trained head
    covers all three cameras.

        0   wrist-to-box distance / span
        1   nearest-fingertip-to-box distance / span
        2   mean-fingertip-to-box distance / span
        3   grip aperture (thumb-index / span)
        4   fraction of fingertips inside the box
        5   wrist inside the box
        6   IoU of the hand's bounding box with the object's
        7   object box diagonal / span              (relative object size)
        8   closing speed of the gap, span/s        (approach vs withdrawal)
        9   object speed, span/s                    (is it being carried)
        10  detector score
        11  hand landmark score
    """
    pts = np.asarray(hand.points_px, dtype=float)
    span = hand.span_px
    tips = hand.fingertips_px

    wrist_gap = det.distance_to_px(hand.wrist_px) / span
    tip_gaps = np.array([det.distance_to_px(tuple(p)) for p in tips]) / span
    inside = np.array([det.contains_px(tuple(p)) for p in tips], dtype=float)

    hand_box = Detection(cls="hand", score=hand.score,
                         box_px=(float(pts[:, 0].min()), float(pts[:, 1].min()),
                                 float(pts[:, 0].max()), float(pts[:, 1].max())))

    gap = float(tip_gaps.min())
    closing = 0.0 if prev_gap is None or dt <= 0 else (prev_gap - gap) / dt

    obj_speed = 0.0
    if prev_obj_centre is not None and dt > 0:
        cx, cy = det.centre_px
        obj_speed = float(np.hypot(cx - prev_obj_centre[0],
                                   cy - prev_obj_centre[1])) / span / dt

    x1, y1, x2, y2 = det.box_px
    diag = float(np.hypot(x2 - x1, y2 - y1)) / span

    return np.array([
        wrist_gap,
        gap,
        float(tip_gaps.mean()),
        hand.grip_aperture,
        float(inside.mean()),
        float(det.contains_px(hand.wrist_px)),
        hand_box.iou(det),
        diag,
        closing,
        obj_speed,
        det.score,
        hand.score,
    ], dtype=float)


# --------------------------------------------------------------------------
# the learned head
# --------------------------------------------------------------------------
@dataclass
class ContactMLP:
    """Three-layer MLP, forward pass only. ~50k parameters.

    Training lives outside the runtime on purpose: the deployed box never
    trains, and a model that can only be evaluated is a model whose behaviour is
    reproducible from a weights hash - which is what the log records.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    sha256: str = ""

    def __post_init__(self) -> None:
        if self.w1.shape[0] != N_FEATURES:
            raise ValueError(
                f"weights expect {self.w1.shape[0]} features, this build produces {N_FEATURES}")
        if self.w3.shape[1] != len(LABELS):
            raise ValueError(
                f"weights emit {self.w3.shape[1]} classes, vocabulary has {len(LABELS)}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        h = np.maximum(0.0, h @ self.w2 + self.b2)
        return _softmax(h @ self.w3 + self.b3)

    @classmethod
    def load(cls, path: str | Path) -> ContactMLP:
        import hashlib
        p = Path(path)
        blob = p.read_bytes()
        d = np.load(p) if p.suffix == ".npz" else json.loads(blob.decode("utf-8"))
        return cls(
            w1=np.asarray(d["w1"], dtype=float), b1=np.asarray(d["b1"], dtype=float),
            w2=np.asarray(d["w2"], dtype=float), b2=np.asarray(d["b2"], dtype=float),
            w3=np.asarray(d["w3"], dtype=float), b3=np.asarray(d["b3"], dtype=float),
            sha256=hashlib.sha256(blob).hexdigest(),
        )


# --------------------------------------------------------------------------
# the fallback
# --------------------------------------------------------------------------
class GeometricContact:
    """Contact from aperture and proximity, with no learned parameters.

    Deliberately conservative, and deliberately limited: it cannot distinguish
    `manipulate` from `grasp`, because that distinction is about motion over
    time and this looks at one frame. It reports `grasp` for both and says so.

    Its job is to make the pipeline runnable and the engine testable before the
    contact head has data, and to be the landing place for PLAN.md risk 7 if the
    head underperforms. A system that degrades to something explainable beats
    one that degrades to a number nobody can defend.
    """

    #: Fingertip-to-box gap, in hand spans, below which contact is asserted.
    TOUCH_SPANS = 0.18
    #: Gap below which the hand is at least reaching toward the object.
    REACH_SPANS = 0.9
    #: Grip aperture below which the hand is closed around something.
    CLOSED_APERTURE = 0.55

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gap, aperture = float(x[1]), float(x[3])
        inside_frac, closing = float(x[4]), float(x[8])

        p = np.zeros(len(LABELS))
        if gap <= self.TOUCH_SPANS and aperture <= self.CLOSED_APERTURE:
            # Closed hand at the object: held. Confidence rises with how much of
            # the hand is actually on it.
            p[LABELS.index("grasp")] = 0.60 + 0.35 * min(1.0, inside_frac + 0.4)
        elif gap <= self.TOUCH_SPANS and closing < -0.15:
            # Touching but opening and moving away: letting go.
            p[LABELS.index("release")] = 0.70
        elif gap <= self.REACH_SPANS and closing > 0.05:
            p[LABELS.index("reach")] = 0.55 + 0.3 * min(1.0, closing)
        else:
            p[LABELS.index("none")] = 0.90
        total = p.sum()
        p[LABELS.index("none")] += max(0.0, 1.0 - total)
        return p / p.sum()


# --------------------------------------------------------------------------
# the head, wired up
# --------------------------------------------------------------------------
class ContactHead:
    """Assigns each hand at most one object, per frame.

    One object per hand, chosen as the best-scoring candidate, because a hand
    holding two things at once is not a state the predicate vocabulary can
    express - `grasped(entity)` has no way to say "partially". Forcing the
    choice here keeps the ambiguity where it can be measured instead of pushing
    it into the engine.
    """

    def __init__(self, model: ContactMLP | GeometricContact | None = None, *,
                 min_score: float = 0.5) -> None:
        self.model = model or GeometricContact()
        self.min_score = min_score
        self._prev_gap: dict[tuple[str, int | None], float] = {}
        self._prev_centre: dict[int | None, tuple[float, float]] = {}

    @property
    def is_learned(self) -> bool:
        return isinstance(self.model, ContactMLP)

    @property
    def weights_sha256(self) -> str:
        """Goes into the run log. A flight record that cannot say which weights
        produced it cannot be audited."""
        return self.model.sha256 if isinstance(self.model, ContactMLP) else "geometric-fallback"

    def __call__(self, hands: Sequence[HandLandmarks],
                 detections: Sequence[Detection], dt: float = 0.1) -> list[ContactState]:
        out: list[ContactState] = []
        for hand in hands:
            best: tuple[float, str, Detection] | None = None
            for det in detections:
                key = (hand.side, det.track_id)
                x = features(hand, det,
                             prev_gap=self._prev_gap.get(key),
                             prev_obj_centre=self._prev_centre.get(det.track_id),
                             dt=dt)
                self._prev_gap[key] = float(x[1])
                probs = self.model(x)
                idx = int(np.argmax(probs))
                score = float(probs[idx])
                if LABELS[idx] == "none":
                    continue
                if best is None or score > best[0]:
                    best = (score, LABELS[idx], det)

            if best is None or best[0] < self.min_score:
                out.append(ContactState(hand=hand.side, label="none", score=1.0))
                continue
            score, label, det = best
            out.append(ContactState(hand=hand.side, label=label, score=score,
                                    entity_cls=det.cls, track_id=det.track_id))

        for det in detections:
            self._prev_centre[det.track_id] = det.centre_px
        return out


# --------------------------------------------------------------------------
class ContactSmoother:
    """Majority vote over a short trailing window.

    Per-frame contact flickers, and a `grasped` clause that flickers turns into
    a step that will not verify. Smoothing here rather than in the engine keeps
    the fix next to the noise it corrects, and keeps the engine's temporal logic
    to the windows the procedure author declared.
    """

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._hist: dict[str, deque[ContactState]] = {}

    def __call__(self, states: Sequence[ContactState]) -> list[ContactState]:
        out: list[ContactState] = []
        for state in states:
            hist = self._hist.setdefault(state.hand, deque(maxlen=self.window))
            hist.append(state)
            counts: dict[str, list[ContactState]] = {}
            for s in hist:
                counts.setdefault(s.label, []).append(s)
            label = max(counts, key=lambda k: (len(counts[k]),
                                               max(s.score for s in counts[k])))
            members = counts[label]
            winner = max(members, key=lambda s: s.score)
            out.append(ContactState(
                hand=state.hand, label=label,
                score=float(np.mean([s.score for s in members])),
                entity_cls=winner.entity_cls, track_id=winner.track_id))
        return out
