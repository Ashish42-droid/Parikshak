"""Model adapters: detector, hands, pose, motion, tags.

Each is a Protocol plus two implementations - a lazily-imported real backend and
a deterministic scripted fake. That shape is not scaffolding, it is the reason
the rest of the system is testable today: mediapipe, onnxruntime and
pupil-apriltags are optional extras, and `pipeline.py` must be exercisable on a
laptop that has none of them installed.

The real adapters are thin on purpose. Everything that makes PARIKSHAK work -
rack-frame canonicalisation, the contact features, the belief assembly - lives
in code we own and test. These classes only convert one library's output shape
into ours, so swapping RT-DETR for something else is an afternoon.

**Honest status.** The scripted backends are complete and used by the tests. The
real backends are correct adapters against published APIs but are UNEXERCISED
until the weights and the recorded runs exist - they are wiring, not results,
and nothing in the eval report is produced by them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from parikshak.perception.types import BodyKeypoints, Detection, HandLandmarks, MotionState

Image = np.ndarray


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------
@runtime_checkable
class ObjectDetector(Protocol):
    def __call__(self, image: Image) -> list[Detection]: ...


@runtime_checkable
class HandTracker(Protocol):
    def __call__(self, image: Image) -> list[HandLandmarks]: ...


@runtime_checkable
class PoseEstimator(Protocol):
    def __call__(self, image: Image) -> BodyKeypoints | None: ...


@runtime_checkable
class MotionClassifier(Protocol):
    def __call__(self, window: np.ndarray) -> MotionState: ...


# --------------------------------------------------------------------------
# scripted backends - what the tests and the replay path use
# --------------------------------------------------------------------------
@dataclass
class ScriptedDetector:
    """Returns pre-authored detections, one list per frame.

    Lets the whole pipeline be driven from a script rather than from pixels, so
    belief assembly can be tested against known ground truth. When the frames
    run out it keeps returning the last one, which models a static scene rather
    than an abrupt blackout.
    """

    frames: list[list[Detection]] = field(default_factory=list)
    _i: int = 0

    def __call__(self, image: Image) -> list[Detection]:
        if not self.frames:
            return []
        out = self.frames[min(self._i, len(self.frames) - 1)]
        self._i += 1
        return list(out)


@dataclass
class ScriptedHands:
    frames: list[list[HandLandmarks]] = field(default_factory=list)
    _i: int = 0

    def __call__(self, image: Image) -> list[HandLandmarks]:
        if not self.frames:
            return []
        out = self.frames[min(self._i, len(self.frames) - 1)]
        self._i += 1
        return list(out)


@dataclass
class ScriptedPose:
    frames: list[BodyKeypoints | None] = field(default_factory=list)
    _i: int = 0

    def __call__(self, image: Image) -> BodyKeypoints | None:
        if not self.frames:
            return None
        out = self.frames[min(self._i, len(self.frames) - 1)]
        self._i += 1
        return out


@dataclass
class ScriptedMotion:
    labels: list[tuple[str, float]] = field(default_factory=list)
    _i: int = 0

    def __call__(self, window: np.ndarray) -> MotionState:
        if not self.labels:
            return MotionState("idle", 0.0)
        label, score = self.labels[min(self._i, len(self.labels) - 1)]
        self._i += 1
        return MotionState(label, score)


# --------------------------------------------------------------------------
# real backends - lazily imported, unexercised until weights exist
# --------------------------------------------------------------------------
class OnnxDetector:
    """RT-DETR-R18 via ONNX Runtime.

    RT-DETR rather than YOLO for a licensing reason as much as an accuracy one:
    Ultralytics is AGPL-3.0, which a procurement-aware reviewer will notice on
    a delivered artefact. Train with whatever is convenient, ship weights whose
    licence can touch the deliverable.
    """

    def __init__(self, model_path: str, class_names: Sequence[str], *,
                 score_threshold: float = 0.35, providers: Sequence[str] | None = None) -> None:
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "OnnxDetector needs onnxruntime: pip install 'parikshak[perception]'") from exc
        import hashlib
        from pathlib import Path

        self.class_names = tuple(class_names)
        self.score_threshold = score_threshold
        self.weights_sha256 = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        self.session = onnxruntime.InferenceSession(
            model_path, providers=list(providers or ["CPUExecutionProvider"]))
        self._input = self.session.get_inputs()[0].name

    def __call__(self, image: Image) -> list[Detection]:  # pragma: no cover - needs weights
        blob = image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        boxes, labels, scores = self.session.run(None, {self._input: blob})[:3]
        out: list[Detection] = []
        for box, label, score in zip(boxes[0], labels[0], scores[0]):
            if score < self.score_threshold:
                continue
            idx = int(label)
            if idx >= len(self.class_names):
                continue
            out.append(Detection(cls=self.class_names[idx], score=float(score),
                                 box_px=tuple(float(v) for v in box[:4])))
        return out


class MediaPipeHands:
    """MediaPipe Hands: BlazePalm plus the landmark model, ~2 MB."""

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "MediaPipeHands needs mediapipe: pip install 'parikshak[perception]'") from exc
        self._solution = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence)

    def __call__(self, image: Image) -> list[HandLandmarks]:  # pragma: no cover - needs mediapipe
        h, w = image.shape[:2]
        result = self._solution.process(image)
        if not result.multi_hand_landmarks:
            return []
        out: list[HandLandmarks] = []
        for lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness or []):
            pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=float)
            label = handed.classification[0].label.lower()
            out.append(HandLandmarks(side="left" if label.startswith("l") else "right",
                                     score=float(handed.classification[0].score),
                                     points_px=pts))
        return out


class AprilTagDetector:
    """AprilTag 36h11 corner detection.

    Prefers pupil-apriltags; falls back to OpenCV's ArUco, which ships an
    AprilTag 36h11 dictionary. Corner ORDER differs between the two, and a
    silent mismatch produces a pose wrong by a rotation while looking perfectly
    plausible - so the layout file states the convention explicitly and the
    rotated-rack test is what catches a mismatch.
    """

    #: pupil-apriltags reports corners as (left-bottom, right-bottom, right-top,
    #: left-top), which is the layout convention. OpenCV's ArUco reports
    #: (top-left, top-right, bottom-right, bottom-left) - the exact reverse.
    #:
    #: Getting this wrong is not a loud failure. Measured on rendered tags, the
    #: reversed order gave 71 px reprojection error while the recovered
    #: translation was still only 4 mm out: a pose that looks entirely plausible
    #: and is wrong by a reflection. Every downstream coordinate then sits in
    #: the wrong place and nothing anywhere says so. Hence the reorder here and
    #: the sub-pixel reprojection assertion in the tests.
    _REVERSE_CORNERS = {"aruco": True, "pupil": False}

    def __init__(self, family: str = "tag36h11") -> None:
        self.family = family
        self._backend = None
        try:  # pragma: no cover - optional extra
            import pupil_apriltags
            self._backend = ("pupil", pupil_apriltags.Detector(families=family))
        except ImportError:
            try:
                import cv2
                dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
                params = cv2.aruco.DetectorParameters()
                # Sub-pixel corner refinement, off by default in OpenCV. On
                # rendered 1280x720 frames it cut rack-lock reprojection from
                # 0.82 to 0.45 px, and brought a 60 mm prop tag at 1.5 m to a
                # median position error of ~6 mm (95th percentile 36 mm, over
                # 80 placements - see perception/markers.py).
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                self._backend = ("aruco", cv2.aruco.ArucoDetector(dictionary, params))
            except Exception:  # pragma: no cover
                self._backend = None

    @property
    def available(self) -> bool:
        return self._backend is not None

    def __call__(self, image: Image):  # pragma: no cover - needs a real image
        from parikshak.perception.rackframe import TagObservation
        if self._backend is None:
            return []
        kind, detector = self._backend
        if kind == "pupil":
            return [TagObservation(int(d.tag_id), np.asarray(d.corners, dtype=float))
                    for d in detector.detect(image)]
        corners, ids, _rejected = detector.detectMarkers(image)
        if ids is None:
            return []
        return [TagObservation(int(np.ravel(i)[0]),
                               np.asarray(c, dtype=float).reshape(4, 2)[::-1])
                for c, i in zip(corners, ids)]
