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


# --------------------------------------------------------------------------
# Deep Learning Backends: YOLOv8 and Motion-TCN
# --------------------------------------------------------------------------
class YoloDetector:
    """Ultralytics YOLOv8 detector with ByteTrack track association.

    Maps detected object classes into Parikshak's Detection format with stable
    track IDs across frames.
    """

    DEFAULT_CLASS_MAPPING = {
        "bottle": "vial",
        "wine glass": "vial",
        "cup": "vial",
        "cell phone": "sample_cartridge",
        "book": "stowage_locker",
        "backpack": "sample_bag",
        "suitcase": "sample_bag",
        "scissors": "cartridge_holder",
    }

    def __init__(self, model_path: str = "yolov8n.pt", *,
                 class_mapping: dict[str, str] | None = None,
                 score_threshold: float = 0.25,
                 track: bool = True) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YoloDetector needs ultralytics: pip install ultralytics") from exc
        import hashlib
        from pathlib import Path

        p = Path(model_path)
        self.weights_sha256 = (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "yolov8n-weights"
        )
        self.model = YOLO(model_path)
        self.class_mapping = class_mapping or dict(self.DEFAULT_CLASS_MAPPING)
        self.score_threshold = score_threshold
        self.track = track

    def __call__(self, image: Image) -> list[Detection]:
        if self.track:
            try:
                results = self.model.track(image, persist=True, verbose=False, imgsz=320)[0]
            except Exception:
                results = self.model(image, verbose=False, imgsz=320)[0]
        else:
            results = self.model(image, verbose=False, imgsz=320)[0]

        out: list[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return out

        for box in results.boxes:
            conf = float(box.conf[0].item())
            if conf < self.score_threshold:
                continue
            cls_id = int(box.cls[0].item())
            raw_cls = self.model.names[cls_id]
            cls_name = self.class_mapping.get(raw_cls, raw_cls)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            track_id = int(box.id[0].item()) if (box.id is not None and len(box.id) > 0) else None

            out.append(Detection(
                cls=cls_name,
                score=conf,
                box_px=(x1, y1, x2, y2),
                track_id=track_id,
            ))
        return out


class YoloPoseEstimator:
    """Ultralytics YOLOv8-pose estimator for crew body keypoints."""

    KEYPOINT_NAMES = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )

    def __init__(self, model_path: str = "yolov8n-pose.pt", *, min_score: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YoloPoseEstimator needs ultralytics: pip install ultralytics") from exc
        import hashlib
        from pathlib import Path

        p = Path(model_path)
        self.weights_sha256 = (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "yolov8n-pose-weights"
        )
        self.model = YOLO(model_path)
        self.min_score = min_score

    def __call__(self, image: Image) -> BodyKeypoints | None:
        results = self.model(image, verbose=False, imgsz=320)[0]
        if results.keypoints is None or len(results.keypoints) == 0:
            return None
        kpts_data = results.keypoints.data[0].cpu().numpy()  # (17, 3) (x, y, conf)
        mean_score = float(np.mean(kpts_data[:, 2]))
        if mean_score < self.min_score:
            return None

        points_px = kpts_data[:, :2].astype(float)
        joints_rack: dict[str, tuple[float, float, float]] = {}
        for idx, name in enumerate(self.KEYPOINT_NAMES):
            if kpts_data[idx, 2] >= self.min_score:
                px, py = float(points_px[idx, 0]), float(points_px[idx, 1])
                joints_rack[name] = ((px - 320.0) / 320.0, (py - 240.0) / 240.0, 1.0)

        if "left_ankle" in joints_rack:
            joints_rack["ankle_l"] = joints_rack["left_ankle"]
        if "right_ankle" in joints_rack:
            joints_rack["ankle_r"] = joints_rack["right_ankle"]

        if "left_hip" in joints_rack and "right_hip" in joints_rack:
            lh = joints_rack["left_hip"]
            rh = joints_rack["right_hip"]
            joints_rack["pelvis"] = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0, 1.0)

        return BodyKeypoints(
            score=mean_score,
            points_px=points_px,
            names=self.KEYPOINT_NAMES,
            points_rack=joints_rack,
        )


class YoloHands:
    """Hand tracker using YOLO pose arm kinematics or MediaPipe."""

    def __init__(self, pose_model: YoloPoseEstimator | None = None) -> None:
        self.pose_estimator = pose_model or YoloPoseEstimator()
        self._mp = None
        try:
            self._mp = MediaPipeHands()
        except Exception:
            self._mp = None

    def __call__(self, image: Image) -> list[HandLandmarks]:
        if self._mp is not None:
            try:
                res = self._mp(image)
                if res:
                    return res
            except Exception:
                pass

        body = self.pose_estimator(image)
        if body is None:
            return []

        out: list[HandLandmarks] = []
        pts = body.points_px

        for wrist_idx, elbow_idx, side in [(9, 7, "left"), (10, 8, "right")]:
            wrist = pts[wrist_idx]
            elbow = pts[elbow_idx]
            arm_vec = wrist - elbow
            arm_len = float(np.linalg.norm(arm_vec)) or 10.0
            span = max(20.0, arm_len * 0.4)
            unit_dir = arm_vec / arm_len

            kpts21 = np.zeros((21, 2), dtype=float)
            kpts21[0] = wrist
            perp_dir = np.array([-unit_dir[1], unit_dir[0]])
            palm_center = wrist + unit_dir * (span * 0.5)

            offsets = [-0.4, -0.2, 0.0, 0.2, 0.4]
            tips = [4, 8, 12, 16, 20]
            for tip_idx, offset in zip(tips, offsets):
                kpts21[tip_idx] = palm_center + unit_dir * (span * 0.5) + perp_dir * (span * offset)

            for i in range(1, 21):
                if i not in tips:
                    kpts21[i] = wrist + (kpts21[tips[min(i // 4, 4)]] - wrist) * (float(i % 4) / 4.0)

            out.append(HandLandmarks(
                side=side,
                score=float(body.score),
                points_px=kpts21,
            ))
        return out


# Re-export TcnMotionClassifier for convenience
from parikshak.perception.motion import TcnMotionClassifier

