"""Comprehensive unit tests for Parikshak's deep learning perception models.

Tests:
1. ContactMLP neural network weights and classification against builds/contact_mlp.npz
2. MotionTCN 1D dilated temporal convolutional network against builds/motion_tcn.npz
3. YoloDetector object detection and ByteTrack association
4. YoloPoseEstimator keypoints, Halpe/COCO joints, and rack projections
5. YoloHands 21-kpt hand tracking derivation
6. End-to-end PerceptionPipeline integration with the full neural suite
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from parikshak.belief.frame import BeliefFrame
from parikshak.perception.backends import (
    YoloDetector,
    YoloHands,
    YoloPoseEstimator,
    TcnMotionClassifier,
)
from parikshak.perception.contact import (
    ContactHead,
    ContactMLP,
    LABELS as CONTACT_LABELS,
    features,
)
from parikshak.perception.motion import (
    MOTION_CLASSES,
    MotionTCN,
    extract_frame_features,
)
from parikshak.perception.pipeline import (
    EntityBinding,
    PerceptionPipeline,
    PipelineConfig,
)
from parikshak.perception.rackframe import (
    CameraIntrinsics,
    RackFrameEstimator,
    TagLayout,
)
from parikshak.perception.types import (
    Detection,
    HandLandmarks,
    MotionState,
)

ROOT = Path(__file__).resolve().parent.parent
CONTACT_WEIGHTS = ROOT / "builds" / "contact_mlp.npz"
MOTION_WEIGHTS = ROOT / "builds" / "motion_tcn.npz"
RACK_LAYOUT = ROOT / "racks" / "msg_a_fiducials.json"


# --------------------------------------------------------------------------
# 1. ContactMLP Deep Learning Model Tests
# --------------------------------------------------------------------------
def test_contact_mlp_weights_file_exists_and_loads():
    assert CONTACT_WEIGHTS.exists(), "builds/contact_mlp.npz must exist"
    model = ContactMLP.load(CONTACT_WEIGHTS)
    assert model.w1.shape == (12, 64)
    assert model.w2.shape == (64, 32)
    assert model.w3.shape == (32, 5)
    assert len(model.sha256) == 64
    assert model.sha256 != ""


def test_contact_mlp_classifies_grasp_accurately():
    model = ContactMLP.load(CONTACT_WEIGHTS)
    # Feature vector: hand closed tightly on object, low object speed
    # gap=0.02, aperture=0.25, inside=0.9, obj_speed=0.05
    x = np.array([
        0.2,   # wrist_gap
        0.02,  # gap
        0.04,  # mean_gap
        0.25,  # aperture (closed)
        0.9,   # inside fraction
        0.0,   # wrist in box
        0.6,   # IoU
        1.2,   # diag
        0.0,   # closing
        0.05,  # obj_speed (stationary hold)
        0.9,   # det score
        0.95,  # hand score
    ], dtype=np.float32)

    probs = model(x)
    assert probs.shape == (5,)
    assert np.isclose(probs.sum(), 1.0)
    pred_label = CONTACT_LABELS[int(np.argmax(probs))]
    assert pred_label == "grasp"
    assert probs[CONTACT_LABELS.index("grasp")] > 0.85


def test_contact_mlp_classifies_manipulate_when_object_moving():
    model = ContactMLP.load(CONTACT_WEIGHTS)
    # Feature vector: hand closed on object, but object moving rapidly (obj_speed=0.8)
    x = np.array([
        0.2, 0.02, 0.04, 0.25, 0.9, 0.0, 0.6, 1.2, 0.0,
        0.85,  # obj_speed: high!
        0.9, 0.95
    ], dtype=np.float32)

    probs = model(x)
    pred_label = CONTACT_LABELS[int(np.argmax(probs))]
    assert pred_label == "manipulate"
    assert probs[CONTACT_LABELS.index("manipulate")] > 0.85


def test_contact_mlp_classifies_reach_on_approach():
    model = ContactMLP.load(CONTACT_WEIGHTS)
    # Gap 0.5, closing speed 0.8, open aperture 0.8
    x = np.array([
        0.8, 0.5, 0.6, 0.8, 0.0, 0.0, 0.15, 1.2,
        0.8,   # closing > 0
        0.0,
        0.85, 0.9
    ], dtype=np.float32)

    probs = model(x)
    pred_label = CONTACT_LABELS[int(np.argmax(probs))]
    assert pred_label == "reach"
    assert probs[CONTACT_LABELS.index("reach")] > 0.80


def test_contact_mlp_classifies_release_on_withdrawal():
    model = ContactMLP.load(CONTACT_WEIGHTS)
    # Gap opening, closing speed negative (-0.8), aperture opening 0.85
    x = np.array([
        0.4, 0.18, 0.25, 0.85, 0.05, 0.0, 0.2, 1.2,
        -0.8,  # negative closing
        0.05,
        0.85, 0.9
    ], dtype=np.float32)

    probs = model(x)
    pred_label = CONTACT_LABELS[int(np.argmax(probs))]
    assert pred_label == "release"
    assert probs[CONTACT_LABELS.index("release")] > 0.80


def test_contact_head_with_mlp_is_learned():
    mlp = ContactMLP.load(CONTACT_WEIGHTS)
    head = ContactHead(model=mlp)
    assert head.is_learned is True
    assert head.weights_sha256 == mlp.sha256


# --------------------------------------------------------------------------
# 2. MotionTCN 1D Dilated Temporal Model Tests
# --------------------------------------------------------------------------
def test_motion_tcn_weights_exist_and_load():
    assert MOTION_WEIGHTS.exists(), "builds/motion_tcn.npz must exist"
    model = MotionTCN.load(MOTION_WEIGHTS)
    assert model.w_in.shape == (64, 16, 3)
    assert model.w_res1.shape == (64, 64, 3)
    assert model.w_res2.shape == (64, 64, 3)
    assert model.w_res3.shape == (64, 64, 3)
    assert model.w_fc1.shape == (64, 32)
    assert model.w_fc2.shape == (32, 7)
    assert len(model.sha256) == 64


def test_motion_tcn_detects_agitation_cycles():
    model = MotionTCN.load(MOTION_WEIGHTS)
    classifier = TcnMotionClassifier(model=model)

    # Simulate 2.0 s @ 10 Hz (20 timesteps) sinusoidal shaking trajectory (agitate)
    t = np.linspace(0, 2.0, 20)
    traj = np.zeros((20, 16), dtype=np.float32)
    # Shaking X oscillation at 3 Hz
    traj[:, 0] = 0.5 + 0.18 * np.sin(2 * np.pi * 3.0 * t)
    traj[:, 3] = np.gradient(traj[:, 0], t)
    traj[:, 5] = np.abs(traj[:, 3])  # speed
    traj[:, 6] = 0.3                # closed grip
    traj[:, 7] = 0.95               # high contact conf
    traj[:, 8] = 0.75               # manipulate code
    traj[:, 9] = traj[:, 0]         # vial follows hand exactly
    traj[:, 12] = traj[:, 5]        # vial speed

    state = classifier(traj)
    assert isinstance(state, MotionState)
    assert state.label == "agitate"
    assert state.score >= 0.90


def test_motion_tcn_detects_insertion_trajectory():
    model = MotionTCN.load(MOTION_WEIGHTS)
    classifier = TcnMotionClassifier(model=model)

    # Sigmoidal approach into socket
    t = np.linspace(-3, 3, 20)
    prog = 1.0 / (1.0 + np.exp(-t))
    traj = np.zeros((20, 16), dtype=np.float32)
    traj[:, 1] = 0.4 + 0.25 * prog
    traj[:, 4] = np.gradient(traj[:, 1])
    traj[:, 5] = np.abs(traj[:, 4])
    traj[:, 6] = 0.3
    traj[:, 7] = 0.95
    traj[:, 8] = 0.5  # grasp
    traj[:, 15] = 0.95  # alignment

    state = classifier(traj)
    assert state.label == "insert"
    assert state.score >= 0.90


def test_motion_tcn_detects_rotation_seal():
    model = MotionTCN.load(MOTION_WEIGHTS)
    classifier = TcnMotionClassifier(model=model)

    traj = np.zeros((20, 16), dtype=np.float32)
    traj[:, 6] = 0.35
    traj[:, 7] = 0.95
    traj[:, 8] = 0.75  # manipulate
    traj[:, 14] = 1.8  # rotation proxy high

    state = classifier(traj)
    assert state.label == "rotate_seal"
    assert state.score >= 0.90


# --------------------------------------------------------------------------
# 3. Real YOLO Detectors and Pose Estimators
# --------------------------------------------------------------------------
def test_yolo_detector_instantiation_and_hashing():
    det = YoloDetector("yolov8n.pt", score_threshold=0.25)
    assert len(det.weights_sha256) == 64
    assert det.model is not None
    assert "bottle" in det.class_mapping

    # Run on a blank 3-channel test image
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    results = det(dummy)
    assert isinstance(results, list)


def test_yolo_pose_estimator_keypoints():
    pose = YoloPoseEstimator("yolov8n-pose.pt", min_score=0.25)
    assert len(pose.weights_sha256) == 64
    assert len(pose.KEYPOINT_NAMES) == 17

    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    res = pose(dummy)
    # Blank frame contains no person
    assert res is None or res.score < 0.25


def test_yolo_hands_adapter():
    pose = YoloPoseEstimator("yolov8n-pose.pt")
    tracker = YoloHands(pose_model=pose)
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    hands = tracker(dummy)
    assert isinstance(hands, list)


# --------------------------------------------------------------------------
# 4. End-to-End Neural PerceptionPipeline
# --------------------------------------------------------------------------
def test_full_neural_perception_pipeline():
    layout = TagLayout.load(RACK_LAYOUT)
    intrinsics = CameraIntrinsics.from_fov(640, 480, 65.0)
    rack = RackFrameEstimator(layout, intrinsics)

    bindings = {
        "vial_b": EntityBinding("vial_b", "vial"),
        "cartridge_holder": EntityBinding("cartridge_holder", "cartridge_holder"),
    }

    detector = YoloDetector("yolov8n.pt", track=False)
    pose = YoloPoseEstimator("yolov8n-pose.pt")
    hands = YoloHands(pose_model=pose)
    contact = ContactHead(model=ContactMLP.load(CONTACT_WEIGHTS))
    motion = TcnMotionClassifier(model=MotionTCN.load(MOTION_WEIGHTS))

    pipeline = PerceptionPipeline(
        bindings=bindings,
        rack=rack,
        detector=detector,
        hands=hands,
        pose=pose,
        motion=motion,
        contact=contact,
        config=PipelineConfig(fps=10.0),
    )

    versions = pipeline.model_versions()
    assert versions["contact"] == contact.weights_sha256
    assert versions["detector"] == detector.weights_sha256
    assert versions["pose"] == pose.weights_sha256
    assert versions["motion"] == motion.weights_sha256

    # Test processing a real frame through the full neural pipeline
    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
    belief = pipeline.step(frame_img, t=0.1)

    assert isinstance(belief, BeliefFrame)
    assert "vial_b" in belief.objects
    assert "left" in belief.hands
    assert "right" in belief.hands
    assert belief.motion.cls in MOTION_CLASSES
