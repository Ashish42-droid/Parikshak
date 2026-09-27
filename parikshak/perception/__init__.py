"""Perception: pixels -> BeliefFrame, in rack coordinates.

    perception/ may not import engine/     (enforced by tests/test_layering.py)

Everything here answers "what does the image contain". What that means for the
procedure is the engine's business, and the two are built by different people
against the frozen BeliefFrame contract.
"""

from parikshak.perception.contact import (
    ContactHead,
    ContactMLP,
    ContactSmoother,
    GeometricContact,
)
from parikshak.perception.pipeline import (
    EntityBinding,
    PerceptionPipeline,
    PipelineConfig,
)
from parikshak.perception.rackframe import (
    CameraIntrinsics,
    Extrinsics,
    RackFrameError,
    RackFrameEstimator,
    TagLayout,
    TagObservation,
    solve_pose,
)
from parikshak.perception.types import (
    BodyKeypoints,
    ContactState,
    Detection,
    HandLandmarks,
    MotionState,
)

__all__ = [
    "BodyKeypoints", "CameraIntrinsics", "ContactHead", "ContactMLP", "ContactSmoother",
    "ContactState", "Detection", "EntityBinding", "Extrinsics", "GeometricContact",
    "HandLandmarks", "MotionState", "PerceptionPipeline", "PipelineConfig",
    "RackFrameError", "RackFrameEstimator", "TagLayout", "TagObservation", "solve_pose",
]
