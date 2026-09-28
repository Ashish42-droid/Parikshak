"""Compose perception into a BeliefFrame stream.

    image -> tags -> rack frame
          -> detector + tracker -> objects
          -> hands -> contact
          -> pose
          -> motion
          =>  BeliefFrame (rack coordinates)

This is where "what the image contains" becomes "what the engine needs to know",
and it is the only place in perception that knows an entity name exists. Below
it everything speaks in detector classes and pixels.

Two rules govern every line of it, and both are about refusing to guess:

  **No rack lock, no geometry.** When AprilTag PnP fails, `pos_rack` is None and
  `frame_lock` is False. Not the last known position, not the image centre
  projected onto a guessed plane - None. The engine turns that into UNVERIFIED,
  which is the correct answer and the one that keeps a rotated rack from
  producing confident nonsense.

  **Not seen is not absent.** An entity the detector did not report is emitted
  with `visible=False`, and `occluded` is set from the occlusion mask rather
  than assumed. `absent` must never fire because a torso was in the way.

The engine is not imported here and never will be - the layering test enforces
it. This module writes BeliefFrames; what reads them is not its business.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from parikshak.belief.frame import (
    BeliefFrame,
    BodyBelief,
    HandBelief,
    MotionBelief,
    ObjectBelief,
    contact_key,
)
from parikshak.perception.contact import ContactHead, ContactSmoother
from parikshak.perception.rackframe import CameraIntrinsics, RackFrameEstimator
from parikshak.perception.types import (
    BodyKeypoints,
    ContactState,
    Detection,
    HandLandmarks,
    MotionState,
)


@dataclass(frozen=True, slots=True)
class EntityBinding:
    """Ties a procedure entity to a detector class, and says how to tell it from
    its lookalikes.

    This mapping is the "procedure is data" boundary made concrete. `vial_a` and
    `vial_b` are both the detector class `vial`; what separates them is an
    instance key the author declared - a blue band versus an orange one. The
    detector never learns "vial A". It learns "vial", and the procedure says
    which one this is.
    """

    entity: str
    detector_class: str
    instance_key: dict[str, Any] = field(default_factory=dict)
    zone_hint: str | None = None

    @classmethod
    def from_procedure(cls, procedure) -> dict[str, EntityBinding]:
        """Build bindings from a loaded Procedure.

        Takes the Procedure rather than importing pdl at module scope, so this
        module stays usable with a plain dict of bindings and perception never
        acquires a hard dependency on the procedure layer.
        """
        return {
            name: cls(entity=name, detector_class=e.detector_class,
                      instance_key=dict(e.instance_key))
            for name, e in procedure.entities.items()
        }


@dataclass
class PipelineConfig:
    fps: float = 10.0
    #: Detector score below which an object is reported as not visible.
    min_detection_score: float = 0.35
    #: Fraction of a zone's area obscured before it is reported occluded.
    occlusion_report_threshold: float = 0.2
    contact_window: int = 5


class PerceptionPipeline:
    """Turns frames into BeliefFrames. One instance per run.

    Deliberately has no `run()` loop: capture, scheduling and recording belong
    to io/, and keeping them out means this class can be driven from a camera, a
    video file, or a list of scripted detections with no difference in
    behaviour. That is what makes the stage failover real - replay goes through
    the identical code.
    """

    def __init__(self, bindings: dict[str, EntityBinding],
                 rack: RackFrameEstimator, *,
                 detector=None, hands=None, pose=None, motion=None, tags=None,
                 contact: ContactHead | None = None,
                 config: PipelineConfig | None = None) -> None:
        self.bindings = bindings
        self.rack = rack
        self.detector = detector
        self.hands = hands
        self.pose = pose
        self.motion = motion
        self.tags = tags
        self.contact = contact or ContactHead()
        self.smoother = ContactSmoother(window=(config or PipelineConfig()).contact_window)
        self.config = config or PipelineConfig()

        self._t0_utc = datetime.now(timezone.utc)
        self._next_track_id = 1
        self._track_of: dict[str, int] = {}

    # ------------------------------------------------------------------
    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self.rack.intrinsics

    def model_versions(self) -> dict[str, str]:
        """What produced this run. Goes into the log header, because a flight
        record that cannot name its weights cannot be audited."""
        return {
            "contact": self.contact.weights_sha256,
            "detector": getattr(self.detector, "weights_sha256", "scripted"),
            "pose": getattr(self.pose, "weights_sha256", "scripted"),
            "motion": getattr(self.motion, "weights_sha256", "scripted"),
            "rack_layout": self.rack.layout.family,
        }

    # ------------------------------------------------------------------
    def step(self, image, t: float, *,
             confirmations: Sequence[str] = (),
             occlusion: dict[str, float] | None = None) -> BeliefFrame:
        """One frame in, one BeliefFrame out."""
        dt = 1.0 / max(self.config.fps, 1e-6)

        if self.tags is not None:
            self.rack.update(list(self.tags(image)), t)
        locked = self.rack.locked

        detections = list(self.detector(image)) if self.detector else []
        detections = self._assign_tracks(detections)
        hands = list(self.hands(image)) if self.hands else []
        body = self.pose(image) if self.pose else None
        contacts = self.smoother(self.contact(hands, detections, dt=dt))
        by_entity = self._bind(detections)

        if hasattr(self.motion, "push_frame"):
            target_det = detections[0] if detections else None
            contact_state = contacts[0] if contacts else None
            hand_state = hands[0] if hands else None
            self.motion.push_frame(
                hand=hand_state,
                contact=contact_state,
                target_det=target_det,
                dt=dt,
            )
            motion = self.motion()
        elif self.motion is not None:
            motion = self.motion(np.zeros(1))
        else:
            motion = MotionState("idle", 0.0)

        return BeliefFrame(
            t_mono=round(t, 3),
            t_utc=self._utc(t),
            frame_lock=locked,
            objects=self._objects(by_entity, locked, contacts),
            hands=self._hands(hands, contacts, by_entity, locked),
            body=self._body(body, locked),
            motion=MotionBelief(cls=motion.label, conf=float(motion.score)),
            occlusion=dict(occlusion or {}),
            contacts=self._object_contacts(by_entity, locked),
            confirmations=tuple(confirmations),
        )

    # ------------------------------------------------------------------
    def _utc(self, t: float) -> str:
        from datetime import timedelta
        return (self._t0_utc + timedelta(seconds=t)).isoformat().replace("+00:00", "Z")

    def _assign_tracks(self, detections: list[Detection]) -> list[Detection]:
        """Stand-in for ByteTrack: stable IDs keyed by class and instance.

        Real tracking maintains identity across occlusion, which is the whole
        reason ByteTrack is in the stack. This keeps IDs stable enough for the
        contact head's velocity features to be meaningful when the detector is
        scripted, and is replaced wholesale when the real tracker lands.
        """
        from dataclasses import replace
        out = []
        for det in detections:
            if det.track_id is not None:
                out.append(det)
                continue
            # By entity when the detector resolved it, so vial A and vial B do
            # not share one track id and hand each other's contact state.
            key = det.entity or det.cls
            if key not in self._track_of:
                self._track_of[key] = self._next_track_id
                self._next_track_id += 1
            out.append(replace(det, track_id=self._track_of[key]))
        return out

    def _bind(self, detections: list[Detection]) -> dict[str, Detection]:
        """Detector classes -> procedure entities.

        When several entities share a detector class - vial A and vial B are
        both `vial` - the instance key decides. Until the instance-key
        classifier exists, candidates are assigned in declaration order and the
        ambiguity is left visible rather than resolved by a coin flip, because
        a confident wrong binding is exactly what produces a WRONG_OBJECT alert
        against an innocent crew member.
        """
        out: dict[str, Detection] = {}
        used: set[int] = set()
        # A detector that resolved the instance itself is taken at its word, as
        # long as the class agrees with what the procedure declares. There is no
        # ambiguity left to be honest about.
        for det in detections:
            if det.entity is None or det.score < self.config.min_detection_score:
                continue
            binding = self.bindings.get(det.entity)
            if binding is not None and binding.detector_class == det.cls and det.entity not in out:
                out[det.entity] = det
                used.add(id(det))

        by_class: dict[str, list[Detection]] = {}
        for det in detections:
            if det.score < self.config.min_detection_score or det.entity is not None:
                continue
            by_class.setdefault(det.cls, []).append(det)

        for name, binding in self.bindings.items():
            if name in out:
                continue
            for det in by_class.get(binding.detector_class, []):
                if id(det) in used:
                    continue
                out[name] = det
                used.add(id(det))
                break
        return out

    def _objects(self, by_entity: dict[str, Detection], locked: bool,
                 contacts: Sequence[ContactState]) -> dict[str, ObjectBelief]:
        held_by: dict[int, str] = {
            c.track_id: c.hand for c in contacts
            if c.track_id is not None and c.label in ("grasp", "manipulate")
        }
        out: dict[str, ObjectBelief] = {}
        for name in self.bindings:
            det = by_entity.get(name)
            if det is None:
                # Not detected. NOT the same as absent - the occlusion mask
                # decides that, and it is the engine's job to tell them apart.
                out[name] = ObjectBelief(
                    visible=False, conf=0.0, pos_rack=None, quat_rack=None,
                    track_id=None, state=None, state_conf=0.0,
                    held_by=None, occluded=False)
                continue
            out[name] = ObjectBelief(
                visible=True,
                conf=float(det.score),
                pos_rack=det.pos_rack if locked else None,
                quat_rack=det.quat_rack if locked else None,
                track_id=det.track_id,
                state=det.state,
                state_conf=float(det.state_score) if det.state else 0.0,
                held_by=held_by.get(det.track_id),
                occluded=False,
            )
        return out

    def _hands(self, hands: Sequence[HandLandmarks], contacts: Sequence[ContactState],
               by_entity: dict[str, Detection], locked: bool) -> dict[str, HandBelief]:
        entity_of_track = {d.track_id: n for n, d in by_entity.items()}
        by_side = {c.hand: c for c in contacts}
        out: dict[str, HandBelief] = {}
        for side in ("left", "right"):
            hand = next((h for h in hands if h.side == side), None)
            if hand is None:
                out[side] = HandBelief(present=False, conf=0.0, wrist_rack=None,
                                       contact_with=None, grasp_type="none", grasp_conf=0.0)
                continue
            state = by_side.get(side)
            out[side] = HandBelief(
                present=True,
                conf=float(hand.score),
                wrist_rack=hand.wrist_rack if locked else None,
                contact_with=(entity_of_track.get(state.track_id)
                              if state and state.label != "none" else None),
                grasp_type=state.label if state else "none",
                grasp_conf=float(state.score) if state and state.label != "none" else 0.0,
            )
        return out

    def _body(self, body: BodyKeypoints | None, locked: bool) -> BodyBelief | None:
        if body is None or not locked:
            return None
        joints = body.rack_joints()
        if not joints:
            return None
        return BodyBelief(conf=float(body.score), joints_rack=joints)

    def _object_contacts(self, by_entity: dict[str, Detection],
                         locked: bool) -> dict[str, float]:
        """Object-object contact, for `contacting(a, b)`.

        Proximity in rack coordinates until a real contact head exists, and
        reported at the detector's own confidence rather than at 1.0 - the
        estimate is geometric, and dressing a geometric guess as a certainty is
        how a step verifies on evidence nobody has.
        """
        if not locked:
            return {}
        names = [n for n, d in by_entity.items() if d.pos_rack is not None]
        out: dict[str, float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pa = np.asarray(by_entity[a].pos_rack, dtype=float)
                pb = np.asarray(by_entity[b].pos_rack, dtype=float)
                if float(np.linalg.norm(pa - pb)) <= 0.05:
                    out[contact_key(a, b)] = min(by_entity[a].score, by_entity[b].score)
        return out
