"""BeliefFrame - the perception -> engine contract. Schema v1.1.

Perception writes it. The engine reads it. Neither imports the other.

The consequence is the entire parallelism plan: the engine can be built,
tested and tuned against hand-written JSONL traces with no camera, no GPU and
no trained model. That only holds while this file is stable, so:

    Adding a field       -> bump BELIEF_SCHEMA_VERSION minor, regenerate traces.
    Removing or renaming -> bump major. Every golden trace becomes invalid.

v1.1 added `quat_rack` and frame-level `contacts`, because CSP-1 step S07
verifies with `aligned` and `contacting` and v1.0 could not express either -
that step would have sat permanently UNVERIFIED. Found in week 1 by running the
procedure against the contract, which is the entire point of doing it in this
order.

Do not "just add one field" in week 9. A 60-trace regression corpus is
downstream of this dataclass.

Units and frame: metres, seconds. All coordinates are in the RACK frame,
recovered per-frame from AprilTag 36h11 fiducials. Gravity is never referenced.
See schema/PREDICATES.md section 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BELIEF_SCHEMA_VERSION = "1.1"

Vec3 = tuple[float, float, float]

#: Orientation in the rack frame as (w, x, y, z). None whenever the deployed
#: build does not estimate 6-DoF pose for that class, which is most of them -
#: `aligned` is the only predicate that needs it and it is expensive to supply.
Quat = tuple[float, float, float, float]

#: Contact-MLP output vocabulary (PLAN.md section 4).
GRASP_TYPES = frozenset({"none", "reach", "grasp", "manipulate", "release"})

#: Hands are named, not indexed. "any"/"both" are predicate-level, not frame-level.
HAND_SIDES = ("left", "right")

#: Separator for object-object contact keys. Contact is symmetric, so the pair
#: is always stored with the two names sorted - one spelling, not two.
CONTACT_SEP = "|"


def contact_key(a: str, b: str) -> str:
    """Canonical key for the symmetric contact relation between two entities."""
    lo, hi = sorted((a, b))
    return f"{lo}{CONTACT_SEP}{hi}"


class TraceFormatError(ValueError):
    """A trace line does not conform to the BeliefFrame schema."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _vec3(value: Any, where: str) -> Vec3 | None:
    """JSON gives lists; the contract is a 3-tuple of floats, or None when the
    quantity is genuinely unknown (occluded, or no rack-frame lock)."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TraceFormatError(f"{where}: expected [x, y, z] or null, got {value!r}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise TraceFormatError(f"{where}: non-numeric coordinate in {value!r}") from exc


def _quat(value: Any, where: str) -> Quat | None:
    """(w, x, y, z), or None when the build does not estimate orientation here.

    Not normalised on read: a de-normalised quaternion means the pose estimator
    is misbehaving, and silently fixing it up would hide that.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TraceFormatError(f"{where}: expected [w, x, y, z] or null, got {value!r}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError) as exc:
        raise TraceFormatError(f"{where}: non-numeric component in {value!r}") from exc


def _prob(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TraceFormatError(f"{where}: confidence must be a number, got {value!r}")
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise TraceFormatError(f"{where}: confidence {v} outside [0, 1]")
    return v


def _require(d: Any, keys: tuple[str, ...], where: str) -> None:
    """The schema is closed in both directions: a missing field is an error and
    so is an unexpected one. A typo'd key that silently rides along in a trace
    is a bug that surfaces six weeks later as an unexplained accuracy drop."""
    if not isinstance(d, dict):
        raise TraceFormatError(f"{where}: expected an object, got {type(d).__name__}")
    missing = [k for k in keys if k not in d]
    if missing:
        raise TraceFormatError(f"{where}: missing field(s) {missing}")
    extra = [k for k in d if k not in keys]
    if extra:
        raise TraceFormatError(
            f"{where}: unknown field(s) {extra} - schema is closed at v{BELIEF_SCHEMA_VERSION}"
        )


def _as_map(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceFormatError(f"{where}: expected an object, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ObjectBelief:
    """What perception believes about one declared entity, this frame.

    visible=False, occluded=True  -> we know it is hidden. Predicates return null.
    visible=False, occluded=False -> we believe it is genuinely absent.

    That distinction is load-bearing: `absent` must not fire because a torso was
    in the way. See PREDICATES.md section 3.1.
    """

    visible: bool
    conf: float
    pos_rack: Vec3 | None
    quat_rack: Quat | None
    track_id: int | None
    state: str | None
    state_conf: float
    held_by: str | None
    occluded: bool

    FIELDS = ("visible", "conf", "pos_rack", "quat_rack", "track_id", "state",
              "state_conf", "held_by", "occluded")

    def to_json(self) -> dict[str, Any]:
        return {
            "visible": self.visible,
            "conf": self.conf,
            "pos_rack": list(self.pos_rack) if self.pos_rack is not None else None,
            "quat_rack": list(self.quat_rack) if self.quat_rack is not None else None,
            "track_id": self.track_id,
            "state": self.state,
            "state_conf": self.state_conf,
            "held_by": self.held_by,
            "occluded": self.occluded,
        }

    @classmethod
    def from_json(cls, d: Any, where: str) -> ObjectBelief:
        _require(d, cls.FIELDS, where)
        held = d["held_by"]
        if held is not None and held not in HAND_SIDES:
            raise TraceFormatError(
                f"{where}.held_by: expected one of {HAND_SIDES} or null, got {held!r}")
        state = d["state"]
        if state is not None and not isinstance(state, str):
            raise TraceFormatError(
                f"{where}.state: expected a string or null, got {state!r} "
                f"({type(state).__name__}) - bare off/on/yes/no in YAML parse as booleans")
        return cls(
            visible=bool(d["visible"]),
            conf=_prob(d["conf"], f"{where}.conf"),
            pos_rack=_vec3(d["pos_rack"], f"{where}.pos_rack"),
            quat_rack=_quat(d["quat_rack"], f"{where}.quat_rack"),
            track_id=None if d["track_id"] is None else int(d["track_id"]),
            state=state,
            state_conf=_prob(d["state_conf"], f"{where}.state_conf"),
            held_by=held,
            occluded=bool(d["occluded"]),
        )


@dataclass(frozen=True, slots=True)
class HandBelief:
    """MediaPipe landmarks reduced to what the predicate vocabulary can ask about.

    The 21 raw keypoints are deliberately NOT in the contract. The engine has no
    business reasoning about knuckles; it asks `grasped(entity)`. Keeping the
    keypoints inside perception is what lets the hand model be swapped without
    touching a single procedure file.
    """

    present: bool
    conf: float
    wrist_rack: Vec3 | None
    contact_with: str | None
    grasp_type: str
    grasp_conf: float

    FIELDS = ("present", "conf", "wrist_rack", "contact_with", "grasp_type", "grasp_conf")

    def to_json(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "conf": self.conf,
            "wrist_rack": list(self.wrist_rack) if self.wrist_rack is not None else None,
            "contact_with": self.contact_with,
            "grasp_type": self.grasp_type,
            "grasp_conf": self.grasp_conf,
        }

    @classmethod
    def from_json(cls, d: Any, where: str) -> HandBelief:
        _require(d, cls.FIELDS, where)
        grasp = d["grasp_type"]
        if grasp not in GRASP_TYPES:
            raise TraceFormatError(
                f"{where}.grasp_type: {grasp!r} not in contact-head vocabulary "
                f"{sorted(GRASP_TYPES)}")
        return cls(
            present=bool(d["present"]),
            conf=_prob(d["conf"], f"{where}.conf"),
            wrist_rack=_vec3(d["wrist_rack"], f"{where}.wrist_rack"),
            contact_with=d["contact_with"],
            grasp_type=grasp,
            grasp_conf=_prob(d["grasp_conf"], f"{where}.grasp_conf"),
        )


@dataclass(frozen=True, slots=True)
class BodyBelief:
    """Crew joints in rack coordinates.

    Sparse by design - only the joints the posture predicates need
    (`body_restrained` wants ankles, `body_in_zone` wants the pelvis root). A
    full 26-kpt Halpe set is welcome but not required, which is what makes
    descope ladder item 3 a config change rather than a rewrite.
    """

    conf: float
    joints_rack: dict[str, Vec3]

    FIELDS = ("conf", "joints_rack")

    def to_json(self) -> dict[str, Any]:
        return {
            "conf": self.conf,
            "joints_rack": {k: list(v) for k, v in self.joints_rack.items()},
        }

    @classmethod
    def from_json(cls, d: Any, where: str) -> BodyBelief:
        _require(d, cls.FIELDS, where)
        joints = _as_map(d["joints_rack"], f"{where}.joints_rack")
        out: dict[str, Vec3] = {}
        for name, xyz in joints.items():
            v = _vec3(xyz, f"{where}.joints_rack.{name}")
            if v is None:
                raise TraceFormatError(
                    f"{where}.joints_rack.{name}: a listed joint may not be null - "
                    f"omit the joint instead")
            out[name] = v
        return cls(conf=_prob(d["conf"], f"{where}.conf"), joints_rack=out)


@dataclass(frozen=True, slots=True)
class MotionBelief:
    """Active class of the 1D-TCN over its trailing window.

    `cls` is free-form here and validated against the procedure's declared motion
    classes by the PDL layer, not by this contract. The frame format must not
    need editing to add a motion class.
    """

    cls: str
    conf: float

    FIELDS = ("cls", "conf")

    def to_json(self) -> dict[str, Any]:
        return {"cls": self.cls, "conf": self.conf}

    @classmethod
    def from_json(cls, d: Any, where: str) -> MotionBelief:
        _require(d, cls.FIELDS, where)
        if not isinstance(d["cls"], str):
            raise TraceFormatError(f"{where}.cls: expected a string, got {d['cls']!r}")
        return cls(cls=d["cls"], conf=_prob(d["conf"], f"{where}.conf"))


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BeliefFrame:
    """One frame of world-state belief, in rack coordinates.

    frame_lock=False means AprilTag PnP failed this frame: the rack extrinsics
    are unknown, so every geometric quantity is stale. The engine must treat
    geometry predicates as null, NOT evaluate them against the last good
    extrinsic. Guessing from a stale transform is how a rotated rack silently
    produces confident nonsense.
    """

    t_mono: float
    t_utc: str
    frame_lock: bool
    objects: dict[str, ObjectBelief]
    hands: dict[str, HandBelief]
    body: BodyBelief | None
    motion: MotionBelief
    occlusion: dict[str, float]
    contacts: dict[str, float]
    confirmations: tuple[str, ...] = ()

    FIELDS = ("t_mono", "t_utc", "frame_lock", "objects", "hands", "body",
              "motion", "occlusion", "contacts", "confirmations")

    def to_json(self) -> dict[str, Any]:
        """Field order here IS the on-disk field order. Keep it stable so trace
        diffs stay readable and byte comparison remains a usable regression check."""
        return {
            "t_mono": self.t_mono,
            "t_utc": self.t_utc,
            "frame_lock": self.frame_lock,
            "objects": {k: v.to_json() for k, v in self.objects.items()},
            "hands": {k: v.to_json() for k, v in self.hands.items()},
            "body": self.body.to_json() if self.body is not None else None,
            "motion": self.motion.to_json(),
            "occlusion": dict(self.occlusion),
            "contacts": dict(self.contacts),
            "confirmations": list(self.confirmations),
        }

    @classmethod
    def from_json(cls, d: Any, where: str = "frame") -> BeliefFrame:
        _require(d, cls.FIELDS, where)
        if not isinstance(d["t_mono"], (int, float)) or isinstance(d["t_mono"], bool):
            raise TraceFormatError(
                f"{where}.t_mono: expected seconds as a number, got {d['t_mono']!r}")
        if not isinstance(d["t_utc"], str):
            raise TraceFormatError(
                f"{where}.t_utc: expected an ISO-8601 string, got {d['t_utc']!r}")

        objects = {
            name: ObjectBelief.from_json(v, f"{where}.objects.{name}")
            for name, v in _as_map(d["objects"], f"{where}.objects").items()
        }
        hands = {
            side: HandBelief.from_json(v, f"{where}.hands.{side}")
            for side, v in _as_map(d["hands"], f"{where}.hands").items()
        }
        for side in hands:
            if side not in HAND_SIDES:
                raise TraceFormatError(
                    f"{where}.hands: unknown side {side!r}, expected {HAND_SIDES}")

        occlusion = {
            zone: _prob(frac, f"{where}.occlusion.{zone}")
            for zone, frac in _as_map(d["occlusion"], f"{where}.occlusion").items()
        }

        contacts: dict[str, float] = {}
        for pair, p in _as_map(d["contacts"], f"{where}.contacts").items():
            if pair.count(CONTACT_SEP) != 1:
                raise TraceFormatError(
                    f"{where}.contacts: key {pair!r} must be 'a{CONTACT_SEP}b' with the two "
                    f"entity names in sorted order")
            a, b = pair.split(CONTACT_SEP)
            if [a, b] != sorted([a, b]):
                raise TraceFormatError(
                    f"{where}.contacts: key {pair!r} is not in sorted order - contact is "
                    f"symmetric, so it must have exactly one spelling")
            contacts[pair] = _prob(p, f"{where}.contacts.{pair}")

        confirmations = d["confirmations"]
        if not isinstance(confirmations, (list, tuple)):
            raise TraceFormatError(f"{where}.confirmations: expected a list of tokens")

        return cls(
            t_mono=float(d["t_mono"]),
            t_utc=d["t_utc"],
            frame_lock=bool(d["frame_lock"]),
            objects=objects,
            hands=hands,
            body=None if d["body"] is None else BodyBelief.from_json(d["body"], f"{where}.body"),
            motion=MotionBelief.from_json(d["motion"], f"{where}.motion"),
            occlusion=occlusion,
            contacts=contacts,
            confirmations=tuple(str(c) for c in confirmations),
        )

    # -- convenience for the engine -------------------------------------
    def object(self, name: str) -> ObjectBelief | None:
        return self.objects.get(name)

    def zone_occlusion(self, zone: str) -> float:
        """0.0 when unlisted. An absent key means "not occluded", not "unknown"."""
        return self.occlusion.get(zone, 0.0)

    def contact(self, a: str, b: str) -> float:
        """Asserted object-object contact probability. 0.0 when unlisted.

        Whether an absent pair means "not touching" or "we could not tell" is
        decided by the caller from the two objects' occlusion, not here - the
        frame reports what perception asserted, nothing more.
        """
        return self.contacts.get(contact_key(a, b), 0.0)


def empty_frame(t_mono: float = 0.0, t_utc: str = "1970-01-01T00:00:00Z") -> BeliefFrame:
    """A frame in which perception knows nothing.

    The starting belief, and the fixture for tests that need a valid frame
    without caring about its contents. frame_lock is False: before AprilTag
    acquisition we genuinely do not have a rack frame.
    """
    return BeliefFrame(
        t_mono=t_mono, t_utc=t_utc, frame_lock=False,
        objects={}, hands={}, body=None,
        motion=MotionBelief(cls="idle", conf=0.0),
        occlusion={}, contacts={}, confirmations=(),
    )
