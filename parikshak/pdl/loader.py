"""Load a PDL v1.0 procedure into typed objects the engine can execute.

The engine never touches raw YAML. It gets a `Procedure`, and by the time it
has one, the file has already been through the same validator the command line
runs - so an engine that holds a Procedure can assume every entity, zone, state
and motion class it references exists, every predicate is in the vocabulary, and
the graph is acyclic with reachable terminals.

That assumption is what keeps engine/ free of defensive `if entity in ...`
checks. Validation happens once, at load, loudly. See PLAN.md section 6.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from parikshak.pdl.validator import Report, check

Vec3 = tuple[float, float, float]

#: PDL step types (mirrors validator.VALID_STEP_TYPES).
STEP_TYPES = frozenset({"discrete", "motion", "wait", "crew_confirm"})


class ProcedureError(ValueError):
    """A procedure file could not be loaded, or did not pass validation."""


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Zone:
    """A named volume in RACK coordinates.

    Because zones are defined in the rack frame rather than the camera frame,
    rotating the rack, the camera or the crew member changes none of these
    numbers. Orientation-agnosticism lives here, not in a learned model.
    """

    name: str
    kind: str            # "box" | "sphere"
    lo: Vec3 | None = None
    hi: Vec3 | None = None
    centre_: Vec3 | None = None
    radius: float = 0.0

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> Zone:
        if d["type"] == "box":
            return cls(name=name, kind="box", lo=tuple(d["min"]), hi=tuple(d["max"]))
        return cls(name=name, kind="sphere", centre_=tuple(d["centre"]), radius=float(d["radius"]))

    @property
    def centre(self) -> Vec3:
        if self.kind == "box":
            assert self.lo is not None and self.hi is not None
            return tuple((a + b) / 2 for a, b in zip(self.lo, self.hi))  # type: ignore[return-value]
        assert self.centre_ is not None
        return self.centre_

    def contains(self, point: Vec3 | None, margin_m: float = 0.0) -> bool | None:
        """None when the point is unknown - occluded, or no rack-frame lock.

        Returning False for an unknown position would place every hidden object
        outside every zone, and `not in_zone(...)` would then read as a
        confident deviation. Unknown must stay unknown all the way up.
        """
        if point is None:
            return None
        if self.kind == "box":
            assert self.lo is not None and self.hi is not None
            return all(lo - margin_m <= p <= hi + margin_m
                       for p, lo, hi in zip(point, self.lo, self.hi))
        assert self.centre_ is not None
        d = math.dist(point, self.centre_)
        return d <= self.radius + margin_m


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Entity:
    """A thing the procedure talks about, bound to a detector class.

    `confusable_with` is what turns WRONG_OBJECT from a guess into a check: the
    author declares that vial A and vial B look alike, so grasping the wrong one
    is a named, expected failure with its own alert rather than an unexplained
    low-confidence reading.
    """

    name: str
    detector_class: str
    label: str = ""
    tts_name: str = ""
    states: tuple[str, ...] = ()
    confusable_with: tuple[str, ...] = ()
    extent_m: float | None = None
    instance_key: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> Entity:
        return cls(
            name=name,
            detector_class=d["detector_class"],
            label=d.get("label", ""),
            tts_name=d.get("tts_name", d.get("label", name)),
            states=tuple(d.get("states", ())),
            confusable_with=tuple(d.get("confusable_with", ())),
            extent_m=float(d["extent_m"]) if "extent_m" in d else None,
            instance_key=dict(d.get("instance_key", {})),
        )

    @property
    def spoken(self) -> str:
        """What Piper should say. Never the internal name - 'sample cartridge',
        not 'cartridge_sc_a'."""
        return self.tts_name or self.label or self.name


@dataclass(frozen=True, slots=True)
class Duration:
    """Duration priors. These are the HSMM's semi-Markov part: they are what let
    the engine conclude a step was skipped rather than merely slow, and what
    carries belief across a three-second occlusion."""

    min_s: float
    nominal_s: float
    sigma_s: float
    max_s: float

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Duration:
        d = d or {}
        nominal = float(d.get("nominal_s", 10.0))
        return cls(
            min_s=float(d.get("min_s", 0.0)),
            nominal_s=nominal,
            sigma_s=float(d.get("sigma_s", max(1.0, nominal / 3))),
            max_s=float(d.get("max_s", nominal * 6)),
        )


@dataclass(frozen=True, slots=True)
class Invariant:
    """A clause that must hold for the WHOLE step, not just at its end.

    This is the only mechanism that catches a hazard when no step was skipped
    and nothing was out of order - opening the latch while the unit runs. Pure
    sequence checking is structurally blind to it.
    """

    expr: dict[str, Any]
    severity: str = "critical"
    tts: str = ""


@dataclass(frozen=True, slots=True)
class Branch:
    """Conditional routing. Used to send a hardware fault to a fault procedure
    instead of blaming the crew, which is the fastest way to lose their trust."""

    when: dict[str, Any]
    goto: str
    severity: str = "info"
    tts: str = ""


@dataclass(frozen=True, slots=True)
class DeviationSpec:
    """What to say, and how loudly, when a particular deviation is detected."""

    severity: str = "advisory"
    tts: str = ""
    reason_template: str = ""
    #: For on_wrong_object: which entities count as the wrong one. Empty means
    #: fall back to the confusable_with declared on the step's own objects.
    confusable: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DeviationSpec | None:
        if not d:
            return None
        return cls(
            severity=d.get("severity", "advisory"),
            tts=d.get("tts", ""),
            reason_template=d.get("reason_template", ""),
            confusable=tuple(d.get("confusable", ())),
        )


@dataclass(frozen=True, slots=True)
class Step:
    id: str
    name: str
    type: str
    prompt_tts: str
    objects: tuple[str, ...]
    critical: bool
    preconditions: dict[str, Any] | None
    verification: dict[str, Any] | None
    effects: tuple[dict[str, Any], ...]
    invariants: tuple[Invariant, ...]
    branch: tuple[Branch, ...]
    duration: Duration
    on_skip: DeviationSpec | None
    on_wrong_object: DeviationSpec | None
    next: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_terminal(self) -> bool:
        return not self.next

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        return cls(
            id=d["id"],
            name=d.get("name", d["id"]),
            type=d.get("type", "discrete"),
            prompt_tts=d.get("prompt_tts", ""),
            objects=tuple(d.get("objects", ())),
            critical=bool(d.get("critical", False)),
            preconditions=d.get("preconditions"),
            verification=d.get("verification"),
            effects=tuple(d.get("effects", ())),
            invariants=tuple(
                Invariant(expr=i["expr"], severity=i.get("severity", "critical"),
                          tts=i.get("tts", ""))
                for i in d.get("invariants", ())
            ),
            branch=tuple(
                Branch(when=b["when"], goto=b["goto"], severity=b.get("severity", "info"),
                       tts=b.get("tts", ""))
                for b in d.get("branch", ())
            ),
            duration=Duration.from_dict(d.get("duration")),
            on_skip=DeviationSpec.from_dict(d.get("on_skip")),
            on_wrong_object=DeviationSpec.from_dict(d.get("on_wrong_object")),
            next=tuple(d.get("next", ())),
            raw=d,
        )


@dataclass(frozen=True, slots=True)
class Group:
    """A set of steps whose internal order does not matter.

    The single most important thing in the file for the false-alarm number: a
    system that alerts on a legal reorder gets its speaker taped over in week
    one of real operations. PLAN.md section 12 records three runs specifically
    to prove this stays silent.
    """

    id: str
    type: str
    members: tuple[str, ...]
    successor: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    """Thresholds live here and nowhere else.

    Never inline in a step: the W8 ROC sweep tunes these numbers, and a
    threshold buried in a step's verification tree is a number that does not
    move when the operating point does.
    """

    complete_threshold: float = 0.85
    deviation_threshold: float = 0.85
    persistence_s: float = 1.5
    unverified_after_s: float = 4.0
    #: Effective independent observations per second when accumulating evidence.
    #: See engine/evidence.py - it is what lets sustained moderate confidence
    #: verify a step while a single frame of it cannot.
    evidence_rate_hz: float = 2.0
    #: Share of a `hold_for` window allowed to disagree with the rest. See
    #: PredicateEvaluator._hold_for - a strict minimum turns a clause that is
    #: true on 93% of frames into one that holds a 7-frame window 60% of the time.
    hold_for_tolerance: float = 0.2
    max_alerts_per_step: int = 2
    cooldown_s: float = 8.0
    advisory_only: bool = True
    severity_tiers: dict[str, Any] = field(default_factory=dict)
    crew_override: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AlertPolicy:
        d = d or {}
        return cls(
            complete_threshold=float(d.get("complete_threshold", 0.85)),
            deviation_threshold=float(d.get("deviation_threshold", 0.85)),
            persistence_s=float(d.get("persistence_s", 1.5)),
            unverified_after_s=float(d.get("unverified_after_s", 4.0)),
            evidence_rate_hz=float(d.get("evidence_rate_hz", 2.0)),
            hold_for_tolerance=float(d.get("hold_for_tolerance", 0.2)),
            max_alerts_per_step=int(d.get("max_alerts_per_step", 2)),
            cooldown_s=float(d.get("cooldown_s", 8.0)),
            advisory_only=bool(d.get("advisory_only", True)),
            severity_tiers=dict(d.get("severity_tiers", {})),
            crew_override=dict(d.get("crew_override", {})),
        )

    @property
    def override_tokens(self) -> tuple[str, ...]:
        if not self.crew_override.get("enabled", False):
            return ()
        return tuple(self.crew_override.get("tokens", ()))


@dataclass(frozen=True, slots=True)
class FrameLossPolicy:
    """What to do when AprilTag lock is lost.

    `hold_last_known` for max_hold_s, then UNVERIFIED - never a guess from a
    stale extrinsic, which is how a rotated rack produces confident nonsense.
    """

    action: str = "hold_last_known"
    max_hold_s: float = 3.0
    then: str = "UNVERIFIED"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FrameLossPolicy:
        d = (d or {}).get("on_frame_loss", {}) if "on_frame_loss" in (d or {}) else (d or {})
        return cls(
            action=d.get("action", "hold_last_known"),
            max_hold_s=float(d.get("max_hold_s", 3.0)),
            then=d.get("then", "UNVERIFIED"),
        )


# --------------------------------------------------------------------------
# the procedure
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Procedure:
    id: str
    title: str
    revision: int
    pdl_version: str
    steps: dict[str, Step]
    order: tuple[str, ...]
    entry: str
    exits: tuple[str, ...]
    groups: tuple[Group, ...]
    zones: dict[str, Zone]
    entities: dict[str, Entity]
    motion_classes: tuple[str, ...]
    alert_policy: AlertPolicy
    frame_loss: FrameLossPolicy
    requires: dict[str, Any]
    logging: dict[str, Any]
    source_sha256: str
    path: Path | None = None
    warnings: tuple[str, ...] = ()
    #: Phrases crew_confirmed may wait for. The voice grammar is built from these.
    crew_tokens: tuple[str, ...] = ()

    # -- lookups ---------------------------------------------------------
    def step(self, step_id: str) -> Step:
        try:
            return self.steps[step_id]
        except KeyError:
            raise ProcedureError(f"no step {step_id!r} in procedure {self.id}") from None

    def entity(self, name: str) -> Entity:
        try:
            return self.entities[name]
        except KeyError:
            raise ProcedureError(f"no entity {name!r} in procedure {self.id}") from None

    def zone(self, name: str) -> Zone:
        try:
            return self.zones[name]
        except KeyError:
            raise ProcedureError(f"no zone {name!r} in procedure {self.id}") from None

    def group_of(self, step_id: str) -> Group | None:
        """The unordered group this step belongs to, if any. The engine consults
        this before calling anything OUT_OF_ORDER."""
        for g in self.groups:
            if step_id in g.members:
                return g
        return None

    def expand(self, target: str) -> tuple[str, ...]:
        """Resolve one `next`/`goto` target to the step IDs it can mean.

        A group is a node in the PDL graph, not a step: `next: [G_GATHER]` means
        "enter the group", and members of a group also declare `next: [G_GATHER]`
        meaning "stay in it until it is satisfied". Callers want steps, so a
        group resolves to its members plus its successor.
        """
        if target in self.steps:
            return (target,)
        for g in self.groups:
            if g.id == target:
                return g.members + self.expand(g.successor)
        raise ProcedureError(f"{self.id}: target {target!r} is neither a step nor a group")

    def successors(self, step_id: str) -> tuple[str, ...]:
        """Every STEP that may legally follow this one.

        Includes branch targets - a hardware fault routing to F01 is a legal
        transition, not a deviation - and, for a member of an unordered group,
        its siblings. Which siblings actually remain is runtime state, so the
        engine filters; the static graph must not forbid any of them, or a legal
        reorder becomes a false alarm.
        """
        step = self.step(step_id)
        out: list[str] = []
        for target in list(step.next) + [b.goto for b in step.branch]:
            for sid in self.expand(target):
                if sid != step_id and sid not in out:
                    out.append(sid)
        return tuple(out)

    @property
    def critical_steps(self) -> tuple[str, ...]:
        return tuple(s for s in self.order if self.steps[s].critical)

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True, build=None) -> Procedure:
        """Parse, validate, and build. Raises ProcedureError on any validation
        error; with strict=True, warnings are errors too.

        There is deliberately no way to skip validation. A procedure that has
        not passed the gate must never reach the engine, because every
        no-defensive-checks assumption in engine/ rests on it having passed.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ProcedureError(f"{path}: file not found") from None
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ProcedureError(f"{path}: YAML will not parse\n{exc}") from None
        if not isinstance(doc, dict):
            raise ProcedureError(f"{path}: expected a mapping at the top level")

        rep = Report()
        check(doc, rep)
        if build is not None:
            # The zero-retraining check. Optional here only because unit tests
            # load procedures without a manifest; tools/check.py always passes one.
            from parikshak.pdl.build import PerceptionBuild, check_build
            deployed = build if isinstance(build, PerceptionBuild) else PerceptionBuild.load(build)
            check_build(doc, deployed, rep)
        if rep.errors:
            joined = "\n  ".join(rep.errors)
            raise ProcedureError(f"{path}: {len(rep.errors)} validation error(s)\n  {joined}")
        if strict and rep.warnings:
            joined = "\n  ".join(rep.warnings)
            raise ProcedureError(
                f"{path}: {len(rep.warnings)} warning(s) under strict loading\n  {joined}")

        return cls.from_dict(doc, path=path, source=text, warnings=tuple(rep.warnings))

    @classmethod
    def from_dict(cls, doc: dict[str, Any], *, path: Path | None = None,
                  source: str | None = None, warnings: tuple[str, ...] = ()) -> Procedure:
        meta = doc["procedure"]
        flow = doc["flow"]
        steps = [Step.from_dict(s) for s in flow["steps"]]

        digest = hashlib.sha256(
            (source if source is not None else yaml.safe_dump(doc, sort_keys=True))
            .encode("utf-8")
        ).hexdigest()

        return cls(
            id=meta["id"],
            title=meta.get("title", meta["id"]),
            revision=int(meta.get("revision", 0)),
            pdl_version=str(doc.get("pdl_version", "1.0")),
            steps={s.id: s for s in steps},
            order=tuple(s.id for s in steps),
            entry=flow["entry"],
            exits=tuple(flow.get("exit", ())),
            groups=tuple(
                Group(id=g["id"], type=g.get("type", "unordered"),
                      members=tuple(g["members"]), successor=g["successor"],
                      note=g.get("note", ""))
                for g in flow.get("groups", ())
            ),
            zones={n: Zone.from_dict(n, z) for n, z in doc.get("zones", {}).items()},
            entities={n: Entity.from_dict(n, e) for n, e in doc.get("entities", {}).items()},
            motion_classes=tuple(doc.get("motion_classes", {}).get("classes", ())),
            alert_policy=AlertPolicy.from_dict(doc.get("alert_policy")),
            frame_loss=FrameLossPolicy.from_dict(doc.get("frame", {})),
            requires=dict(doc.get("requires", {})),
            logging=dict(doc.get("logging", {})),
            source_sha256=digest,
            path=path,
            warnings=warnings,
            crew_tokens=tuple(t for t in doc.get("crew_tokens") or () if isinstance(t, str)),
        )

    def __repr__(self) -> str:
        return (f"<Procedure {self.id} rev{self.revision} "
                f"{len(self.steps)} steps, {len(self.entities)} entities, "
                f"sha256:{self.source_sha256[:12]}>")


def load_procedure(path: str | Path, *, strict: bool = True, build=None) -> Procedure:
    """Module-level convenience: `load_procedure("procedures/csp1_....yaml")`."""
    return Procedure.load(path, strict=strict, build=build)
