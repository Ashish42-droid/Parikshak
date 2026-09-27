"""The deployed perception build, and whether a procedure fits inside it.

This is where "a new experiment onboards with zero retraining" stops being a
slogan and becomes a check.

Before this file existed, a procedure's `requires.detector_classes` was checked
only against the procedure's own entity list - which is self-referential. A
procedure could declare a class the deployed detector has never seen, list it
under `requires`, and pass. The build manifest is the external thing it is
checked against: the classes, states, motion classes and capabilities one
specific set of trained weights actually provides.

What can and cannot change without retraining, stated once:

    free (procedure edit)          costs a data campaign (model release)
    ---------------------------    --------------------------------------
    steps, order, groups           a new detector class
    predicates over known things   a new state for an existing class
    zones, durations, thresholds   a new motion class
    alert wording, crew tokens     a new capability (e.g. 6-DoF pose)

Knowing which column a change is in is the whole skill of authoring a procedure.
`tools/onboard.py` prints the verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Where build manifests live. Procedures refer to one by name through
#: `requires.min_model_version`; the file is `<name>.json` in this directory.
BUILDS_DIR = Path(__file__).resolve().parents[2] / "builds"


class BuildError(ValueError):
    """A build manifest is missing or malformed."""


@dataclass(frozen=True, slots=True)
class PerceptionBuild:
    name: str
    detector_classes: frozenset[str]
    capabilities: frozenset[str]
    motion_registry: str
    motion_classes: frozenset[str]
    #: detector class -> the states its state classifier emits
    state_heads: dict[str, tuple[str, ...]] = field(default_factory=dict)
    position_sigma_m: float | None = None
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> PerceptionBuild:
        p = Path(path)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise BuildError(f"no build manifest at {p}") from None
        except json.JSONDecodeError as exc:
            raise BuildError(f"{p}: not valid JSON - {exc.msg}") from None
        try:
            motion = d["motion_registry"]
            return cls(
                name=d["build"],
                detector_classes=frozenset(d["detector_classes"]),
                capabilities=frozenset(d["capabilities"]),
                motion_registry=motion["version"],
                motion_classes=frozenset(motion["classes"]),
                state_heads={k: tuple(v) for k, v in (d.get("state_heads") or {}).items()},
                position_sigma_m=d.get("position_sigma_m"),
                path=p,
            )
        except KeyError as exc:
            raise BuildError(f"{p}: build manifest is missing {exc}") from None

    @classmethod
    def named(cls, name: str, search: Path = BUILDS_DIR) -> PerceptionBuild:
        """The build a procedure names in `requires.min_model_version`."""
        return cls.load(search / f"{name}.json")


def _err(rep: Any, where: str, msg: str) -> None:
    rep.err(where, msg)


def check_build(doc: dict, build: PerceptionBuild, rep: Any) -> None:
    """Report every requirement of `doc` that `build` does not satisfy.

    Every message names what would have to happen instead, because "invalid" is
    useless to a payload engineer and "this needs a new detection dataset" tells
    them it is not an afternoon's work.
    """
    requires = doc.get("requires") or {}

    target = requires.get("min_model_version")
    if target and target != build.name:
        _err(rep, "requires.min_model_version",
             f"procedure targets '{target}' but the deployed build is '{build.name}' - "
             f"validate against the build that will actually run it")

    for cls_name in requires.get("detector_classes") or []:
        if cls_name not in build.detector_classes:
            _err(rep, "requires.detector_classes",
                 f"'{cls_name}' is not in {build.name}'s class map - onboarding this "
                 f"procedure needs a NEW detection dataset and a retrained detector, "
                 f"not a procedure edit")

    for cap in requires.get("capabilities") or []:
        if cap not in build.capabilities:
            _err(rep, "requires.capabilities",
                 f"'{cap}' is not provided by {build.name} - that is a new perception "
                 f"component, not a procedure edit")

    motion = doc.get("motion_classes") or {}
    registry = motion.get("registry_version")
    if registry and registry != build.motion_registry:
        _err(rep, "motion_classes.registry_version",
             f"procedure expects TCN registry '{registry}', {build.name} ships "
             f"'{build.motion_registry}'")
    for m in motion.get("classes") or []:
        if isinstance(m, str) and m not in build.motion_classes:
            _err(rep, "motion_classes.classes",
                 f"motion class '{m}' is not emitted by {build.name}'s TCN "
                 f"({build.motion_registry}) - needs labelled motion data and a "
                 f"retrained TCN")

    for name, entity in (doc.get("entities") or {}).items():
        states = entity.get("states")
        if not states:
            continue
        cls_name = entity.get("detector_class")
        head = build.state_heads.get(cls_name)
        if head is None:
            _err(rep, f"entities.{name}.states",
                 f"declares states, but {build.name} has no state classifier for "
                 f"'{cls_name}' - a state head must be trained before state_is can "
                 f"ask about it")
            continue
        for st in states:
            if isinstance(st, str) and st not in head:
                _err(rep, f"entities.{name}.states",
                     f"state '{st}' is not emitted by {build.name}'s '{cls_name}' state "
                     f"head (it emits {list(head)}) - that is a retrained state head")


# --------------------------------------------------------------------------
@dataclass
class OnboardingReport:
    """The verdict on one procedure against one build."""

    procedure_id: str
    title: str
    build: str
    errors: list[str] = field(default_factory=list)
    rows: list[tuple[str, str, bool]] = field(default_factory=list)
    crew_tokens: tuple[str, ...] = ()

    @property
    def retraining_required(self) -> bool:
        return bool(self.errors)


class _Collector:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")


def onboard(doc: dict, build: PerceptionBuild) -> OnboardingReport:
    """Everything a procedure needs from the build, and whether it is there."""
    rep = _Collector()
    check_build(doc, build, rep)

    meta = doc.get("procedure") or {}
    requires = doc.get("requires") or {}
    classes = list(requires.get("detector_classes") or [])
    caps = list(requires.get("capabilities") or [])
    motions = [m for m in (doc.get("motion_classes") or {}).get("classes") or []
               if isinstance(m, str)]
    states = {
        f"{e.get('detector_class')}": [s for s in e.get("states") or [] if isinstance(s, str)]
        for e in (doc.get("entities") or {}).values() if e.get("states")
    }

    def row(label: str, needed: list[str], have: frozenset[str] | set[str]) -> None:
        missing = [x for x in needed if x not in have]
        detail = f"{len(needed)} needed, {len(needed) - len(missing)} in build"
        if missing:
            detail += f" - missing {missing}"
        rep_row.append((label, detail, not missing))

    rep_row: list[tuple[str, str, bool]] = []
    row("detector classes", classes, build.detector_classes)
    row("capabilities", caps, build.capabilities)
    row("motion classes", motions, build.motion_classes)
    state_missing = []
    for cls_name, sts in states.items():
        head = set(build.state_heads.get(cls_name, ()))
        state_missing += [f"{cls_name}.{s}" for s in sts if s not in head]
    total_states = sum(len(v) for v in states.values())
    rep_row.append((
        "state vocabulary",
        f"{total_states} needed across {len(states)} state head(s)"
        + (f" - missing {state_missing}" if state_missing else ""),
        not state_missing,
    ))

    return OnboardingReport(
        procedure_id=str(meta.get("id", "?")),
        title=str(meta.get("title", "")),
        build=build.name,
        errors=rep.errors,
        rows=rep_row,
        crew_tokens=tuple(t for t in doc.get("crew_tokens") or [] if isinstance(t, str)),
    )
