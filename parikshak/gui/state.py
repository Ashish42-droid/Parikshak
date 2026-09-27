"""What the GUI shows, as data.

A view model with no Qt in it. The panels PLAN.md section 15 asks for - live
overlay, checklist, timeline, deviation feed, replay, export - are all views
over this structure, so what the crew sees can be asserted in a test instead of
verified by squinting at a screenshot.

It also means the display is identical whether frames came from a camera or from
a replayed trace, which is the stage failover: if a camera fails, the same
`DisplayState` is built from a file and the screen looks the same.

Presentation rules that are decisions, not styling:

  **Status is never colour alone.** Every status carries a symbol and a word.
  Colour-blind judges exist, venue projectors wash out, and a checklist whose
  meaning survives neither is not a checklist.

  **UNVERIFIED reads as its own thing, not as a failure.** It is the system
  declining to guess, and showing it in the same red as a skip would train the
  crew to read restraint as error.

  **The reason is on screen, not behind a hover.** "Deviation detected" is
  useless at a rack; "latch never closed between 43.0 s and 46.0 s" is what a
  crew member can act on and a reviewer can check.

  **"Cannot see" and "cannot tell with this setup" are different sentences.**
  A blocked camera is a moment; a step whose check needs a hand model the box
  does not have is permanent for the run. Saying "cannot verify" for both would
  make a camera-only run look broken rather than honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from parikshak.engine.hsmm import StepStatus

#: Symbol, ASCII twin, and word for every status, so meaning never rests on
#: colour - and never on a font either. The ASCII column is not cosmetic: a
#: Jetson serial console and a cp1252 Windows terminal both refuse U+2713, and
#: a display that raises UnicodeEncodeError mid-run is a display that is not
#: there when it is needed.
STATUS_GLYPH: dict[str, tuple[str, str, str]] = {
    "PENDING": ("·", ".", "pending"),
    "ACTIVE": ("▸", ">", "in progress"),
    "COMPLETE": ("✓", "+", "complete"),
    "SKIPPED": ("✗", "X", "not done"),
    "UNVERIFIED": ("?", "?", "cannot verify"),
}

#: Severity ordering for the deviation feed, worst first.
SEVERITY_RANK = {"critical": 0, "caution": 1, "advisory": 2, "info": 3}


@dataclass(frozen=True, slots=True)
class StepRow:
    """One line of the checklist."""

    step_id: str
    name: str
    status: str
    critical: bool
    is_active: bool
    t_start: float | None = None
    t_end: float | None = None
    reason: str = ""
    unknown_s: float = 0.0
    objects: tuple[str, ...] = ()
    #: Seconds undecidable while the camera could not see - see
    #: StepTracker.ever_unverified. Waiting for a window to fill is not blindness.
    unseen_s: float = 0.0
    #: Why this setup cannot verify the step at all, when it cannot - a missing
    #: capability, not a blocked view. Empty when it can.
    unverifiable: str = ""

    @property
    def glyph(self) -> str:
        return STATUS_GLYPH.get(self.status, ("·", ".", ""))[0]

    @property
    def ascii_glyph(self) -> str:
        return STATUS_GLYPH.get(self.status, ("·", ".", ""))[1]

    @property
    def status_word(self) -> str:
        return STATUS_GLYPH.get(self.status, ("", "", self.status.lower()))[2]

    @property
    def elapsed(self) -> float | None:
        if self.t_start is None:
            return None
        return (self.t_end or self.t_start) - self.t_start

    @property
    def was_ever_unverified(self) -> bool:
        """A step can finish COMPLETE and still have been blind for part of it.
        Worth showing: it is the difference between a clean run and a lucky one."""
        return self.unseen_s > 0.0 and self.status == "COMPLETE"


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    t: float
    kind: str                  # completed | skipped | unverified | hazard | branched | alert
    step_id: str
    label: str = ""
    severity: str = "info"

    @property
    def is_deviation(self) -> bool:
        return self.kind in ("skipped", "hazard", "alert")


@dataclass(frozen=True, slots=True)
class DeviationRow:
    """One entry in the feed. Carries its justification, not just its name."""

    t: float
    kind: str
    step_id: str
    severity: str
    reason: str
    spoken: str = ""
    confidence: float = 1.0
    acknowledged: bool = False

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 9)


@dataclass
class DisplayState:
    """Everything on screen, at one moment."""

    run_id: str = ""
    procedure_id: str = ""
    procedure_title: str = ""
    t: float = 0.0
    frame_lock: bool = False
    active_step: str | None = None
    prompt: str = ""
    steps: list[StepRow] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    deviations: list[DeviationRow] = field(default_factory=list)
    occluded_zones: tuple[str, ...] = ()
    unverified_notice: str = ""
    log_bytes: int = 0
    frames: int = 0
    source: str = "live"

    # -- headline counters ---------------------------------------------
    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed(self) -> int:
        return sum(1 for s in self.steps if s.status == "COMPLETE")

    @property
    def progress(self) -> float:
        return self.completed / self.total_steps if self.total_steps else 0.0

    @property
    def skipped(self) -> tuple[str, ...]:
        return tuple(s.step_id for s in self.steps if s.status == "SKIPPED")

    @property
    def unverified(self) -> tuple[str, ...]:
        return tuple(s.step_id for s in self.steps if s.status == "UNVERIFIED")

    @property
    def worst_severity(self) -> str | None:
        """Drives the status banner. None when there is nothing to say."""
        live = [d for d in self.deviations if not d.acknowledged]
        return min(live, key=lambda d: d.rank).severity if live else None

    @property
    def alerts_per_45min(self) -> float:
        if self.t <= 0:
            return 0.0
        return len(self.deviations) * 45 * 60 / self.t

    # -- panels ---------------------------------------------------------
    def feed(self, limit: int = 8) -> list[DeviationRow]:
        """The deviation feed: most severe first, then most recent.

        Not purely chronological. A critical hazard scrolling off the top
        because four advisories arrived after it is exactly the failure this
        panel exists to prevent.
        """
        return sorted(self.deviations, key=lambda d: (d.rank, -d.t))[:limit]

    def recent_timeline(self, window_s: float = 120.0) -> list[TimelineEvent]:
        return [e for e in self.timeline if e.t >= self.t - window_s]

    def active_row(self) -> StepRow | None:
        return next((s for s in self.steps if s.is_active), None)

    def status_line(self) -> str:
        """One line summarising the run, for the window title and the log."""
        bits = [f"{self.completed}/{self.total_steps} complete"]
        if self.skipped:
            bits.append(f"{len(self.skipped)} not done")
        if self.unverified:
            bits.append(f"{len(self.unverified)} unverified")
        if not self.frame_lock:
            bits.append("NO RACK LOCK")
        # ASCII separator: this string goes to the window title AND to a
        # console that may not be able to encode anything else.
        return "  |  ".join(bits)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_engine(cls, engine, *, frame=None, source: str = "live",
                    acknowledged: Sequence[str] = (),
                    unverifiable: Mapping[str, str] | None = None) -> DisplayState:
        """Build the whole display from the engine's current state.

        A pull rather than a push: the GUI asks what is true now instead of
        accumulating incremental updates, so a dropped update cannot leave the
        screen disagreeing with the log.

        `unverifiable` maps a step to why this setup cannot verify it at all -
        from `parikshak.live.capability_gaps`. Empty for replay and for a full
        perception build.
        """
        proc = engine.proc
        tracker = engine.tracker
        acked = set(acknowledged)
        gaps = dict(unverifiable or {})

        # The author's own wording, where they wrote one. The tracker's reason
        # is engine bookkeeping ("S09 satisfied while S08 verification stayed at
        # 0.00"); the procedure's reason_template is what a crew member can act
        # on and a reviewer can check. The checklist should show the latter.
        authored = {d.step_id: d.reason for d in engine.detector.deviations}

        steps = []
        for sid in proc.order:
            rec = tracker.records[sid]
            step = proc.step(sid)
            steps.append(StepRow(
                step_id=sid, name=step.name,
                status=rec.status.value if isinstance(rec.status, StepStatus)
                else str(rec.status),
                critical=step.critical,
                is_active=sid in tracker.frontier,
                t_start=rec.t_start, t_end=rec.t_end,
                reason=authored.get(sid, rec.reason), unknown_s=rec.unknown_s,
                objects=step.objects,
                unseen_s=getattr(rec, "unseen_s", 0.0),
                unverifiable=gaps.get(sid, ""),
            ))

        timeline = [
            TimelineEvent(tr.t, tr.kind, tr.step_id, tr.detail)
            for tr in tracker.transitions if tr.kind != "entered"
        ]
        timeline += [
            TimelineEvent(a.t, "alert", a.step_id, a.text, a.severity.label)
            for a in engine.policy.alerts
        ]
        timeline.sort(key=lambda e: e.t)

        spoken = {(a.step_id, a.kind.value): a.text for a in engine.policy.alerts}
        deviations = [
            DeviationRow(
                t=d.t, kind=d.kind.value, step_id=d.step_id, severity=d.severity,
                reason=d.reason, spoken=spoken.get((d.step_id, d.kind.value), ""),
                confidence=d.confidence,
                acknowledged=f"{d.step_id}:{d.kind.value}" in acked)
            for d in engine.detector.deviations
        ]

        occluded: tuple[str, ...] = ()
        if frame is not None:
            occluded = tuple(z for z, f in frame.occlusion.items() if f > 0.2)

        # Blindness first: the same measure the engine's own notice uses, so the
        # screen and the log never disagree about when the camera could not see.
        notice = ""
        blind = [s for s in steps if s.is_active and s.unseen_s >=
                 proc.alert_policy.unverified_after_s]
        if blind:
            where = f" ({', '.join(occluded)})" if occluded else ""
            notice = f"Cannot verify {blind[0].step_id}{where}"
        else:
            beyond = next((s for s in steps if s.is_active and s.unverifiable), None)
            if beyond is not None:
                notice = (f"{beyond.step_id} is not verifiable with this setup - "
                          f"{beyond.unverifiable}")

        return cls(
            run_id=engine.run_id,
            procedure_id=proc.id,
            procedure_title=proc.title,
            t=engine.t,
            frame_lock=bool(frame.frame_lock) if frame is not None else False,
            active_step=engine.active_step,
            prompt=engine.prompt,
            steps=steps, timeline=timeline, deviations=deviations,
            occluded_zones=occluded,
            unverified_notice=notice,
            log_bytes=engine.log.size_bytes if engine.log else 0,
            frames=engine.frames,
            source=source,
        )

    # -- export ---------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """For the GUI's export button, and for asserting on the display."""
        return {
            "run_id": self.run_id,
            "procedure": f"{self.procedure_id} - {self.procedure_title}",
            "t": round(self.t, 2),
            "source": self.source,
            "frame_lock": self.frame_lock,
            "status": self.status_line(),
            "active_step": self.active_step,
            "prompt": self.prompt,
            "completed": self.completed,
            "total_steps": self.total_steps,
            "skipped": list(self.skipped),
            "unverified": list(self.unverified),
            "unverifiable_here": {s.step_id: s.unverifiable for s in self.steps
                                  if s.unverifiable},
            "deviations": [
                {"t": round(d.t, 2), "kind": d.kind, "step_id": d.step_id,
                 "severity": d.severity, "reason": d.reason}
                for d in self.deviations
            ],
            "log_bytes": self.log_bytes,
        }
