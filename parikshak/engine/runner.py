"""ProcedureEngine - the loop, composed.

    BeliefFrame -> predicates -> step tracker -> deviations -> alerts -> log

One frame in, a decision and its justification out. Nothing here knows whether
the frame came from a camera or a file, which is the entire reason a trace can
be replayed through the identical engine: on stage, if a camera fails, the
replay path produces the same GUI from the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parikshak.belief.frame import BeliefFrame
from parikshak.engine.alerts import Alert, AlertPolicyEngine, UnverifiedNotice
from parikshak.engine.deviations import Deviation, DeviationDetector
from parikshak.engine.hsmm import StepStatus, StepTracker
from parikshak.engine.logger import RunLogger
from parikshak.pdl.loader import Procedure


@dataclass
class FrameResult:
    """What this frame produced. Everything the GUI and TTS need, nothing more."""

    t: float
    active_step: str | None
    prompt: str
    transitions: list[Any] = field(default_factory=list)
    deviations: list[Deviation] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    notices: list[UnverifiedNotice] = field(default_factory=list)


@dataclass
class RunSummary:
    run_id: str
    procedure_id: str
    duration_s: float
    frames: int
    status: dict[str, str]
    complete: tuple[str, ...]
    skipped: tuple[str, ...]
    unverified: tuple[str, ...]
    deviations: tuple[Deviation, ...]
    alerts: tuple[Alert, ...]
    notices: tuple[UnverifiedNotice, ...]
    log_path: Path | None = None
    #: When each step became active, None if never. The eval harness needs it to
    #: tell an alarm the injected error could have caused from one it could not.
    started_at: dict[str, float | None] = field(default_factory=dict)

    @property
    def alerts_per_45min(self) -> float:
        """The adoption metric, extrapolated from this run.

        Reported per 45 minutes because that is the unit PLAN.md section 2 sets
        the target in. Extrapolating from a two-minute trace is not a claim
        about the fielded system - it is a comparable number across runs.
        """
        if self.duration_s <= 0:
            return 0.0
        return len(self.alerts) * (45 * 60) / self.duration_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "procedure_id": self.procedure_id,
            "duration_s": round(self.duration_s, 2),
            "frames": self.frames,
            "complete": len(self.complete),
            "skipped": list(self.skipped),
            "unverified": list(self.unverified),
            "deviations": len(self.deviations),
            "alerts": len(self.alerts),
            "alerts_per_45min": round(self.alerts_per_45min, 2),
        }


class ProcedureEngine:
    """One run. Feed it frames; ask it for a summary at the end."""

    def __init__(self, procedure: Procedure, *, run_id: str = "run",
                 log_path: str | Path | None = None,
                 model_versions: dict[str, str] | None = None,
                 belief_schema: str = "1.1") -> None:
        self.proc = procedure
        self.run_id = run_id
        self.tracker = StepTracker(procedure)
        self.detector = DeviationDetector(procedure, self.tracker)
        self.policy = AlertPolicyEngine(procedure)
        self.frames = 0
        self.t = 0.0

        self.log: RunLogger | None = None
        if log_path is not None:
            self.log = RunLogger(
                path=Path(log_path),
                procedure_id=procedure.id,
                procedure_sha256=procedure.source_sha256,
                run_id=run_id,
                belief_schema=belief_schema,
                operating_point=self.policy.false_alarm_budget,
                model_versions=model_versions or {},
                mirror_text=bool(procedure.logging.get("mirror_text", True)),
            )

    # ------------------------------------------------------------------
    @property
    def active_step(self) -> str | None:
        frontier = self.tracker.frontier
        return frontier[0] if frontier else None

    @property
    def prompt(self) -> str:
        """What to say next. The prompt is the product as much as the alerts
        are - a witness that only speaks when you get it wrong is a nag."""
        sid = self.active_step
        return self.proc.step(sid).prompt_tts if sid else ""

    # ------------------------------------------------------------------
    def step(self, frame: BeliefFrame) -> FrameResult:
        self.frames += 1
        self.t = frame.t_mono

        transitions = self.tracker.update(frame)
        deviations = self.detector.update(frame, transitions)

        active = self.active_step
        if frame.confirmations and active:
            self.policy.observe_overrides(frame.t_mono, frame.confirmations, active)
            if self.log:
                for token in frame.confirmations:
                    if token.lower() in {t.lower() for t in self.policy.policy.override_tokens}:
                        self.log.override(frame.t_mono, active, token)

        # A step that recovered inside the persistence window retracts its own
        # deviation - the crew went back and fixed it, and nothing is said.
        recovered = frozenset(
            tr.step_id for tr in transitions
            if tr.kind == "completed" and self.tracker.records[tr.step_id].was_passed
        )
        alerts = self.policy.update(frame.t_mono, deviations, retracted=recovered)

        notices: list[UnverifiedNotice] = []
        for sid in self.tracker.frontier:
            rec = self.tracker.records[sid]
            # Time undecidable while blind, not time waiting for a window to
            # fill - see StepTracker.ever_unverified.
            notice = self.policy.unverified(
                frame.t_mono, sid, rec.unseen_s,
                tuple(z for z, f in frame.occlusion.items() if f > 0.5))
            if notice is not None:
                notices.append(notice)

        self._log(transitions, deviations, alerts, notices)
        return FrameResult(t=frame.t_mono, active_step=active, prompt=self.prompt,
                           transitions=transitions, deviations=deviations,
                           alerts=alerts, notices=notices)

    # ------------------------------------------------------------------
    def finish(self) -> RunSummary:
        transitions = self.tracker.finalise()
        deviations = self.detector.finalise(transitions)
        alerts = self.policy.update(self.t, deviations)
        alerts += self.policy.flush(self.t)
        self._log(transitions, deviations, alerts, [])

        summary = RunSummary(
            run_id=self.run_id,
            procedure_id=self.proc.id,
            duration_s=self.t,
            frames=self.frames,
            status=self.tracker.summary(),
            complete=self.tracker.steps_with(StepStatus.COMPLETE),
            skipped=self.tracker.steps_with(StepStatus.SKIPPED),
            unverified=self.tracker.unverified_steps,
            deviations=tuple(self.detector.deviations),
            alerts=tuple(self.policy.alerts),
            notices=tuple(self.policy.notices),
            started_at={sid: rec.t_start for sid, rec in self.tracker.records.items()},
        )
        if self.log:
            self.log.close(self.t, summary.as_dict())
            summary.log_path = self.log.path
        return summary

    # ------------------------------------------------------------------
    def _log(self, transitions, deviations, alerts, notices) -> None:
        if self.log is None:
            return
        for tr in transitions:
            if tr.kind == "entered":
                continue  # the prompt is the interesting event, not the bookkeeping
            self.log.step_event(tr.t, tr.step_id, tr.kind.upper(), tr.detail, tr.confidence)
        for dev in deviations:
            self.log.deviation(dev)
        for alert in alerts:
            self.log.alert(alert)
        for notice in notices:
            self.log.notice(notice)
