"""The deviation taxonomy: SKIP, OUT_OF_ORDER, WRONG_OBJECT, REPEAT, DURATION, HAZARD.

Categories are borrowed from CaptainCook4D, IndustReal and EgoPER rather than
invented, so the eval section can cite prior art instead of defending a bespoke
ontology (PLAN.md section 11).

A deviation here is a raw detection with a justification attached. It is not yet
an alert: gating, persistence, escalation and cooldown all live in alerts.py.
Keeping them apart is what makes the W8 ROC sweep possible - the detector's
output is fixed, and the operating point moves over it.

Every Deviation carries a `reason` that names the predicate and the times
involved. That is not for debugging. It is the thing the crew hears, the thing
the ground PI reads six weeks later, and the reason the system is certifiable as
advisory software rather than an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from parikshak.belief.frame import BeliefFrame
from parikshak.engine.hsmm import HOLDING_STATUSES, StepStatus, StepTracker, Transition
from parikshak.engine.evidence import sigmoid
from parikshak.engine.truth import FALSE, UNKNOWN, Truth
from parikshak.pdl.loader import DeviationSpec, Procedure


class DeviationKind(str, Enum):
    SKIP = "SKIP"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    WRONG_OBJECT = "WRONG_OBJECT"
    REPEAT = "REPEAT"
    DURATION = "DURATION"
    HAZARD = "HAZARD"


@dataclass(frozen=True, slots=True)
class Deviation:
    t: float
    kind: DeviationKind
    step_id: str
    severity: str
    confidence: float
    reason: str
    tts: str = ""
    entities: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"[{self.t:7.1f}s] {self.kind.value:<13} {self.step_id:<10} {self.reason}"


DEFAULT_SEVERITY = {
    DeviationKind.SKIP: "caution",
    DeviationKind.OUT_OF_ORDER: "caution",
    DeviationKind.WRONG_OBJECT: "caution",
    DeviationKind.REPEAT: "info",
    DeviationKind.DURATION: "advisory",
    DeviationKind.HAZARD: "critical",
}


class DeviationDetector:
    """Turns tracker transitions plus per-frame checks into typed deviations.

    Reads the tracker; never writes to it. The tracker decides what happened,
    this decides what to call it.
    """

    def __init__(self, procedure: Procedure, tracker: StepTracker) -> None:
        self.proc = procedure
        self.tracker = tracker
        self.deviations: list[Deviation] = []
        self._wrong_object_seen: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    def update(self, frame: BeliefFrame, transitions: list[Transition]) -> list[Deviation]:
        found: list[Deviation] = []
        found += self._from_transitions(transitions)
        found += self._check_wrong_object(frame)
        self.deviations.extend(found)
        return found

    def finalise(self, transitions: list[Transition]) -> list[Deviation]:
        found = self._from_transitions(transitions)
        self.deviations.extend(found)
        return found

    # ------------------------------------------------------------------
    def _from_transitions(self, transitions: list[Transition]) -> list[Deviation]:
        out: list[Deviation] = []
        for tr in transitions:
            if tr.kind == "skipped":
                out.append(self._skip(tr))
            elif tr.kind == "hazard":
                out.append(self._hazard(tr))
            elif tr.kind == "repeated":
                out.append(self._repeat(tr))
            elif tr.kind == "completed":
                dev = self._duration(tr)
                if dev is not None:
                    out.append(dev)
                dev = self._out_of_order(tr)
                if dev is not None:
                    out.append(dev)
        return out

    # ------------------------------------------------------------------
    def _skip(self, tr: Transition) -> Deviation:
        step = self.proc.step(tr.step_id)
        spec = step.on_skip
        rec = self.tracker.records[tr.step_id]
        reason = self._render(spec, tr.step_id, rec) or tr.detail
        return Deviation(
            t=tr.t, kind=DeviationKind.SKIP, step_id=tr.step_id,
            severity=self._severity(spec, DeviationKind.SKIP, step.critical),
            confidence=tr.confidence, reason=reason,
            tts=spec.tts if spec else f"Step {tr.step_id} was not completed.",
            entities=step.objects,
        )

    def _hazard(self, tr: Transition) -> Deviation:
        step = self.proc.step(tr.step_id)
        sev = step.invariants[0].severity if step.invariants else "critical"
        return Deviation(
            t=tr.t, kind=DeviationKind.HAZARD, step_id=tr.step_id,
            severity=sev, confidence=tr.confidence,
            reason=f"invariant violated during {tr.step_id}: {tr.detail}",
            tts=tr.detail, entities=step.objects,
        )

    def _repeat(self, tr: Transition) -> Deviation:
        return Deviation(
            t=tr.t, kind=DeviationKind.REPEAT, step_id=tr.step_id,
            severity=DEFAULT_SEVERITY[DeviationKind.REPEAT],
            confidence=tr.confidence,
            reason=f"{tr.step_id} was already complete and has been performed again",
            tts="", entities=self.proc.step(tr.step_id).objects,
        )

    def _duration(self, tr: Transition) -> Deviation | None:
        """Completed outside its duration prior.

        Too fast usually means the verification matched something other than the
        action - a step "done" in a fifth of its minimum is a reason to look at
        the predicate, not to congratulate the crew.
        """
        rec = self.tracker.records[tr.step_id]
        dm = self.tracker.durations[tr.step_id]
        elapsed = rec.performed_s
        if elapsed is None:
            return None
        if rec.was_passed:
            # Done out of sequence, or confirmed after the crew had moved on:
            # the time between its start and its completion includes whatever
            # the crew did in between, and is not a duration of this step.
            return None
        if rec.sat_on_entry:
            # The world already satisfied this step when we reached it, so the
            # elapsed time measures nothing the crew did. Calling that a timing
            # violation is a false alarm, and false alarms are the metric that
            # decides whether any of this gets switched on.
            return None
        if dm.too_fast(elapsed):
            if not rec.start_known:
                # "Too fast" is a claim about when the step started, and nobody
                # saw the previous one finish. After an occlusion clears, the
                # step before is recognised late, the next one appears to have
                # started late, and a correctly paced step reads as rushed - on
                # the corpus that was most of the timing false alarms. A step
                # whose start is unknown can still be flagged too SLOW: a late
                # start only ever shortens the measured time.
                return None
            what = f"completed in {elapsed:.1f}s, below min_s {dm.d.min_s:.0f}s"
        elif dm.too_slow(elapsed):
            what = f"took {elapsed:.1f}s, above max_s {dm.d.max_s:.0f}s"
        else:
            return None
        return Deviation(
            t=tr.t, kind=DeviationKind.DURATION, step_id=tr.step_id,
            severity=DEFAULT_SEVERITY[DeviationKind.DURATION],
            confidence=1.0,
            reason=f"{tr.step_id} {what} (nominal {dm.d.nominal_s:.0f}s)",
            tts="", entities=self.proc.step(tr.step_id).objects,
        )

    def _out_of_order(self, tr: Transition) -> Deviation | None:
        """A step completing after it was already reported skipped.

        Online, "skipped" and "performed out of order" look identical at the
        moment the crew moves on - the difference is whether they come back to
        it. So the engine says SKIP when the step is passed, and if it is later
        completed, says so rather than leaving a wrong verdict in the log. The
        flight record ends up with both events and their times, which is what a
        reviewing PI actually needs.
        """
        rec = self.tracker.records[tr.step_id]
        if rec.status is not StepStatus.COMPLETE:
            return None
        if not rec.was_passed:
            return None
        if rec.passed_as is not StepStatus.SKIPPED:
            # Passed as UNVERIFIED means nobody saw it either way. Seeing its end
            # state later is a late confirmation, not evidence it was done out
            # of order - frame_loss_S07 reported OUT_OF_ORDER on a correct run
            # for exactly this.
            return None
        return Deviation(
            t=tr.t, kind=DeviationKind.OUT_OF_ORDER, step_id=tr.step_id,
            severity=DEFAULT_SEVERITY[DeviationKind.OUT_OF_ORDER],
            confidence=tr.confidence,
            reason=(f"{tr.step_id} was passed at {rec.passed_at:.1f}s and completed at "
                    f"{tr.t:.1f}s - performed out of sequence, not omitted"),
            tts="", entities=self.proc.step(tr.step_id).objects,
        )

    # ------------------------------------------------------------------
    def _check_wrong_object(self, frame: BeliefFrame) -> list[Deviation]:
        """The crew is holding something the step declares as confusable.

        This is why `confusable_with` is authored rather than inferred: vial A
        and vial B differ by a colour band, so the detector's own confidence is
        not the signal. The procedure says these two are mistakable, and the
        engine watches for exactly that substitution.

        Reported once per (step, entity) - a crew member holding the wrong vial
        for six seconds has made one mistake, not thirty.

        **Evidence accumulates over the holding episode**, exactly as a step's
        verification does. This used to report the first frame on which the
        wrong object was seen held, with that frame's contact confidence, and
        mark the pair as reported. The first frame of a grasp is the weakest -
        the hand is still closing - so under a less confident detector it came
        in at 0.77, the alert layer dropped it below deviation_threshold, and
        the latch meant six further seconds of plainly holding the wrong vial
        could never be reported: half of CRX-2's wrong-object runs, and one of
        CSP-1's, on the degraded corpus. Now the pair is reported once, when the
        accumulated evidence clears deviation_threshold, carrying that belief.
        """
        out: list[Deviation] = []
        bar = self.proc.alert_policy.deviation_threshold
        dt = getattr(self.tracker, "_dt", 0.0)
        for sid in self.tracker.frontier:
            step = self.proc.step(sid)
            spec = step.on_wrong_object
            if spec is None:
                continue
            if self.tracker.records[sid].status not in HOLDING_STATUSES:
                continue
            for wrong in self._confusables(sid, spec):
                if (sid, wrong) in self._wrong_object_seen:
                    continue
                held = self._held(frame, wrong)
                key = f"{sid}.wrong_object.{wrong}"
                self.tracker.evidence.update(key, self._holding(frame, wrong, held), dt)
                if held is None or not self.tracker.evidence.verified(key, bar):
                    continue
                self._wrong_object_seen.add((sid, wrong))
                # The object the step wants, not everything the step names. S02
                # names vial B AND the locker it comes from; the mistake is about
                # the vial, so the reason should say "expects vial_b" - the
                # entities sharing the wrong object's detector class.
                wrong_cls = self.proc.entity(wrong).detector_class
                others = [o for o in step.objects if o != wrong]
                right = ([o for o in others
                          if self.proc.entity(o).detector_class == wrong_cls] or others)
                out.append(Deviation(
                    t=frame.t_mono, kind=DeviationKind.WRONG_OBJECT, step_id=sid,
                    severity=self._severity(spec, DeviationKind.WRONG_OBJECT, step.critical),
                    # From support alone. Before the grasp the wrong object sits
                    # in view un-held, which fills the contrary statistic - but
                    # "not yet holding the wrong vial" is no evidence against a
                    # mistake that has not happened. The net belief carried that
                    # in, came in under deviation_threshold under a less
                    # confident detector, and the alert layer dropped a report
                    # the evidence had already cleared. Support itself decays
                    # on contrary frames, so a momentary brush still does not
                    # add up.
                    confidence=sigmoid(self.tracker.evidence.support(key)),
                    reason=(f"{sid} expects {', '.join(right) or 'its declared object'} but "
                            f"{wrong} is being held (contact confidence {held:.2f})"),
                    tts=spec.tts, entities=(wrong,),
                ))
        return out

    def _confusables(self, sid: str, spec: DeviationSpec) -> tuple[str, ...]:
        """Explicit list if the step gave one, otherwise the union of what the
        step's own objects declare themselves confusable with."""
        if spec.confusable:
            return spec.confusable
        out: list[str] = []
        for name in self.proc.step(sid).objects:
            for other in self.proc.entity(name).confusable_with:
                if other not in out:
                    out.append(other)
        return tuple(out)

    @staticmethod
    def _holding(frame: BeliefFrame, entity: str, held: float | None) -> Truth:
        """One frame's evidence that `entity` is being held.

        Seen held: that confidence. Seen NOT held - the object in view and a
        hand tracked - is evidence against, so a momentary brush does not add
        up across a whole step. Anything else - the object hidden, no hand in
        view - is no evidence either way.
        """
        if held is not None:
            return Truth.known(held)
        ob = frame.objects.get(entity)
        if ob is None or ob.occluded or not ob.visible:
            return UNKNOWN
        if not any(h.present for h in frame.hands.values()):
            return UNKNOWN
        return FALSE

    @staticmethod
    def _held(frame: BeliefFrame, entity: str) -> float | None:
        for hand in frame.hands.values():
            if hand.contact_with == entity and hand.grasp_type in ("grasp", "manipulate"):
                return hand.grasp_conf
        ob = frame.objects.get(entity)
        if ob is not None and ob.held_by is not None and not ob.occluded:
            return ob.conf
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _severity(spec: DeviationSpec | None, kind: DeviationKind, critical: bool) -> str:
        if spec is not None and spec.severity:
            return spec.severity
        base = DEFAULT_SEVERITY[kind]
        return "critical" if critical and base == "caution" else base

    @staticmethod
    def _render(spec: DeviationSpec | None, sid: str, rec) -> str:
        """Fill the author's reason_template. Unknown placeholders are left as
        written rather than raising - a malformed template must not take the
        alert down with it, because the alert is the part that matters."""
        if spec is None or not spec.reason_template:
            return rec.reason
        values = {
            "step_id": sid,
            "t_start": f"{rec.t_start:.1f}s" if rec.t_start is not None else "?",
            "t_now": f"{rec.t_end:.1f}s" if rec.t_end is not None else "?",
            "elapsed": f"{rec.elapsed:.1f}s" if rec.elapsed is not None else "?",
        }
        out = spec.reason_template
        for key, value in values.items():
            out = out.replace("{" + key + "}", value)
        return out
