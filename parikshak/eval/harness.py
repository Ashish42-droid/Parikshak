"""Evaluation harness: the numbers PLAN.md section 14 says to report.

    step accuracy                    the headline
    deviation recall PER TYPE        an aggregate hides WRONG_OBJECT at 40%
    false alarms per 45 min + ROC    the number that decides adoption
    accuracy stratified by occlusion proves the UNVERIFIED design works
    UNVERIFIED rate                  too high = useless, too low = guessing
    alert latency p50 / p95          the edge claim
    legal-reorder false alarms       must be zero

Scoring rests on the `injected` ground truth carried in each trace header, so
"false alarm" has a checkable definition rather than a rhetorical one:

    true positive   an alert whose (kind, step) was injected
    false positive  an alert that matches nothing injected, on a run with nothing
                    injected - OR on any run, about a step that comes before
                    every injected step, which no injected error can have caused
    cascade         an alert about a later step on a run with an injection
    miss            an injected deviation that produced no alert

Two deliberate scoring choices, both of which make our own numbers worse:

  - An UNVERIFIED notice is never counted as a false alarm. It is the system
    declining to guess, and scoring it as an error would punish exactly the
    behaviour the design exists to produce.
  - A SKIP alert on a step that was injected as OUT_OF_ORDER counts as a true
    positive on the STEP but is reported separately in the confusion matrix, so
    the type confusion stays visible instead of being averaged away.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from parikshak.belief.frame import BeliefFrame
from parikshak.belief.trace import read_trace
from parikshak.engine.runner import ProcedureEngine, RunSummary
from parikshak.pdl.loader import Procedure

SECONDS_PER_45_MIN = 45 * 60

#: Kinds that are advisory bookkeeping rather than "the crew made a mistake".
#: Counted, but reported apart from the safety-relevant ones.
SOFT_KINDS = frozenset({"DURATION", "REPEAT"})


@dataclass
class RunScore:
    """One replayed run, scored against its own injected ground truth."""

    run_id: str
    scenario: str
    profile: str
    duration_s: float
    frames: int
    injected: tuple[tuple[str, str], ...]        # (kind, step_id)
    alerted: tuple[tuple[str, str], ...]
    true_positives: tuple[tuple[str, str], ...]
    false_positives: tuple[tuple[str, str], ...]
    cascade: tuple[tuple[str, str], ...]
    misses: tuple[tuple[str, str], ...]
    latencies: tuple[float, ...]
    unverified_steps: tuple[str, ...]
    steps_total: int
    steps_correct: int
    #: Every step in the procedure, not just the ones this trace constrains.
    #: The UNVERIFIED numerator counts all of them, so the denominator must too.
    steps_in_procedure: int
    occlusion_fraction: float
    #: The stretch of this run in which an alarm can only be false: all of a run
    #: with nothing injected, and the part before the earliest injected step
    #: began on a run with an injection.
    fa_window_s: float = 0.0

    @property
    def false_alarms_per_45min(self) -> float:
        if self.fa_window_s <= 0:
            return 0.0
        return len(self.false_positives) * SECONDS_PER_45_MIN / self.fa_window_s

    @property
    def step_accuracy(self) -> float:
        return self.steps_correct / self.steps_total if self.steps_total else 1.0


# --------------------------------------------------------------------------
def _occlusion_fraction(frames: Iterable[BeliefFrame]) -> float:
    """Fraction of frames in which anything was meaningfully occluded or the
    rack frame was lost. The stratification variable for PLAN.md section 14."""
    total = blocked = 0
    for f in frames:
        total += 1
        if (not f.frame_lock
                or any(v > 0.5 for v in f.occlusion.values())
                or any(o.occluded for o in f.objects.values())):
            blocked += 1
    return blocked / total if total else 0.0


def _expected_status(expect: dict[str, Any], step_id: str) -> str | None:
    """What the trace says this step should end as. None means unconstrained."""
    if step_id in expect.get("complete", []):
        return "COMPLETE"
    if step_id in expect.get("skipped", []):
        return "SKIPPED"
    return None


def score_run(summary: RunSummary, header_extra: dict[str, Any],
              frames: list[BeliefFrame]) -> RunScore:
    injected = tuple((d["kind"], d["step_id"])
                     for d in header_extra.get("injected", []))
    expect = header_extra.get("expect", {})

    alerted = tuple((a.kind.value, a.step_id) for a in summary.alerts)
    injected_steps = {step for _k, step in injected}

    # Procedure order, which RunSummary.status preserves. The earliest injected
    # step bounds what the injection can have caused.
    order = {sid: i for i, sid in enumerate(summary.status)}
    first_injected = min((order[s] for s in injected_steps if s in order), default=None)

    tps: list[tuple[str, str]] = []
    fps: list[tuple[str, str]] = []
    cascade: list[tuple[str, str]] = []
    for kind, step in alerted:
        # Matching on the STEP, not the kind: an omission the engine calls SKIP
        # and the corpus injected as OUT_OF_ORDER is the same real event caught.
        # Kind confusion is reported separately in the confusion matrix rather
        # than scored as a false alarm, which would make the FA number junk.
        if step in injected_steps:
            tps.append((kind, step))
        elif injected and (first_injected is None or order.get(step, len(order)) >= first_injected):
            # A CASCADE, not a false alarm. Skip "open the latch" and "insert
            # the cartridge" genuinely cannot be verified either - saying so is
            # correct behaviour, not crying wolf. Counted and reported, but not
            # against the false-alarm budget.
            cascade.append((kind, step))
        else:
            # Nothing injected - or an alarm about a step BEFORE every injected
            # one, which the injection cannot have caused. These used to be
            # filed as cascades too, which kept real false alarms on deviation
            # runs out of the budget: a rushed S12 run also told the crew that a
            # correctly inserted S06 was skipped, and the headline number never
            # counted it.
            fps.append((kind, step))

    fa_window = summary.duration_s
    if injected and first_injected is not None:
        first_sid = next(s for s, i in order.items() if i == first_injected)
        began = (summary.started_at or {}).get(first_sid)
        fa_window = min(summary.duration_s, began) if began is not None else summary.duration_s

    alerted_steps = {step for _k, step in alerted}
    misses = tuple((k, s) for k, s in injected if s not in alerted_steps)

    latencies: list[float] = []
    for alert in summary.alerts:
        origin = [d.t for d in summary.deviations
                  if d.step_id == alert.step_id and d.kind is alert.kind]
        if origin:
            latencies.append(alert.t - min(origin))

    correct = total = 0
    for step_id, status in summary.status.items():
        want = _expected_status(expect, step_id)
        if want is None:
            continue
        total += 1
        correct += int(status == want)

    return RunScore(
        run_id=summary.run_id,
        scenario=header_extra.get("scenario", summary.run_id),
        profile=header_extra.get("profile", "unknown"),
        duration_s=summary.duration_s,
        frames=summary.frames,
        injected=injected,
        alerted=alerted,
        true_positives=tuple(tps),
        false_positives=tuple(fps),
        cascade=tuple(cascade),
        misses=misses,
        latencies=tuple(latencies),
        unverified_steps=summary.unverified,
        steps_total=total,
        steps_correct=correct,
        steps_in_procedure=len(summary.status),
        occlusion_fraction=_occlusion_fraction(frames),
        fa_window_s=fa_window,
    )


# --------------------------------------------------------------------------
@dataclass
class EvalReport:
    scores: list[RunScore] = field(default_factory=list)

    # -- headline ------------------------------------------------------
    @property
    def step_accuracy(self) -> float:
        total = sum(s.steps_total for s in self.scores)
        correct = sum(s.steps_correct for s in self.scores)
        return correct / total if total else 1.0

    @property
    def clean_runs(self) -> list[RunScore]:
        """Runs with nothing injected: nominal, legal reorder, occlusion, lock
        loss. The false-alarm target is a claim about these."""
        return [s for s in self.scores if not s.injected]

    @property
    def false_alarms_per_45min(self) -> float:
        """False alarms per 45 minutes of operation in which the crew had done
        nothing wrong.

        Pooled over monitored time rather than averaged over runs: averaging
        per-run rates lets a ten-second run with one alert dominate a
        forty-minute clean one, and the fielded number is alarms per hour of
        operation.

        That operation is every run with nothing injected, plus the part of
        each deviation run before its earliest injected step began - an alarm
        there cannot be a consequence of the injection. The time after it is
        left out of the denominator, because no alarm there can be counted as
        false, and including it would dilute the rate in the flattering
        direction.
        """
        window = sum(s.fa_window_s for s in self.scores)
        fps = sum(len(s.false_positives) for s in self.scores)
        return fps * SECONDS_PER_45_MIN / window if window else 0.0

    @property
    def cascade_alerts_per_deviation_run(self) -> float:
        """Extra alerts on runs that did contain an injected deviation.

        Reported because a cascade is still noise in the crew's ear even when
        every alert in it is technically correct.
        """
        runs = [s for s in self.scores if s.injected]
        if not runs:
            return 0.0
        return sum(len(s.cascade) for s in runs) / len(runs)

    def recall(self, kind: str | None = None) -> float:
        inj = [i for s in self.scores for i in s.injected
               if kind is None or i[0] == kind]
        if not inj:
            return float("nan")
        hit = [i for s in self.scores for i in s.injected
               if (kind is None or i[0] == kind) and i not in s.misses]
        return len(hit) / len(inj)

    @property
    def recall_by_type(self) -> dict[str, float]:
        kinds = {k for s in self.scores for k, _ in s.injected}
        return {k: self.recall(k) for k in sorted(kinds)}

    @property
    def unverified_rate(self) -> float:
        """Fraction of scored steps that ended or spent time UNVERIFIED.

        Reported because a system that says "cannot verify" half the time is
        useless even though it never lies - and one that never says it is
        guessing.

        Denominator is every step of every run, not only the steps a trace
        constrains: the numerator counts all of them, and mixing the two
        produced rates above 100%.
        """
        total = sum(s.steps_in_procedure for s in self.scores)
        unv = sum(len(s.unverified_steps) for s in self.scores)
        return unv / total if total else 0.0

    @property
    def latency_p50(self) -> float:
        lat = [x for s in self.scores for x in s.latencies]
        return statistics.median(lat) if lat else 0.0

    @property
    def latency_p95(self) -> float:
        lat = sorted(x for s in self.scores for x in s.latencies)
        if not lat:
            return 0.0
        return lat[min(len(lat) - 1, int(0.95 * len(lat)))]

    # -- the ones that catch self-deception ----------------------------
    @property
    def legal_reorder_false_alarms(self) -> int:
        """Must be zero. Proves the unordered group, and is the row that
        separates a procedure engine from an if-else chain."""
        return sum(len(s.false_positives) for s in self.scores
                   if "legal_reorder" in s.scenario)

    @property
    def nominal_false_alarms(self) -> int:
        return sum(len(s.false_positives) for s in self.scores
                   if s.scenario == "nominal")

    @property
    def occlusion_false_alarms(self) -> int:
        """Alerts raised on runs whose only feature was that we could not see.
        Every one of these is the failure mode the whole design exists to
        prevent."""
        return sum(len(s.false_positives) for s in self.scores
                   if s.scenario.startswith(("occlusion", "frame_loss")))

    # -- stratification -------------------------------------------------
    def by_profile(self) -> dict[str, EvalReport]:
        out: dict[str, list[RunScore]] = defaultdict(list)
        for s in self.scores:
            out[s.profile].append(s)
        return {k: EvalReport(v) for k, v in out.items()}

    def by_occlusion_stratum(self) -> dict[str, EvalReport]:
        """PLAN.md section 14: accuracy stratified by occlusion fraction."""
        bands = {"0-10%": [], "10-25%": [], "25-50%": [], "50%+": []}
        for s in self.scores:
            f = s.occlusion_fraction
            key = "0-10%" if f < 0.10 else "10-25%" if f < 0.25 else "25-50%" if f < 0.50 else "50%+"
            bands[key].append(s)
        return {k: EvalReport(v) for k, v in bands.items() if v}

    @property
    def kind_confusion(self) -> dict[str, Counter]:
        """What the engine called each injected kind. Keeps type confusion
        visible instead of hiding it inside a step-level match."""
        out: dict[str, Counter] = defaultdict(Counter)
        for s in self.scores:
            for kind, step in s.injected:
                called = [k for k, st in s.alerted if st == step]
                for c in called or ["<none>"]:
                    out[kind][c] += 1
        return dict(out)

    # -- summary --------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": len(self.scores),
            "monitored_s": round(sum(s.duration_s for s in self.scores), 1),
            "step_accuracy": round(self.step_accuracy, 4),
            "deviation_recall": round(self.recall(), 4),
            "recall_by_type": {k: round(v, 4) for k, v in self.recall_by_type.items()},
            "false_alarms_per_45min": round(self.false_alarms_per_45min, 3),
            "clean_runs": len(self.clean_runs),
            "cascade_per_deviation_run": round(self.cascade_alerts_per_deviation_run, 2),
            "unverified_rate": round(self.unverified_rate, 4),
            "latency_p50_s": round(self.latency_p50, 3),
            "latency_p95_s": round(self.latency_p95, 3),
            "nominal_false_alarms": self.nominal_false_alarms,
            "legal_reorder_false_alarms": self.legal_reorder_false_alarms,
            "occlusion_false_alarms": self.occlusion_false_alarms,
        }


# --------------------------------------------------------------------------
def evaluate(procedure: Procedure, traces: Iterable[Path]) -> EvalReport:
    """Replay every trace and score it. No side effects, no files written."""
    report = EvalReport()
    for path in traces:
        header, frames = read_trace(path)
        engine = ProcedureEngine(procedure, run_id=header.run_id or path.stem)
        for f in frames:
            engine.step(f)
        summary = engine.finish()
        report.scores.append(score_run(summary, header.extra, frames))
    return report


# --------------------------------------------------------------------------
# Replay cache - what makes the ROC sweep affordable
# --------------------------------------------------------------------------
@dataclass
class ReplayCache:
    """One replayed run, kept so the alert layer can be re-gated without
    re-running perception and sequence tracking.

    This is the separation in engine/ paying for itself: `deviation_threshold`
    and `persistence_s` only affect alerts.py, so a seven-point ROC costs one
    replay pass plus seven cheap re-gates instead of seven full passes over
    ninety thousand frames. `complete_threshold` is different - it changes what
    the tracker concludes - so varying it needs a full replay, and `sweep`
    refuses to fake that.
    """

    run_id: str
    header_extra: dict[str, Any]
    duration_s: float
    frames_n: int
    status: dict[str, str]
    unverified: tuple[str, ...]
    deviations: tuple[Any, ...]
    occlusion_fraction: float
    #: Everything the alert layer was given, frame by frame, so a re-gate can
    #: drive it exactly as the live runner does. See regate.
    inputs: tuple["FrameInput", ...] = ()
    #: Deviations raised by finish(), after the last frame.
    final_deviations: tuple[Any, ...] = ()
    started_at: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrameInput:
    """What ProcedureEngine.step handed the alert policy on one frame."""

    t: float
    deviations: tuple[Any, ...]
    #: Steps that recovered on this frame; they retract their own pending alerts.
    retracted: frozenset[str]
    confirmations: tuple[str, ...]
    active_step: str | None


def replay_all(procedure: Procedure, traces: Iterable[Path]) -> list[ReplayCache]:
    out: list[ReplayCache] = []
    for path in traces:
        header, frames = read_trace(path)
        engine = ProcedureEngine(procedure, run_id=header.run_id or path.stem)
        inputs: list[FrameInput] = []
        for f in frames:
            n_dev = len(engine.detector.deviations)
            n_tr = len(engine.tracker.transitions)
            engine.step(f)
            fresh = engine.tracker.transitions[n_tr:]
            inputs.append(FrameInput(
                t=f.t_mono,
                deviations=tuple(engine.detector.deviations[n_dev:]),
                retracted=frozenset(
                    tr.step_id for tr in fresh
                    if tr.kind == "completed" and engine.tracker.records[tr.step_id].was_passed),
                confirmations=tuple(f.confirmations),
                active_step=engine.active_step,
            ))
        n_dev = len(engine.detector.deviations)
        summary = engine.finish()
        out.append(ReplayCache(
            run_id=summary.run_id,
            header_extra=header.extra,
            duration_s=summary.duration_s,
            frames_n=summary.frames,
            status=summary.status,
            unverified=summary.unverified,
            deviations=summary.deviations,
            occlusion_fraction=_occlusion_fraction(frames),
            inputs=tuple(inputs),
            final_deviations=tuple(summary.deviations[n_dev:]),
            started_at=dict(summary.started_at),
        ))
    return out


def regate(procedure: Procedure, cache: list[ReplayCache]) -> EvalReport:
    """Re-run only the alert policy over a cached replay.

    It must drive the policy exactly as ProcedureEngine.step and finish do -
    every frame, with that frame's recoveries as retractions and the crew's
    override tokens - or the ROC measures a different system from the one
    being shipped. The first version fed only the deviations, at the moments
    they were raised: a SKIP the crew recovered from inside the persistence
    window was never retracted, persistence ran on the wrong clock, and the
    sweep reported 1.77 false alarms / 45 min at the very thresholds where the
    engine itself produced 0.88. tests/test_eval.py pins the equivalence.
    """
    from parikshak.engine.alerts import AlertPolicyEngine

    report = EvalReport()
    for entry in cache:
        policy = AlertPolicyEngine(procedure)
        for fi in entry.inputs:
            if fi.confirmations and fi.active_step:
                policy.observe_overrides(fi.t, fi.confirmations, fi.active_step)
            policy.update(fi.t, list(fi.deviations), retracted=fi.retracted)
        policy.update(entry.duration_s, list(entry.final_deviations))
        policy.flush(entry.duration_s)

        summary = RunSummary(
            run_id=entry.run_id, procedure_id=procedure.id,
            duration_s=entry.duration_s, frames=entry.frames_n,
            status=entry.status,
            complete=tuple(s for s, v in entry.status.items() if v == "COMPLETE"),
            skipped=tuple(s for s, v in entry.status.items() if v == "SKIPPED"),
            unverified=entry.unverified,
            deviations=entry.deviations,
            alerts=tuple(policy.alerts),
            notices=(),
            started_at=entry.started_at,
        )
        score = score_run(summary, entry.header_extra, [])
        report.scores.append(replace(score, occlusion_fraction=entry.occlusion_fraction))
    return report


# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OperatingPoint:
    complete_threshold: float
    deviation_threshold: float
    persistence_s: float
    step_accuracy: float
    recall: float
    false_alarms_per_45min: float
    unverified_rate: float
    legal_reorder_false_alarms: int

    def meets_targets(self) -> bool:
        """PLAN.md section 2: step accuracy >= 95%, deviation recall >= 90%,
        and <= 1 false alarm per 45 min - with zero on a legal reorder."""
        return (self.step_accuracy >= 0.95
                and self.recall >= 0.90
                and self.false_alarms_per_45min <= 1.0
                and self.legal_reorder_false_alarms == 0)


def sweep(procedure: Procedure, traces: list[Path], *,
          complete_thresholds: Iterable[float] = (0.70, 0.75, 0.80, 0.85),
          deviation_thresholds: Iterable[float] = (0.70, 0.80, 0.85, 0.90),
          persistences: Iterable[float] = (1.5,),
          progress: bool = False) -> list[OperatingPoint]:
    """The ROC. Every knob lives in the procedure file, so moving the operating
    point builds a modified Procedure rather than editing engine code.

    `complete_threshold` is swept too, and it is the one that matters most.
    Because `all` combines by MIN, a step is only as confident as its weakest
    clause, so this threshold is effectively a floor under every sensor at once:
    set it at 0.85 and a detector that degrades to 0.82 makes EVERY step
    permanently unverifiable. That interaction is invisible on clean traces and
    obvious on degraded ones, which is the whole reason this corpus exists.

    It changes what the tracker concludes, so each value costs a full replay.
    `deviation_threshold` and `persistence_s` only gate alerts, so they are
    swept cheaply from the cached replay at each complete_threshold.
    """
    points: list[OperatingPoint] = []
    for ct in complete_thresholds:
        base = replace(
            procedure,
            alert_policy=replace(procedure.alert_policy, complete_threshold=ct),
        )
        if progress:
            print(f"    replaying corpus at complete_threshold={ct} ...", flush=True)
        cache = replay_all(base, traces)
        for dt in deviation_thresholds:
            for persist in persistences:
                tuned = replace(
                    base,
                    alert_policy=replace(base.alert_policy,
                                         deviation_threshold=dt,
                                         persistence_s=persist),
                )
                report = regate(tuned, cache)
                points.append(OperatingPoint(
                    complete_threshold=ct,
                    deviation_threshold=dt,
                    persistence_s=persist,
                    step_accuracy=report.step_accuracy,
                    recall=report.recall(),
                    false_alarms_per_45min=report.false_alarms_per_45min,
                    unverified_rate=report.unverified_rate,
                    legal_reorder_false_alarms=report.legal_reorder_false_alarms,
                ))
    return points


def choose_operating_point(points: list[OperatingPoint]) -> OperatingPoint | None:
    """Highest recall among points that meet every target.

    Returns None when no point does, which is a result rather than a failure:
    PLAN.md risk 2 says publish the tradeoff. A tuned-down system with a curve
    beats an untuned one with a claim.
    """
    viable = [p for p in points if p.meets_targets()]
    if not viable:
        return None
    return max(viable, key=lambda p: (p.recall, p.step_accuracy,
                                      -p.false_alarms_per_45min))
