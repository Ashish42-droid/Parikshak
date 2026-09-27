"""Step-index tracking over the procedure graph, with duration priors.

Hidden semi-Markov in the sense that matters here: the hidden variable is which
step the crew is on, transitions are constrained by the PDL graph, and each
state carries an explicit duration distribution rather than the geometric
dwell-time a plain HMM would impose. The duration prior is what makes "skipped"
a probabilistic inference instead of a timeout, and what carries belief across a
three-second occlusion.

It is deliberately not a full forward-backward over a lattice. Two reasons:

  - The graph has unordered groups and conditional branches. Flattening those
    into a linear state sequence loses exactly the structure that keeps a legal
    reorder from being reported as a deviation.
  - The engine must answer online, within 2 s, with a human-readable reason.
    A posterior over lattices is not a reason a crew member can act on.

So: every step's verification tree is evaluated every frame, the frontier of
legally-completable steps is tracked explicitly, and the duration prior scores
how surprising it is that a frontier step has not completed yet.

Evaluating every step every frame (rather than only the active one) is not
wasteful, it is required: `hold_for` accumulates its window per evaluation, so a
tree checked intermittently has a window full of holes.

Decisions are taken on accumulated BELIEF, not on one frame's confidence. A
detector that reports the latch open at 0.84 for ten seconds has said something
very different from one that reported it once, and a hard per-frame bar cannot
tell them apart. Measured on the degraded corpus, that single bar left correct
evidence permanently just short of 0.85, and froze 104 of 129 failing steps
behind one unverifiable predecessor. See evidence.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from parikshak.belief.frame import BeliefFrame
from parikshak.engine.evidence import EvidenceAccumulator, sigmoid
from parikshak.engine.predicates import PredicateEvaluator, RunFacts
from parikshak.engine.truth import Truth, t_not
from parikshak.pdl.loader import Duration, Group, Procedure


class StepStatus(str, Enum):
    """PENDING -> ACTIVE -> one of COMPLETE / SKIPPED / UNVERIFIED.

    UNVERIFIED and SKIPPED are different claims and must never be merged.
    SKIPPED says we watched and it did not happen. UNVERIFIED says we could not
    see. Reporting the second as the first is how an occluded camera
    manufactures a deviation.
    """

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"
    SKIPPED = "SKIPPED"
    UNVERIFIED = "UNVERIFIED"


#: Statuses in which a step is still being performed, so a wrong-object
#: substitution is still attributable to it.
HOLDING_STATUSES = frozenset({StepStatus.ACTIVE})


@dataclass
class StepRecord:
    step_id: str
    status: StepStatus = StepStatus.PENDING
    t_start: float | None = None
    t_end: float | None = None
    best: Truth | None = None
    unknown_since: float | None = None
    #: Seconds this step spent ACTIVE with its verification undecidable.
    #: A step can finish COMPLETE and still have been UNVERIFIED for part of its
    #: life - that is the normal shape of an occlusion that later clears, and it
    #: is the raw material for the UNVERIFIED-rate metric.
    unknown_s: float = 0.0
    #: True once the crew moved past this step without completing it. If it
    #: later completes, the run was out of order rather than an omission - and
    #: the log should end up carrying both facts with their times.
    was_passed: bool = False
    passed_at: float | None = None
    #: How it was passed: SKIPPED (seen not done) or UNVERIFIED (not seen). Only
    #: the first can later be "out of order"; the second is just confirmed late.
    passed_as: StepStatus | None = None
    #: Verification already read as satisfied the instant this step was entered.
    #: Its elapsed time is then not a duration - nothing was performed - so
    #: timing checks must not treat it as one.
    sat_on_entry: bool = False
    reason: str = ""
    #: When evidence that this step was done FIRST appeared (the onset of its
    #: final climb to verified), as opposed to `t_end`, when that evidence had
    #: accumulated far enough to be believed. The gap is recognition lag, and it
    #: varies with confidence - see `performed_s`.
    t_onset: float | None = None
    #: The earliest moment the crew could have begun: the last time the step
    #: before it was SEEN still undone.
    t_begun: float | None = None
    #: Whether `t_begun` is a real observation. False when the previous step was
    #: never seen finishing - blind, jumped past, or branched away from.
    start_known: bool = True
    #: Seconds this step was on the frontier with its own objects hidden or the
    #: rack frame lost. Unlike `unknown_s`, a `hold_for` window filling does not
    #: count: that is waiting, not blindness.
    blind_s: float = 0.0
    #: Seconds undecidable AND blind at once - what "cannot verify" means. This,
    #: not `unknown_s`, is what the crew is told about and what counts as the
    #: step having been UNVERIFIED. See ever_unverified.
    unseen_s: float = 0.0

    @property
    def elapsed(self) -> float | None:
        if self.t_start is None:
            return None
        return (self.t_end if self.t_end is not None else self.t_start) - self.t_start

    @property
    def performed_s(self) -> float | None:
        """The LONGEST time the crew could have spent on this step, given what
        was observed.

        From the last time the previous step was seen still undone - it cannot
        have been finished before then - to the onset of this step's own
        evidence - it was finished by then. A "too fast" claim is only honest
        if the step was too fast even on that generous reading.

        Both earlier clocks produced false alarms. Recognition to recognition
        moved with confidence, because belief lags evidence by an amount that
        depends on it. Onset to onset was late whenever the previous step's
        evidence sat undecided before it appeared - a hold_for window refilling
        after an occlusion made a correctly paced S08 read 0.8 s quick. And
        blindness needs no special case: nothing is sighted while a step is
        hidden, so the bracket simply widens across it.
        """
        if self.t_onset is not None and self.t_begun is not None:
            return max(0.0, self.t_onset - self.t_begun)
        return self.elapsed


@dataclass
class Transition:
    """One thing that happened, for the deviation layer and the log."""

    t: float
    step_id: str
    kind: str          # entered | completed | skipped | unverified | branched | repeated
    detail: str = ""
    confidence: float = 1.0


class DurationModel:
    """P(step duration) as a Gaussian truncated to [min_s, max_s].

    Used for two questions the engine actually asks:

      overdue(elapsed)  - how surprising is it that this has not finished?
                          Feeds skip confidence, so a step abandoned after
                          three nominal durations is a stronger claim than one
                          abandoned after half of one.

      too_fast(elapsed) - did it complete below min_s? A step done faster than
                          physically plausible usually means the verification
                          matched something else.
    """

    def __init__(self, d: Duration) -> None:
        self.d = d
        self.sigma = max(d.sigma_s, 1e-3)

    def overdue(self, elapsed: float) -> float:
        """P(duration <= elapsed): the step should already have finished."""
        if elapsed >= self.d.max_s:
            return 1.0
        z = (elapsed - self.d.nominal_s) / (self.sigma * math.sqrt(2.0))
        return max(0.0, min(1.0, 0.5 * (1.0 + math.erf(z))))

    def too_fast(self, elapsed: float) -> bool:
        return elapsed < self.d.min_s

    def too_slow(self, elapsed: float) -> bool:
        return elapsed > self.d.max_s


class StepTracker:
    """Tracks which step the crew is on and how each one resolved.

    Owns the run's facts (flags, completion times, confirmations) because those
    are consequences of steps the ENGINE accepted. Perception must never be able
    to assert that a step completed.
    """

    def __init__(self, procedure: Procedure, evaluator: PredicateEvaluator | None = None) -> None:
        self.proc = procedure
        self.ev = evaluator or PredicateEvaluator(procedure)
        self.facts = RunFacts()
        self.records: dict[str, StepRecord] = {
            sid: StepRecord(sid) for sid in procedure.order
        }
        self.durations = {sid: DurationModel(procedure.step(sid).duration)
                          for sid in procedure.order}
        self.truths: dict[str, Truth] = {}
        #: Accumulated belief that each step's verification holds. Every
        #: completion, jump and skip decision reads this, never one frame.
        policy = procedure.alert_policy
        self.evidence = EvidenceAccumulator(
            rate_hz=policy.evidence_rate_hz,
            bars=(policy.complete_threshold, policy.deviation_threshold))
        self.belief: dict[str, float] = {}
        #: When each step's verification was last OBSERVED to be false - an
        #: informative frame pointing the wrong way. See refuted_belief.
        self._last_contrary: dict[str, float] = {}
        #: When each step was last SEEN still undone: contrary readings that
        #: persisted for one independent observation, not a single noisy frame.
        #: The next step's earliest start is taken from this. See update().
        self._seen_undone: dict[str, float] = {}
        #: Start of the current unbroken run of contrary readings, per step.
        self._contrary_since: dict[str, float] = {}
        #: When each step's support last rose from zero, and the last contrary
        #: sighting before that. See StepRecord.performed_s.
        self._onset: dict[str, float] = {}
        self._onset_after: dict[str, float | None] = {}
        self._support_zero_since: dict[str, float] = {}
        #: (earliest moment the next steps could have begun, whether that moment
        #: was observed), handed to the steps a completion lets the crew start.
        #: See _complete and _enter.
        self._begun_hint: tuple[float, bool] | None = None
        #: The latest moment the run was OBSERVED to reach a step - the newest
        #: `t_begun` that was a real sighting rather than a guess. When a step is
        #: entered without a known start (its predecessor was never seen
        #: finishing), this is the most recent moment anything can honestly be
        #: said to have begun, and it bounds `occurred` evidence. See _enter.
        self._known_begin = 0.0
        self._initialising = True
        #: Frontier steps whose own objects are hidden, or the rack frame lost,
        #: on the current frame.
        self._blind_now: set[str] = set()
        self.transitions: list[Transition] = []
        self.expected_states: dict[str, str] = {}
        self.finished = False
        self.t = 0.0
        self._prev_t: float | None = None
        self._dt = 0.0
        #: Steps whose invariant is being violated right now. A violated
        #: invariant suppresses jump detection for that step - see
        #: _check_invariants for why that ordering is load-bearing.
        self._hazard_now: set[str] = set()
        self._hazard_latched: set[tuple[str, int]] = set()

        self.active: str = procedure.entry
        self.group: Group | None = procedure.group_of(procedure.entry)
        self._enter(procedure.entry, 0.0)
        if self.group:
            for m in self.group.members:
                self._enter(m, 0.0)
        self._initialising = False

    # ------------------------------------------------------------------
    @property
    def frontier(self) -> tuple[str, ...]:
        """Steps that may legally complete right now.

        For an unordered group that is every member still pending - which is
        precisely what stops a legal reorder from being called a deviation.
        """
        if self.group is not None:
            return tuple(m for m in self.group.members
                         if self.records[m].status is StepStatus.ACTIVE)
        return (self.active,) if self.records[self.active].status is StepStatus.ACTIVE else ()

    @property
    def threshold(self) -> float:
        """The bar a verification must clear to count as COMPLETE."""
        return self.proc.alert_policy.complete_threshold

    @property
    def deviation_threshold(self) -> float:
        return self.proc.alert_policy.deviation_threshold

    def contradicted(self, t: Truth) -> bool:
        """Is there positive evidence that this is NOT so?

        Deliberately not `t.refuted(complete_threshold)`. That asks "can this
        reach the completion bar", which a clause that is TRUE at 0.84
        confidence also fails - and answering yes there turns every
        low-confidence frame into an accusation. Requiring the negation to be
        confidently true means occlusion can widen a verdict into UNVERIFIED
        but never into SKIPPED.
        """
        return t_not(t).satisfied(self.deviation_threshold)

    def verified(self, sid: str) -> bool:
        """Has evidence that this step was done built up past the bar?"""
        return self.evidence.verified(sid, self.threshold)

    def refuted_belief(self, sid: str) -> bool:
        """May we say this step was NOT done?

        Two conditions, and the second is the one that matters:

          - contrary evidence gathered since the step became active has built up
            past the deviation bar, and
          - the step was last SEEN undone more recently than it could possibly
            have been performed - its `min_s`.

        Every step is honestly "not done yet" for its first moments, so contrary
        evidence reaches the bar within half a second of any step starting. If
        the camera then goes blind, that half-second-old observation is all
        there is - and without the recency condition it was carried, frozen,
        to the moment the crew moved on and reported as a skip. It did exactly
        that on CRX-2 occlusion_S05 and CSP-1 frame_loss_S07, both runs in
        which the crew did the step and nobody could see it.

        If the last contrary sighting is older than `min_s`, the crew could have
        finished in the unobserved time since. Then the honest verdict is
        UNVERIFIED. This can only ever move a verdict from SKIPPED toward
        UNVERIFIED, never the other way.
        """
        if not self.evidence.refuted(sid, self.deviation_threshold):
            return False
        seen = self._last_contrary.get(sid)
        if seen is None:
            return False
        window = max(self.durations[sid].d.min_s,
                     1.0 / self.proc.alert_policy.evidence_rate_hz)
        return (self.t - seen) < window

    def status(self, step_id: str) -> StepStatus:
        return self.records[step_id].status

    def steps_with(self, status: StepStatus) -> tuple[str, ...]:
        return tuple(s for s in self.proc.order if self.records[s].status is status)

    # ------------------------------------------------------------------
    def update(self, frame: BeliefFrame) -> list[Transition]:
        """Advance one frame. Returns the transitions produced by this frame."""
        dt = 0.0 if self._prev_t is None else max(0.0, frame.t_mono - self._prev_t)
        self._prev_t = frame.t_mono
        self.t = frame.t_mono
        self._dt = dt
        self.ev.ingest(frame)
        self.facts.confirmations.update(frame.confirmations)

        produced: list[Transition] = []
        if self.finished:
            return produced

        #: One independent observation - how long a reading must persist to count.
        watch_s = 1.0 / self.proc.alert_policy.evidence_rate_hz

        # Every tree, every frame - see the module docstring.
        for sid in self.proc.order:
            step = self.proc.step(sid)
            self.facts.current_step = sid
            self.truths[sid] = (
                Truth(1.0, 1.0) if step.verification is None
                else self.ev.evaluate(step.verification, self.facts, f"{sid}.verification")
            )
            contrary_before = self._seen_undone.get(sid)
            self.belief[sid] = self.evidence.update(sid, self.truths[sid], dt)
            informativeness, point = self.evidence.weight(self.truths[sid])
            if informativeness >= 0.5 and point < 0.5:
                self._last_contrary[sid] = frame.t_mono
                # "Seen still undone" - the moment the NEXT step's start is taken
                # from - needs the contrary reading to persist for one independent
                # observation. One noisy frame is not a sighting: in
                # occlusion_S06__harsh_2 a single frame at 38.8s read S06 as not
                # done between frames that agreed it was, moved S07's start from
                # 29.8s to 38.8s, and a correctly paced step was reported as 3.8s
                # against a 5s minimum. That was 1 of the 5 false "too fast"
                # alarms on the CSP-1 corpus; the rest have other causes.
                # _track_onset already refuses to let one frame end an episode of
                # support; this is the same rule for the opposite reading.
                began = self._contrary_since.setdefault(sid, frame.t_mono)
                if frame.t_mono - began >= watch_s:
                    self._seen_undone[sid] = frame.t_mono
            else:
                # Any other frame ends the run - one that says it IS done, and an
                # undecided one too. "Seen still undone" means seen, continuously:
                # letting undecided frames bridge the run turned nine seconds of
                # [0.00, 0.95] readings in occlusion_S06__harsh_2 into nine seconds
                # of "undone", and the one noisy frame at the end still counted.
                self._contrary_since.pop(sid, None)
            self._track_onset(sid, contrary_before, frame.t_mono)

        # Blindness, for the duration checks: the step's own objects hidden or
        # the rack frame lost. Deliberately narrower than "undecided", which a
        # hold_for window filling also produces.
        self._blind_now = set()
        for sid in self.frontier:
            step = self.proc.step(sid)
            hidden = not frame.frame_lock or any(
                frame.objects[o].occluded for o in step.objects if o in frame.objects)
            if hidden:
                self.records[sid].blind_s += dt
                self._blind_now.add(sid)

        # Order matters. Invariants first: a hazard can produce exactly the
        # world state a later step's verification looks for, and if the jump
        # detector runs first it reports the crew as having moved on instead.
        produced += self._check_invariants()
        produced += self._resolve_frontier()
        produced += self._recover_passed()
        produced += self._detect_jumps()
        produced += self._check_branches()
        self.transitions.extend(produced)
        return produced

    # ------------------------------------------------------------------
    def _track_onset(self, sid: str, contrary_before: float | None, t: float) -> None:
        """Remember when support for `sid` began, surviving brief dips.

        One noisy frame that drains support to zero is not the end of the
        evidence episode, and treating it as one moved the onset later and made
        the NEXT step look rushed. Only a gap longer than one independent
        observation (1 / evidence_rate_hz) ends the episode.
        """
        if self.evidence.support(sid) > 0.0:
            self._support_zero_since.pop(sid, None)
            if sid not in self._onset:
                self._onset[sid] = t
                self._onset_after[sid] = contrary_before
            return
        since = self._support_zero_since.setdefault(sid, t)
        if t - since >= 1.0 / self.proc.alert_policy.evidence_rate_hz:
            self._onset.pop(sid, None)
            self._onset_after.pop(sid, None)

    def _resolve_frontier(self) -> list[Transition]:
        out: list[Transition] = []
        for sid in self.frontier:
            rec = self.records[sid]
            t = self.truths[sid]
            if rec.best is None or t.lo > rec.best.lo:
                rec.best = t

            if self.verified(sid):
                out.append(self._complete(sid, t))
            elif t.undecided(self.threshold):
                # Undecidable: we can neither confirm nor refute. Latch when it
                # started and accumulate, because "how long were we blind" is a
                # reported metric, not an internal detail - a system that is
                # UNVERIFIED half the time is useless even if it never lies.
                if rec.unknown_since is None:
                    rec.unknown_since = self.t
                rec.unknown_s += getattr(self, "_dt", 0.0)
                if sid in self._blind_now:
                    rec.unseen_s += getattr(self, "_dt", 0.0)
            else:
                rec.unknown_since = None
        return out

    def _complete(self, sid: str, t: Truth) -> Transition:
        rec = self.records[sid]
        rec.status = StepStatus.COMPLETE
        rec.t_end = self.t
        rec.best = t
        belief = self.belief.get(sid, 0.5)
        rec.reason = f"verification belief {belief:.2f} (latest evidence {t})"
        rec.t_onset = self._onset.get(sid, self.t)
        self.facts.step_completed_at[sid] = self.t
        self._apply_effects(sid)

        performed = rec.performed_s or 0.0
        detail = f"performed in {performed:.1f}s"
        dm = self.durations[sid]
        if dm.too_fast(performed) and rec.start_known:
            detail += f" (below min_s {dm.d.min_s})"
        elif dm.too_slow(performed):
            detail += f" (above max_s {dm.d.max_s})"

        # The steps this completion lets the crew start could have begun as soon
        # as this step was last seen still undone. If it was never seen undone,
        # nobody knows when it was finished, so no start time can be claimed.
        seen_undone = self._onset_after.get(sid)
        self._begun_hint = (seen_undone if seen_undone is not None else rec.t_onset,
                            seen_undone is not None)
        self._advance_from(sid)
        self._begun_hint = None
        return Transition(self.t, sid, "completed", detail, belief)

    def _apply_effects(self, sid: str) -> None:
        """Flags and expected states.

        `set_state` records what the procedure SAYS should now be true. It is
        deliberately not fed back into the belief state: an engine that can
        assert a latch is closed would satisfy its own verification, which is
        the one thing an independent witness must never do.
        """
        for eff in self.proc.step(sid).effects:
            kind, value = next(iter(eff.items()))
            if kind == "assert":
                self.facts.flags[value] = True
            elif kind == "retract":
                self.facts.flags[value] = False
            elif kind == "set_state":
                self.expected_states[value["entity"]] = value["state"]

    # ------------------------------------------------------------------
    def _advance_from(self, sid: str) -> None:
        """Move the frontier after `sid` completed."""
        if self.group is not None and sid in self.group.members:
            if any(self.records[m].status is StepStatus.ACTIVE for m in self.group.members):
                self.active = next(m for m in self.group.members
                                   if self.records[m].status is StepStatus.ACTIVE)
                return
            target = self.group.successor
            self.group = None
            self._goto(target)
            return

        nxt = self.proc.step(sid).next
        if not nxt:
            self.finished = True
            return
        self._goto(nxt[0])

    def _goto(self, target: str) -> None:
        """Enter a step or a group by name."""
        group = next((g for g in self.proc.groups if g.id == target), None)
        if group is not None:
            self.group = group
            for m in group.members:
                if self.records[m].status is StepStatus.PENDING:
                    self._enter(m, self.t)
            self.active = group.members[0]
            return

        self.group = self.proc.group_of(target)
        if self.group is not None:
            for m in self.group.members:
                if self.records[m].status is StepStatus.PENDING:
                    self._enter(m, self.t)
        self.active = target
        if self.records[target].status is StepStatus.PENDING:
            self._enter(target, self.t)

    def _enter(self, sid: str, t: float) -> None:
        rec = self.records[sid]
        if rec.status is StepStatus.COMPLETE:
            # Re-entering a finished step means the crew is doing it again.
            self.transitions.append(Transition(t, sid, "repeated",
                                               f"{sid} re-entered after completing at "
                                               f"{rec.t_end:.1f}s" if rec.t_end else ""))
            return
        if rec.status is not StepStatus.PENDING:
            return
        rec.status = StepStatus.ACTIVE
        rec.t_start = t
        if self._begun_hint is not None:
            rec.t_begun, rec.start_known = self._begun_hint
        else:
            # The run starting is a known moment. Any other entry without a
            # hint - a jump past an unverified step, a branch - is not.
            rec.t_begun = t
            rec.start_known = self._initialising
        rec.sat_on_entry = self.verified(sid)
        # Contrary evidence from before this step was active says nothing about
        # whether the crew did it now: "the latch is not closed" was true for
        # every step before anyone was asked to close it. Carried into the skip
        # decision, that history got a step performed while the rack frame was
        # lost - nothing seen either way - reported as SKIPPED.
        self.evidence.reset_against(sid)

        # Bound `occurred` evidence by WHEN this step could have begun, rather
        # than clearing it at some moment of the engine's own choosing.
        #
        # Every step's tree is evaluated every frame, so a momentary action -
        # "press start", "rotate the seal" - is latched whenever it happens.
        # Clearing that latch on entry discards actions the camera saw while the
        # engine was still behind the crew; clearing the SUCCESSORS' latches on
        # this step's entry, as this did before, discards them one step earlier
        # still. In nominal__harsh_0 the insert was seen at 30.0-33.8s and wiped
        # when S05 was entered at 34.4s, so a correctly inserted cartridge was
        # reported SKIPPED. Trimming to `since` keeps the bound that matters - a
        # press during step 4 cannot satisfy step 9 - while keeping evidence
        # from the window in which this step was the one being performed.
        #
        # When this step's own start was never observed, the honest anchor is
        # the most recent moment that WAS observed: the crew cannot have started
        # it before then either.
        if rec.start_known and rec.t_begun is not None:
            self._known_begin = max(self._known_begin, rec.t_begun)
        since = (rec.t_begun if rec.start_known and rec.t_begun is not None
                 else self._known_begin)
        self.ev.begin_step(sid, since=since)
        for succ in self.proc.successors(sid):
            self.ev.begin_step(succ, since=since)
        self.transitions.append(Transition(t, sid, "entered"))

    # ------------------------------------------------------------------
    def _detect_jumps(self) -> list[Transition]:
        """A step beyond the frontier satisfied its verification.

        The crew has moved on without finishing what they were on. Whether the
        abandoned step is SKIPPED or UNVERIFIED is decided by what we could see:
        a refuted tree means we watched and it did not happen; an undecidable
        one means we could not tell, and saying "skipped" there would be the
        occluded camera manufacturing a deviation.
        """
        out: list[Transition] = []
        frontier = set(self.frontier)
        if not frontier:
            return out

        watch_s = 1.0 / self.proc.alert_policy.evidence_rate_hz
        for sid in frontier:
            if sid in self._hazard_now:
                continue  # a live hazard is not evidence the crew moved on
            rec = self.records[sid]
            if rec.t_start is not None and self.t - rec.t_start < watch_s:
                # Not yet watched for one independent observation. Contrary
                # evidence is reset on entry, so a step abandoned in the frame it
                # was entered can only ever be UNVERIFIED - when the predecessor
                # was recognised late, a latch left open in plain view went
                # unreported for exactly that reason.
                continue
            for succ in self.proc.successors(sid):
                if succ in frontier:
                    continue  # a group sibling: legal reorder, not a jump
                if self.records[succ].status is not StepStatus.PENDING:
                    continue
                if not self.verified(succ):
                    continue
                # Rising edge, not level - judged against when this step could
                # have BEGUN, not when the engine got round to entering it.
                #
                # A successor whose evidence was already there before this step
                # was even possible is not evidence the crew performed it. In
                # skip_S08 the latch is never closed, so S11 ("open the latch")
                # reads as satisfied from frame one - a level test abandons S10
                # on the strength of a state nobody created.
                #
                # But "already true when the step was entered" is the wrong
                # clock. When a predecessor is recognised late, the crew may
                # already have moved on: in out_of_order_S09_before_S08 under
                # moderate noise, S07 was recognised at 46.6 s, start was pressed
                # at 44.0 s, and S09 was verified before S08 was entered - so the
                # entry-time snapshot called the press "already true" and an open
                # latch with the unit running was never reported as a skip.
                onset = self._onset.get(succ)
                began = rec.t_begun
                if onset is not None and began is not None and onset <= began:
                    continue
                out.append(self._abandon(sid, succ))
                break
        return out

    def _check_invariants(self) -> list[Transition]:
        """Clauses that must hold for the WHOLE step, not just at its end.

        This is the only mechanism that catches "the latch was opened while the
        unit was running": no step was skipped and nothing was out of order, so
        sequence checking alone is structurally blind to it.

        It also has to run BEFORE jump detection. In CSP-1 the hazard - latch
        open during S10 - is character-for-character the verification of S11
        ("open the glovebox latch"). Detect jumps first and a critical hazard is
        silently reported as normal progress to the next step.
        """
        out: list[Transition] = []
        self._hazard_now = set()
        for sid in self.frontier:
            for i, inv in enumerate(self.proc.step(sid).invariants):
                self.facts.current_step = sid
                t = self.ev.evaluate(inv.expr, self.facts, f"{sid}.invariant[{i}]")
                # Evidence of the VIOLATION accumulates over time, like every
                # other decision. Judged one frame at a time against the
                # deviation bar, a hazard seen for three seconds at 0.75-0.82 -
                # a vial agitated right beside the powered unit, under a less
                # confident detector - was never raised at all.
                key = f"{sid}.invariant.{i}"
                self.evidence.update(key, t_not(t), self._dt)
                violated = self.evidence.verified(key, self.deviation_threshold)
                if not violated and not self.contradicted(t):
                    # Only a POSITIVELY satisfied invariant clears the latch.
                    # Clearing on merely-not-contradicted lets confidence jitter
                    # reset it every few frames, and a critical alert that
                    # bypasses cooldown then becomes a siren - twelve
                    # "stop, the latch is open" in ninety seconds, on a run
                    # where the latch was closed the whole time.
                    if t.satisfied(self.threshold):
                        self._hazard_latched.discard((sid, i))
                    continue
                # A frame that confidently shows the violation is a live hazard
                # for jump detection even before the evidence has built up.
                self._hazard_now.add(sid)
                if not violated or (sid, i) in self._hazard_latched:
                    continue  # not yet established, or onset already reported
                self._hazard_latched.add((sid, i))
                out.append(Transition(
                    self.t, sid, "hazard",
                    inv.tts or f"invariant violated during {sid}",
                    sigmoid(self.evidence.support(key))))
        return out

    def _abandon(self, sid: str, because_of: str) -> Transition:
        rec = self.records[sid]
        t = self.truths[sid]
        elapsed = self.t - (rec.t_start or self.t)
        overdue = self.durations[sid].overdue(elapsed)

        belief = self.belief.get(sid, 0.5)
        if self.refuted_belief(sid):
            rec.status = StepStatus.SKIPPED
            rec.reason = (f"{because_of} verified while belief in {sid} stayed at "
                          f"{belief:.2f} (latest evidence {t}); {elapsed:.1f}s elapsed "
                          f"against nominal {self.durations[sid].d.nominal_s:.0f}s")
            kind, conf = "skipped", min(1.0, max(overdue, 1.0 - belief))
        else:
            rec.status = StepStatus.UNVERIFIED
            rec.reason = (f"{because_of} verified but {sid} could not be decided "
                          f"(belief {belief:.2f}, latest evidence {t}); evidence was "
                          f"unavailable or too weak, not contradicted")
            kind, conf = "unverified", overdue
        rec.t_end = self.t
        rec.was_passed = True
        rec.passed_at = self.t
        rec.passed_as = rec.status
        self._advance_from(sid)
        return Transition(self.t, sid, kind, rec.reason, conf)

    # ------------------------------------------------------------------
    def _recover_passed(self) -> list[Transition]:
        """A step the crew moved past has now been completed after all.

        Online, "skipped" and "performed out of order" are the same observation
        at the moment it happens; the difference is whether the crew comes back.
        Rather than guess, the engine reports the omission when it happens and
        records the recovery when it happens, so the flight record carries both
        events with their times instead of one confident wrong verdict.

        **Evidence is attributed to the frontier first.** If a step completed
        during the very episode of evidence that would recover this one, that
        step explains the evidence. CSP-1 S05 is "open the latch"; when it was
        genuinely skipped, S11 - also "open the latch" - reproduced its end state
        ninety seconds later, and S05 was credited as done out of order on every
        such run. A crew member going back for a missed step does so while the
        procedure has moved on to something else, not in the same breath as
        completing a later step that looks exactly like it.
        """
        out: list[Transition] = []
        for sid in self.proc.order:
            rec = self.records[sid]
            if not rec.was_passed or rec.status is StepStatus.COMPLETE:
                continue
            t = self.truths.get(sid)
            if t is None or not self.verified(sid):
                continue
            onset = self._onset.get(sid, self.t)
            # Only a step acting on the same objects can explain the evidence.
            # S11 re-opens the latch S05 opened, so it explains S05 reading true.
            # S08 closes the latch and has nothing to do with a vial attached to a
            # cartridge; when S07's evidence happened to arrive in the frame S08
            # was recognised, "any later completion" blocked S07's recovery, the
            # SKIP alert was never retracted, and a correct run raised an alarm.
            mine = set(self.proc.step(sid).objects)
            explained = any(
                done_at >= onset and other != sid and not self.records[other].was_passed
                and mine & set(self.proc.step(other).objects)
                for other, done_at in self.facts.step_completed_at.items())
            if explained:
                continue
            rec.status = StepStatus.COMPLETE
            rec.t_end = self.t
            rec.best = t
            rec.t_onset = onset
            rec.reason = (f"verification satisfied at {t}, after the step was "
                          f"passed at {rec.passed_at:.1f}s")
            self.facts.step_completed_at[sid] = self.t
            self._apply_effects(sid)
            out.append(Transition(self.t, sid, "completed",
                                  f"recovered after being passed at {rec.passed_at:.1f}s",
                                  self.belief.get(sid, t.lo)))
        return out

    def _check_branches(self) -> list[Transition]:
        """Conditional routing. A hardware fault sending the run to F01 is a
        legal transition, not a crew deviation."""
        out: list[Transition] = []
        for sid in self.frontier:
            for br in self.proc.step(sid).branch:
                self.facts.current_step = sid
                t = self.ev.evaluate(br.when, self.facts, f"{sid}.branch.{br.goto}")
                key = f"{sid}.branch.{br.goto}"
                belief = self.evidence.update(key, t, self._dt)
                if not self.evidence.verified(key, self.threshold):
                    continue
                if br.goto == self.active or self.records.get(br.goto) is None:
                    continue
                if self.records[br.goto].status is not StepStatus.PENDING:
                    continue
                rec = self.records[sid]
                if rec.status is StepStatus.ACTIVE:
                    rec.status = StepStatus.COMPLETE
                    rec.t_end = self.t
                    rec.reason = f"branched to {br.goto}"
                    self.facts.step_completed_at[sid] = self.t
                self.group = None
                self._goto(br.goto)
                out.append(Transition(self.t, br.goto, "branched",
                                      f"from {sid}: {br.tts or br.goto}", belief))
                return out
        return out

    # ------------------------------------------------------------------
    def finalise(self) -> list[Transition]:
        """Close the run. Anything still ACTIVE never resolved.

        It becomes UNVERIFIED, never SKIPPED - even when its verification reads
        as contradicted at the final frame. The run stopping is not evidence
        about the crew: a `hold_for` window that had two more seconds to run, or
        a stability check straddling the last frame, says nothing about whether
        the step was performed. Steps never reached stay PENDING.

        An abandoned procedure still shows up plainly, as a completion count
        below the step count, without inventing an accusation per step.
        """
        out: list[Transition] = []
        for sid in self.proc.order:
            rec = self.records[sid]
            if rec.status is not StepStatus.ACTIVE:
                continue
            t = self.truths.get(sid, Truth(0.0, 1.0))
            rec.t_end = self.t
            rec.status = StepStatus.UNVERIFIED
            rec.reason = f"run ended with {sid} still in progress ({t})"
            out.append(Transition(self.t, sid, "unverified", rec.reason, 0.5))
        self.finished = True
        self.transitions.extend(out)
        return out

    # ------------------------------------------------------------------
    def ever_unverified(self, step_id: str) -> bool:
        """Did this step spend long enough unable to be seen to be reported as such?

        `unverified_after_s` is the same policy number the alert layer uses, so
        a step that reads UNVERIFIED to the crew is exactly a step that counts
        here.

        Measured as time undecidable while BLIND - the step's objects hidden or
        the rack frame lost - not all undecidable time. A window still filling
        is waiting, not blindness: once `stable` read "just set down" as
        undecided rather than false, a bag settling in the locker on a correct
        CSP-1 run spent four seconds undecided and the crew was told "Cannot
        verify step S14" with the camera seeing everything.
        """
        rec = self.records[step_id]
        if rec.status is StepStatus.UNVERIFIED:
            return True
        return rec.unseen_s >= self.proc.alert_policy.unverified_after_s

    @property
    def unverified_steps(self) -> tuple[str, ...]:
        return tuple(s for s in self.proc.order if self.ever_unverified(s))

    def summary(self) -> dict[str, str]:
        return {sid: self.records[sid].status.value for sid in self.proc.order}
