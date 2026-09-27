"""Evaluate a PDL predicate tree against a BeliefFrame.

This is the whole learned/symbolic boundary, in one file. Above it the engine
reasons about steps; below it perception reports what it saw. The vocabulary is
closed (schema/PREDICATES.md) and the validator has already proved the file only
uses predicates the deployed build can answer, so nothing here needs to handle
an unknown predicate at run time - it is a programming error, not a data error.

Three rules run through every implementation:

1. **Occluded is not false.** An entity we cannot see returns UNKNOWN, never 0.
   Return 0 and `not in_zone(...)` becomes a confident deviation, and the system
   alerts because someone's torso was in the way.

2. **No rack lock means no geometry.** When AprilTag PnP fails, every geometric
   quantity is stale. Geometry predicates return UNKNOWN rather than evaluating
   against the last good extrinsic, which is how a rotated rack would otherwise
   produce confident nonsense.

3. **No thresholds here.** Predicates return truth; `alert_policy` decides what
   counts as satisfied. A threshold inlined in a predicate is a number the W8
   ROC sweep cannot move.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from parikshak.belief.frame import BeliefFrame
from parikshak.engine.truth import FALSE, TRUE, UNKNOWN, Truth, t_all, t_any, t_not
from parikshak.pdl.loader import Procedure

#: Contact-head classes that count as holding something. `reach` does not - the
#: hand is on its way, which is exactly the moment a naive detector fires early.
HOLDING = frozenset({"grasp", "manipulate"})

#: The stability check inside `released`. Not author-visible, so it cannot be
#: tuned per procedure - which is why it uses the same 3 cm tolerance CSP-1
#: settled on after measurement, over a window long enough to clear the noise
#: floor (validator.stable_noise_floor). At 0.5 s and 2 cm it compared two or
#: three raw samples and read "still drifting" for 52-76% of frames on objects
#: that had been let go. At 1 s the floor under 15 mm of per-axis noise is
#: 4.2 cm, still above the tolerance; at 2 s it is 2.7 cm.
RELEASE_SETTLE_S = 2.0
RELEASE_TOL_M = 0.03

#: Motion confidence below which a frame carries no motion evidence at all, for
#: `motion_count`, which has no author-set gate of its own. A pipeline with no
#: motion model reports confidence 0 - see _p_motion_is.
MOTION_EVIDENCE_MIN_CONF = 0.5

#: Predicates whose answer depends on the rack frame being locked this frame.
GEOMETRIC = frozenset({
    "inside", "near", "aligned", "in_zone", "stable",
    "hand_in_zone", "body_restrained", "body_in_zone",
})


class PredicateError(RuntimeError):
    """The tree contains something the validator should have rejected."""


@dataclass
class RunFacts:
    """Run-scoped facts the engine owns and the predicates read.

    Kept separate from BeliefFrame because none of it is perception: these are
    consequences of steps the engine has already accepted. Perception must never
    be able to assert that a step completed.
    """

    flags: dict[str, bool] = field(default_factory=dict)
    step_completed_at: dict[str, float] = field(default_factory=dict)
    confirmations: set[str] = field(default_factory=set)
    current_step: str | None = None


class PredicateEvaluator:
    """Evaluates predicate trees, and owns the history the temporal ones need.

    One instance per run. `ingest(frame)` once per frame, then `evaluate(...)`
    for each tree you care about. History is keyed by the node's path in the
    tree, which is stable across frames because the tree is.

    **`hold_for` and `occurred` accumulate history per evaluate() call, not per
    ingest().** A tree evaluated only occasionally therefore has a window full
    of holes and will never satisfy a `hold_for`. The engine evaluates the
    active step's tree - and every live invariant - once per frame, and any
    other caller must do the same or accept that temporal clauses are wrong.
    """

    def __init__(self, procedure: Procedure, *, history_s: float = 30.0) -> None:
        self.proc = procedure
        self.history_s = history_s
        self.t = 0.0
        self.frame: BeliefFrame | None = None

        self._node_hist: dict[str, deque[tuple[float, Truth]]] = {}
        self._pos_hist: dict[str, deque[tuple[float, tuple | None]]] = {}
        self._motion_hist: deque[tuple[float, str, float]] = deque()
        #: `occurred` evidence, timestamped: {key: [(t, Truth), ...]}. Entries a
        #: later one dominates in both bounds are dropped, so this stays short.
        #: Timestamps are what let a latch be trimmed to the window in which the
        #: step could have been performed, rather than thrown away wholesale.
        self._occurred: dict[str, list[tuple[float, Truth]]] = {}
        #: Per STEP, the earliest moment its `occurred` evidence may come from.
        self._occurred_from: dict[str, float] = {}
        self._ever_held: set[str] = set()
        #: When each entity was last seen held. See _p_released.
        self._last_held: dict[str, float] = {}
        #: Whether any hand has been observed this run - without hand tracking,
        #: never having seen an object held is not evidence it was never held.
        self._hands_seen = False

    # ------------------------------------------------------------------
    # per-frame bookkeeping
    # ------------------------------------------------------------------
    def ingest(self, frame: BeliefFrame) -> None:
        """Record what the temporal predicates will need later.

        Position history stores None for an unseen object rather than skipping
        it, so `stable` can tell "did not move" from "was not watched".
        """
        self.frame = frame
        self.t = frame.t_mono

        for name, ob in frame.objects.items():
            hist = self._pos_hist.setdefault(name, deque())
            hist.append((frame.t_mono, ob.pos_rack if frame.frame_lock else None))
            _trim(hist, frame.t_mono, self.history_s)
            if ob.held_by is not None:
                self._ever_held.add(name)
                self._last_held[name] = frame.t_mono

        for hand in frame.hands.values():
            if hand.present:
                self._hands_seen = True
            if hand.contact_with and hand.grasp_type in HOLDING:
                self._ever_held.add(hand.contact_with)
                self._last_held[hand.contact_with] = frame.t_mono

        self._motion_hist.append((frame.t_mono, frame.motion.cls, frame.motion.conf))
        _trim(self._motion_hist, frame.t_mono, self.history_s)

    def begin_step(self, step_id: str, since: float | None = None) -> None:
        """Bound this step's `occurred` latches to when it could have happened.

        `since` is the earliest moment the crew could have begun this step.
        Evidence from before it belongs to an earlier step - a press during step
        4 must not satisfy step 9 - and evidence after it is this step being
        performed, even when the engine only reached the step later.

        Dropping the latches outright instead threw away actions the camera
        plainly saw whenever the engine was behind the crew: in nominal__harsh_0
        an insert seen at 30.0-33.8s was wiped when the PREVIOUS step was entered
        at 34.4s, the insert never came again - it had already happened - and a
        correctly inserted cartridge was reported SKIPPED. That mechanism was
        all 4 false SKIPs of S06 on the 130-run CSP-1 corpus.

        `since=None` drops them entirely: the strictest bound, for when nothing
        is known about when this step could have started.
        """
        self.proc.step(step_id)  # raises if unknown
        prefix = f"{step_id}."
        if since is None:
            self._occurred_from.pop(step_id, None)
            self._occurred = {k: v for k, v in self._occurred.items()
                              if not k.startswith(prefix)}
            return
        self._occurred_from[step_id] = since
        for key, hist in self._occurred.items():
            if key.startswith(prefix):
                self._occurred[key] = [e for e in hist if e[0] >= since]

    def reset_occurred(self) -> None:
        self._occurred.clear()

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def evaluate(self, node: Any, facts: RunFacts, path: str = "root") -> Truth:
        """Evaluate a predicate tree. Returns a Truth interval, never a bool."""
        if self.frame is None:
            raise PredicateError("evaluate() before the first ingest(); no frame to read")
        if node is None:
            return TRUE  # an absent tree constrains nothing
        if not isinstance(node, dict) or len(node) != 1:
            raise PredicateError(f"{path}: malformed predicate node {node!r}")

        key, val = next(iter(node.items()))

        if key == "all":
            return t_all(self.evaluate(c, facts, f"{path}.all[{i}]") for i, c in enumerate(val))
        if key == "any":
            return t_any(self.evaluate(c, facts, f"{path}.any[{i}]") for i, c in enumerate(val))
        if key == "not":
            return t_not(self.evaluate(val, facts, f"{path}.not"))
        if key == "hold_for":
            return self._hold_for(val, facts, f"{path}.hold_for")
        if key == "occurred":
            return self._occurred_since(val, facts, f"{path}.occurred")

        fn = getattr(self, f"_p_{key}", None)
        if fn is None:
            raise PredicateError(
                f"{path}: predicate {key!r} is in the vocabulary but has no implementation")

        # Rule 2: geometry is meaningless without a rack frame.
        if key in GEOMETRIC and not self.frame.frame_lock:
            return UNKNOWN
        return fn(val, facts)

    # -- temporal combinators -------------------------------------------
    def _hold_for(self, spec: dict[str, Any], facts: RunFacts, path: str) -> Truth:
        """The inner expression held across the trailing window - tolerating a
        bounded share of frames that disagree.

        The window's bounds are taken at the `hold_for_tolerance` quantile rather
        than at its minimum. A strict minimum is a noise amplifier: a clause true
        on 93% of frames holds a seven-frame window only 60% of the time, so
        `hold_for(stable(...))` read a flat FALSE at the moment the crew moved on
        in four of six false SKIPs measured on the degraded corpus. With a
        tolerance of 0.2, one frame in five may disagree. A genuine change of
        state is still caught - it disagrees on every frame after it happens -
        just a frame or two later than a single outlier would have been.

        Until the window is fully covered the lower bound stays at 0: "has held
        for 1.5 s" cannot be true 0.4 s in.
        """
        seconds = float(spec["seconds"])
        inner = self.evaluate(spec["expr"], facts, f"{path}.expr")

        hist = self._node_hist.setdefault(path, deque())
        hist.append((self.t, inner))
        _trim(hist, self.t, max(seconds, self.history_s))

        window = [t for (ts, t) in hist if ts >= self.t - seconds]
        if not window:
            return UNKNOWN
        tolerated = int(len(window) * self.proc.alert_policy.hold_for_tolerance)
        hi = sorted(t.hi for t in window)[tolerated]
        # Coverage is a property of the WHOLE history, not of the filtered
        # window: every entry in the window is >= t - seconds by construction,
        # so testing the window's own oldest entry can never show coverage and
        # would pin lo at 0 forever.
        if _covers(hist, self.t, seconds):
            lo = sorted(t.lo for t in window)[tolerated]
        else:
            lo = 0.0
        return Truth(min(lo, hi), hi)

    def _occurred_since(self, expr: Any, facts: RunFacts, path: str) -> Truth:
        """Did this ever hold during the current step?

        A step like "press start" is satisfied by a moment, not by a state: by
        the time the unit is running, the press is over. Without this, every
        momentary action would have to be caught in the single frame it happened.

        Entries carry their time so the latch can be trimmed to the window in
        which the step could have been performed (see `begin_step`) instead of
        being thrown away wholesale. An entry that a later one matches or beats
        in both bounds can never be the maximum again, so it is dropped - the
        list stays a few entries long however long the step runs.
        """
        step = facts.current_step or "?"
        key = f"{step}.{path}"
        now = self.evaluate(expr, facts, f"{path}.expr")
        hist = self._occurred.setdefault(key, [])
        while hist and hist[-1][1].lo <= now.lo and hist[-1][1].hi <= now.hi:
            hist.pop()
        hist.append((self.t, now))
        since = self._occurred_from.get(step)
        return t_any(v for t, v in hist if since is None or t >= since)

    # ------------------------------------------------------------------
    # leaf predicates - presence
    # ------------------------------------------------------------------
    def _p_visible(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        ob = self._obj(a["entity"])
        if ob is None or ob.occluded:
            return UNKNOWN
        if not ob.visible:
            return FALSE
        if ob.conf < float(a.get("min_conf", 0.5)):
            # Seen, but below the author's confidence gate. That is not a
            # sighting the author accepts - and it is not an absence either.
            # Returning FALSE here turned "the detector is a little less sure
            # than the author asked" into accumulated evidence that the crew
            # failed: on the degraded corpus CSP-1 S04 ("read the cartridge
            # barcode", min_conf 0.90) was called SKIPPED 13 times on runs where
            # the label was checked, because the cartridge was seen at 0.82.
            return UNKNOWN
        return Truth.known(ob.conf)

    def _p_absent(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        """Confidently not present. Rule 1 in its sharpest form: an occluded
        object is not absent, it is hidden, and conflating the two is how a
        crew member's shoulder gets reported as a missing tool."""
        ob = self._obj(a["entity"])
        if ob is None or ob.occluded:
            return UNKNOWN
        return FALSE if ob.visible else TRUE

    def _p_count_of(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        assert self.frame is not None
        base = a["entity"]
        instances = [ob for name, ob in self.frame.objects.items()
                     if name == base or name.startswith(f"{base}#")]
        if not instances or any(ob.occluded for ob in instances):
            return UNKNOWN
        seen = [ob for ob in instances if ob.visible]
        if len(seen) != int(a["n"]):
            return FALSE
        return Truth.known(min((ob.conf for ob in seen), default=1.0))

    # -- manipulation ---------------------------------------------------
    def _p_grasped(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        assert self.frame is not None
        entity = a["entity"]
        ob = self._obj(entity)
        if ob is not None and ob.occluded:
            return UNKNOWN

        wanted = a.get("hand", "any")
        sides = ("left", "right") if wanted in ("any", "both") else (wanted,)
        per_side: list[Truth] = []
        for side in sides:
            hand = self.frame.hands.get(side)
            if hand is None or not hand.present:
                per_side.append(UNKNOWN)
                continue
            hit = hand.contact_with == entity and hand.grasp_type in HOLDING
            per_side.append(Truth.known(hand.grasp_conf) if hit else FALSE)
        return t_all(per_side) if wanted == "both" else t_any(per_side)

    def _p_released(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        """Previously grasped, no longer held, and not drifting.

        The stability clause is what makes this mean "let go of it where it
        belongs" rather than "let go of it". In microgravity a released object
        that is still moving has not been stowed, it has been launched.
        """
        assert self.frame is not None
        entity = a["entity"]
        ob = self._obj(entity)
        if ob is None or ob.occluded:
            return UNKNOWN
        if entity not in self._ever_held:
            # Never seen held is only evidence when hands are being tracked. A
            # camera-only run with no hand model would otherwise read every
            # stowing step as "never released" - false evidence from a model
            # that is not there.
            return FALSE if self._hands_seen else UNKNOWN
        held_now = any(h.contact_with == entity and h.grasp_type in HOLDING
                       for h in self.frame.hands.values())
        if held_now or ob.held_by is not None:
            return FALSE
        # Until RELEASE_SETTLE_S has passed since the let-go this is undecided,
        # not false - see _stability.
        return t_all([Truth.known(ob.conf), self._stability(entity, RELEASE_SETTLE_S, RELEASE_TOL_M)])

    def _p_contacting(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        assert self.frame is not None
        oa, ob_ = self._obj(a["a"]), self._obj(a["b"])
        if oa is None or ob_ is None or oa.occluded or ob_.occluded:
            return UNKNOWN
        return Truth.known(self.frame.contact(a["a"], a["b"]))

    def _p_hand_in_zone(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        assert self.frame is not None
        zone = self.proc.zone(a["zone"])
        wanted = a["hand"]
        sides = ("left", "right") if wanted in ("any", "both") else (wanted,)
        per_side: list[Truth] = []
        for side in sides:
            hand = self.frame.hands.get(side)
            if hand is None or not hand.present:
                per_side.append(UNKNOWN)
                continue
            inside = zone.contains(hand.wrist_rack)
            per_side.append(UNKNOWN if inside is None
                            else (Truth.known(hand.conf) if inside else FALSE))
        return t_all(per_side) if wanted == "both" else t_any(per_side)

    # -- geometry -------------------------------------------------------
    def _p_in_zone(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        ob = self._obj(a["entity"])
        if ob is None or ob.occluded:
            return UNKNOWN
        inside = self.proc.zone(a["zone"]).contains(ob.pos_rack)
        if inside is None:
            return UNKNOWN
        return Truth.known(ob.conf) if inside else FALSE

    def _p_inside(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        """Containment in a zone, or in another entity.

        An entity container needs a declared `extent_m` - the procedure author
        says how big the thing is, because the detector reports a centroid and
        cannot know. Undeclared extent is UNKNOWN rather than a guessed radius:
        a guess here silently decides whether a critical step passed.
        """
        ob = self._obj(a["a"])
        if ob is None or ob.occluded:
            return UNKNOWN
        margin = float(a.get("margin_m", 0.0))
        container = a["container"]

        if container in self.proc.zones:
            inside = self.proc.zone(container).contains(ob.pos_rack, margin_m=margin)
            if inside is None:
                return UNKNOWN
            return Truth.known(ob.conf) if inside else FALSE

        holder = self._obj(container)
        if holder is None or holder.occluded:
            return UNKNOWN
        extent = self.proc.entity(container).extent_m
        if extent is None:
            return UNKNOWN
        if ob.pos_rack is None or holder.pos_rack is None:
            return UNKNOWN
        near = math.dist(ob.pos_rack, holder.pos_rack) <= extent + margin
        return Truth.known(min(ob.conf, holder.conf)) if near else FALSE

    def _p_near(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        oa, ob_ = self._obj(a["a"]), self._obj(a["b"])
        if oa is None or ob_ is None or oa.occluded or ob_.occluded:
            return UNKNOWN
        if oa.pos_rack is None or ob_.pos_rack is None:
            return UNKNOWN
        close = math.dist(oa.pos_rack, ob_.pos_rack) <= float(a["dist_m"])
        return Truth.known(min(oa.conf, ob_.conf)) if close else FALSE

    def _p_aligned(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        """Relative orientation within tol_deg. Needs 6-DoF pose for both, which
        most classes never get - hence UNKNOWN when either quaternion is absent,
        not a false alignment claim."""
        oa, ob_ = self._obj(a["a"]), self._obj(a["b"])
        if oa is None or ob_ is None or oa.occluded or ob_.occluded:
            return UNKNOWN
        if oa.quat_rack is None or ob_.quat_rack is None:
            return UNKNOWN
        angle = _quat_angle_deg(oa.quat_rack, ob_.quat_rack)
        if angle is None:
            return UNKNOWN
        return Truth.known(min(oa.conf, ob_.conf)) if angle <= float(a["tol_deg"]) else FALSE

    def _p_stable(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        return self._stability(a["entity"], float(a["seconds"]), float(a.get("tol_m", 0.02)))

    # -- object state ---------------------------------------------------
    def _p_state_is(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        ob = self._obj(a["entity"])
        if ob is None or ob.occluded or ob.state is None:
            return UNKNOWN
        return Truth.known(ob.state_conf) if ob.state == a["state"] else FALSE

    # -- motion ---------------------------------------------------------
    def _p_motion_is(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        assert self.frame is not None
        m = self.frame.motion
        if m.conf < float(a.get("min_conf", 0.6)):
            # Below the author's gate the classifier is not claiming anything -
            # whichever class it names. This used to apply only to the requested
            # class, so a DIFFERENT class at any confidence read FALSE; a
            # pipeline with no motion model reports "idle" at 0.0, and every
            # "press" or "insert" clause in a camera-only run became evidence the
            # crew had not done it. A different class ABOVE the gate is still
            # FALSE: the model is saying something else is happening.
            return UNKNOWN
        if m.cls != a["motion_class"]:
            return FALSE
        return Truth.known(m.conf)

    def _p_motion_count(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        """Counts rising edges, not frames. Ten shakes is ten transitions into
        the class, not sixty frames of being in it."""
        window_s = float(a.get("window_s", 60))
        want = a["motion_class"]
        edges, prev, informative = 0, None, False
        for ts, cls, conf in self._motion_hist:
            if ts < self.t - window_s:
                prev = cls
                continue
            informative = informative or conf >= MOTION_EVIDENCE_MIN_CONF
            if cls == want and prev != want:
                edges += 1
            prev = cls
        if edges >= int(a["n"]):
            return TRUE
        # Too few counted is only evidence if the classifier was saying anything
        # in the window. With no confident frame at all - no motion model, or a
        # camera that saw nothing - the count is unknown, not short.
        return FALSE if informative else UNKNOWN

    # -- crew posture ---------------------------------------------------
    def _p_body_restrained(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        """Both ankles inside the restraint zone, in rack coordinates - so it
        works with the crew member inverted, which is the point."""
        assert self.frame is not None
        body = self.frame.body
        if body is None:
            return UNKNOWN
        zone = self.proc.zone(a["zone"])
        checks: list[Truth] = []
        for joint in ("ankle_l", "ankle_r"):
            pos = body.joints_rack.get(joint)
            inside = zone.contains(pos)
            checks.append(UNKNOWN if inside is None
                          else (Truth.known(body.conf) if inside else FALSE))
        return t_all(checks)

    def _p_body_in_zone(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        assert self.frame is not None
        body = self.frame.body
        if body is None:
            return UNKNOWN
        inside = self.proc.zone(a["zone"]).contains(body.joints_rack.get("pelvis"))
        if inside is None:
            return UNKNOWN
        return Truth.known(body.conf) if inside else FALSE

    # -- engine ---------------------------------------------------------
    def _p_elapsed_since(self, a: dict[str, Any], facts: RunFacts) -> Truth:
        done_at = facts.step_completed_at.get(a["step_id"])
        if done_at is None:
            return FALSE  # it has not completed, so no time has elapsed since
        return TRUE if (self.t - done_at) >= float(a["seconds"]) else FALSE

    def _p_crew_confirmed(self, a: dict[str, Any], facts: RunFacts) -> Truth:
        return TRUE if a["token"] in facts.confirmations else FALSE

    def _p_occluded(self, a: dict[str, Any], _f: RunFacts) -> Truth:
        """Never UNKNOWN. This is the predicate that gates the others, so if it
        could itself be unknown there would be nothing to stand on."""
        assert self.frame is not None
        return Truth.known(self.frame.zone_occlusion(a["zone"]))

    def _p_assert_true(self, a: dict[str, Any], facts: RunFacts) -> Truth:
        return TRUE if facts.flags.get(a["flag"], False) else FALSE

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _obj(self, name: str):
        assert self.frame is not None
        return self.frame.objects.get(name)

    def _stability(self, entity: str, seconds: float, tol_m: float) -> Truth:
        """Centroid moved less than tol_m over the trailing window.

        In microgravity this is the difference between "stowed" and "drifting",
        which no amount of single-frame position accuracy can tell you.
        """
        ob = self._obj(entity)
        if ob is None or ob.occluded:
            return UNKNOWN
        hist = self._pos_hist.get(entity)
        if not hist:
            return UNKNOWN
        window = [(ts, p) for ts, p in hist if ts >= self.t - seconds]
        if not window:
            return UNKNOWN
        if not _covers(hist, self.t, seconds):
            if any(p is None for _ts, p in window):
                return UNKNOWN
            return Truth(0.0, ob.conf)  # not yet provable, not yet refuted
        if any(p is None for _ts, p in window):
            return self._partial_stability(window, seconds, tol_m, ob.conf)
        pts = [p for _ts, p in window]
        drift = _centroid_drift(pts)
        if drift <= tol_m:
            # Still across the window, however it was held. A cartridge held
            # motionless in its seat and then let go was stable all along, and
            # throwing that away delayed CSP-1 S06 by most of a second.
            return Truth.known(ob.conf)
        if drift - 2.0 * _drift_noise(window) <= tol_m:
            # Beyond the tolerance, but not by more than this window's own jitter
            # explains. Two half-window centroids of a vial that never moved
            # differ by 3-4 cm under 2 cm of per-axis noise, and reading that as
            # a definite "it moved" refuted CSP-1 S07 for seconds: in
            # skip_S13__harsh_1 a correctly attached vial was called SKIPPED -
            # five of the corpus's nine remaining false alarms. Movement is
            # proven only when the drift clears the tolerance by two standard
            # errors; short of that the honest answer is "cannot tell".
            return Truth(0.0, ob.conf)
        held_now = ob.held_by is not None or (self.frame is not None and any(
            h.contact_with == entity and h.grasp_type in HOLDING
            for h in self.frame.hands.values()))
        if not held_now and self._last_held.get(entity, float("-inf")) > self.t - seconds:
            # It moved - but it was let go inside the window, so the motion may
            # be the crew carrying it there. Carried motion is not drift: this
            # cannot show the object is drifting, only that it has not yet been
            # free for a whole window. Reading it as FALSE turned "just put it
            # down" into strong evidence the step was never done: CRX-2 S07
            # checks a vial stowed with hold_for(stable), the carry into the
            # locker read false, an occlusion kept those frames in the hold
            # window, and a correct stow was called SKIPPED when the crew said
            # "run complete" seven seconds later. A genuinely drifting object is
            # still caught, one window after it is let go. An object STILL held
            # and moving is judged as before.
            return Truth(0.0, ob.conf)
        return FALSE

    def _partial_stability(self, window: list, seconds: float, tol_m: float,
                           conf: float) -> Truth:
        """`stable` over a window with hidden frames in it: may confirm, never refute.

        Demanding an unbroken window left CRX-2's settle and stow steps
        UNVERIFIED on most harsh runs - at 15-30% occlusion an unbroken 5 s
        stretch is rare - although the positions before and after each short
        gap agreed. The halves are split by time, not by index, so a gap does not
        shift samples between them, and each half must be at least half observed.

        Only a confirmation is allowed. Fewer samples mean a higher noise floor,
        so drift beyond tolerance here is not trustworthy evidence of motion; it
        reads unknown. Losing information may move a verdict toward UNVERIFIED,
        never toward a deviation.
        """
        mid = self.t - seconds / 2.0
        halves = ([p for ts, p in window if ts < mid], [p for ts, p in window if ts >= mid])
        seen = [[p for p in half if p is not None] for half in halves]
        if any(not half or len(obs) * 2 < len(half) for half, obs in zip(halves, seen)):
            return UNKNOWN
        a, b = (tuple(sum(p[i] for p in obs) / len(obs) for i in range(3)) for obs in seen)
        return Truth.known(conf) if math.dist(a, b) <= tol_m else UNKNOWN


# --------------------------------------------------------------------------
def _centroid_drift(points: list) -> float:
    """How far the centroid moved across a window.

    The distance between the mean position over the window's first half and the
    mean over its second half. The first version took the largest distance of
    any sample from the first one, which measures detector jitter rather than
    motion: it is the MAXIMUM of many noisy distances, so it grows with every
    extra frame in the window, and no amount of observation can rescue it.
    Under 8-15 mm of per-axis centroid noise it read a stowed, stationary
    object as drifting 70-100% of the time, the step waiting on it never
    verified, and the whole run froze there.

    Averaging each half divides the jitter by the square root of the samples in
    it, so a longer window makes the estimate MORE certain, which is the right
    direction. A genuine drift still shows up in full: an object moving steadily
    across the window displaces the second half's mean from the first half's
    by about half the distance travelled.
    """
    n = len(points)
    if n < 2:
        return 0.0
    half = n // 2
    first, second = points[:half], points[n - half:]
    a = tuple(sum(p[i] for p in first) / half for i in range(3))
    b = tuple(sum(p[i] for p in second) / half for i in range(3))
    return math.dist(a, b)


def _drift_noise(window: list) -> float:
    """The spread `_centroid_drift` would show from jitter alone, in this window.

    Per-axis jitter is estimated from the residuals about a straight-line fit in
    time, so an object drifting steadily does not count its own motion as noise.
    From that: each half-window mean has variance sigma^2 / half, the difference
    of the two has twice that, and a 3-D magnitude adds sqrt(3). Exact,
    noise-free positions give zero, so clean data is judged exactly as before.
    `window` is [(t, (x, y, z)), ...] with no hidden frames.
    """
    n = len(window)
    half = n // 2
    if n < 4 or half < 2:
        return 0.0
    ts = [t for t, _p in window]
    t_mean = sum(ts) / n
    s_tt = sum((t - t_mean) ** 2 for t in ts)
    if s_tt <= 0.0:
        return 0.0
    ss = 0.0
    for axis in range(3):
        xs = [p[axis] for _t, p in window]
        x_mean = sum(xs) / n
        slope = sum((t - t_mean) * (x - x_mean) for t, x in zip(ts, xs)) / s_tt
        ss += sum((x - x_mean - slope * (t - t_mean)) ** 2 for t, x in zip(ts, xs))
    per_axis_var = ss / (3 * (n - 2))
    return math.sqrt(3.0 * 2.0 * per_axis_var / half)


def _trim(hist: deque, now: float, window_s: float) -> None:
    while hist and hist[0][0] < now - window_s:
        hist.popleft()


def _covers(hist: deque, now: float, seconds: float) -> bool:
    """Does the history reach back at least `seconds`?

    Until it does, a "has held for N seconds" claim cannot be true yet - not
    because it is false, but because we have not been watching long enough.
    """
    return bool(hist) and hist[0][0] <= now - seconds + 1e-9


def _quat_angle_deg(qa: Iterable[float], qb: Iterable[float]) -> float | None:
    """Angle of the rotation taking qa to qb, in degrees.

    Returns None for a degenerate quaternion rather than normalising it - a
    zero-norm quaternion means the pose estimator is broken, and quietly
    repairing its output would hide that from the eval harness.
    """
    a, b = list(qa), list(qb)
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return None
    dot = abs(sum(x * y for x, y in zip(a, b))) / (na * nb)
    return math.degrees(2.0 * math.acos(min(1.0, dot)))
