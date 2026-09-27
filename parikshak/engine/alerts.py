"""Alert policy: what the crew actually hears, and when.

The detector says what happened. This decides whether it is worth interrupting
someone whose hands are busy. That is a different question, and it is the one
that decides adoption: one spurious alert in real operations and the speaker
gets taped over.

Every knob lives in `alert_policy` in the procedure file, never in this code, so
the W8 ROC sweep can move the operating point without a code change:

    complete_threshold   when a step counts as done
    deviation_threshold  confidence a deviation needs before it can be spoken
    persistence_s        how long it must survive before it is spoken
    unverified_after_s   how long undecidable before the crew is told we cannot see
    cooldown_s           minimum gap between alerts on the same step
    max_alerts_per_step  hard cap, so one confused step cannot become a siren
    advisory_only        alerts never block; the crew is always in command

The persistence delay does more than debounce. A crew member who skips a latch
and immediately goes back to it retracts their own deviation inside the window
and is never interrupted - the system notices, logs it, and stays quiet. Being
right and silent is worth more than being right and loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from parikshak.engine.deviations import Deviation, DeviationKind
from parikshak.pdl.loader import AlertPolicy, Procedure


class Severity(IntEnum):
    """Ordered so escalation is a comparison, not a lookup table."""

    INFO = 0
    ADVISORY = 1
    CAUTION = 2
    CRITICAL = 3

    @classmethod
    def parse(cls, name: str) -> Severity:
        try:
            return cls[name.upper()]
        except KeyError:
            return cls.ADVISORY

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Alert:
    """One thing said to the crew, with the reason it was said."""

    t: float
    kind: DeviationKind
    step_id: str
    severity: Severity
    channels: tuple[str, ...]
    text: str
    reason: str
    confidence: float
    requires_ack: bool = False
    repeat_s: float | None = None

    def __str__(self) -> str:
        chan = "+".join(self.channels)
        return (f"[{self.t:7.1f}s] {self.severity.label.upper():<8} {self.kind.value:<13} "
                f"{self.step_id:<10} ({chan}) {self.text or self.reason}")


@dataclass
class _Pending:
    deviation: Deviation
    first_seen: float


@dataclass
class UnverifiedNotice:
    """Not a deviation. The system telling the crew it cannot see.

    Kept as its own type on purpose: an UNVERIFIED report must never be counted
    as a false alarm in the eval, because it is the system declining to guess.
    Merging the two would make the honest behaviour look like the failure mode.
    """

    t: float
    step_id: str
    zones: tuple[str, ...]
    text: str


class AlertPolicyEngine:
    """Gates deviations into alerts. One instance per run."""

    def __init__(self, procedure: Procedure) -> None:
        self.proc = procedure
        self.policy: AlertPolicy = procedure.alert_policy
        self.alerts: list[Alert] = []
        self.notices: list[UnverifiedNotice] = []
        self.overrides: dict[str, float] = {}

        self._pending: dict[tuple[str, str], _Pending] = {}
        self._count: dict[str, int] = {}
        self._last_alert_at: dict[str, float] = {}
        self._last_critical_at: dict[tuple[str, str], float] = {}
        self._notified_unverified: set[str] = set()

    # ------------------------------------------------------------------
    def observe_overrides(self, t: float, confirmations: tuple[str, ...], step_id: str) -> None:
        """A crew override silences a step and is itself logged.

        The override is not a way to hide a mistake. It is recorded with its
        timestamp and the step it applied to, because the whole point of the log
        is that the ground PI can see what the crew decided and when.
        """
        tokens = {t_.lower() for t_ in self.policy.override_tokens}
        for token in confirmations:
            if token.lower() in tokens:
                self.overrides[step_id] = t

    # ------------------------------------------------------------------
    def update(self, t: float, deviations: list[Deviation],
               retracted: frozenset[str] = frozenset()) -> list[Alert]:
        """Buffer new deviations, retract resolved ones, emit what survives."""
        for dev in deviations:
            key = (dev.step_id, dev.kind.value)
            if key not in self._pending:
                self._pending[key] = _Pending(dev, t)

        for key in [k for k in self._pending if k[0] in retracted]:
            # The crew fixed it inside the persistence window. Nothing is said.
            del self._pending[key]

        fired: list[Alert] = []
        for key, pending in list(self._pending.items()):
            if t - pending.first_seen < self.policy.persistence_s:
                continue
            del self._pending[key]
            alert = self._admit(t, pending.deviation)
            if alert is not None:
                fired.append(alert)
        self.alerts.extend(fired)
        return fired

    def flush(self, t: float) -> list[Alert]:
        """End of run: emit anything still inside its persistence window.

        A skip detected two seconds before the run ended is still a skip. The
        window is there to catch retractions, not to swallow late findings.
        """
        fired: list[Alert] = []
        for _key, pending in list(self._pending.items()):
            alert = self._admit(t, pending.deviation)
            if alert is not None:
                fired.append(alert)
        self._pending.clear()
        self.alerts.extend(fired)
        return fired

    # ------------------------------------------------------------------
    def _admit(self, t: float, dev: Deviation) -> Alert | None:
        sev = Severity.parse(dev.severity)

        tier_cfg = self.policy.severity_tiers.get(sev.label, {})
        if sev is Severity.CRITICAL:
            # A critical hazard is exempt from the per-step budget - losing
            # "the latch is open while the unit is running" to a rate limiter
            # would be the worst possible trade. It is NOT exempt from a repeat
            # interval: an unbounded critical is a siren, and a siren is
            # something the crew silences.
            repeat_s = float(tier_cfg.get("repeat_s", self.policy.cooldown_s))
            last = self._last_critical_at.get((dev.step_id, dev.kind.value))
            if last is not None and (t - last) < repeat_s:
                return None
            self._last_critical_at[(dev.step_id, dev.kind.value)] = t
        else:
            if dev.confidence < self.policy.deviation_threshold:
                return None
            if dev.step_id in self.overrides:
                return None
            if self._count.get(dev.step_id, 0) >= self.policy.max_alerts_per_step:
                return None
            last = self._last_alert_at.get(dev.step_id)
            if last is not None and (t - last) < self.policy.cooldown_s:
                return None

        self._count[dev.step_id] = self._count.get(dev.step_id, 0) + 1
        self._last_alert_at[dev.step_id] = t

        tier = tier_cfg
        return Alert(
            t=t, kind=dev.kind, step_id=dev.step_id, severity=sev,
            channels=tuple(tier.get("channel", ("gui",))),
            text=dev.tts or self._default_text(dev),
            reason=dev.reason, confidence=dev.confidence,
            requires_ack=bool(tier.get("require_ack", False)),
            repeat_s=tier.get("repeat_s"),
        )

    def _default_text(self, dev: Deviation) -> str:
        """Spoken fallback when the author wrote no `tts`.

        Uses the step's human name and the entity's tts_name, never internal
        identifiers - "the glovebox latch", not "latch" and certainly not
        "S08.verification".
        """
        step = self.proc.step(dev.step_id)
        name = step.name.lower()
        if dev.kind is DeviationKind.SKIP:
            return f"Step {dev.step_id}, {name}, was not completed."
        if dev.kind is DeviationKind.WRONG_OBJECT and dev.entities:
            return f"Wrong object. You are holding {self.proc.entity(dev.entities[0]).spoken}."
        if dev.kind is DeviationKind.OUT_OF_ORDER:
            return f"Step {dev.step_id}, {name}, was performed out of sequence."
        if dev.kind is DeviationKind.DURATION:
            return f"Check step {dev.step_id}, {name}. The timing was outside the expected range."
        if dev.kind is DeviationKind.REPEAT:
            return f"Step {dev.step_id}, {name}, has already been completed."
        return f"Check step {dev.step_id}, {name}."

    # ------------------------------------------------------------------
    def unverified(self, t: float, step_id: str, unknown_s: float,
                   zones: tuple[str, ...]) -> UnverifiedNotice | None:
        """Tell the crew we cannot see, once per step.

        This is the restraint the demo leads with: blocking the camera produces
        "workspace occluded, cannot verify", not an alert. Judges notice
        restraint more than accuracy.
        """
        if unknown_s < self.policy.unverified_after_s:
            return None
        if step_id in self._notified_unverified:
            return None
        self._notified_unverified.add(step_id)
        where = f" - {', '.join(zones)} occluded" if zones else ""
        notice = UnverifiedNotice(
            t=t, step_id=step_id, zones=zones,
            text=f"Cannot verify step {step_id}{where}.")
        self.notices.append(notice)
        return notice

    # ------------------------------------------------------------------
    @property
    def false_alarm_budget(self) -> str:
        """A human-readable statement of the operating point, for the log header
        and the slide. The number nobody can argue with is the one you declared
        before the run rather than after."""
        p = self.policy
        return (f"complete_threshold={p.complete_threshold} "
                f"deviation_threshold={p.deviation_threshold} "
                f"evidence_rate_hz={p.evidence_rate_hz} "
                f"hold_for_tolerance={p.hold_for_tolerance} "
                f"persistence_s={p.persistence_s} cooldown_s={p.cooldown_s} "
                f"max_alerts_per_step={p.max_alerts_per_step} "
                f"advisory_only={p.advisory_only}")
