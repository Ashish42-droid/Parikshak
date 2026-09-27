"""Temporal evidence accumulation: decide on sustained evidence, not one frame.

The failure this exists to fix, measured on the degraded corpus: a detector that
reports the latch open at 0.84 every frame for ten seconds was treated exactly
like one that reported it at 0.84 once. Both fall short of a 0.85 bar, so the
step never completed - and because jump detection used the same bar, the step
after it never verified either, and the run froze. 104 of the 129 steps that
should have completed were never even reached.

The model is two one-sided CUSUM statistics per decision (Page's test), fed with
per-frame log-likelihood evidence:

    llr     =  kappa * rate_hz * dt * logit(p)
    support =  clamp(support + llr, 0, limit)       evidence it HAS become true
    against =  clamp(against - llr, 0, limit)       evidence it is NOT true

    verified  <=>  support >= logit(complete_threshold)
    refuted   <=>  against >= logit(deviation_threshold)

Each term is doing something specific:

  p          the frame's evidence, the midpoint of the Truth interval, kept off
             0 and 1 so that a flat FALSE carries strong but finite weight.

  kappa      how informative the frame is: 1 - interval width. A known point
             is 1; UNKNOWN - an occluded object, a lost rack frame - is 0.
             **Unknown evidence moves neither statistic**, which is the property
             the whole design rests on, carried into time.

  rate_hz    effective independent observations per second. Consecutive video
             frames of one scene are highly correlated - the same pixels fail
             in the same way - so evidence is counted per SECOND, not per frame.
             That also makes the result independent of frame rate, which
             matters because dropped frames change it.

  dt cap     one frame never stands for more than one independent observation
             (dt <= 1 / rate_hz). A frame arriving after a two-second dropout is
             still one look at the world, not four.

  limit      both statistics are clamped just past the strictest decision bar,
             so a real change of state registers within about a second.

**Why two one-sided statistics and not one signed belief.** The first version
accumulated a single clamped log-odds, and replaying it against the golden
corpora showed exactly why that is wrong. A signed belief cannot tell "not true
YET" from "false":

  - CRX-2 S04 verifies on the tenth shake. For the eight seconds of shaking
    before it, the tree is honestly false, so a signed belief sat saturated at
    the negative clamp and needed a full second to climb back after the tenth
    shake. An occlusion began inside that second; S04 froze, and every later
    step in the run stayed PENDING.
  - CSP-1 S07 was "not done" for the whole run before anyone attempted it. With
    the rack frame lost for its entire duration no new evidence arrived, so the
    stale negative belief survived to the moment the crew moved on - and the
    engine called a step it never saw "skipped".

A one-sided `support` statistic cannot go below zero, so the evidence before an
onset costs nothing: detection latency is set by the evidence after it, about
0.4 s at 0.9 confidence. And `against` is reset when a step becomes active
(`reset_against`), so a SKIP rests only on evidence gathered while the crew
could actually have been doing the step.

This is deliberately not a calibration model. Calibrating raw scores needs
held-out labelled data, and fitting it on the evaluation corpus would be tuning
on the test set. Accumulation needs no such data: it only assumes that evidence
which keeps agreeing with itself is stronger than evidence seen once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from parikshak.engine.truth import Truth

#: Keeps a flat 0 or 1 off the asymptote. FALSE still carries ~2.6x the weight
#: of a 0.84 detection, so a genuine mismatch outvotes a weak match quickly.
EVIDENCE_EPS = 0.02

#: How far past the strictest decision bar a statistic may go, in log-odds
#: units. Enough that a verified step is not one frame away from unverified;
#: small enough that the other statistic can take over promptly.
HEADROOM = 1.0

#: Bars are kept off 0 and 1 so logit stays finite. A bar of exactly 1.0 means
#: "never" and stays unreachable rather than raising.
_BAR_EPS = 1e-6


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _bar(threshold: float) -> float:
    return logit(min(max(threshold, _BAR_EPS), 1.0 - _BAR_EPS))


@dataclass
class EvidenceAccumulator:
    """Per-key support/against statistics. One key per step verification tree
    (or branch condition); keys never interact."""

    rate_hz: float
    #: The thresholds the statistics will be compared against. The clamp is
    #: derived from them so that every bar stays reachable when a sweep moves it.
    bars: tuple[float, ...] = (0.85,)
    headroom: float = HEADROOM
    support_scores: dict[str, float] = field(default_factory=dict)
    against_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rate_hz > 0:
            raise ValueError(f"evidence rate must be positive, got {self.rate_hz!r}")
        strictest = max((max(b, 1.0 - b) for b in self.bars), default=0.85)
        self.limit = _bar(strictest) + self.headroom

    # ------------------------------------------------------------------
    def weight(self, truth: Truth) -> tuple[float, float]:
        """(informativeness, point estimate) for one frame's evidence."""
        kappa = max(0.0, 1.0 - (truth.hi - truth.lo))
        p = min(1.0 - EVIDENCE_EPS, max(EVIDENCE_EPS, (truth.lo + truth.hi) / 2.0))
        return kappa, p

    def update(self, key: str, truth: Truth, dt: float) -> float:
        """Fold in one frame's evidence for `key`. Returns the summary belief."""
        dt_eff = min(max(dt, 0.0), 1.0 / self.rate_hz)
        kappa, p = self.weight(truth)
        if kappa > 0.0 and dt_eff > 0.0:
            llr = kappa * self.rate_hz * dt_eff * logit(p)
            self.support_scores[key] = min(self.limit,
                                           max(0.0, self.support_scores.get(key, 0.0) + llr))
            self.against_scores[key] = min(self.limit,
                                           max(0.0, self.against_scores.get(key, 0.0) - llr))
        return self.belief(key)

    # ------------------------------------------------------------------
    def support(self, key: str) -> float:
        return self.support_scores.get(key, 0.0)

    def against(self, key: str) -> float:
        return self.against_scores.get(key, 0.0)

    def verified(self, key: str, bar: float) -> bool:
        """Has evidence that `key` holds built up past `bar`?"""
        return self.support(key) >= _bar(bar)

    def refuted(self, key: str, bar: float) -> bool:
        """Has evidence that `key` does NOT hold built up past `bar`?"""
        return self.against(key) >= _bar(bar)

    def reset_against(self, key: str) -> None:
        """Forget contrary evidence gathered before now.

        Called when a step becomes active: "the latch is not closed" was true
        for every step before the crew was asked to close it, and none of that
        is evidence that they failed to.
        """
        self.against_scores.pop(key, None)

    def belief(self, key: str) -> float:
        """One number for logs and displays: 0.5 with no evidence either way,
        toward 1 as support builds, toward 0 as contrary evidence does. Decisions
        use `verified` and `refuted`, never this."""
        return sigmoid(self.support(key) - self.against(key))
