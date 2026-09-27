"""Temporal evidence accumulation, and the regressions it exists to fix.

Measured on the degraded corpus before this existed: correct evidence scaled to
0.84 could never clear a per-frame 0.85 bar, so steps neither completed nor
could be jumped past, and 104 of 129 failing steps were never even reached.

The accumulator is two one-sided statistics, not one signed belief. The signed
version was built first and two golden traces broke it; the onset and
reset_against tests below pin down exactly those two failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parikshak.belief.trace import read_trace
from parikshak.engine.evidence import EvidenceAccumulator, logit
from parikshak.engine.runner import ProcedureEngine
from parikshak.engine.truth import FALSE, TRUE, UNKNOWN, Truth
from parikshak.eval.perturb import Perturbation, apply
from parikshak.pdl import load_procedure

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "traces" / "golden"
BAR = 0.85


def acc(rate: float = 2.0) -> EvidenceAccumulator:
    return EvidenceAccumulator(rate_hz=rate, bars=(BAR, BAR))


def feed(a: EvidenceAccumulator, truth: Truth, seconds: float, fps: float,
         key: str = "s") -> EvidenceAccumulator:
    for _ in range(int(round(seconds * fps))):
        a.update(key, truth, 1.0 / fps)
    return a


def seconds_until(a: EvidenceAccumulator, truth: Truth, done, fps: float = 5.0,
                  cap: float = 30.0) -> float:
    elapsed = 0.0
    while not done(a) and elapsed < cap:
        a.update("s", truth, 1.0 / fps)
        elapsed += 1.0 / fps
    return elapsed


# -- the properties --------------------------------------------------------
def test_unknown_evidence_moves_neither_statistic():
    """The rule the whole design rests on, carried into time: a minute of an
    occluded camera is still no evidence at all, in either direction."""
    a = feed(acc(), UNKNOWN, 60, 5)
    assert a.support("s") == 0.0 and a.against("s") == 0.0
    assert a.belief("s") == 0.5
    assert not a.verified("s", BAR) and not a.refuted("s", BAR)


def test_sustained_moderate_evidence_verifies():
    assert feed(acc(), Truth.known(0.82), 3, 5).verified("s", BAR)


def test_one_frame_of_moderate_evidence_does_not():
    a = acc()
    a.update("s", Truth.known(0.82), 0.2)
    assert not a.verified("s", BAR)


def test_contradicting_evidence_refutes():
    assert feed(acc(), FALSE, 1.0, 5).refuted("s", BAR)


def test_onset_is_detected_promptly_however_long_it_was_false_before():
    """The first failure of the signed design. CRX-2 S04 is honestly false for
    eight seconds of shaking and true on the tenth shake; a signed belief took a
    second to climb back from its negative clamp, an occlusion began inside that
    second, and the whole run froze. Support cannot go below zero, so the time
    spent false beforehand costs nothing."""
    a = feed(acc(), FALSE, 60, 5)
    assert seconds_until(a, Truth.known(0.85), lambda x: x.verified("s", BAR)) <= 0.8


def test_contrary_evidence_from_before_a_step_was_active_is_forgotten():
    """The second failure. CSP-1 S07 was "not done" for the whole run before
    anyone attempted it, the rack frame was then lost for its entire duration,
    and the stale contrary evidence got a step nobody saw called SKIPPED."""
    a = feed(acc(), FALSE, 10, 5)
    assert a.refuted("s", BAR)
    a.reset_against("s")
    feed(a, UNKNOWN, 10, 5)
    assert not a.refuted("s", BAR)


def test_accumulation_is_per_second_not_per_frame():
    """Dropped frames change the frame rate. The same second of evidence must
    mean the same thing at 5 Hz and at 20 Hz."""
    # A whole second, so both rates take an exact number of frames - round(2.5)
    # is 2 under Python's half-to-even rounding, which silently fed the 5 Hz
    # side 0.4 s of evidence in an earlier version of this test. And 0.7, so a
    # second of it stays under the clamp.
    slow = feed(acc(), Truth.known(0.7), 1.0, 5)
    fast = feed(acc(), Truth.known(0.7), 1.0, 20)
    assert slow.support("s") == pytest.approx(fast.support("s"), abs=1e-9)


def test_a_frame_after_a_long_dropout_is_still_one_observation():
    after_gap, one_look = acc(), acc()
    after_gap.update("s", Truth.known(0.8), dt=10.0)
    one_look.update("s", Truth.known(0.8), dt=0.5)
    assert after_gap.support("s") == pytest.approx(one_look.support("s"))


def test_statistics_are_bounded_so_a_real_change_registers_promptly():
    """Without a clamp, a latch closed for a minute would need a minute of
    contrary evidence before it could be believed open."""
    a = feed(acc(), TRUE, 60, 5)
    assert a.support("s") <= a.limit
    assert seconds_until(a, FALSE, lambda x: x.refuted("s", BAR)) <= 1.0


@pytest.mark.parametrize("bar", [0.70, 0.85, 0.95])
def test_every_decision_bar_stays_reachable(bar):
    """The clamp is derived from the thresholds, so sweeping a threshold can
    never move it out of reach."""
    assert EvidenceAccumulator(rate_hz=2.0, bars=(bar, bar)).limit > logit(bar)


def test_a_bar_of_one_is_never_reached_and_does_not_raise():
    a = feed(acc(), TRUE, 60, 5)
    assert not a.verified("s", 1.0)


def test_a_wide_interval_carries_almost_no_evidence():
    """A hold_for window still filling reads [0, 0.91]. That is not a claim the
    step failed, and must not drift toward one."""
    a = feed(acc(), Truth(0.0, 0.91), 3, 5)
    assert not a.verified("s", BAR) and not a.refuted("s", BAR)
    assert a.belief("s") == pytest.approx(0.5, abs=0.05)


def test_keys_are_independent():
    a = feed(acc(), TRUE, 2, 5, key="done")
    assert a.support("other") == 0.0 and a.belief("other") == 0.5


def test_rate_must_be_positive():
    with pytest.raises(ValueError):
        EvidenceAccumulator(rate_hz=0.0)


# -- the regressions, on real traces ---------------------------------------
@pytest.fixture(scope="module")
def csp1():
    return load_procedure(ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")


def replay(proc, name: str, scale: float | None = None):
    """A golden trace, optionally as a uniformly less confident detector would
    report it: every confidence multiplied by `scale`, nothing else changed."""
    _, frames = read_trace(GOLDEN / f"{name}.jsonl")
    if scale is not None:
        frames = apply(frames, Perturbation("scaled", conf_scale=scale, seed=1))
    engine = ProcedureEngine(proc, run_id=name)
    for f in frames:
        engine.step(f)
    return engine.finish()


SAFETY_KINDS = {"SKIP", "WRONG_OBJECT", "HAZARD", "OUT_OF_ORDER"}


def test_a_less_confident_detector_still_completes_a_correct_run(csp1):
    """At 0.88 scale the foot-restraint clause reads 0.77 and the latch 0.80.
    A per-frame 0.85 bar completed 3 of 14 steps on this; every step was done.

    Every step completes except S04, and S04 is the correct exception. Its
    author requires the cartridge to be seen at 0.90 before a barcode read
    counts, and the scaled detector reaches 0.82. The honest verdict is that
    this detector cannot verify what the author asked for - UNVERIFIED - and
    never that the crew skipped the label check.
    """
    summary = replay(csp1, "nominal", 0.88)
    assert summary.status["S04"] == "UNVERIFIED", summary.status
    others = [s for s in csp1.order[:14] if s != "S04"]
    assert all(summary.status[s] == "COMPLETE" for s in others), summary.status
    assert not [d for d in summary.deviations if d.kind.value in SAFETY_KINDS]


def test_a_less_confident_detector_still_catches_the_skip(csp1):
    summary = replay(csp1, "skip_S08_latch", 0.88)
    assert "S08" in summary.skipped
    assert any(d.kind.value == "SKIP" and d.step_id == "S08" for d in summary.deviations)


def test_lower_confidence_never_turns_occlusion_into_a_skip(csp1):
    summary = replay(csp1, "occlusion_S06", 0.88)
    assert "S06" not in summary.skipped
    assert not any(d.kind.value == "SKIP" and d.step_id == "S06" for d in summary.deviations)


def test_a_step_performed_while_blind_is_not_called_skipped(csp1):
    """frame_loss_S07: the rack frame is lost for all of S07. Nothing was seen,
    so nothing may be claimed - UNVERIFIED, never SKIPPED."""
    summary = replay(csp1, "frame_loss_S07")
    assert "S07" not in summary.skipped
    assert not any(d.kind.value == "SKIP" and d.step_id == "S07" for d in summary.deviations)


# -- recovery and timing ---------------------------------------------------
def test_a_skipped_step_is_not_credited_when_a_later_step_recreates_its_end_state(csp1):
    """CSP-1 S05 is "open the latch", and so is S11. When S05 is genuinely
    skipped, S11 opens the latch ninety seconds later - and before this, S05 was
    credited as done out of order on every such run."""
    import yaml

    from parikshak.eval import synth
    doc = yaml.safe_load((ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
                         .read_text(encoding="utf-8"))
    engine = ProcedureEngine(csp1, run_id="skip_S05")
    for f in synth.run(doc, skip={"S05"}).frames:
        engine.step(f)
    summary = engine.finish()
    assert summary.status["S05"] == "SKIPPED", summary.status
    assert not any(d.kind.value == "OUT_OF_ORDER" and d.step_id == "S05"
                   for d in summary.deviations)


def test_a_step_confirmed_late_after_being_unverified_is_not_out_of_order(csp1):
    """Passed as UNVERIFIED means nobody saw it either way; seeing its end state
    afterwards is a late confirmation, not an out-of-sequence performance."""
    summary = replay(csp1, "frame_loss_S07")
    assert not any(d.kind.value == "OUT_OF_ORDER" and d.step_id == "S07"
                   for d in summary.deviations)


def test_moving_on_before_the_previous_step_is_recognised_is_still_a_jump(csp1):
    """Start pressed at 44.0 s, S07 recognised at 46.6 s, S08 entered after S09
    was already verified. Judged against entry time, the press was "already
    true" and the open latch was never reported; judged against when S08 could
    have begun, it is a jump."""
    import yaml

    from parikshak.eval.corpus import build_corpus
    doc = yaml.safe_load((ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
                         .read_text(encoding="utf-8"))
    entry = next(e for e in build_corpus(doc, seeds=1)
                 if e.scenario == "out_of_order_S09_before_S08" and e.profile == "moderate")
    engine = ProcedureEngine(csp1, run_id=entry.run_id)
    for f in entry.frames:
        engine.step(f)
    summary = engine.finish()
    assert any(d.step_id == "S08" and d.kind.value in {"SKIP", "OUT_OF_ORDER"}
               for d in summary.deviations), [str(d) for d in summary.deviations]


def test_a_recovery_is_only_explained_away_by_a_step_on_the_same_objects(csp1):
    """nominal, mild degradation: S07's evidence arrived in the very frame S08
    (close the latch) was recognised. Treating ANY later completion as the
    explanation blocked S07's recovery, and the SKIP reached the crew on a
    correct run. S08 shares no object with S07, so it cannot explain it."""
    import yaml

    from parikshak.eval.corpus import build_corpus
    doc = yaml.safe_load((ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
                         .read_text(encoding="utf-8"))
    entry = next(e for e in build_corpus(doc, seeds=1)
                 if e.scenario == "nominal" and e.profile == "mild")
    engine = ProcedureEngine(csp1, run_id=entry.run_id)
    for f in entry.frames:
        engine.step(f)
    summary = engine.finish()
    assert not [a for a in summary.alerts if a.kind.value == "SKIP"], \
        [a.reason for a in summary.alerts]


def test_a_hazard_seen_for_seconds_below_the_bar_is_still_raised():
    """CRX-2, harsh degradation: the vial is agitated 2-8 cm from the powered
    unit for about three seconds, but "near" reads only 0.70-0.82. Judged frame
    by frame against 0.85 the hazard was never raised; accumulated, it is."""
    import yaml

    from parikshak.eval.corpus import build_corpus
    from parikshak.eval.crx2 import corpus_scenarios
    path = ROOT / "procedures" / "crx2_colloid_resuspension.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = next(e for e in build_corpus(doc, seeds=1, scenario_list=corpus_scenarios())
                 if e.scenario == "hazard_vial_near_unit" and e.profile == "harsh")
    engine = ProcedureEngine(load_procedure(path), run_id=entry.run_id)
    for f in entry.frames:
        engine.step(f)
    summary = engine.finish()
    assert any(a.kind.value == "HAZARD" and a.step_id == "S04" for a in summary.alerts), \
        [str(d) for d in summary.deviations]


def test_timing_is_not_flagged_on_a_noisy_correct_run(csp1):
    """Durations run from the last sighting of the previous step still undone to
    the onset of this step's evidence - the longest reading consistent with
    observation - so detector noise alone does not make a correct run "rushed"."""
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    noisy = apply(frames, Perturbation("noisy", conf_scale=0.93, conf_jitter=0.04,
                                       pos_noise_m=0.008, drop_rate=0.05, seed=3))
    engine = ProcedureEngine(csp1, run_id="noisy")
    for f in noisy:
        engine.step(f)
    summary = engine.finish()
    assert not [d for d in summary.deviations if d.kind.value == "DURATION"], \
        [str(d) for d in summary.deviations]
