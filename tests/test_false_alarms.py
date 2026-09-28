"""Two false-alarm causes from the failure reel, fixed and pinned to traces.

Honest scoring on the 130-run degraded CSP-1 corpus put false alarms at 2.98 per
45 minutes - 14 alarms - against a target of 1. Two of the mechanisms behind
them, each confirmed by replaying the trace frame by frame, are fixed here.
Together they removed 5 of the 14, bringing the rate to 1.92:

  A. Evidence latched before the engine caught up was thrown away.
     `occurred` latches for a step were cleared when its PREDECESSOR was
     entered. In nominal__harsh_0 the crew's insert was seen at 30.0-33.8 s and
     latched, S04 was hidden so S05 was only entered at 34.4 s, and the latch
     was wiped one frame before S06 needed it. The insert never came again -
     it had already happened - so a correctly inserted cartridge was reported
     SKIPPED. All 4 false SKIPs of S06.

  B. One noisy frame moved the next step's start.
     A step's start is the last moment the step before it was SEEN still
     undone. In occlusion_S06__harsh_2 a single frame at 38.8 s read S06 as not
     done - the frames either side disagree - which moved S07's start from
     29.8 s to 38.8 s and made a correctly paced step read "3.8 s, below the 5 s
     minimum". 1 of the 5 false "too fast" alarms.

The other 9 have different causes, still open and shown in the failure reel:
S07 called skipped when position noise reads as motion (5), and "too fast"
claims within a frame or two of the minimum (4).

The last two tests are the guards: a fix that silences these by weakening
"skipped" or "too fast" in general would pass the first three and destroy the
recall the system is for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parikshak.belief.trace import read_trace
from parikshak.engine.predicates import PredicateEvaluator, RunFacts
from parikshak.engine.runner import ProcedureEngine, RunSummary
from parikshak.pdl import load_procedure
try:
    from tests.test_predicates import THR, frame
except ImportError:
    from test_predicates import THR, frame

ROOT = Path(__file__).resolve().parent.parent
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
INSERT = {"occurred": {"motion_is": {"motion_class": "insert", "min_conf": 0.55}}}


@pytest.fixture(scope="module")
def proc():
    return load_procedure(CSP1)


def replay(trace: Path, procedure) -> RunSummary:
    _, frames = read_trace(trace)
    engine = ProcedureEngine(procedure, run_id=trace.stem)
    for f in frames:
        engine.step(f)
    return engine.finish()


def alerts(summary: RunSummary, kind: str, step: str) -> list:
    return [a for a in summary.alerts if a.kind.value == kind and a.step_id == step]


# -- A: the latch window ---------------------------------------------------
def test_an_action_seen_before_the_step_was_entered_still_counts(proc):
    """The engine reaches a step later than the crew does. Evidence gathered in
    between is the crew doing the step, not noise from an earlier one."""
    ev = PredicateEvaluator(proc)
    facts = RunFacts(current_step="S06")
    ev.ingest(frame(10.0, motion=("insert", 0.9)))
    assert ev.evaluate(INSERT, facts, "v").satisfied(THR)

    # S06 is entered at 12.0 s, but it could have begun at 9.0 - the moment its
    # predecessor was last seen undone. The insert at 10.0 is inside that.
    ev.begin_step("S06", since=9.0)
    ev.ingest(frame(12.0, motion=("idle", 0.9)))
    assert ev.evaluate(INSERT, facts, "v").satisfied(THR)


def test_an_action_from_before_the_step_could_have_begun_is_not_evidence(proc):
    """The bound that makes the latch honest: a press during step 4 must not
    satisfy step 9."""
    ev = PredicateEvaluator(proc)
    facts = RunFacts(current_step="S06")
    ev.ingest(frame(10.0, motion=("insert", 0.9)))
    assert ev.evaluate(INSERT, facts, "v").satisfied(THR)

    ev.begin_step("S06", since=11.0)
    ev.ingest(frame(12.0, motion=("idle", 0.9)))
    assert not ev.evaluate(INSERT, facts, "v").satisfied(THR)


def test_a_correct_insert_is_not_reported_skipped(proc):
    """nominal__harsh_0: nothing was injected, so every alert on it is false."""
    summary = replay(ROOT / "traces" / "eval" / "nominal__harsh_0.jsonl", proc)
    assert summary.status["S06"] == "COMPLETE", summary.status
    assert not summary.alerts, [str(a) for a in summary.alerts]


# -- B: one frame must not move a start ------------------------------------
def test_one_contrary_frame_does_not_make_the_next_step_look_rushed(proc):
    summary = replay(ROOT / "traces" / "eval" / "occlusion_S06__harsh_2.jsonl", proc)
    assert not alerts(summary, "DURATION", "S07"), [str(a) for a in summary.alerts]


def test_a_step_one_frame_under_its_minimum_is_not_called_too_fast(proc):
    """duration_rush_S12__mild_0: nothing was injected at S07, and it used to
    measure 4.8 s against a 5 s minimum. A uniform one-observation margin was
    tried and rejected - it also silenced the genuinely rushed S12 on a clean
    run, which measures just as close to its own minimum. What cleared this one
    is fix C below: with jitter no longer read as movement, S07's evidence
    begins where the vial is actually seated, and the step measures its length."""
    summary = replay(ROOT / "traces" / "eval" / "duration_rush_S12__mild_0.jsonl", proc)
    assert not alerts(summary, "DURATION", "S07"), [str(a) for a in summary.alerts]


# -- C: jitter is not movement ---------------------------------------------
STILL = {"stable": {"entity": "vial_a", "seconds": 2.0, "tol_m": 0.03}}


def _track(seed: int, speed_mps: float, sigma_m: float, seconds: float = 6.0, fps: float = 5.0):
    """Vial A at a fixed point (or moving along x), seen with per-axis noise."""
    import random

    from tests.test_predicates import obj

    rng = random.Random(seed)
    frames = []
    for i in range(int(seconds * fps)):
        t = i / fps
        pos = (0.20 + speed_mps * t + rng.gauss(0, sigma_m),
               0.04 + rng.gauss(0, sigma_m), 0.12 + rng.gauss(0, sigma_m))
        frames.append(frame(t, objects={"vial_a": obj(pos)}))
    return frames


def test_jitter_alone_is_never_read_as_movement(proc):
    """A stationary vial under 2 cm of per-axis jitter. Its half-window centroids
    differ by 3-4 cm by chance alone, which read as a definite 'it moved' - and
    in skip_S13__harsh_1 that refuted S07 for seconds and a correctly attached
    vial was reported SKIPPED. Movement is only proven when the drift exceeds the
    tolerance by more than the window's own noise can explain."""
    for seed in range(20):
        ev = PredicateEvaluator(proc)
        facts = RunFacts()
        for f in _track(seed, 0.0, 0.02):
            ev.ingest(f)
            value = ev.evaluate(STILL, facts)
            assert not value.refuted(THR), (seed, f.t_mono, value)


def test_an_object_floating_away_is_still_caught(proc):
    """The guard: 15 cm/s under the same noise is movement, and must read so."""
    ev = PredicateEvaluator(proc)
    facts = RunFacts()
    refuted = False
    for f in _track(1, 0.15, 0.015):
        ev.ingest(f)
        refuted = refuted or ev.evaluate(STILL, facts).refuted(THR)
    assert refuted


@pytest.mark.xfail(strict=True, reason=(
    "Known limitation, kept as evidence. With jitter no longer read as movement "
    "the noise-driven refutations are gone, but S07 is still refuted by the vial "
    "genuinely being carried into place (39.0-41.6 s), and the crew closes the "
    "latch 4.8 s later - 0.2 s inside S07's 5 s minimum, which is the recency "
    "rule's line for calling a skip. Moving that line is the same kind of blanket "
    "margin that silenced a real rush, so it stays."))
def test_an_attached_vial_under_noise_is_not_called_skipped(proc):
    summary = replay(ROOT / "traces" / "eval" / "skip_S13__harsh_1.jsonl", proc)
    assert not alerts(summary, "SKIP", "S07"), [str(a) for a in summary.alerts]


# -- the guards ------------------------------------------------------------
def test_a_genuinely_rushed_step_is_still_flagged(proc):
    summary = replay(ROOT / "traces" / "eval" / "duration_rush_S12__clean.jsonl", proc)
    assert alerts(summary, "DURATION", "S12"), [str(a) for a in summary.alerts]


def test_a_latch_left_open_is_still_reported_skipped(proc):
    summary = replay(ROOT / "traces" / "golden" / "skip_S08_latch.jsonl", proc)
    assert alerts(summary, "SKIP", "S08"), [str(a) for a in summary.alerts]
