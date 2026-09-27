"""The eval layer: synthesis, degradation, scoring.

The harness only means anything if its own scoring is trustworthy, so most of
these tests are about the definitions rather than the machinery - what counts
as a false alarm, what counts as a cascade, and the guarantee that degrading a
run can never turn a correct verdict into an accusation.

The corpus itself is generated on demand rather than read from disk: it is
~300 MB and reproducible from its generator, so it is not committed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from parikshak.belief.trace import read_trace, write_trace
from parikshak.engine.runner import ProcedureEngine
from parikshak.eval.corpus import build_corpus, scenarios
from parikshak.eval.harness import EvalReport, evaluate, score_run
from parikshak.eval.perturb import PROFILES_BY_NAME, Perturbation, apply
from parikshak.eval.synth import RunBuilder, run
from parikshak.pdl import load_procedure

ROOT = Path(__file__).resolve().parent.parent
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"

#: Tightest geometric tolerance CSP-1 asks for. Bounds the centroid noise
#: the procedure can be evaluated under - see the validator's precision check.
TIGHTEST_TOLERANCE_M = 0.03


@pytest.fixture(scope="module")
def proc():
    return load_procedure(CSP1)


@pytest.fixture(scope="module")
def doc():
    return yaml.safe_load(CSP1.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clean_frames(doc):
    return run(doc).frames


# -- synthesis -------------------------------------------------------------
def test_a_scripted_run_produces_frames(clean_frames):
    assert len(clean_frames) > 500
    assert clean_frames[0].t_mono == 0.0
    assert clean_frames[-1].t_mono > 100


def test_synthesis_is_deterministic(doc):
    """Same script, same bytes. Without this the corpus cannot be a regression
    corpus - every diff would be noise."""
    a = [f.to_json() for f in run(doc).frames]
    b = [f.to_json() for f in run(doc).frames]
    assert a == b


def test_track_ids_do_not_depend_on_the_process(doc):
    """Guards the hash() defect: Python randomises string hashing per process."""
    ids = {n: o.track_id for n, o in run(doc).frames[0].objects.items()}
    assert ids == {n: o.track_id for n, o in run(doc).frames[0].objects.items()}
    assert all(v is not None for v in ids.values())


def test_a_stowed_bag_carries_the_cartridge_sealed_in_it(doc):
    """Stowing used to move the bag and leave its sealed-in cartridge floating
    at the workspace, which made S13's `inside` false as soon as S14 began and
    turned an occlusion-delayed S13 into a false SKIP."""
    last = run(doc).frames[-1].objects
    assert last["cartridge_sc_a"].pos_rack == last["sample_bag_sb01"].pos_rack


def test_a_cartridge_never_bagged_is_not_stowed_with_the_bag(doc):
    last = run(doc, skip={"S13"}).frames[-1].objects
    assert last["cartridge_sc_a"].pos_rack != last["sample_bag_sb01"].pos_rack


def test_a_corpus_session_outlasts_its_final_steps_verification(doc, proc):
    """CSP-1 S14 cannot verify in under 6 s, and the corpus used to stop recording
    2 s after it: correctly performed final steps scored as UNVERIFIED, and a
    skipped S13 was never passed. That measured truncation, not perception."""
    from parikshak.eval.corpus import (RECOGNITION_ALLOWANCE_S, session_tail_s,
                                       terminal_verification)
    from parikshak.pdl.validator import min_satisfaction_time
    floor = min_satisfaction_time(terminal_verification(doc))
    assert session_tail_s(doc) >= floor + RECOGNITION_ALLOWANCE_S

    entry = next(e for e in build_corpus(doc, seeds=1)
                 if e.scenario == "nominal" and e.profile == "harsh")
    engine = ProcedureEngine(proc, run_id=entry.run_id)
    for f in entry.frames:
        engine.step(f)
    assert engine.finish().status["S14"] == "COMPLETE"


# -- degradation -----------------------------------------------------------
def test_dropped_video_frames_do_not_drop_what_the_crew_said(clean_frames):
    """A confirmation is a voice event, not a pixel. Dropping it with the frame
    it rode on made CRX-2's spoken steps vanish under harsh degradation."""
    said = [c for f in clean_frames for c in f.confirmations]
    assert said, "the nominal run should contain crew confirmations"
    degraded = apply(clean_frames, Perturbation("drops", drop_rate=0.5, seed=1))
    assert len(degraded) < len(clean_frames)
    assert [c for f in degraded for c in f.confirmations] == said


def test_clean_profile_is_the_identity(clean_frames):
    out = apply(clean_frames, PROFILES_BY_NAME["clean"])
    assert [f.to_json() for f in out] == [f.to_json() for f in clean_frames]


def test_degradation_is_reproducible_for_a_seed(clean_frames):
    p = Perturbation("x", conf_scale=0.9, conf_jitter=0.05, pos_noise_m=0.01,
                     drop_rate=0.1, occlusion_rate=0.2, seed=17)
    a = [f.to_json() for f in apply(clean_frames, p)]
    b = [f.to_json() for f in apply(clean_frames, p)]
    assert a == b


def test_different_seeds_give_different_traces(clean_frames):
    p = Perturbation("x", conf_jitter=0.05, pos_noise_m=0.01, drop_rate=0.1)
    a = [f.to_json() for f in apply(clean_frames, p.with_seed(1))]
    b = [f.to_json() for f in apply(clean_frames, p.with_seed(2))]
    assert a != b


def test_dropping_frames_shortens_the_trace(clean_frames):
    out = apply(clean_frames, Perturbation("d", drop_rate=0.2, seed=3))
    assert len(out) < len(clean_frames)


def test_degradation_never_revives_an_undetected_object(clean_frames):
    """A zero confidence means "not detected". Noise must not push it above
    zero, or the corpus contains detections the detector never made."""
    out = apply(clean_frames, Perturbation("j", conf_jitter=0.3, seed=5))
    for a, b in zip(clean_frames, out):
        for name, ob in a.objects.items():
            if ob.conf == 0.0 and name in b.objects:
                assert b.objects[name].conf == 0.0


def test_degraded_frames_still_satisfy_the_schema(clean_frames):
    """Whatever the corpus does to a run, the result must remain a valid
    BeliefFrame - otherwise the harness is measuring a format bug."""
    from parikshak.belief.frame import BeliefFrame
    for profile in PROFILES_BY_NAME.values():
        for f in apply(clean_frames, profile.with_seed(11))[:40]:
            assert BeliefFrame.from_json(f.to_json()) == f


def test_occlusion_arrives_in_runs_not_as_speckle(clean_frames):
    """A torso blocks the view for seconds, not for one frame in seven. The
    difference decides whether `hold_for` windows survive, so independent
    per-frame dropout would make the system look far more robust than it is."""
    out = apply(clean_frames, Perturbation("o", occlusion_rate=0.5, seed=2))
    occluded = [any(o.occluded for o in f.objects.values()) for f in out]
    runs, current = [], 0
    for flag in occluded:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    assert runs, "no occlusion produced at rate 0.5"
    assert max(runs) >= 4  # >= ~1 s at 5 fps


# -- the corpus ------------------------------------------------------------
def test_every_scenario_declares_its_ground_truth():
    """A scenario whose injected list is wrong makes every metric wrong."""
    for sc in scenarios():
        for inj in sc.injected:
            assert inj.kind in {"SKIP", "OUT_OF_ORDER", "WRONG_OBJECT",
                                "REPEAT", "DURATION", "HAZARD"}
            assert inj.step_id.startswith(("S", "F"))


def test_the_corpus_contains_runs_with_nothing_injected():
    """The false-alarm number is measured on these. Without them there is
    nothing to measure it against."""
    clean = [sc for sc in scenarios() if not sc.injected]
    names = {sc.name for sc in clean}
    assert "nominal" in names
    assert any("legal_reorder" in n for n in names)
    assert any(n.startswith(("occlusion", "frame_loss")) for n in names)


def test_corpus_build_is_reproducible(doc):
    a = build_corpus(doc, seeds=1)
    b = build_corpus(doc, seeds=1)
    assert [e.run_id for e in a] == [e.run_id for e in b]
    assert [f.to_json() for f in a[3].frames] == [f.to_json() for f in b[3].frames]


# -- scoring ---------------------------------------------------------------
def _score(proc, frames, extra):
    engine = ProcedureEngine(proc, run_id="t")
    for f in frames:
        engine.step(f)
    return score_run(engine.finish(), extra, frames)


def test_an_alert_on_a_clean_run_is_a_false_alarm(proc, clean_frames):
    """Nothing injected, so anything said is a false alarm by definition."""
    degraded = apply(clean_frames, PROFILES_BY_NAME["harsh"].with_seed(4))
    score = _score(proc, degraded, {"injected": [], "expect": {}})
    assert len(score.false_positives) == len(score.alerted)
    assert score.cascade == ()


def test_downstream_alerts_are_cascade_not_false_alarms(proc, doc):
    """Skip "open the latch" and "insert the cartridge" genuinely cannot be
    verified either. Charging that to the false-alarm budget would make the
    number fall as we injected MORE errors, which is worse than useless."""
    frames = run(doc, skip={"S05"}).frames
    extra = {"injected": [{"kind": "SKIP", "step_id": "S05"}], "expect": {}}
    score = _score(proc, frames, extra)
    assert score.false_positives == ()
    assert ("SKIP", "S05") in score.true_positives


def test_nominal_run_scores_zero_false_alarms(proc, clean_frames):
    score = _score(proc, clean_frames, {"injected": [], "expect": {}})
    assert score.false_positives == ()
    assert score.false_alarms_per_45min == 0.0


def test_false_alarm_rate_is_pooled_over_clean_runs_only(proc):
    """Averaging per-run rates lets a ten-second run dominate a forty-minute
    one; and including deviation runs would count cascades twice."""
    report = EvalReport()
    assert report.false_alarms_per_45min == 0.0  # no runs, no claim


def test_missing_an_injected_deviation_is_a_miss(proc, doc):
    frames = run(doc).frames  # correct run...
    extra = {"injected": [{"kind": "SKIP", "step_id": "S08"}], "expect": {}}
    score = _score(proc, frames, extra)  # ...but ground truth claims a skip
    assert ("SKIP", "S08") in score.misses


def test_occlusion_fraction_is_measured(proc, clean_frames):
    clean = _score(proc, clean_frames, {"injected": [], "expect": {}})
    harsh = _score(proc, apply(clean_frames, PROFILES_BY_NAME["harsh"].with_seed(1)),
                   {"injected": [], "expect": {}})
    assert clean.occlusion_fraction < harsh.occlusion_fraction


# -- the guarantee ---------------------------------------------------------
def test_degradation_never_converts_a_completion_into_a_skip(proc, clean_frames):
    """The load-bearing property, measured rather than asserted.

    Losing information must be able to move a step out of COMPLETE and into
    UNVERIFIED, but never into SKIPPED. If a noisier camera can manufacture an
    accusation, every false-alarm number in the report is meaningless.

    Scoped to the noise the PROCEDURE claims to tolerate. That is not a
    loophole - it is the same rule the validator applies: `stable(tol_m)` holds
    above ~95% of the time only when tol_m >= 5 sigma, so a procedure whose
    tolerances are tighter than the detector's noise is misconfigured, and the
    validator says so at load time. Asserting the guarantee beyond that point
    would be testing a configuration the tooling already rejects.
    """
    from parikshak.pdl.validator import MIN_TOLERANCE_SIGMAS

    supported_sigma = TIGHTEST_TOLERANCE_M / MIN_TOLERANCE_SIGMAS
    checked = []
    for profile in PROFILES_BY_NAME.values():
        if profile.pos_noise_m > supported_sigma:
            continue
        for seed in (0, 1, 2):
            frames = apply(clean_frames, profile.with_seed(seed))
            engine = ProcedureEngine(proc, run_id=f"{profile.name}{seed}")
            for f in frames:
                engine.step(f)
            summary = engine.finish()
            checked.append(profile.name)
            assert not summary.skipped, (
                f"{profile.name} seed {seed}: {list(summary.skipped)} reported SKIPPED "
                f"on a run where the crew did everything correctly")
    assert checked, "no profile was within the procedure's declared precision"


def test_a_procedure_tolerance_below_the_noise_floor_is_flagged(doc):
    """The companion to the test above: when a procedure DOES ask for precision
    the detector cannot deliver, the validator warns instead of the engine
    silently reporting skips."""
    import copy

    from parikshak.pdl.validator import Report, check

    bad = copy.deepcopy(doc)
    bad["requires"]["position_sigma_m"] = 0.02  # far above what 0.03 m supports
    rep = Report()
    check(bad, rep)
    assert not rep.errors
    assert any("position_sigma_m" in w for w in rep.warnings)


def test_an_alarm_before_every_injected_step_is_false_not_a_cascade():
    """A rushed S12 cannot cause an alarm about S06, which came first. Filing it
    as a cascade kept real false alarms on deviation runs out of the budget."""
    from types import SimpleNamespace

    from parikshak.engine.deviations import DeviationKind
    from parikshak.engine.runner import RunSummary

    status = {sid: "COMPLETE" for sid in ("S05", "S06", "S12", "S13")}
    alert = lambda kind, sid, t: SimpleNamespace(kind=kind, step_id=sid, t=t)  # noqa: E731
    summary = RunSummary(
        run_id="rush", procedure_id="CSP-1", duration_s=140.0, frames=700, status=status,
        complete=tuple(status), skipped=(), unverified=(), deviations=(),
        alerts=(alert(DeviationKind.SKIP, "S06", 45.6),
                alert(DeviationKind.DURATION, "S12", 125.8),
                alert(DeviationKind.DURATION, "S13", 129.4)),
        notices=(), started_at={"S05": 20.0, "S06": 30.0, "S12": 123.4, "S13": 125.8})
    score = score_run(summary, {"injected": [{"kind": "DURATION", "step_id": "S12"}]}, [])

    assert score.true_positives == (("DURATION", "S12"),)
    assert score.false_positives == (("SKIP", "S06"),)
    assert score.cascade == (("DURATION", "S13"),)
    assert score.fa_window_s == pytest.approx(123.4)


def test_regate_at_the_procedures_own_thresholds_reproduces_the_engine(proc, doc, tmp_path):
    """The ROC is only honest if re-gating a cached replay gives the same
    alerts the live engine gives. The first re-gate never passed recoveries as
    retractions, and reported double the engine's false-alarm rate at the very
    thresholds in the procedure file. mild nominal is the run that exposed it:
    a SKIP on S07 is recovered inside the persistence window and never spoken."""
    from parikshak.eval.harness import regate, replay_all

    wanted = {("nominal", "mild"), ("legal_reorder_S03_S02", "harsh"),
              ("skip_S08", "moderate"), ("occlusion_S06", "harsh")}
    paths = []
    for e in build_corpus(doc, seeds=1):
        if (e.scenario, e.profile) in wanted:
            p = tmp_path / f"{e.run_id}.jsonl"
            write_trace(p, e.frames, procedure_id="CSP-1", run_id=e.run_id,
                        source="synthetic", notes=e.notes, extra=e.header_extra())
            paths.append(p)
    assert len(paths) == len(wanted)

    direct = evaluate(proc, paths)
    gated = regate(proc, replay_all(proc, paths))

    def shape(report):
        return [(s.run_id, s.false_positives, s.misses, s.cascade, s.latencies)
                for s in report.scores]

    assert shape(gated) == shape(direct)
    assert gated.false_alarms_per_45min == direct.false_alarms_per_45min


def test_harness_runs_end_to_end(proc, doc, tmp_path):
    entries = build_corpus(doc, seeds=1)[:6]
    paths = []
    for e in entries:
        p = tmp_path / f"{e.run_id}.jsonl"
        write_trace(p, e.frames, procedure_id="CSP-1", run_id=e.run_id,
                    source="synthetic", notes=e.notes, extra=e.header_extra())
        paths.append(p)
    report = evaluate(proc, paths)
    d = report.as_dict()
    assert d["runs"] == len(paths)
    assert 0.0 <= d["step_accuracy"] <= 1.0
    assert d["monitored_s"] > 0
    assert read_trace(paths[0])[0].extra["injected"] is not None
