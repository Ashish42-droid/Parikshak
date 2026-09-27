"""The engine, end to end, against the golden corpus.

The corpus is the specification. Each trace is a statement about what the crew
did and what the engine must therefore conclude, and its `expect` block travels
with the data rather than living in this file - so adding a case is a data
change, not a code change.

The two most important tests here are not the ones that catch deviations. They
are `test_nominal_run_raises_no_alerts` and `test_legal_reorder_raises_no_alerts`:
a system that cries wolf on a correct run gets its speaker taped over in week
one, and no accuracy number recovers from that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parikshak.belief.trace import read_trace
from parikshak.engine.alerts import Severity
from parikshak.engine.deviations import DeviationKind
from parikshak.engine.hsmm import DurationModel, StepStatus, StepTracker
from parikshak.engine.logger import read_log, verify_chain, verify_log_file
from parikshak.engine.runner import ProcedureEngine
from parikshak.pdl.loader import Duration
from parikshak.pdl import load_procedure
from parikshak.replay import check_expectations

ROOT = Path(__file__).resolve().parent.parent
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
GOLDEN = ROOT / "traces" / "golden"
TRACES = sorted(GOLDEN.glob("*.jsonl"))


@pytest.fixture(scope="module")
def proc():
    return load_procedure(CSP1)


def run_trace(proc, path, **kw):
    header, frames = read_trace(path)
    engine = ProcedureEngine(proc, run_id=header.run_id or path.stem, **kw)
    for f in frames:
        engine.step(f)
    return header, engine.finish()


@pytest.fixture(scope="module")
def runs(proc):
    """Every golden trace, replayed once. Module-scoped: the corpus is ~6000
    frames and there is no reason to replay it per test."""
    return {p.stem: run_trace(proc, p) for p in TRACES}


# -- the corpus is the specification --------------------------------------
@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_every_golden_trace_meets_its_own_expectations(runs, path):
    header, summary = runs[path.stem]
    problems = check_expectations(summary, header.expect)
    assert not problems, "\n  ".join([f"{path.stem}:"] + problems)


# -- the false-alarm requirement ------------------------------------------
def test_nominal_run_raises_no_alerts(runs):
    """A correct run must be silent. This is the number that decides adoption,
    and it is the one every other team's demo quietly fails."""
    _, summary = runs["nominal"]
    assert summary.alerts == (), [str(a) for a in summary.alerts]
    assert summary.deviations == (), [str(d) for d in summary.deviations]
    assert summary.alerts_per_45min == 0.0


def test_legal_reorder_raises_no_alerts(runs):
    """S02 and S03 are an unordered group, so doing them in either order is
    correct. PLAN.md section 12 records three runs specifically to prove this
    stays silent - it is the test nobody else writes."""
    _, summary = runs["legal_reorder_S03_S02"]
    assert summary.alerts == (), [str(a) for a in summary.alerts]
    assert len(summary.complete) == 14


def test_nominal_completes_every_step(runs):
    _, summary = runs["nominal"]
    assert len(summary.complete) == 14
    assert summary.skipped == ()
    assert summary.unverified == ()


# -- occlusion must never manufacture a deviation -------------------------
def test_occlusion_reports_unverified_not_skipped(runs):
    """The single most important rule in the system. A blocked camera says "I
    cannot see", never "you skipped it"."""
    _, summary = runs["occlusion_S06"]
    assert "S06" in summary.unverified
    assert "S06" not in summary.skipped
    assert not [d for d in summary.deviations
                if d.step_id == "S06" and d.kind is DeviationKind.SKIP]


def test_lost_rack_frame_reports_unverified_not_skipped(runs):
    """Losing AprilTag lock makes geometry unknown, not false."""
    _, summary = runs["frame_loss_S07"]
    assert "S07" in summary.unverified
    assert not [d for d in summary.deviations
                if d.step_id == "S07" and d.kind is DeviationKind.SKIP]


def test_unverified_notices_are_not_deviations(runs):
    """UNVERIFIED is the system declining to guess. Counting it as a false alarm
    would make the honest behaviour look like the failure mode."""
    _, summary = runs["occlusion_S06"]
    assert summary.notices
    assert all(n.step_id not in summary.skipped for n in summary.notices)


# -- the deviation taxonomy ------------------------------------------------
def test_skip_is_detected_with_a_named_reason(runs):
    _, summary = runs["skip_S08_latch"]
    skips = [d for d in summary.deviations if d.kind is DeviationKind.SKIP]
    assert [d.step_id for d in skips] == ["S08"]
    # The reason is the product. "Deviation detected" helps nobody on a rack.
    assert "state_is(latch, closed)" in skips[0].reason


def test_hazard_fires_when_no_step_was_skipped(runs):
    """The latch opened while the unit was running. Nothing was skipped and
    nothing was out of order, so sequence checking alone is blind to it."""
    _, summary = runs["hazard_latch_open_while_running"]
    hazards = [d for d in summary.deviations if d.kind is DeviationKind.HAZARD]
    assert "S10" in [d.step_id for d in hazards]
    assert len(summary.complete) == 14  # and it is NOT reported as a skip
    assert summary.skipped == ()


def test_wrong_object_names_the_object(runs):
    _, summary = runs["wrong_object_vial_b"]
    wrong = [d for d in summary.deviations if d.kind is DeviationKind.WRONG_OBJECT]
    assert [d.step_id for d in wrong] == ["S03"]
    assert wrong[0].entities == ("vial_b",)
    assert "vial B" in wrong[0].tts


def test_hardware_fault_branches_instead_of_blaming_the_crew(runs, proc):
    """A red indicator is a payload fault. Routing it to F01 rather than
    reporting a crew deviation is what keeps the crew trusting the system."""
    header, summary = runs["fault_branch_red_indicator"]
    assert header.expect["branch"] == "F01"
    assert not [d for d in summary.deviations if d.kind is DeviationKind.WRONG_OBJECT]


# -- alert policy ----------------------------------------------------------
def test_alert_latency_is_within_two_seconds(runs, proc):
    """PLAN.md section 2 targets alert latency <= 2 s after the deviation."""
    for name in ("skip_S08_latch", "skip_S13_seal", "wrong_object_vial_b"):
        _, summary = runs[name]
        for alert in summary.alerts:
            matching = [d for d in summary.deviations
                        if d.kind is alert.kind and d.step_id == alert.step_id]
            assert matching, f"{name}: alert with no originating deviation"
            latency = alert.t - min(d.t for d in matching)
            assert latency <= 2.0, f"{name}: {alert.kind.value} took {latency:.2f}s"


def test_alerts_wait_out_the_persistence_window(runs, proc):
    persistence = proc.alert_policy.persistence_s
    _, summary = runs["skip_S08_latch"]
    for alert in summary.alerts:
        origin = min(d.t for d in summary.deviations
                     if d.kind is alert.kind and d.step_id == alert.step_id)
        assert alert.t - origin >= persistence - 1e-6


def test_critical_hazards_reach_the_voice_channel(runs):
    _, summary = runs["hazard_latch_open_while_running"]
    critical = [a for a in summary.alerts if a.severity is Severity.CRITICAL]
    assert critical
    assert "voice" in critical[0].channels
    assert critical[0].requires_ack


def test_severity_ordering_is_comparable():
    assert Severity.CRITICAL > Severity.CAUTION > Severity.ADVISORY > Severity.INFO
    assert Severity.parse("caution") is Severity.CAUTION
    assert Severity.parse("nonsense") is Severity.ADVISORY  # never crash on bad input


def test_alert_budget_per_step_is_capped(proc, runs):
    """One confused step must not become a siren."""
    cap = proc.alert_policy.max_alerts_per_step
    for _name, (_h, summary) in runs.items():
        per_step: dict[str, int] = {}
        for a in summary.alerts:
            if a.severity is Severity.CRITICAL:
                continue  # criticals are deliberately exempt
            per_step[a.step_id] = per_step.get(a.step_id, 0) + 1
        assert all(n <= cap for n in per_step.values()), per_step


def test_advisory_only_is_declared(proc):
    """The system never commands hardware and never blocks the crew. It is
    certifiable as advisory, non-critical software precisely because of this."""
    assert proc.alert_policy.advisory_only is True


# -- duration priors -------------------------------------------------------
def test_duration_model_scores_overdue_monotonically():
    dm = DurationModel(Duration(min_s=5, nominal_s=30, sigma_s=10, max_s=120))
    assert dm.overdue(5) < dm.overdue(30) < dm.overdue(90)
    assert dm.overdue(30) == pytest.approx(0.5, abs=0.01)
    assert dm.overdue(200) == 1.0


def test_duration_model_flags_both_tails():
    dm = DurationModel(Duration(min_s=5, nominal_s=30, sigma_s=10, max_s=120))
    assert dm.too_fast(2) and not dm.too_fast(10)
    assert dm.too_slow(200) and not dm.too_slow(60)


def test_a_step_already_true_on_entry_is_not_a_timing_violation(runs):
    """Its elapsed time measures nothing the crew did, so calling it "too fast"
    is a false alarm - and false alarms are the metric that matters."""
    for _name, (_h, summary) in runs.items():
        for dev in summary.deviations:
            if dev.kind is DeviationKind.DURATION:
                assert "0.2s" not in dev.reason, dev.reason


# -- the flight record -----------------------------------------------------
def test_log_chain_verifies(proc, tmp_path):
    _, summary = run_trace(proc, GOLDEN / "skip_S08_latch.jsonl",
                           log_path=tmp_path / "run.jsonl")
    ok, msg = verify_log_file(summary.log_path)
    assert ok, msg


def test_altering_one_character_breaks_the_chain(proc, tmp_path):
    """This is the whole claim of a tamper-evident record. If a single edited
    reason string still verified, the log would be a summary, not evidence."""
    _, summary = run_trace(proc, GOLDEN / "skip_S08_latch.jsonl",
                           log_path=tmp_path / "run.jsonl")
    records = read_log(summary.log_path)
    target = next(i for i, r in enumerate(records) if r["kind"] == "deviation")
    records[target]["payload"]["reason"] += "."
    ok, msg = verify_chain(records)
    assert not ok
    assert str(target) in msg


def test_removing_a_record_breaks_the_chain(proc, tmp_path):
    """seq and prev_hash are inside the hashed body precisely so that deleting
    an inconvenient record cannot go unnoticed."""
    _, summary = run_trace(proc, GOLDEN / "skip_S08_latch.jsonl",
                           log_path=tmp_path / "run.jsonl")
    records = read_log(summary.log_path)
    del records[5]
    ok, _msg = verify_chain(records)
    assert not ok


def test_log_pins_the_procedure_and_operating_point(proc, tmp_path):
    """A record that does not say which procedure and which thresholds produced
    it cannot be audited six weeks later."""
    _, summary = run_trace(proc, GOLDEN / "nominal.jsonl",
                           log_path=tmp_path / "run.jsonl")
    head = read_log(summary.log_path)[0]["payload"]
    assert head["procedure_sha256"] == proc.source_sha256
    assert head["procedure_id"] == "CSP-1"
    assert "deviation_threshold" in head["operating_point"]
    assert head["belief_schema"] == "1.1"


def test_log_is_kilobytes_not_gigabytes(proc, tmp_path):
    """The compression argument: a structured log instead of the video."""
    _, summary = run_trace(proc, GOLDEN / "nominal.jsonl",
                           log_path=tmp_path / "run.jsonl")
    assert summary.log_path.stat().st_size < 40 * 1024


def test_log_has_a_human_mirror(proc, tmp_path):
    _, summary = run_trace(proc, GOLDEN / "skip_S08_latch.jsonl",
                           log_path=tmp_path / "run.jsonl")
    text = summary.log_path.with_suffix(".txt").read_text(encoding="utf-8")
    assert "PARIKSHAK run log" in text
    assert "S08" in text and "SKIPPED" in text


def test_unverified_is_logged_as_its_own_kind(proc, tmp_path):
    _, summary = run_trace(proc, GOLDEN / "occlusion_S06.jsonl",
                           log_path=tmp_path / "run.jsonl")
    kinds = {r["kind"] for r in read_log(summary.log_path)}
    assert "unverified" in kinds


# -- determinism -----------------------------------------------------------
@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_replay_is_deterministic(proc, path):
    """Same trace, same verdict, every time. Without this a regression corpus
    cannot tell a real behaviour change from noise."""
    _, a = run_trace(proc, path)
    _, b = run_trace(proc, path)
    assert a.status == b.status
    assert [str(d) for d in a.deviations] == [str(d) for d in b.deviations]
    assert [str(x) for x in a.alerts] == [str(x) for x in b.alerts]


# -- the tracker -----------------------------------------------------------
def test_frontier_covers_both_members_of_an_unordered_group(proc):
    tracker = StepTracker(proc)
    assert set(tracker.frontier) >= {"S01"}
    assert tracker.status("S01") is StepStatus.ACTIVE


def test_skipped_and_unverified_are_never_merged(runs):
    """They are different claims: "we watched and it did not happen" versus
    "we could not see". Merging them is exactly how an occluded camera
    manufactures a deviation."""
    for _name, (_h, summary) in runs.items():
        assert not set(summary.skipped) & set(summary.unverified)
