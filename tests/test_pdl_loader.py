"""The PDL loader is the gate between an authored file and the engine.

Its contract: if you are holding a Procedure, the file passed validation. Every
"no defensive checks needed" assumption in engine/ rests on that, so the tests
that matter most here are the ones proving a bad file cannot get through.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from parikshak.pdl import Procedure, ProcedureError, Zone, load_procedure

ROOT = Path(__file__).resolve().parent.parent
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
TEMPLATE = ROOT / "procedures" / "_template.yaml"


@pytest.fixture(scope="module")
def proc() -> Procedure:
    return load_procedure(CSP1)


@pytest.fixture
def doc() -> dict:
    return yaml.safe_load(CSP1.read_text(encoding="utf-8"))


# -- loading ---------------------------------------------------------------
def test_csp1_loads_clean(proc):
    assert proc.id == "CSP-1"
    assert proc.pdl_version == "1.0"
    assert len(proc.steps) == 16
    assert proc.warnings == ()


def test_missing_file_is_a_procedure_error():
    with pytest.raises(ProcedureError, match="not found"):
        load_procedure(ROOT / "procedures" / "does_not_exist.yaml")


def test_the_template_is_refused(tmp_path):
    """PLAN.md section 20: the unmodified template fails by design. If it ever
    loads clean, the placeholders have stopped being placeholders."""
    with pytest.raises(ProcedureError):
        load_procedure(TEMPLATE)


def test_validation_cannot_be_skipped(doc, tmp_path):
    """A file with a predicate outside the closed vocabulary must never reach
    the engine, strict or not."""
    doc["flow"]["steps"][3]["verification"] = {"looks_right": {"entity": "vial_a"}}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    for strict in (True, False):
        with pytest.raises(ProcedureError, match="validation error"):
            load_procedure(p, strict=strict)


def test_unparseable_yaml_is_reported_as_such(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("procedure: [unclosed\n", encoding="utf-8")
    with pytest.raises(ProcedureError, match="YAML"):
        load_procedure(p)


# -- identity and hashing --------------------------------------------------
def test_source_hash_is_stable_and_content_addressed(proc):
    """The procedure hash goes into every log record. Two loads of the same file
    must agree, or the flight record cannot prove which procedure ran."""
    again = load_procedure(CSP1)
    assert proc.source_sha256 == again.source_sha256
    assert len(proc.source_sha256) == 64


def test_editing_the_file_changes_the_hash(doc, tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text(CSP1.read_text(encoding="utf-8"), encoding="utf-8")
    b = tmp_path / "b.yaml"
    doc["procedure"]["revision"] = 99
    b.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert load_procedure(a).source_sha256 != load_procedure(b).source_sha256


# -- graph -----------------------------------------------------------------
def test_group_members_may_run_in_either_order(proc):
    """The legal-reorder case. S02 and S03 are an unordered group, so each must
    accept the other as a successor. This is the false-alarm test that PLAN.md
    section 12 records three runs to prove."""
    assert "S03" in proc.successors("S02")
    assert "S02" in proc.successors("S03")


def test_group_id_is_not_leaked_as_a_step(proc):
    """`next: [G_GATHER]` names a group node, not a step. Callers want steps."""
    for sid in proc.order:
        for succ in proc.successors(sid):
            assert succ in proc.steps, f"{sid} -> {succ!r} is not a step"


def test_group_successor_is_reachable_from_its_members(proc):
    assert "S04" in proc.successors("S02")
    assert "S04" in proc.successors("S03")


def test_branch_targets_are_legal_successors(proc):
    """A hardware fault routing to F01 is a legal transition. If the engine did
    not know that, a fault would be reported as an out-of-order deviation and
    the crew would be blamed for a broken payload."""
    assert set(proc.successors("S10")) == {"S11", "F01"}


def test_terminals_have_no_successors(proc):
    assert proc.successors("S14") == ()
    assert proc.step("S14").is_terminal
    assert set(proc.exits) == {"S14", "FAULT_END"}


def test_no_step_is_its_own_successor(proc):
    for sid in proc.order:
        assert sid not in proc.successors(sid)


def test_expand_rejects_an_unknown_target(proc):
    with pytest.raises(ProcedureError, match="neither a step nor a group"):
        proc.expand("S99")


# -- declarations ----------------------------------------------------------
def test_critical_steps_are_the_ones_marked_critical(proc):
    assert proc.critical_steps == ("S08", "S13", "S14")


def test_invariants_are_loaded_with_severity(proc):
    """The hazard path: latch opened while the unit runs. No step is skipped, so
    only an invariant can catch it."""
    inv = proc.step("S09").invariants
    assert len(inv) == 1
    assert inv[0].severity == "critical"
    assert "latch" in inv[0].tts.lower()


def test_confusable_entities_are_declared(proc):
    """WRONG_OBJECT is a declared expectation, not an inference from low
    confidence."""
    assert proc.entity("vial_a").confusable_with == ("vial_b",)


def test_entities_speak_their_tts_name_not_their_key(proc):
    """Piper must say "vial A", never "vial_a"."""
    assert proc.entity("vial_a").spoken == "vial A"
    assert "_" not in proc.entity("cartridge_sc_a").spoken


def test_thresholds_come_from_alert_policy_only(proc):
    ap = proc.alert_policy
    assert ap.complete_threshold == 0.85
    assert ap.persistence_s == 1.5
    assert ap.unverified_after_s == 4.0
    assert ap.advisory_only is True
    assert "override" in ap.override_tokens


def test_frame_loss_policy_ends_in_unverified(proc):
    """Never a guess from a stale extrinsic."""
    assert proc.frame_loss.then == "UNVERIFIED"
    assert proc.frame_loss.max_hold_s == 3.0


def test_unknown_lookups_name_the_procedure(proc):
    with pytest.raises(ProcedureError, match="CSP-1"):
        proc.step("S99")
    with pytest.raises(ProcedureError, match="CSP-1"):
        proc.entity("wrench")
    with pytest.raises(ProcedureError, match="CSP-1"):
        proc.zone("airlock")


# -- zones -----------------------------------------------------------------
def test_box_zone_containment(proc):
    z = proc.zone("workspace")
    assert z.contains(z.centre) is True
    assert z.contains((5.0, 5.0, 5.0)) is False


def test_sphere_zone_containment(proc):
    z = proc.zone("start_button")
    cx, cy, cz = z.centre
    assert z.contains((cx, cy, cz)) is True
    assert z.contains((cx + z.radius * 0.5, cy, cz)) is True
    assert z.contains((cx + z.radius * 2, cy, cz)) is False


def test_margin_widens_a_zone(proc):
    z = proc.zone("holder_h1_seat")
    just_outside = (z.hi[0] + 0.01, z.centre[1], z.centre[2])
    assert z.contains(just_outside) is False
    assert z.contains(just_outside, margin_m=0.05) is True


def test_unknown_position_is_unknown_not_outside():
    """The single most important rule in the system, at its lowest level. If an
    occluded object read as "outside every zone", `not in_zone(...)` would
    become a confident deviation and an occluded camera would manufacture
    alerts."""
    z = Zone.from_dict("z", {"type": "box", "min": [0, 0, 0], "max": [1, 1, 1]})
    assert z.contains(None) is None
    assert z.contains(None) is not False
