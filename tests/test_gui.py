"""The operator display: view model, overlay geometry, and presentation rules.

PySide6 is not installed here, and none of these tests need it. That is the
design: everything the crew sees is computed in `DisplayState` and
`build_overlay`, so "the screen shows X" is an assertion rather than a
screenshot somebody squinted at.

The presentation tests are not styling checks. Whether UNVERIFIED reads as a
failure, whether a critical hazard can scroll out of the feed, and whether the
checklist survives a projector that eats colour are all decisions about whether
the crew trusts the system.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from parikshak.belief.trace import read_trace
from parikshak.engine.runner import ProcedureEngine
from parikshak.gui.app import (
    HeadlessDisplay,
    format_clock,
    format_deviation,
    format_step_row,
    format_timeline,
    render_text,
    supports_unicode,
)
from parikshak.gui.overlay import build_overlay
from parikshak.gui.state import STATUS_GLYPH, DeviationRow, DisplayState, TimelineEvent
from parikshak.gui.theme import THEME, contrast_ratio, stylesheet
from parikshak.pdl import load_procedure
from parikshak.perception.rackframe import (
    CameraIntrinsics,
    RackFrameEstimator,
    TagLayout,
    TagObservation,
)

ROOT = Path(__file__).resolve().parent.parent
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
GOLDEN = ROOT / "traces" / "golden"


@pytest.fixture(scope="module")
def proc():
    return load_procedure(CSP1)


def run(proc, name: str):
    header, frames = read_trace(GOLDEN / f"{name}.jsonl")
    engine = ProcedureEngine(proc, run_id=name)
    last = None
    for f in frames:
        engine.step(f)
        last = f
    engine.finish()
    return engine, last


@pytest.fixture(scope="module")
def skip_state(proc):
    engine, frame = run(proc, "skip_S08_latch")
    return DisplayState.from_engine(engine, frame=frame, source="replay")


@pytest.fixture(scope="module")
def nominal_state(proc):
    engine, frame = run(proc, "nominal")
    return DisplayState.from_engine(engine, frame=frame, source="replay")


# ==========================================================================
# the view model
# ==========================================================================
def test_the_display_is_built_from_engine_state(skip_state):
    assert skip_state.procedure_id == "CSP-1"
    assert skip_state.run_id == "skip_S08_latch"
    assert skip_state.total_steps == 16
    assert skip_state.t > 100


def test_a_skipped_step_shows_as_not_done_with_its_reason(skip_state):
    row = next(s for s in skip_state.steps if s.step_id == "S08")
    assert row.status == "SKIPPED"
    assert row.status_word == "not done"
    assert "state_is(latch, closed)" in row.reason


def test_a_clean_run_shows_nothing_wrong(nominal_state):
    assert nominal_state.skipped == ()
    assert nominal_state.unverified == ()
    assert nominal_state.deviations == []
    assert nominal_state.worst_severity is None
    assert nominal_state.progress == 14 / 16     # F01/FAULT_END are never entered


def test_critical_steps_are_marked(skip_state):
    assert {s.step_id for s in skip_state.steps if s.critical} == {"S08", "S13", "S14"}


def test_deviations_carry_their_justification(skip_state):
    """"Deviation detected" is useless at a rack."""
    for dev in skip_state.deviations:
        assert dev.reason
        assert dev.step_id and dev.severity


def test_the_feed_puts_the_worst_first_not_the_newest(skip_state):
    """A critical hazard scrolling off the top because four advisories arrived
    after it is the failure this panel exists to prevent."""
    feed = skip_state.feed()
    assert feed[0].severity == "critical"
    assert [d.rank for d in feed] == sorted(d.rank for d in feed)


def test_the_feed_is_bounded(skip_state):
    assert len(skip_state.feed(limit=2)) <= 2


def test_acknowledged_deviations_stop_driving_the_banner(proc):
    engine, frame = run(proc, "skip_S08_latch")
    acked = [f"{d.step_id}:{d.kind.value}" for d in engine.detector.deviations]
    state = DisplayState.from_engine(engine, frame=frame, acknowledged=acked)
    assert state.worst_severity is None
    assert state.deviations, "the deviations are still on record, just acknowledged"


def test_a_step_that_completed_after_being_blind_is_flagged(proc):
    """A step can finish COMPLETE and still have been unverifiable for part of
    its life. That is the difference between a clean run and a lucky one."""
    engine, frame = run(proc, "occlusion_S06")
    state = DisplayState.from_engine(engine, frame=frame)
    s06 = next(s for s in state.steps if s.step_id == "S06")
    assert s06.unknown_s > 0
    assert s06.was_ever_unverified


def test_status_line_summarises_the_run(skip_state, nominal_state):
    assert "13/16 complete" in skip_state.status_line()
    assert "1 not done" in skip_state.status_line()
    assert "not done" not in nominal_state.status_line()


def test_status_line_warns_when_the_rack_frame_is_lost(proc):
    engine, _ = run(proc, "nominal")
    state = DisplayState.from_engine(engine, frame=None)
    assert "NO RACK LOCK" in state.status_line()


def test_the_display_serialises_for_export(skip_state):
    d = skip_state.as_dict()
    assert d["run_id"] == "skip_S08_latch"
    assert d["skipped"] == ["S08"]
    assert all("reason" in dev for dev in d["deviations"])
    json.dumps(d)          # must be serialisable, it is the export button


def test_replay_and_live_produce_the_same_shape(proc):
    """The stage failover. If a camera fails, the screen is built from a file
    and looks the same, because the widgets never learn the source."""
    engine, frame = run(proc, "nominal")
    a = DisplayState.from_engine(engine, frame=frame, source="live")
    b = DisplayState.from_engine(engine, frame=frame, source="replay")
    assert a.as_dict()["status"] == b.as_dict()["status"]
    assert [s.status for s in a.steps] == [s.status for s in b.steps]


# ==========================================================================
# presentation rules
# ==========================================================================
def test_every_status_has_a_glyph_and_a_word_not_just_a_colour():
    """Roughly one man in twelve has a colour vision deficiency, and venue
    projectors crush saturation. A checklist that means nothing under either is
    not a checklist."""
    for status, (glyph, ascii_glyph, word) in STATUS_GLYPH.items():
        assert glyph and ascii_glyph and word, status
        assert THEME.status(status) != THEME.status("__unknown__")


def test_unverified_is_not_shown_in_the_failure_colour():
    """Showing "I cannot see" in the same red as "you skipped it" would train
    the crew to read the system's restraint as its failure."""
    assert THEME.status("UNVERIFIED") != THEME.status("SKIPPED")


def test_status_colours_are_readable_on_the_background():
    """WCAG AA wants 4.5:1 for body text. "Looks fine on my laptop" is not a
    standard, and the venue projector will not be my laptop."""
    for status in STATUS_GLYPH:
        ratio = contrast_ratio(THEME.status(status))
        assert ratio >= 4.5, f"{status} at {ratio:.1f}:1"


def test_severity_colours_are_readable():
    for severity in ("critical", "caution", "advisory", "info"):
        assert contrast_ratio(THEME.severity(severity)) >= 4.5, severity


def test_the_active_step_is_the_most_prominent_colour():
    """The eye should find "what do I do now" first."""
    assert contrast_ratio(THEME.status("ACTIVE")) > contrast_ratio(THEME.status("PENDING"))


def test_the_stylesheet_is_plain_text():
    css = stylesheet()
    assert THEME.background in css and "QLabel#prompt" in css


# ==========================================================================
# text rendering
# ==========================================================================
def test_render_survives_a_console_that_cannot_encode_glyphs():
    """A cp1252 Windows terminal and a Jetson serial console both refuse U+2713.
    A display that dies mid-run with UnicodeEncodeError is not a display."""
    assert not supports_unicode(io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    assert supports_unicode(io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))


def test_ascii_rendering_contains_no_high_codepoints(skip_state):
    text = render_text(skip_state, ascii_only=True)
    assert text.isascii(), [c for c in text if not c.isascii()][:5]
    text.encode("cp1252")           # must not raise


def test_the_rendered_display_names_what_went_wrong(skip_state):
    text = render_text(skip_state, ascii_only=True)
    assert "S08" in text
    assert "DEVIATIONS" in text
    assert "state_is(latch, closed)" in text


def test_the_rendered_display_leads_with_the_next_step(proc):
    engine, frame = run(proc, "nominal")
    engine.tracker.finished = False        # pretend the run is still going
    state = DisplayState.from_engine(engine, frame=frame)
    state.prompt = "Close the glovebox latch."
    assert "NEXT: Close the glovebox latch." in render_text(state, ascii_only=True)


def test_clock_formatting():
    assert format_clock(0.0) == "00:00.0"
    assert format_clock(66.5) == "01:06.5"


def test_a_checklist_row_leads_with_its_glyph():
    from parikshak.gui.state import StepRow
    row = StepRow("S08", "Close glovebox latch", "SKIPPED", True, False,
                  t_start=43.0, t_end=46.0)
    line = format_step_row(row, ascii_only=True)
    assert line.startswith("X")
    assert "S08" in line and "3.0s" in line


def test_a_step_this_setup_cannot_verify_says_so_on_its_row():
    from parikshak.gui.state import StepRow
    row = StepRow("S04", "Resuspend vial B by agitation", "ACTIVE", False, True,
                  t_start=10.0, unverifiable="motion_count needs motion_tcn")
    assert "(not verifiable here)" in format_step_row(row, ascii_only=True)


def test_missing_capability_is_not_reported_as_a_blocked_camera(proc):
    """Waiting on a model the box does not have is not blindness. The screen
    must say which it is, and never "cannot verify" while the camera sees."""
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    engine = ProcedureEngine(proc, run_id="nominal")
    for f in frames:
        if f.t_mono > 40.0:
            break
        engine.step(f)
        frame = f
    active = engine.active_step
    assert active is not None
    state = DisplayState.from_engine(
        engine, frame=frame, unverifiable={active: "motion_is needs motion_tcn"})
    assert not state.unverified_notice.startswith("Cannot verify")
    assert "not verifiable with this setup" in state.unverified_notice
    assert state.as_dict()["unverifiable_here"] == {active: "motion_is needs motion_tcn"}


@pytest.mark.parametrize("image, view, expected", [
    ((1280, 720), (1280, 720), (1.0, 0.0, 0.0)),
    ((1280, 720), (640, 720), (0.5, 0.0, 180.0)),     # pillar-boxed top and bottom
    ((1280, 720), (1920, 720), (1.0, 320.0, 0.0)),     # letter-boxed left and right
])
def test_the_overlay_and_the_image_share_one_letterbox(image, view, expected):
    from parikshak.gui.app import letterbox
    assert letterbox(*image, *view) == pytest.approx(expected)


def test_timeline_marks_share_one_scale_and_put_deviations_on_top():
    from parikshak.gui.app import timeline_marks
    events = [TimelineEvent(10.0, "completed", "S01"),
              TimelineEvent(50.0, "hazard", "S02", severity="critical"),
              TimelineEvent(100.0, "completed", "S03")]
    marks = timeline_marks(events, duration=100.0, width=200.0)
    assert [m.x for m in marks if not m.is_deviation] == [20.0, 200.0]
    assert marks[-1].is_deviation and marks[-1].x == 100.0
    # An event past the stated duration stretches the scale instead of
    # falling off the strip.
    late = timeline_marks([TimelineEvent(150.0, "completed", "S04")], duration=100.0, width=200.0)
    assert late[0].x == 200.0


def test_the_banner_shows_the_worst_open_deviation_until_it_is_acknowledged(skip_state):
    from parikshak.gui.app import banner_text, deviation_key
    text, severity = banner_text(skip_state)
    assert severity == skip_state.feed()[0].severity
    assert skip_state.feed()[0].step_id in text
    everything = [deviation_key(d) for d in skip_state.deviations]
    assert banner_text(skip_state, everything) == ("No open deviations", None)


def test_export_writes_what_is_on_screen(skip_state, tmp_path):
    from parikshak.gui.app import export_state
    path = export_state(skip_state, tmp_path, ["S08:SKIP"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "skip_S08_latch"
    assert payload["acknowledged"] == ["S08:SKIP"]
    assert payload["skipped"] == ["S08"]


def test_timeline_never_hides_a_deviation_behind_a_step_boundary():
    """Both land in the same cell at coarse widths. The deviation must win."""
    events = [TimelineEvent(10.0, "completed", "S01"),
              TimelineEvent(10.1, "hazard", "S01", severity="critical")]
    assert "!" in format_timeline(events, width=10, duration=20.0)


def test_timeline_of_an_empty_run_is_blank():
    assert set(format_timeline([], width=12)) == {"."}


def test_deviation_line_carries_time_severity_and_reason():
    row = DeviationRow(46.0, "SKIP", "S08", "caution", "latch never closed")
    line = format_deviation(row)
    assert "00:46.0" in line and "CAUTION" in line and "latch never closed" in line


# ==========================================================================
# headless display and export
# ==========================================================================
def test_headless_display_renders_and_exports(skip_state, tmp_path):
    out = []
    display = HeadlessDisplay(sink=out.append)
    display.update(skip_state)
    display.show()
    assert out and "CSP-1" in out[0]

    path = display.export(tmp_path / "view.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["skipped"] == ["S08"]
    assert b"\r\n" not in path.read_bytes()


def test_headless_display_before_any_state():
    assert "no state" in HeadlessDisplay().render()


# ==========================================================================
# overlay geometry
# ==========================================================================
@pytest.fixture(scope="module")
def rack_setup():
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    intrinsics = CameraIntrinsics.from_fov(1600, 1200, 65.0)
    R = np.diag([1.0, -1.0, -1.0])
    t = np.array([0.0, 0.0, 1.4])
    ids = layout.tag_ids
    obj = layout.object_points(ids)
    px = intrinsics.project((R @ obj.T).T + t)
    obs = [TagObservation(tid, px[4 * i:4 * i + 4]) for i, tid in enumerate(ids)]
    ext = RackFrameEstimator(layout, intrinsics).update(obs, 0.0)
    return ext, intrinsics


def test_no_rack_lock_means_no_overlay(proc, rack_setup):
    """Drawing the last known zone positions over a rack that has since moved
    is the stale-extrinsic bug, rendered."""
    _ext, intrinsics = rack_setup
    _, frames = read_trace(GOLDEN / "frame_loss_S07.jsonl")
    unlocked = next(f for f in frames if not f.frame_lock)
    scene = build_overlay(proc, unlocked, None, intrinsics)
    assert scene.is_empty
    assert "NO RACK LOCK" in scene.notice


def test_zones_and_axes_are_drawn_when_locked(proc, rack_setup):
    ext, intrinsics = rack_setup
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    scene = build_overlay(proc, frames[100], ext, intrinsics)
    kinds = scene.kinds()
    assert "zone" in kinds
    assert "axis" in kinds, "the rack axes are the orientation-agnostic beat"
    assert not scene.notice


def test_the_active_step_zones_are_highlighted(proc, rack_setup):
    ext, intrinsics = rack_setup
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    scene = build_overlay(proc, frames[100], ext, intrinsics,
                          active_zones=["glovebox_interior"])
    assert any(p.kind == "zone_active" for p in scene.polylines)


def test_objects_are_marked_with_their_state(proc, rack_setup):
    ext, intrinsics = rack_setup
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    scene = build_overlay(proc, frames[100], ext, intrinsics)
    latch = next(m for m in scene.markers if "latch" in m.label.lower())
    assert "closed" in latch.detail


def test_a_held_object_is_drawn_differently(proc, rack_setup):
    ext, intrinsics = rack_setup
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    held = next((f for f in frames
                 if any(o.held_by for o in f.objects.values())), None)
    assert held is not None
    scene = build_overlay(proc, held, ext, intrinsics)
    assert any(m.kind == "object_held" for m in scene.markers)


def test_the_overlay_follows_the_rack_not_the_screen(proc, rack_setup):
    """Demo beat three: rotate the rack and the overlay stays locked to it.

    The zone MOVES on screen because the camera's view of it changed - that is
    the visible part - while the rack coordinates the engine reasons about do
    not change at all.
    """
    _ext, intrinsics = rack_setup
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    frame = frames[100]

    def scene_at(deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        Rw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R = np.diag([1.0, -1.0, -1.0]) @ Rw.T
        t = np.array([0.0, 0.0, 1.4])
        ids = layout.tag_ids
        px = intrinsics.project((R @ layout.object_points(ids).T).T + t)
        obs = [TagObservation(tid, px[4 * i:4 * i + 4]) for i, tid in enumerate(ids)]
        ext = RackFrameEstimator(layout, intrinsics).update(obs, 0.0)
        return build_overlay(proc, frame, ext, intrinsics)

    upright, rotated = scene_at(0), scene_at(30)
    assert not upright.is_empty and not rotated.is_empty
    # Same things drawn, different pixels - the rack moved, the model did not.
    assert upright.kinds() == rotated.kinds()
    assert upright.polylines[0].points != rotated.polylines[0].points


def test_geometry_behind_the_camera_is_clipped(proc, rack_setup):
    """An unclipped segment with one endpoint behind the lens draws a diagonal
    streak across the whole view."""
    _ext, intrinsics = rack_setup
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    ids = layout.tag_ids
    R = np.diag([1.0, -1.0, -1.0])
    t = np.array([0.0, 0.0, 0.30])          # camera pushed in close to the face
    px = intrinsics.project((R @ layout.object_points(ids).T).T + t)
    obs = [TagObservation(tid, px[4 * i:4 * i + 4]) for i, tid in enumerate(ids)]
    ext = RackFrameEstimator(layout, intrinsics).update(obs, 0.0)
    scene = build_overlay(proc, frames[100], ext, intrinsics)
    for line in scene.polylines:
        for x, y in line.points:
            assert np.isfinite(x) and np.isfinite(y)
