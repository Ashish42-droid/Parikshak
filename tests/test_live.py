"""The live path, on a synthetic camera with known ground truth.

A live run is the replay path with a camera attached. These tests drive the
whole front half - rendered frames, tag detection, rack lock, marker props,
crew phrases - into the unchanged engine, and hold it to the promises that
matter most when models are missing: nothing it cannot see becomes evidence,
and what it recorded replays to the same verdicts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")

from parikshak.belief.frame import BeliefFrame  # noqa: E402
from parikshak.belief.trace import read_trace  # noqa: E402
from parikshak.engine.runner import ProcedureEngine  # noqa: E402
from parikshak.eval.scene import TagScene  # noqa: E402
from parikshak.io.camera import ScriptedProp, SyntheticCamera  # noqa: E402
from parikshak.io.tts import NullVoice, SpeechQueue  # noqa: E402
from parikshak.live import LiveSession, capability_gaps, MARKER_CAPABILITIES  # noqa: E402
from parikshak.pdl import load_procedure  # noqa: E402
from parikshak.perception.markers import PropMarkers  # noqa: E402
from parikshak.perception.rackframe import TagLayout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
CRX2 = ROOT / "procedures" / "crx2_colloid_resuspension.yaml"
PROPS = PropMarkers.load(ROOT / "racks" / "props_crx2.json")


@pytest.fixture(scope="module")
def crx2():
    return load_procedure(CRX2)


def camera(duration=16.0, blackouts=()):
    """The rack face with vial B stowed in the cold locker, the latch closed and
    the unit idle - the world CRX-2 S03 asks the crew to confirm."""
    scene = TagScene.facing_rack(width=960, height=540)
    props = [ScriptedProp(203, 0.0, 1e9, (-0.48, 0.21, 0.03)),
             ScriptedProp(211, 0.0, 1e9, (0.10, 0.00, 0.10)),
             ScriptedProp(231, 0.0, 1e9, (0.30, 0.10, 0.15))]
    return SyntheticCamera(scene, LAYOUT, props, tag_size_m=PROPS.tag_size_m, fps=5.0,
                           blackouts=list(blackouts), duration_s=duration)


def session(procedure, source, **kw):
    return LiveSession(procedure, source, LAYOUT, props=PROPS, run_id="live-test",
                       speech=SpeechQueue(NullVoice()), fps=5.0, **kw)


def drive(live, say_at=None, text=""):
    ticks = []
    while True:
        if say_at is not None and live.source.now() >= say_at and text:
            live.submit(text)
            text = ""
        tick = live.tick()
        if tick is None:
            return ticks
        ticks.append(tick)


def test_the_setup_names_the_steps_it_cannot_verify(crx2):
    """CRX-2 S04 counts agitations with a motion model and checks the vial is
    held; a camera with markers has neither. S03 is a spoken confirmation plus
    object states, all of which it can answer."""
    gaps = {g.step_id: g for g in capability_gaps(crx2, MARKER_CAPABILITIES)}
    assert "S04" in gaps and any("motion_tcn" in need for need in gaps["S04"].needs)
    assert "S03" not in gaps


def test_an_entity_with_no_tag_is_left_unobserved_not_absent(crx2):
    live = session(crx2, camera(duration=0.4))
    assert "crew_hand" in live.unobserved
    tick = live.tick()
    assert "crew_hand" not in tick.frame.objects


def test_a_live_run_locks_sees_its_props_and_raises_nothing_it_cannot_justify(crx2):
    live = session(crx2, camera())
    ticks = drive(live)
    summary = live.finish()
    assert any(t.frame.frame_lock for t in ticks)
    vial = ticks[-1].frame.objects["vial_b"]
    assert vial.visible and vial.pos_rack is not None
    # Steps needing hands, motion or body pose read UNKNOWN - never a skip.
    assert summary.alerts == ()
    assert not summary.skipped


def test_a_typed_crew_phrase_confirms_its_step(crx2):
    live = session(crx2, camera())
    drive(live, say_at=4.0, text="unit idle")
    summary = live.finish()
    assert summary.status["S03"] == "COMPLETE", summary.status
    assert summary.alerts == ()


def test_words_outside_the_grammar_confirm_nothing(crx2):
    live = session(crx2, camera(duration=1.0))
    assert live.submit("launch the rocket") is None
    ticks = drive(live)
    assert all(t.frame.confirmations == () for t in ticks)


def test_a_blocked_camera_drops_the_lock_after_the_hold_and_recovers(crx2):
    live = session(crx2, camera(duration=16.0, blackouts=[(4.0, 10.0)]))
    ticks = drive(live)
    lock = {round(t.frame.t_mono, 1): t.frame.frame_lock for t in ticks}
    assert lock[3.0] is True
    assert lock[9.0] is False          # blind for 5 s, past the 3 s hold
    assert lock[12.0] is True
    live.finish()


def test_a_recorded_live_run_replays_to_the_same_verdicts(crx2, tmp_path):
    """A live run is a regression case: its trace, replayed through a fresh
    engine, must reach exactly the verdicts the live engine reached."""
    live = session(crx2, camera())
    drive(live, say_at=4.0, text="unit idle")
    trace = tmp_path / "live.jsonl"
    summary = live.finish(trace_path=trace)

    header, frames = read_trace(trace)
    assert header.source == "recorded" and len(frames) == summary.frames
    replay = ProcedureEngine(crx2, run_id="replay")
    for f in frames:
        replay.step(f)
    again = replay.finish()
    assert again.status == summary.status
    assert [str(d) for d in again.deviations] == [str(d) for d in summary.deviations]


def test_a_streamed_live_trace_is_the_trace_a_buffered_run_writes(crx2, tmp_path):
    """`parikshak run` streams its trace so a long session holds no frames in
    memory. Nothing downstream may be able to tell: same bytes, same replay."""
    buffered = tmp_path / "buffered.jsonl"
    held = session(crx2, camera())
    drive(held, say_at=4.0, text="unit idle")
    held.finish(trace_path=buffered)

    streamed = tmp_path / "streamed.jsonl"
    live = session(crx2, camera(), trace_path=streamed)
    ticks = drive(live, say_at=4.0, text="unit idle")
    assert live.frames == [], "a streaming session must not also hold its frames"
    assert live.recorded_frames == len(ticks)
    assert (tmp_path / "streamed.jsonl.partial").exists() and not streamed.exists()
    live.finish(trace_path=streamed)          # the same path again is fine
    assert not (tmp_path / "streamed.jsonl.partial").exists()

    # Two runs are two runs: each frame's wall-clock stamp (t_utc) is when it
    # happened. Everything else - header byte for byte, every other field of
    # every frame - must match. (Byte identity on the same frames is
    # test_trace_writer's job.)
    import json
    s_lines = streamed.read_text(encoding="utf-8").splitlines()
    b_lines = buffered.read_text(encoding="utf-8").splitlines()
    assert s_lines[0] == b_lines[0]
    assert len(s_lines) == len(b_lines) == len(ticks) + 1

    def without_clock(line):
        d = json.loads(line)
        d.pop("t_utc")
        return d

    assert [without_clock(x) for x in s_lines[1:]] == [without_clock(x) for x in b_lines[1:]]


def test_a_streaming_session_refuses_to_finish_under_another_name(crx2, tmp_path):
    live = session(crx2, camera(duration=0.4), trace_path=tmp_path / "a.jsonl")
    drive(live)
    with pytest.raises(ValueError, match="streams its trace"):
        live.finish(trace_path=tmp_path / "b.jsonl")
    live.finish()
    assert (tmp_path / "a.jsonl").exists() and not (tmp_path / "b.jsonl").exists()


def test_live_session_with_deep_learning_suite(crx2):
    live = session(crx2, camera(duration=0.3), use_deep_learning=True)
    assert "motion_tcn" in live.capabilities
    assert "hand_object_contact" in live.capabilities
    assert "body_pose" in live.capabilities
    assert "contact" in live.pipeline.model_versions()
    assert "motion" in live.pipeline.model_versions()
    tick = live.tick()
    assert tick is not None
    assert isinstance(tick.frame, BeliefFrame)

