"""PLAN.md's Phase 6 gate: a replayed trace and a live run look identical in the GUI.

That is the stage insurance. If a camera fails in front of the judges, the
recording replays through the same engine into the same window, and the screen
must not tell the difference. These tests hold the window to it on an
offscreen Qt platform, and check the one thing only a replay adds - seeking -
against playing straight through.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("cv2")

from parikshak.eval.scene import TagScene  # noqa: E402
from parikshak.gui.app import build_window, run_session_window, to_qimage  # noqa: E402
from parikshak.io.camera import ScriptedProp, SyntheticCamera  # noqa: E402
from parikshak.live import LiveSession, ReplaySession  # noqa: E402
from parikshak.pdl import load_procedure  # noqa: E402
from parikshak.perception.markers import PropMarkers  # noqa: E402
from parikshak.perception.rackframe import TagLayout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CRX2 = ROOT / "procedures" / "crx2_colloid_resuspension.yaml"
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def comparable(state) -> dict:
    d = state.as_dict()
    d.pop("source")
    return d


def test_a_replay_of_a_live_run_looks_identical_in_the_window(app, tmp_path):
    procedure = load_procedure(CRX2)
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    props = PropMarkers.load(ROOT / "racks" / "props_crx2.json")
    camera = SyntheticCamera(TagScene.facing_rack(width=960, height=540), layout,
                             [ScriptedProp(203, 0.0, 1e9, (-0.48, 0.21, 0.03)),
                              ScriptedProp(211, 0.0, 1e9, (0.10, 0.00, 0.10)),
                              ScriptedProp(231, 0.0, 1e9, (0.30, 0.10, 0.15))],
                             fps=5.0, duration_s=8.0)
    live = LiveSession(procedure, camera, layout, props=props, run_id="gate", fps=5.0)
    live_window, replay_window = build_window(), build_window(seekable=True)

    live_states = []
    said = False
    while True:
        if not said and camera.now() >= 3.0:
            assert live.submit("unit idle") is not None
            said = True
        tick = live.tick()
        if tick is None:
            break
        live_states.append(tick.state)
    trace = tmp_path / "gate.jsonl"
    live.finish(trace_path=trace)
    live_window.update_state(live_states[-1])

    replay = ReplaySession(procedure, trace)
    replay_states = []
    while (tick := replay.tick()) is not None:
        replay_states.append(tick.state)
    replay_window.update_state(replay_states[-1])

    assert len(replay_states) == len(live_states)
    assert [comparable(s) for s in replay_states] == [comparable(s) for s in live_states]
    assert replay_window.checklist_texts() == live_window.checklist_texts()
    assert replay_window.feed_texts() == live_window.feed_texts()
    assert any("(not verifiable here)" in row for row in replay_window.checklist_texts()), \
        "a replay of a live run keeps the live run's capability limits"


def test_seeking_a_replay_reaches_the_same_state_as_playing_through(app):
    procedure = load_procedure(CSP1)
    trace = ROOT / "traces" / "golden" / "skip_S08_latch.jsonl"

    straight = ReplaySession(procedure, trace, render=False)
    for _ in range(300):
        straight.tick()
    expected = comparable(straight.last.state)

    jumpy = ReplaySession(procedure, trace, render=False)
    jumpy.seek(500)          # past the moment
    jumpy.seek(120)          # back before it: a silent re-run from the start
    jumpy.seek(299)
    tick = jumpy.tick()
    assert jumpy.position == 300
    assert comparable(tick.state) == expected


def test_the_replay_window_runs_and_exits(app):
    procedure = load_procedure(CSP1)
    session = ReplaySession(procedure, ROOT / "traces" / "golden" / "skip_S08_latch.jsonl")
    lines: list[str] = []
    assert run_session_window(session, seconds=1.0, report=lines.append) == 0
    assert len(lines) == 1
    assert session.position > 0
