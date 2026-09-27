"""The live operator window, run for real on an offscreen Qt platform.

Everything the window shows is asserted in test_gui.py without Qt. This proves
the part that cannot be: the window opens over a live session, ticks it on a
timer, paints frames, and - however the run ends - finishes it exactly once and
writes the flight record.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("cv2")

from parikshak.belief.trace import read_trace  # noqa: E402
from parikshak.eval.scene import TagScene  # noqa: E402
from parikshak.gui.app import run_live_window  # noqa: E402
from parikshak.io.camera import ScriptedProp, SyntheticCamera  # noqa: E402
from parikshak.io.tts import NullVoice, SpeechQueue  # noqa: E402
from parikshak.live import LiveSession  # noqa: E402
from parikshak.pdl import load_procedure  # noqa: E402
from parikshak.perception.markers import PropMarkers  # noqa: E402
from parikshak.perception.rackframe import TagLayout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_the_window_runs_a_live_session_to_the_end_and_keeps_the_record(tmp_path):
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    props = PropMarkers.load(ROOT / "racks" / "props_crx2.json")
    source = SyntheticCamera(TagScene.facing_rack(width=640, height=360), layout,
                             [ScriptedProp(203, 0.0, 1e9, (-0.48, 0.21, 0.03))],
                             tag_size_m=props.tag_size_m, fps=10.0, duration_s=1.5)
    speech = SpeechQueue(NullVoice())
    session = LiveSession(load_procedure(ROOT / "procedures" / "crx2_colloid_resuspension.yaml"),
                          source, layout, props=props, run_id="window-test",
                          speech=speech, fps=10.0)
    trace = tmp_path / "window.jsonl"
    lines: list[str] = []

    assert run_live_window(session, speech=speech, trace_path=trace, report=lines.append) == 0

    header, frames = read_trace(trace)
    assert header.run_id == "window-test" and len(frames) >= 10
    assert len(lines) == 1, "the run must be finished exactly once"
    assert speech.voice.spoken, "the first prompt should have been spoken"
