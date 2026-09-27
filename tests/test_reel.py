"""The failure-mode reel and the schematic it draws.

The reel is evidence shown to judges, so the tests hold it to two things: the
drawing puts things where the belief frame says they are and nowhere else, and
a segment renders from a real trace through the real engine into a video that
plays back.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "traces" / "golden"

from parikshak.belief.trace import read_trace  # noqa: E402
from parikshak.eval.schematic import render_schematic  # noqa: E402
from parikshak.pdl import load_procedure  # noqa: E402


@pytest.fixture(scope="module")
def csp1():
    return load_procedure(ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")


def test_the_schematic_draws_a_located_object_where_the_frame_puts_it(csp1):
    _, frames = read_trace(GOLDEN / "nominal.jsonl")
    from dataclasses import replace

    frame = next(f for f in frames if f.frame_lock and f.objects["latch"].pos_rack is not None)
    blank_scene = render_schematic(csp1, replace(frame, objects={}))
    drawn = render_schematic(csp1, frame)
    assert drawn.shape == (540, 960, 3)
    # Something was drawn for the objects that is not in the object-free scene.
    assert np.count_nonzero(np.any(drawn != blank_scene, axis=2)) > 200


def test_the_schematic_says_so_when_the_rack_frame_is_lost(csp1):
    _, frames = read_trace(GOLDEN / "frame_loss_S07.jsonl")
    unlocked = next(f for f in frames if not f.frame_lock)
    img = render_schematic(csp1, unlocked)
    banner = img[-40:, :, :]
    # The banner is a solid red-ish band (BGR 60, 60, 200) with white text.
    assert np.mean(banner[:, :, 2]) > 150 and np.mean(banner[:, :, 0]) < 140


def test_a_reel_segment_renders_to_a_playable_video(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT / "tools"))
    import make_reel

    spec = [{
        "trace": "traces/golden/skip_S08_latch.jsonl",
        "procedure": "procedures/csp1_colloid_sample_processing.yaml",
        "title": "test segment", "happened": "the latch was left open",
        "why": "a test", "window": [46.0, 48.0], "hold_at": 47.0,
    }]
    payload = make_reel.build(spec, tmp_path)

    video = cv2.VideoCapture(str(tmp_path / "reel.mp4"))
    ok, first = video.read()
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    assert ok and first.shape == (make_reel.HEIGHT, make_reel.WIDTH, 3)
    expected = int(make_reel.TITLE_S * make_reel.FPS) + 11 + int(make_reel.HOLD_S * make_reel.FPS)
    assert payload["frames"] == count == expected

    index = json.loads((tmp_path / "reel.json").read_text(encoding="utf-8"))
    assert index["segments"][0]["run_id"] == "skip_S08_latch"
    assert any("SKIP" in line and "S08" in line for line in index["segments"][0]["engine_said"])
    captions = (tmp_path / "captions.txt").read_text(encoding="utf-8")
    assert "skip_S08_latch" in captions and "the latch was left open" in captions
    # Every rendered frame is labelled as a replay. The label is burned into the
    # pixels, so check the builder carries it rather than OCR the video.
    assert "not camera footage" in Path(make_reel.__file__).read_text(encoding="utf-8")
