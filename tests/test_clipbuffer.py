"""Deviation clips without ffmpeg: what is written, and what is admitted.

A clip is evidence. These tests hold the laptop path to the same standard as the
flight path in test_clips: the window is right, a clip that could not cover its
whole window says so, and memory does not grow with the length of the run.
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from parikshak.io.clipbuffer import ClipBuffer  # noqa: E402

FPS = 10.0


def image(i: int):
    img = np.zeros((240, 320, 3), np.uint8)
    cv2.putText(img, str(i), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    return img


def run(buf: ClipBuffer, until_s: float, *, alert_at: float | None = None, start_s: float = 0.0):
    done = []
    i = int(round(start_s * FPS))
    while i / FPS <= until_s + 1e-9:
        t = i / FPS
        done += buf.push(t, image(i))
        if alert_at is not None and abs(t - alert_at) < 1e-9:
            buf.request("run", "S08", "SKIP", t)
        i += 1
    return done


def frames_in(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_a_clip_covers_the_window_around_the_alert(tmp_path):
    buf = ClipBuffer(tmp_path, pre_s=3.0, post_s=3.0, fps=FPS)
    done = run(buf, 12.0, alert_at=8.0)
    assert len(done) == 1
    clip = done[0]
    assert clip.available and clip.reason == "", clip.reason
    assert clip.path.suffix == ".mp4" and clip.path.exists()
    assert abs(frames_in(clip.path) - 61) <= 2       # 3 s either side at 10 Hz, inclusive


def test_a_clip_is_written_only_once_its_post_roll_has_been_seen(tmp_path):
    buf = ClipBuffer(tmp_path, pre_s=3.0, post_s=3.0, fps=FPS)
    assert run(buf, 10.9, alert_at=8.0) == []
    assert run(buf, 11.0, start_s=11.0)


def test_an_alert_early_in_the_run_says_its_pre_roll_is_short(tmp_path):
    buf = ClipBuffer(tmp_path, pre_s=3.0, post_s=3.0, fps=FPS)
    (clip,) = run(buf, 5.0, alert_at=1.0)
    assert clip.available and clip.reason.startswith("partial coverage"), clip.reason


def test_a_run_that_ends_before_the_post_roll_still_keeps_its_evidence(tmp_path):
    buf = ClipBuffer(tmp_path, pre_s=3.0, post_s=3.0, fps=FPS)
    assert run(buf, 9.0, alert_at=8.0) == []
    (clip,) = buf.flush()
    assert clip.available and clip.path.exists() and "partial coverage" in clip.reason


def test_a_live_session_with_clips_runs_and_writes_nothing_without_an_alert(tmp_path):
    from parikshak.pdl import load_procedure
    from tests.test_live import CRX2, camera, drive, session

    clips = ClipBuffer(tmp_path / "clips", pre_s=2.0, post_s=2.0, fps=5.0)
    live = session(load_procedure(CRX2), camera(duration=6.0), clips=clips)
    drive(live, say_at=3.0, text="unit idle")
    summary = live.finish()
    assert summary.alerts == ()
    assert clips.results == [] and not (tmp_path / "clips").exists()
    assert 0 < clips.buffered_frames <= 2.0 * 5.0 + 2


def test_memory_is_bounded_by_the_pre_roll(tmp_path):
    buf = ClipBuffer(tmp_path, pre_s=3.0, post_s=3.0, fps=FPS)
    run(buf, 60.0)
    assert buf.buffered_frames <= 3.0 * FPS + 2
