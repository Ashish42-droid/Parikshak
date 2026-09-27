#!/usr/bin/env python3
"""The failure-mode reel: the cases where PARIKSHAK does NOT work, and why.

    python tools/make_reel.py docs/reel/segments.json --out runs/reel

PLAN.md section 14: judges spend all day being told things work perfectly; a
team that opens with "here are the cases where we fail, and why" gets believed
about everything else. So every segment is a real, measured failure from the
evaluation corpus, replayed through the unchanged engine and drawn by the real
operator window - and every frame says, on screen, that it is a synthetic replay
under a named degradation profile, not camera footage.

A segment spec (JSON list) gives, per case:
    trace, procedure, title, happened, why, window [t0, t1], hold_at (optional)

Writes reel.mp4, reel.json (the index, with what the engine actually said in
each window), and captions.txt (the same words, for subtitles or a script).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WIDTH, HEIGHT = 1280, 720
WINDOW_H = 540
FPS = 10
TITLE_S = 3.5
HOLD_S = 2.5
PAPER = (247, 247, 244)


def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The offscreen platform ships no font database: every label, checklist row
    # and feed line rendered as empty boxes in the first reel. Point it at the
    # system's fonts unless someone already chose a directory.
    if "QT_QPA_FONTDIR" not in os.environ:
        for candidate in (Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
                          Path("/usr/share/fonts"), Path("/System/Library/Fonts")):
            if candidate.is_dir():
                os.environ["QT_QPA_FONTDIR"] = str(candidate)
                break
    from PySide6 import QtGui, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app, QtGui


def grab(window, QtGui) -> np.ndarray:
    """The window as a BGR array of exactly WIDTH x WINDOW_H."""
    import cv2
    image = window.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    w, h, line = image.width(), image.height(), image.bytesPerLine()
    arr = np.frombuffer(image.constBits(), np.uint8)[: line * h].reshape(h, line)[:, : w * 3]
    rgb = arr.reshape(h, w, 3)
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (WIDTH, WINDOW_H),
                      interpolation=cv2.INTER_AREA)


def caption_bar(lines: list[tuple[str, tuple[int, int, int], float]],
                label: str) -> np.ndarray:
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar = np.full((HEIGHT - WINDOW_H, WIDTH, 3), PAPER, dtype=np.uint8)
    label_top = bar.shape[0] - 22           # reserved: the replay label never gets overwritten
    usable = WIDTH - 36

    def wrap(text: str, size: float) -> list[str]:
        """Wrap by measured pixel width, not by a guessed character count."""
        rows, row = [], ""
        for word in text.split():
            trial = f"{row} {word}".strip()
            if cv2.getTextSize(trial, font, size, 1)[0][0] <= usable or not row:
                row = trial
            else:
                rows.append(row)
                row = word
        return rows + ([row] if row else [])

    y = 22
    for text, colour, size in lines:
        step = int(30 * size) + 6
        for chunk in wrap(text, size):
            if y > label_top - 4:
                break
            cv2.putText(bar, chunk, (18, y), font, size, colour, 1, cv2.LINE_AA)
            y += step
    cv2.rectangle(bar, (0, label_top), (WIDTH, bar.shape[0]), PAPER, -1)
    cv2.putText(bar, label, (18, bar.shape[0] - 7), font, 0.45, (90, 90, 200), 1, cv2.LINE_AA)
    return bar


def title_card(index: int, total: int, seg: dict, label: str) -> np.ndarray:
    import cv2
    card = np.full((HEIGHT, WIDTH, 3), (30, 30, 34), dtype=np.uint8)
    cv2.putText(card, f"FAILURE {index} OF {total}", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (120, 170, 255), 2, cv2.LINE_AA)
    y = 250
    for chunk in textwrap.wrap(seg["title"], 40):
        cv2.putText(card, chunk, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 240), 2,
                    cv2.LINE_AA)
        y += 60
    for chunk in textwrap.wrap(seg.get("measured", ""), 70):
        cv2.putText(card, chunk, (70, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200),
                    1, cv2.LINE_AA)
        y += 40
    cv2.putText(card, label, (70, HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 220),
                1, cv2.LINE_AA)
    return card


def build(segments: list[dict], out_dir: Path) -> dict:
    import cv2

    from parikshak.belief.trace import read_trace
    from parikshak.engine.runner import ProcedureEngine
    from parikshak.eval.schematic import render_schematic
    from parikshak.gui.app import AppConfig, build_window, to_qimage
    from parikshak.gui.overlay import OverlayScene
    from parikshak.gui.state import DisplayState
    from parikshak.pdl import load_procedure

    app, QtGui = _qt()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "reel.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {video_path}")

    index: list[dict] = []
    captions: list[str] = []
    frames_written = 0

    def write(img: np.ndarray, n: int = 1) -> None:
        nonlocal frames_written
        for _ in range(n):
            writer.write(img)
            frames_written += 1

    for number, seg in enumerate(segments, 1):
        header, frames = read_trace(ROOT / seg["trace"])
        procedure = load_procedure(ROOT / seg["procedure"])
        profile = header.extra.get("profile", "unknown")
        label = (f"SYNTHETIC REPLAY - {header.run_id} - degradation profile '{profile}' - "
                 f"not camera footage")
        write(title_card(number, len(segments), seg, label), int(TITLE_S * FPS))

        engine = ProcedureEngine(procedure, run_id=header.run_id)
        window = build_window(AppConfig(window_title=f"PARIKSHAK - {procedure.id}"))
        window.resize(WIDTH, WINDOW_H)
        t0, t1 = seg["window"]
        hold_at = seg.get("hold_at")
        held = False
        said: list[str] = []
        start_frame = frames_written
        for frame in frames:
            result = engine.step(frame)
            # Only what happened inside this segment's window (plus the alert
            # delay), so the caption file never pins another case's alarm here.
            if t0 <= frame.t_mono <= t1 + procedure.alert_policy.persistence_s + 1.0:
                said += [f"[{a.t:6.1f}s] {a.severity.label.upper()} {a.kind.value} "
                         f"{a.step_id}: {a.text}" for a in result.alerts]
                said += [f"[{n.t:6.1f}s] {n.text}" for n in result.notices]
                said += [f"[{tr.t:6.1f}s] verdict {tr.kind.upper()} {tr.step_id}: {tr.detail}"
                         for tr in result.transitions if tr.kind in ("skipped", "unverified")]
            if frame.t_mono < t0 or frame.t_mono > t1:
                continue
            active = engine.active_step
            state = DisplayState.from_engine(engine, frame=frame, source="replay")
            schematic = render_schematic(procedure, frame, 960, 540,
                                         active_objects=procedure.step(active).objects
                                         if active else ())
            window.update_state(state, OverlayScene(), to_qimage(schematic))
            app.processEvents()
            spoken = [a.text for a in result.alerts]
            bar = caption_bar([
                ("WHAT HAPPENED: " + seg["happened"], (30, 30, 30), 0.62),
                ("WHY: " + seg["why"], (40, 40, 150), 0.58),
                (("SAID: " + " | ".join(spoken)) if spoken else "", (20, 110, 40), 0.55),
            ], label)
            composed = np.vstack([grab(window, QtGui), bar])
            write(composed)
            if hold_at is not None and not held and frame.t_mono >= hold_at:
                write(composed, int(HOLD_S * FPS))
                held = True
        engine.finish()
        window.close()

        entry = {"number": number, "title": seg["title"], "trace": seg["trace"],
                 "run_id": header.run_id, "profile": profile, "window": [t0, t1],
                 "happened": seg["happened"], "why": seg["why"],
                 "measured": seg.get("measured", ""),
                 "engine_said": said, "video_frames": [start_frame, frames_written]}
        index.append(entry)
        captions += [f"FAILURE {number}: {seg['title']}", f"  run: {header.run_id} ({profile})",
                     f"  what happened: {seg['happened']}", f"  why: {seg['why']}"]
        if seg.get("measured"):
            captions.append(f"  measured: {seg['measured']}")
        captions += [f"  engine said: {line}" for line in said] + [""]

    writer.release()
    payload = {"video": video_path.name, "fps": FPS, "size": [WIDTH, HEIGHT],
               "frames": frames_written, "segments": index}
    (out_dir / "reel.json").write_text(json.dumps(payload, indent=2), encoding="utf-8",
                                       newline="\n")
    (out_dir / "captions.txt").write_text("\n".join(captions), encoding="utf-8", newline="\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the failure-mode reel.")
    ap.add_argument("spec", type=Path, help="segment spec (JSON list)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "reel")
    args = ap.parse_args(argv)
    segments = json.loads(args.spec.read_text(encoding="utf-8"))
    payload = build(segments, args.out)
    print(f"{len(payload['segments'])} segment(s), {payload['frames']} frames "
          f"({payload['frames'] / FPS:.0f} s) -> {args.out / payload['video']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
