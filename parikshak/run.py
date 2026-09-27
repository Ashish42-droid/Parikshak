"""Run a procedure live, from a camera.

    python -m parikshak.run --procedure procedures/crx2_colloid_resuspension.yaml \\
        --props racks/props_crx2.json --camera 0

    python -m parikshak.run --camera synthetic --seconds 20 --headless

Objects are seen through AprilTags on the props (racks/props_*.json), the rack
frame through the four rack fiducials (racks/msg_a_fiducials.json). Type a crew
phrase - "unit idle", "run complete", "override" - and press Enter to say it;
Vosk is optional and absent here, and the closed grammar is the same either way.

`--camera synthetic` plays a rendered scene instead of a camera, so the live
loop can be rehearsed with no hardware and no printed tags.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from parikshak.io.camera import CameraError, OpenCVCamera, ScriptedProp, SyntheticCamera
from parikshak.io.clipbuffer import ClipBuffer
from parikshak.io.tts import CallbackVoice, SpeechQueue
from parikshak.live import LiveSession
from parikshak.pdl.loader import Procedure, ProcedureError
from parikshak.perception.markers import PropMarkerError, PropMarkers
from parikshak.perception.rackframe import TagLayout

DEFAULT_PROCEDURE = Path("procedures/csp1_colloid_sample_processing.yaml")
DEFAULT_RACK = Path("racks/msg_a_fiducials.json")


def _synthetic_source(layout: TagLayout, props: PropMarkers | None, fps: float,
                      seconds: float | None) -> SyntheticCamera:
    """A rendered rack with every tagged prop sitting in view, and a blocked
    lens for a few seconds in the middle - lock, see, lose, recover."""
    from parikshak.eval.scene import TagScene

    scene = TagScene.facing_rack()
    placed: list[ScriptedProp] = []
    if props is not None:
        seen_entities: set[str] = set()
        spots = [(-0.30 + 0.15 * (i % 5), -0.30 + 0.15 * (i // 5), 0.05) for i in range(20)]
        for tag in sorted(props.tags.values(), key=lambda t: t.tag_id):
            # One tag per entity, the first listed - a state pair would be a rig
            # that cannot tell, which is its own test, not a demo scene.
            if tag.entity in seen_entities:
                continue
            seen_entities.add(tag.entity)
            placed.append(ScriptedProp(tag.tag_id, 0.0, float("inf"),
                                       spots[len(placed) % len(spots)]))
    duration = seconds if seconds is not None else 30.0
    return SyntheticCamera(scene, layout, placed, tag_size_m=props.tag_size_m if props else 0.06,
                           fps=fps, blackouts=[(duration * 0.4, duration * 0.4 + 5.0)],
                           duration_s=duration)


def _stdin_lines(out: queue.Queue) -> None:  # pragma: no cover - interactive
    for line in sys.stdin:
        out.put(line.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a procedure live from a camera.")
    ap.add_argument("--procedure", type=Path, default=DEFAULT_PROCEDURE)
    ap.add_argument("--props", type=Path, default=None,
                    help="fiducial prop map (racks/props_*.json); without it no object is seen")
    ap.add_argument("--rack", type=Path, default=DEFAULT_RACK, help="rack fiducial layout")
    ap.add_argument("--camera", default="0", help="camera index, or 'synthetic'")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=70.0,
                    help="horizontal field of view in degrees, until the camera is calibrated")
    ap.add_argument("--fps", type=float, default=10.0, help="perception rate")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--headless", action="store_true", help="no window; print to the console")
    ap.add_argument("--no-voice", action="store_true", help="print speech instead of saying it")
    ap.add_argument("--no-clips", action="store_true",
                    help="do not write video clips around alerts")
    ap.add_argument("--log-dir", type=Path, default=Path("runs/live"),
                    help="hash-chained log and recorded trace go here")
    args = ap.parse_args(argv)

    try:
        procedure = Procedure.load(args.procedure)
        layout = TagLayout.load(args.rack)
        props = PropMarkers.load(args.props) if args.props else None
    except (ProcedureError, PropMarkerError, OSError, ValueError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 2

    try:
        if args.camera == "synthetic":
            source = _synthetic_source(layout, props, args.fps, args.seconds)
        else:
            source = OpenCVCamera(int(args.camera), width=args.width, height=args.height)
    except (CameraError, ValueError) as exc:
        print(f"cannot open camera: {exc}", file=sys.stderr)
        return 2

    if args.no_voice or args.headless:
        speech = SpeechQueue(CallbackVoice(lambda text: print(f"    [voice] {text}")))
    else:
        try:
            from parikshak.io.tts import Pyttsx3Voice
            speech = SpeechQueue(Pyttsx3Voice())
        except ImportError:
            speech = SpeechQueue(CallbackVoice(lambda text: print(f"    [voice] {text}")))

    run_id = f"live-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.log_dir / f"{run_id}.trace.jsonl"
    try:
        # The trace streams to disk as the run goes: a 45-minute session holds
        # no frames in memory, and a crash leaves <trace>.partial behind.
        session = LiveSession(procedure, source, layout, props=props, hfov_deg=args.hfov,
                              run_id=run_id, log_path=args.log_dir / f"{run_id}.log.jsonl",
                              speech=speech, fps=args.fps, trace_path=trace_path,
                              clips=None if args.no_clips else ClipBuffer(
                                  args.log_dir / "clips", fps=args.fps))
    except PropMarkerError as exc:
        source.close()
        print(f"cannot start: {exc}", file=sys.stderr)
        return 2

    print(f"{procedure.id} {procedure.title} - {run_id}")
    print(f"  source {args.camera} {source.width}x{source.height}, rack {args.rack}, "
          f"props {args.props or 'none'}")
    if session.unobserved:
        print(f"  not observable in this setup: {', '.join(session.unobserved)}")
    if session.gaps:
        print("  steps that will read UNVERIFIED here, and why:")
        for gap in session.gaps:
            print(f"    {gap}")

    if not args.headless:
        try:
            from parikshak.gui.app import run_live_window
        except ImportError as exc:
            print(f"  no window available ({exc}); running headless", file=sys.stderr)
        else:
            return run_live_window(session, seconds=args.seconds, speech=speech,
                                   trace_path=trace_path)

    return _run_headless(session, args.fps, args.seconds, args.camera == "synthetic",
                         trace_path)


def _run_headless(session: LiveSession, fps: float, seconds: float | None,
                  as_fast_as_possible: bool, trace_path: Path) -> int:
    typed: queue.Queue = queue.Queue()
    if sys.stdin is not None and sys.stdin.isatty():  # pragma: no cover - interactive
        threading.Thread(target=_stdin_lines, args=(typed,), daemon=True).start()
        print("  type a crew phrase and press Enter; Ctrl+C ends the run")

    period = 1.0 / fps
    started = time.monotonic()
    try:
        while seconds is None or (time.monotonic() - started) < seconds or as_fast_as_possible:
            while not typed.empty():
                heard = session.submit(typed.get())
                print(f"  heard: {heard.command.value} {heard.token or heard.phrase}"
                      if heard else "  not in the grammar - ignored")
            t0 = time.monotonic()
            tick = session.tick()
            if tick is None:
                break
            for tr in tick.result.transitions:
                if tr.kind != "entered":
                    print(f"  {tr.t:8.1f}s  {tr.kind.upper():<11} {tr.step_id:<10} {tr.detail}")
            for alert in tick.result.alerts:
                print(f"  {alert}")
            for notice in tick.result.notices:
                print(f"  {notice.t:8.1f}s  UNVERIFIED  {notice.step_id:<10} {notice.text}")
            if not as_fast_as_possible:
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass

    from parikshak.gui.app import render_text
    if session.last is not None:
        print(render_text(session.last.state))
    summary = session.finish(trace_path=trace_path)
    print(f"  {summary.frames} frames, {summary.duration_s:.1f}s, "
          f"{len(summary.alerts)} alert(s); trace {trace_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
