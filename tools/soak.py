#!/usr/bin/env python3
"""Soak test: does a long run stay as fast and as small as a short one?

    python tools/soak.py --minutes 45 --stream     # how `parikshak run` records
    python tools/soak.py --minutes 45              # frames held in memory
    python tools/soak.py --minutes 45 --no-record

A demo lasts four minutes; a real session lasts forty-five, and a leak or a
history that never trims shows up only in the second. This runs the live loop
over a static rendered rack - the bench scene, every tagged prop in view, the
camera blacked out for five seconds every minute - as fast as the machine
allows, for the given number of SIMULATED minutes, and samples it once per
simulated minute:

    latency     p50 / p95 of LiveSession.tick over the minute
    locked      share of ticks with the rack frame locked - proof the soak ran
                the real detection path, not the cheap nothing-found one
    memory      resident set size
    recorded    frames recorded, held in memory or streamed to disk
    histories   the evaluator's temporal histories, which must stay bounded

It passes when the last minute is no more than 1.5x slower than the first and
every history stays within its trim window. Memory growth is reported, not
gated. Writes runs/soak.json (or --out).

Read latency drift with the machine in mind: on a laptop on battery the first
and last minute can differ by more than any code change would. Compare modes
back to back, or A/B in one process, before blaming the code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench import rss_bytes, synthetic_source  # noqa: E402

PROCEDURE = ROOT / "procedures" / "crx2_colloid_resuspension.yaml"
DRIFT_LIMIT = 1.5
BLACKOUT_EVERY_S = 60.0
BLACKOUT_FOR_S = 5.0
MODES = ("memory", "stream", "off")


class StaticRack:
    """Two pre-rendered images and a deterministic clock. Rendering is not what
    is being soaked, so it is done once rather than per frame."""

    def __init__(self, normal, blocked, fps: float, frames: int) -> None:
        self.normal, self.blocked = normal, blocked
        self.height, self.width = normal.shape[:2]
        self.fps, self.frames = fps, frames
        self._i = 0

    def now(self) -> float:
        return (self._i - 1) / self.fps   # time of the frame just read

    def read(self):
        if self._i >= self.frames:
            return False, None
        t = self._i / self.fps
        self._i += 1
        blind = (t % BLACKOUT_EVERY_S) >= BLACKOUT_EVERY_S - BLACKOUT_FOR_S
        return True, (self.blocked if blind else self.normal)

    def close(self) -> None:
        pass


def longest_hold_for_s(path: Path) -> float:
    """The longest `hold_for` window in the procedure. A `hold_for` history is
    trimmed to the larger of that and the evaluator's history_s, so this is part
    of the bound the soak checks against."""
    import yaml

    found = [0.0]

    def walk(node) -> None:
        if isinstance(node, dict):
            spec = node.get("hold_for")
            if isinstance(spec, dict) and "seconds" in spec:
                found.append(float(spec["seconds"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(yaml.safe_load(path.read_text(encoding="utf-8")))
    return max(found)


def history_sizes(session) -> dict[str, int]:
    ev = session.engine.tracker.ev
    return {
        "node_history_entries": sum(len(h) for h in ev._node_hist.values()),
        "position_history_entries": sum(len(h) for h in ev._pos_hist.values()),
        "motion_history_entries": len(ev._motion_hist),
        "transitions": len(session.engine.tracker.transitions),
    }


def drift_thirds(windows: list[dict]) -> float | None:
    """Late over early tick time, as a trend rather than a pair of samples.

    Minute one is skipped (caches warming, the evaluator's windows filling). The
    median p50 of the last third of the remaining windows is divided by the
    median of the first third. Comparing the first and last window alone swung
    from 0.56x to 6.7x on unchanged code, on a laptop whose CPU speed wandered 3x
    within a run - perception, engine and display all slowing together while the
    engine's state never changed. A leak or an unbounded history is a trend
    across many windows, and this keeps it; it cannot separate a machine that
    slows for the whole last third from code that does, so read it beside the
    per-stage split.
    """
    body = [w["p50_ms"] for w in windows[1:]] or [w["p50_ms"] for w in windows]
    if not body:
        return None
    k = max(1, len(body) // 3)
    early = statistics.median(body[:k])
    return statistics.median(body[-k:]) / early if early else None


def run(minutes: float = 5.0, fps: float = 10.0, mode: str = "memory",
        sample_every_s: float = 60.0, report=None, trace_path: Path | None = None) -> dict:
    """`mode` is how frames are recorded: "memory" (held for a trace written at
    the end), "stream" (written as they arrive, to `trace_path`), or "off"."""
    import numpy as np

    from parikshak.live import LiveSession
    from parikshak.pdl import load_procedure
    from parikshak.perception.markers import PropMarkers
    from parikshak.perception.rackframe import TagLayout

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "stream" and trace_path is None:
        trace_path = ROOT / "runs" / "soak_stream.trace.jsonl"
    procedure = load_procedure(PROCEDURE)
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    props = PropMarkers.load(ROOT / "racks" / "props_crx2.json")
    ok, normal = synthetic_source(layout, fps, 1).read()
    assert ok, "the bench scene rendered nothing"
    blocked = np.full_like(normal, 90)
    frames = int(round(minutes * 60.0 * fps))
    source = StaticRack(normal, blocked, fps, frames)
    session = LiveSession(procedure, source, layout, props=props, run_id="soak", fps=fps,
                          record=(mode == "memory"),
                          trace_path=trace_path if mode == "stream" else None)
    ev = session.engine.tracker.ev

    windows: list[dict] = []
    tick_s: list[float] = []
    locked: list[bool] = []
    window_start = 0.0
    rss0 = rss_bytes()
    wall0 = time.monotonic()
    while True:
        t0 = time.perf_counter()
        tick = session.tick()
        elapsed = time.perf_counter() - t0
        if tick is None:
            break
        tick_s.append(elapsed)
        locked.append(bool(tick.frame.frame_lock))
        sim_t = tick.frame.t_mono
        if sim_t - window_start >= sample_every_s or source._i >= frames:
            ms = sorted(x * 1000 for x in tick_s)
            windows.append({
                "sim_minute": round(sim_t / 60.0, 2),
                "frames": len(tick_s),
                "rack_locked_fraction": round(sum(locked) / len(locked), 3),
                "p50_ms": round(statistics.median(ms), 2),
                "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 2),
                "rss_bytes": rss_bytes(),
                "recorded_frames": session.recorded_frames,
                **history_sizes(session),
            })
            if report is not None:
                w = windows[-1]
                report(f"  minute {w['sim_minute']:6.2f}: {w['p50_ms']:6.2f} / {w['p95_ms']:6.2f} ms, "
                       f"locked {w['rack_locked_fraction']:.0%}, "
                       f"rss {(w['rss_bytes'] or 0) / 1e6:7.1f} MB, recorded {w['recorded_frames']}, "
                       f"history entries {w['node_history_entries'] + w['position_history_entries']}")
            tick_s, locked = [], []
            window_start = sim_t
    summary = session.finish()
    wall = time.monotonic() - wall0

    trace_bytes = None
    if mode == "stream" and trace_path is not None and trace_path.exists():
        trace_bytes = trace_path.stat().st_size
        trace_path.unlink()   # ~2 KB a frame; a soak trace is not worth keeping

    # Bounded: every temporal history is trimmed by time, so no series may hold
    # more than its window's worth of frames however long the run. The window is
    # the evaluator's history_s or the procedure's longest hold_for, whichever is
    # larger, plus a few frames of slack for the entries at the trim boundary.
    window_s = max(ev.history_s, longest_hold_for_s(PROCEDURE))
    per_series_bound = int(window_s * fps) + 3
    longest = max([len(h) for h in ev._node_hist.values()]
                  + [len(h) for h in ev._pos_hist.values()]
                  + [len(ev._motion_hist)])
    first, last = windows[0], windows[-1]
    single = last["p50_ms"] / first["p50_ms"] if first["p50_ms"] else None
    drift = drift_thirds(windows)
    rss_growth = (last["rss_bytes"] - rss0) if rss0 and last["rss_bytes"] else None
    sim_hours = (frames / fps) / 3600.0
    result = {
        "simulated_minutes": minutes, "fps": fps, "frames": frames, "recording": mode,
        "wall_seconds": round(wall, 1),
        "windows": windows,
        # The gate: a trend across the run (see drift_thirds).
        "latency_drift_thirds": round(drift, 3) if drift else None,
        # Kept for information only - one window against another measures the CPU.
        "latency_drift_last_over_first_p50": round(single, 3) if single else None,
        "longest_history_series": longest, "history_window_s": window_s,
        "history_series_bound": per_series_bound,
        "histories_bounded": longest <= per_series_bound,
        "rss_growth_bytes": rss_growth,
        "rss_growth_mb_per_simulated_hour": round(rss_growth / 1e6 / sim_hours, 1)
        if rss_growth is not None else None,
        "streamed_trace_bytes": trace_bytes,
        "alerts": len(summary.alerts), "notices": len(summary.notices),
    }
    result["verdict"] = "PASS" if (result["histories_bounded"] and drift is not None
                                   and drift <= DRIFT_LIMIT) else "FAIL"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Soak-test the live loop.")
    ap.add_argument("--minutes", type=float, default=5.0, help="simulated minutes")
    ap.add_argument("--fps", type=float, default=10.0)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--stream", action="store_true",
                       help="stream the trace to disk as the run goes (how parikshak run records)")
    group.add_argument("--no-record", action="store_true", help="record nothing")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "soak.json")
    args = ap.parse_args(argv)
    mode = "stream" if args.stream else "off" if args.no_record else "memory"
    result = run(args.minutes, args.fps, mode=mode, report=print)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(f"\n  {result['frames']} frames ({args.minutes:g} simulated min) in "
          f"{result['wall_seconds']} s wall, recording: {mode}")
    print(f"  latency drift (last third / first third, p50): {result['latency_drift_thirds']}x, "
          f"limit {DRIFT_LIMIT}x  [last/first minute alone: "
          f"{result['latency_drift_last_over_first_p50']}x, not gated]")
    print(f"  longest history series {result['longest_history_series']} "
          f"(bound {result['history_series_bound']}): "
          f"{'bounded' if result['histories_bounded'] else 'UNBOUNDED'}")
    print(f"  memory growth: {result['rss_growth_mb_per_simulated_hour']} MB per simulated hour")
    if result["streamed_trace_bytes"] is not None:
        print(f"  streamed trace: {result['streamed_trace_bytes'] / 1e6:.1f} MB on disk (deleted)")
    print(f"  alerts {result['alerts']}, unverified notices {result['notices']}")
    print(f"  {result['verdict']} -> {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
