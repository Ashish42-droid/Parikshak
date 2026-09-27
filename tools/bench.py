#!/usr/bin/env python3
"""Latency and throughput on THIS machine, stage by stage.

    python tools/bench.py                     # synthetic camera, 300 frames
    python tools/bench.py --camera 0 --seconds 20

PLAN.md Phase 7 gates the edge build on >= 10 FPS at <= 25 W on a meter. There is
no Jetson and no meter here, so this is the plan's own fallback made concrete:
published numbers from the machine at hand, with the machine named. It measures
what the software costs - it cannot say what a Jetson would do, and the report
says so.

Per stage, over the same frames:
    tags        AprilTag detection on the image
    rack        rack-frame pose from the tags
    props       marker prop poses (one per tagged prop in view)
    perception  the whole PerceptionPipeline.step (includes tags, rack, props)
    engine      ProcedureEngine.step
    display     DisplayState.from_engine
    overlay     build_overlay
    system      one LiveSession.tick, minus the time the camera took to deliver
                the frame - the number the 10 FPS gate is about

and engine-only throughput replaying the golden CSP-1 trace. Writes
runs/bench.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GATE_FPS = 10.0


def rss_bytes() -> int | None:
    """Resident memory of this process, without psutil."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32 = ctypes.WinDLL("kernel32")
            # ctypes defaults every return and argument to a 32-bit int, which
            # truncates the 64-bit process handle - declare the real types.
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = getattr(kernel32, "K32GetProcessMemoryInfo", None) \
                or ctypes.WinDLL("psapi").GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
            return None
        with open("/proc/self/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, AttributeError, ValueError):
        return None


def summarise(samples_s: list[float]) -> dict:
    ms = sorted(s * 1000.0 for s in samples_s)
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "p50_ms": round(statistics.median(ms), 3),
            "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 3),
            "max_ms": round(ms[-1], 3), "mean_ms": round(statistics.fmean(ms), 3)}


def machine() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version()}
    try:
        import cv2
        info["opencv"] = cv2.__version__
        info["opencv_threads"] = cv2.getNumThreads()
    except ImportError:
        info["opencv"] = None
    return info


def synthetic_source(layout, fps: float, frames: int):
    from parikshak.eval.scene import TagScene
    from parikshak.io.camera import ScriptedProp, SyntheticCamera
    props = [ScriptedProp(203, 0.0, 1e9, (-0.48, 0.21, 0.03)),
             ScriptedProp(211, 0.0, 1e9, (0.10, 0.00, 0.10)),
             ScriptedProp(231, 0.0, 1e9, (0.30, 0.10, 0.15))]
    return SyntheticCamera(TagScene.facing_rack(), layout, props, fps=fps,
                           duration_s=(frames - 1) / fps)


def bench_stages(frames: int) -> dict:
    """Each stage timed separately on the same rendered images."""
    from parikshak.engine.runner import ProcedureEngine
    from parikshak.gui.overlay import build_overlay
    from parikshak.gui.state import DisplayState
    from parikshak.pdl import load_procedure
    from parikshak.perception.backends import AprilTagDetector
    from parikshak.perception.markers import MarkerDetector, PropMarkers, SharedTags
    from parikshak.perception.pipeline import EntityBinding, PerceptionPipeline, PipelineConfig
    from parikshak.perception.rackframe import RackFrameEstimator, TagLayout

    procedure = load_procedure(ROOT / "procedures" / "crx2_colloid_resuspension.yaml")
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    props = PropMarkers.load(ROOT / "racks" / "props_crx2.json")
    source = synthetic_source(layout, 10.0, frames)
    images = []
    while True:
        ok, image = source.read()
        if not ok:
            break
        images.append(image)

    detector = AprilTagDetector()
    rack = RackFrameEstimator(layout, source.scene.intrinsics)
    times: dict[str, list[float]] = {k: [] for k in
                                     ("tags", "rack", "props", "perception", "engine",
                                      "display", "overlay")}
    # Stage by stage on a private rack estimator and marker detector...
    for i, image in enumerate(images):
        t0 = time.perf_counter()
        obs = list(detector(image))
        t1 = time.perf_counter()
        rack.update(obs, i / 10.0)
        t2 = time.perf_counter()
        markers = MarkerDetector(props, procedure, rack, SharedTags(lambda _img, o=obs: o))
        t3 = time.perf_counter()
        markers(image)
        t4 = time.perf_counter()
        times["tags"].append(t1 - t0)
        times["rack"].append(t2 - t1)
        times["props"].append(t4 - t3)

    # ...then the assembled pipeline, engine, display and overlay in sequence.
    tagged = {t.entity for t in props.tags.values()}
    bindings = {n: b for n, b in EntityBinding.from_procedure(procedure).items() if n in tagged}
    rack2 = RackFrameEstimator(layout, source.scene.intrinsics)
    shared = SharedTags(AprilTagDetector())
    pipeline = PerceptionPipeline(bindings, rack2,
                                  detector=MarkerDetector(props, procedure, rack2, shared),
                                  tags=shared, config=PipelineConfig(fps=10.0))
    engine = ProcedureEngine(procedure, run_id="bench")
    for i, image in enumerate(images):
        t0 = time.perf_counter()
        frame = pipeline.step(image, i / 10.0)
        t1 = time.perf_counter()
        engine.step(frame)
        t2 = time.perf_counter()
        DisplayState.from_engine(engine, frame=frame)
        t3 = time.perf_counter()
        build_overlay(procedure, frame, rack2.extrinsics, source.scene.intrinsics)
        t4 = time.perf_counter()
        times["perception"].append(t1 - t0)
        times["engine"].append(t2 - t1)
        times["display"].append(t3 - t2)
        times["overlay"].append(t4 - t3)
    return {"image": [source.width, source.height], "frames": len(images),
            "stages": {k: summarise(v) for k, v in times.items()}}


class TimedSource:
    """Wraps a frame source and records how long each read took, so the time
    spent producing a frame is not billed to the system consuming it."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.width, self.height = inner.width, inner.height
        self.read_s: list[float] = []

    def read(self):
        t0 = time.perf_counter()
        out = self.inner.read()
        self.read_s.append(time.perf_counter() - t0)
        return out

    def now(self) -> float:
        return self.inner.now()

    def close(self) -> None:
        self.inner.close()


def bench_end_to_end(frames: int, camera: str, seconds: float | None) -> dict:
    """LiveSession.tick, as fast as it will go, minus frame delivery time."""
    from parikshak.io.camera import OpenCVCamera
    from parikshak.live import LiveSession
    from parikshak.pdl import load_procedure
    from parikshak.perception.markers import PropMarkers
    from parikshak.perception.rackframe import TagLayout

    procedure = load_procedure(ROOT / "procedures" / "crx2_colloid_resuspension.yaml")
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    props = PropMarkers.load(ROOT / "racks" / "props_crx2.json")
    inner = (synthetic_source(layout, 10.0, frames) if camera == "synthetic"
             else OpenCVCamera(int(camera)))
    source = TimedSource(inner)
    session = LiveSession(procedure, source, layout, props=props, run_id="bench",
                          fps=10.0, record=False)
    tick_s: list[float] = []
    rss_start = rss_bytes()
    started = time.monotonic()
    while True:
        if camera != "synthetic" and seconds is not None and time.monotonic() - started > seconds:
            break
        if camera == "synthetic" and len(tick_s) >= frames:
            break
        t0 = time.perf_counter()
        tick = session.tick()
        elapsed = time.perf_counter() - t0
        if tick is None:
            break
        tick_s.append(elapsed)
    session.finish()
    reads = source.read_s[:len(tick_s)]
    system = [max(0.0, t - r) for t, r in zip(tick_s, reads)]
    summary = summarise(system)
    fps = 1000.0 / summary["p50_ms"] if summary.get("p50_ms") else None
    fps95 = 1000.0 / summary["p95_ms"] if summary.get("p95_ms") else None
    locked = session.last.frame.frame_lock if session.last else None
    return {"camera": camera, "image": [source.width, source.height],
            "frames": len(tick_s), "rack_locked_at_end": locked,
            "frame_delivery": summarise(reads), "system_per_frame": summary,
            "fps_at_p50": round(fps, 1) if fps else None,
            "fps_at_p95": round(fps95, 1) if fps95 else None,
            "rss_bytes_start": rss_start, "rss_bytes_end": rss_bytes()}


def bench_engine_replay(repeats: int = 3) -> dict:
    from parikshak.belief.trace import read_trace
    from parikshak.engine.runner import ProcedureEngine
    from parikshak.pdl import load_procedure
    procedure = load_procedure(ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
    _, frames = read_trace(ROOT / "traces" / "golden" / "nominal.jsonl")
    per_frame: list[float] = []
    for _ in range(repeats):
        engine = ProcedureEngine(procedure, run_id="bench")
        for f in frames:
            t0 = time.perf_counter()
            engine.step(f)
            per_frame.append(time.perf_counter() - t0)
        engine.finish()
    s = summarise(per_frame)
    return {"procedure": procedure.id, "steps": len(procedure.order), "frames": len(frames),
            "per_frame": s, "frames_per_second_at_p50": round(1000.0 / s["p50_ms"], 1)}


def run(frames: int = 300, camera: str = "synthetic", seconds: float | None = None) -> dict:
    report = {
        "machine": machine(),
        "stages": bench_stages(min(frames, 200)),
        "end_to_end": bench_end_to_end(frames, camera, seconds),
        "engine_replay": bench_engine_replay(),
        "gate": {"fps_required": GATE_FPS, "power_w_limit": 25.0,
                 "power_measured": False,
                 "note": ("Measured on the machine named above, CPU only. No Jetson and no "
                          "power meter were available: this is not the edge gate, it is the "
                          "fallback the plan names - published CPU numbers.")},
    }
    fps = report["end_to_end"]["fps_at_p95"]
    report["gate"]["fps_at_p95"] = fps
    report["gate"]["fps_gate_on_this_cpu"] = "PASS" if fps and fps >= GATE_FPS else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark PARIKSHAK on this machine.")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--camera", default="synthetic", help="'synthetic' or a camera index")
    ap.add_argument("--seconds", type=float, default=20.0, help="duration for a real camera")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bench.json")
    args = ap.parse_args(argv)

    report = run(args.frames, args.camera, args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    m = report["machine"]
    print(f"{m['processor']} ({m['logical_cpus']} threads), Python {m['python']}, "
          f"OpenCV {m['opencv']}")
    print(f"\n  per stage, {report['stages']['image'][0]}x{report['stages']['image'][1]}, "
          f"{report['stages']['frames']} frames      p50 ms    p95 ms    max ms")
    for name, s in report["stages"]["stages"].items():
        print(f"    {name:<12} {s['p50_ms']:>30.2f} {s['p95_ms']:>9.2f} {s['max_ms']:>9.2f}")
    e = report["end_to_end"]
    s = e["system_per_frame"]
    print(f"\n  end to end ({e['camera']}, {e['frames']} frames): {s['p50_ms']:.1f} ms p50, "
          f"{s['p95_ms']:.1f} ms p95 -> {e['fps_at_p50']} FPS at p50, {e['fps_at_p95']} at p95")
    r = report["engine_replay"]
    print(f"  engine only ({r['procedure']}, {r['steps']} steps): "
          f"{r['per_frame']['p50_ms']:.2f} ms/frame p50 -> {r['frames_per_second_at_p50']} FPS")
    g = report["gate"]
    print(f"\n  >= {g['fps_required']:g} FPS gate on this CPU: {g['fps_gate_on_this_cpu']} "
          f"(p95 basis). Power: not measured - no meter.")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
