#!/usr/bin/env python3
"""Board power from a tegrastats log, against the 25 W edge budget.

    tegrastats --interval 200 --logfile runs/power/tegrastats.log   (on the Jetson)
    python tools/power_report.py runs/power/tegrastats.log --bench runs/bench.json

PLAN.md Phase 7 gates the edge build on >= 10 FPS at <= 25 W "on a meter". A USB
meter on the barrel jack is the number to put on a slide; tegrastats is the
number to log while the benchmark runs, because it is timestamped against the
run. This reads it.

Two rail formats, because the name changed between Jetson generations:
    Orin (JetPack 5/6):   VDD_IN 5316mW/5316mW         instantaneous/average
    Nano / TX2 (older):   POM_5V_IN 3405/3405           instantaneous/average, mW

Only the board input rail is summed. Adding the GPU and CPU rails on top of it
would count their power twice - they are already inside the input rail.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

POWER_LIMIT_W = 25.0
FPS_REQUIRED = 10.0

_INPUT_RAIL = re.compile(r"\b(?:VDD_IN|POM_5V_IN)\s+(\d+)(?:mW)?/(\d+)(?:mW)?")


def parse(lines) -> list[float]:
    """Instantaneous board input power per sample, in watts."""
    out: list[float] = []
    for line in lines:
        m = _INPUT_RAIL.search(line)
        if m:
            out.append(int(m.group(1)) / 1000.0)
    return out


def summarise(watts: list[float], interval_ms: float = 200.0,
              fps: float | None = None) -> dict:
    if not watts:
        return {"samples": 0, "verdict": "NO DATA",
                "note": "no VDD_IN or POM_5V_IN field found - is this a tegrastats log?"}
    ordered = sorted(watts)
    mean = statistics.fmean(watts)
    result = {
        "samples": len(watts),
        "duration_s": round(len(watts) * interval_ms / 1000.0, 1),
        "mean_w": round(mean, 2),
        "p95_w": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        "peak_w": round(ordered[-1], 2),
        "power_limit_w": POWER_LIMIT_W,
        "power_ok": ordered[-1] <= POWER_LIMIT_W,
    }
    if fps is not None:
        result["fps"] = fps
        result["fps_ok"] = fps >= FPS_REQUIRED
        result["frames_per_watt"] = round(fps / mean, 2) if mean else None
    gate = result["power_ok"] and result.get("fps_ok", True)
    result["verdict"] = "PASS" if gate else "FAIL"
    if fps is None:
        result["verdict"] += " (power only - pass --bench for the FPS half)"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise tegrastats board power.")
    ap.add_argument("log", type=Path)
    ap.add_argument("--interval-ms", type=float, default=200.0)
    ap.add_argument("--bench", type=Path, default=None,
                    help="runs/bench.json from the same board, for the FPS half of the gate")
    args = ap.parse_args(argv)

    fps = None
    if args.bench is not None:
        bench = json.loads(args.bench.read_text(encoding="utf-8"))
        fps = bench["end_to_end"]["fps_at_p95"]
    result = summarise(parse(args.log.read_text(encoding="utf-8", errors="replace").splitlines()),
                       args.interval_ms, fps)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
