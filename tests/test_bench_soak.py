"""The benchmark and soak tools produce every number they report, on tiny runs.

The real runs are minutes long and their numbers go in the edge report; these
tests keep the tools from rotting between those runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import bench  # noqa: E402
import soak  # noqa: E402


def test_the_benchmark_reports_every_stage_and_the_gate():
    report = bench.run(frames=12)
    stages = report["stages"]["stages"]
    assert set(stages) == {"tags", "rack", "props", "perception", "engine", "display", "overlay"}
    assert all(s["n"] > 0 and s["p50_ms"] > 0 for s in stages.values())
    assert report["end_to_end"]["frames"] == 12
    assert report["gate"]["power_measured"] is False
    assert report["gate"]["fps_gate_on_this_cpu"] in ("PASS", "FAIL")
    assert report["engine_replay"]["per_frame"]["p50_ms"] > 0


def test_a_short_soak_keeps_its_histories_bounded():
    result = soak.run(minutes=0.5, fps=10.0, sample_every_s=10.0)
    assert len(result["windows"]) >= 2
    assert result["histories_bounded"], (result["longest_history_series"],
                                         result["history_series_bound"])
    assert result["windows"][-1]["recorded_frames"] == result["frames"]
    assert result["latency_drift_last_over_first_p50"] is not None
    # A soak over a scene the detector cannot lock onto would time the cheap path.
    assert result["windows"][0]["rack_locked_fraction"] > 0.9
    assert all(w["rss_bytes"] for w in result["windows"]), "resident memory went unmeasured"


def test_drift_is_a_trend_not_one_fast_minute():
    def windows(p50s):
        return [{"p50_ms": p} for p in p50s]
    # One quick warm-up minute must not fail a flat run...
    assert soak.drift_thirds(windows([3.0] + [8.0] * 44)) == 1.0
    # ...a noisy flat run stays near 1...
    noisy = [8.0, 12.0, 7.0, 9.0, 15.0, 8.0] * 7 + [8.0, 9.0, 8.0]
    assert 0.8 <= soak.drift_thirds(windows([5.0] + noisy)) <= 1.25
    # ...a history that never trims, slowing every minute, is caught...
    assert soak.drift_thirds(windows([8.0 + 0.5 * m for m in range(45)])) > soak.DRIFT_LIMIT
    # ...and a short run neither divides by zero nor needs three windows.
    assert soak.drift_thirds(windows([4.0, 8.0, 8.0])) == 1.0
    assert soak.drift_thirds(windows([8.0])) == 1.0
    assert soak.drift_thirds([]) is None


def test_a_streaming_soak_records_to_disk_and_cleans_up(tmp_path):
    trace = tmp_path / "soak.trace.jsonl"
    result = soak.run(minutes=0.2, fps=10.0, mode="stream", sample_every_s=6.0, trace_path=trace)
    assert result["recording"] == "stream"
    assert result["windows"][-1]["recorded_frames"] == result["frames"]
    assert result["streamed_trace_bytes"] > 0
    assert not trace.exists() and not trace.with_name(trace.name + ".partial").exists()
    assert result["histories_bounded"]
