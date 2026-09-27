# PARIKSHAK on a Jetson — edge deployment package

> **Status: not yet run on hardware.** Everything in this folder is written and
> reviewed, but no Jetson has been available. Commands are correct against
> NVIDIA's published JetPack tooling; they are unexercised until a board exists.
> Treat every number in the gate table below as empty until it is measured.

PLAN.md Phase 7: *ONNX → TensorRT INT8, Jetson bring-up, power measurement,
latency p50/p95. Gate: ≥ 10 FPS at ≤ 25 W on a meter.* The plan's own fallback if
the Jetson slips is "laptop GPU + published CPU numbers" — that half is done:
`runs/bench.json` (see *Measured without a Jetson* below).

---

## What is measured already, without a Jetson

| | Value | Where |
|---|---|---|
| Machine | Intel Core 11th gen (Tiger Lake), 8 threads, Windows 11, Python 3.14, OpenCV 5.0 | `runs/bench.json` |
| End to end, 1280×720, CPU only | **34.6 ms p50 → 28.9 FPS; 74.8 ms p95 → 13.4 FPS** | `python tools/bench.py` |
| ≥ 10 FPS gate on this CPU | PASS on the p95 basis | same |
| Largest stages | tag detection 18 ms, prop poses 6 ms, overlay 3 ms (p50) | same |
| Engine alone (CSP-1, 16 steps) | ~1.2 ms per frame | same |
| Power | **not measured** — no meter | — |

The overlay was 29 ms p50 until projection was batched; tag detection at reduced
resolution was tried and rejected (≤ 2 ms faster at 0.75×, camera-position
error ×3; at 0.5× tags start to be missed).

**Read single CPU numbers as ±2×.** The soaks below ran on the same laptop on
battery with other software loading the CPU, and median tick time on unchanged
code ranged from 6 ms to 34 ms between runs. Only comparisons made back to back,
or A/B inside one process, are evidence. Re-run `tools/bench.py` on mains power
before quoting it.

### Soak: 45 simulated minutes (`tools/soak.py`)

CRX-2 on the bench scene, all 7 tags in view, the camera blacked out 5 s every
minute, rack locked on 97% of ticks. Checks that latency does not drift and that
every temporal history stays inside its trim window.

| Recording | Histories (longest / bound) | Memory growth | Drift, last third / first third | Verdict |
|---|---|---|---|---|
| off | 301 / 303, bounded | 7.8 MB per hour | 0.93× | PASS |
| frames held in memory | 301 / 303, bounded | **105 MB per hour** | 2.06× | FAIL on drift - machine, not code (below) |
| **streamed to disk** (`parikshak run` default) | 301 / 303, bounded | **7.4 MB per hour** | 1.35× | PASS - trace 45.8 MB on disk |

Streaming brings recording down to the memory cost of not recording at all.

**How drift is judged, and why it changed.** It was the last minute's median
tick over the first minute's. On this laptop that one ratio read 0.56×, 6.7× and
2.7× across three runs of code that does the same work every frame, so it
measured the CPU, not the code. A per-stage split of a full 45-minute run
settled it: the engine sat on the same step with one frontier step, one
transition and one notice for all 45 minutes, while tick time wandered between
6.6 and 15.6 ms - tag detection, engine and display rising and falling together,
in a fixed 7 : 1 : 2.5 proportion. A leak is a trend, so the gate is now the
median of the last third of minutes over the first third, minute one skipped
(`soak.drift_thirds`), limit 1.5×. It still cannot tell a machine that slows for
a whole third of the run from code that does; on a Jetson in MAXN with
`jetson_clocks`, the CPU does not wander and the ratio means what it says.

The held-in-memory run was flat at ~12 ms for 32 minutes while 19 000 frames
accumulated, then climbed as the machine slowed; its wall time was 86 minutes
for 6 minutes of compute. An interleaved A/B in one process settles it: holding
27 000 recorded frames (418 k tracked objects against 40 k) gave tick p50
16.5 ms against 19.4 ms without, and **no full garbage collection ran during any
of 1 800 ticks** in either arm. Holding frames does not slow the loop.

The memory growth is real, though, and it is why `parikshak run` now streams
its trace (`TraceWriter`): frames go to `<trace>.partial` as they arrive, and the
closed file is byte-identical to the buffered one. A crash leaves the partial
readable.

---

## 1. Bring-up

1. Flash **JetPack 6** (Orin Nano / NX) with SDK Manager. Record the L4T version: `cat /etc/nv_tegra_release`.
2. Maximum performance mode for measurement, and note it in the report — power numbers mean nothing without the mode:
   ```
   sudo nvpmodel -q          # record the current mode
   sudo nvpmodel -m 0        # MAXN
   sudo jetson_clocks
   ```
3. Python environment. Use JetPack's system OpenCV, which is built with GStreamer and CUDA; a pip `opencv-python` wheel is not:
   ```
   sudo apt install python3-venv python3-opencv python3-gi gstreamer1.0-tools
   python3 -m venv --system-site-packages ~/parikshak-venv
   . ~/parikshak-venv/bin/activate
   pip install -e ".[gui]"
   ```
4. Copy the repository to `/opt/parikshak` and run the checks before anything else:
   ```
   python tools/check.py
   ```

## 2. Camera

- **USB camera:** `python -m parikshak.run --camera 0 --headless ...` (OpenCV / V4L2).
- **CSI camera (IMX296):** the GStreamer capture graph in `parikshak/io/capture.py` builds `nvarguscamerasrc` → tee → recorder + perception + RTSP. Check it parses on the board:
  ```
  python -c "from parikshak.io.capture import CaptureConfig, build_pipeline; print(build_pipeline(CaptureConfig(source='csi', record_dir='runs/rec')))"
  gst-launch-1.0 <paste the printed pipeline>
  ```

## 3. Run at boot

`parikshak.service` starts the live run headless when the board powers on, and
restarts it if it exits. Logs and traces go to `/opt/parikshak/runs/live`.

```
sudo cp deploy/jetson/parikshak.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now parikshak
journalctl -u parikshak -f
```

Cold-boot test (PLAN.md W12): power off at the wall, power on, and time until the
first `RACK LOCKED` line appears in the journal. Record it below.

## 4. Latency and power, together

Run both at once so the power log covers exactly the benchmark:

```
./deploy/jetson/log_power.sh runs/power/tegrastats.log &     # starts tegrastats
python tools/bench.py --camera 0 --seconds 60
kill %1
python tools/power_report.py runs/power/tegrastats.log --bench runs/bench.json
```

`power_report.py` reads the board input rail (`VDD_IN` on Orin, `POM_5V_IN` on
older boards) and reports mean, p95 and peak watts, frames per watt, and the gate
verdict. **tegrastats is not the meter the gate names** — put a USB-C or barrel
power meter inline as well and record its reading beside it.

## 5. Models: ONNX → TensorRT INT8 (when weights exist)

There are no trained weights yet — objects are seen through AprilTag props. When
the RT-DETR detector exists as `models/detector.onnx`:

```
/usr/src/tensorrt/bin/trtexec --onnx=models/detector.onnx \
    --saveEngine=models/detector_int8.engine \
    --int8 --calib=models/calibration.cache \
    --memPoolSize=workspace:2048 --shapes=images:1x3x640x640
```

INT8 needs a calibration cache built from ~500 representative frames from the
recording campaign (not synthetic renders). Validate the INT8 engine against the
FP16 engine on the held-out set before trusting it: a calibration that shifts
detector scores moves every threshold the ROC sweep chose.

## 6. The gate — fill in on hardware

| Board | JetPack / mode | Camera | FPS p50 | FPS p95 | Watts mean (tegrastats) | Watts peak (meter) | Cold boot → lock | Verdict |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

Gate: **FPS p95 ≥ 10 and meter peak ≤ 25 W.**
