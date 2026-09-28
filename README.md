# PARIKSHAK — on-board procedure witness

SIH 2026 · problem statement 26174 · Space Technology

An astronaut runs a science procedure on the space station. PARIKSHAK watches
through a camera, checks every step against the procedure file, and speaks up
only when it is sure something went wrong — naming the step and the reason.
When it cannot see, it says "cannot verify" instead of guessing. Every run
writes a small, tamper-evident log that goes down to the ground first.

## See it in two minutes

```bash
python -m demo
```

Opens a page in your browser with two ways in.

**Perform an experiment.** Choose CSP-1 or CRX-2 and the engine guides you
through it, step by step: it prompts, you decide what the crew does — do the
step, skip it, take the wrong item, work with the camera blocked, or wait — and
it responds. Nothing is scripted. Skip a step and the engine notices for itself
when a later one verifies, names it, and says why. At the end you get the run's
summary and its hash-chained record, verified.

**Watch a recorded run.** Fourteen recorded runs — a skipped latch, the wrong
vial, a hazard, a blocked camera, lost markers, a second experiment — replayed
through the engine with playback controls, a timeline and the flight record.

Both show the rack as the engine believed it, the prompt the crew hears, and
every alert with its reason. Nothing on the page is canned: the engine decides
live, from belief frames, while you watch.

To send the demo to someone who does not have Python:

```bash
python tools/make_demo_page.py
```

This writes `runs/demo/parikshak_demo.html`, one file with every run embedded.
It opens by double-click, offline.

## Other ways to run it

| What | Command |
|---|---|
| Desktop operator window, replaying a run | `python -m parikshak.replay traces/golden/skip_S08_latch.jsonl --window` |
| Replay in the terminal, spoken | `python -m parikshak.replay traces/golden/skip_S08_latch.jsonl --speak` |
| Live run on a rendered camera (no hardware) | `python -m parikshak.run --camera synthetic --procedure procedures/crx2_colloid_resuspension.yaml --props racks/props_crx2.json` |
| Live run on a webcam with printed tags | Print `runs/tags/tags.pdf` at 100 %, then the same command with `--camera 0` |
| Check a run log has not been altered | `python tools/verify_log.py runs/live/<run>.log.jsonl` |
| Every gate: tests, validators, golden replays | `python tools/check.py` |
| Accuracy on the degraded corpus | `python tools/eval_report.py` (CSP-1) · add `--traces traces/eval_crx2 --procedure procedures/crx2_colloid_resuspension.yaml` for CRX-2 |
| Speed and 45-minute endurance | `python tools/bench.py` · `python tools/soak.py --minutes 45 --stream` |

A live run writes its log, its trace and a ±10 s video clip around every alert
into `runs/live/`.

## Install

Python 3.12 or newer.

```bash
pip install -e ".[web,gui,dev]" opencv-contrib-python pyttsx3
```

The engine alone needs only NumPy, PyYAML and jsonschema. OpenCV is needed for
live runs, PySide6 for the desktop window, FastAPI and Uvicorn for the browser
demo. Licences: [docs/LICENCE_STATEMENT.md](docs/LICENCE_STATEMENT.md).

## What is measured

Degraded synthetic runs — added noise, dropped frames, occlusion — each scored
against the error injected into it. Numbers from `runs/eval.json` and
`runs/eval_crx2.json`, 22 September 2026.

| Measure | Target | CSP-1 (14 steps, 130 runs) | CRX-2 (8 steps, 70 runs) |
|---|---|---|---|
| Step accuracy | ≥ 95 % | 98.7 % | 95.6 % |
| Deviations caught | ≥ 90 % | 97.8 % | 97.5 % |
| False alarms per 45 min | ≤ 1 | 0.85 | 0.00 |
| False alarms on permitted reorders | 0 | 0 | 0 |
| Alert delay, median / 95th percentile | ≤ 2 s | 1.6 / 1.6 s | 1.6 / 1.8 s |

On a laptop CPU the full loop runs at 13.4 frames per second in its slowest 5 %
of frames (target ≥ 10), and a 45-minute run grows memory by 7.4 MB per hour.

## Where it stands

| Phase | Status |
|---|---|
| 0 Project spine, procedure format, validator | Done |
| 1 Belief frame (shared contract) | Done |
| 2 Procedure engine: steps, evidence, deviations, alerts, hash-chained log | Done |
| 3 Synthetic runs and evaluation | Done — every target met on both experiments |
| 4 Perception | Done — AprilTag fiducials + deep learning suite (YOLOv8, YOLO-Pose, ContactMLP, MotionTCN in pure NumPy) |
| 5 Camera, voice commands, speech | Done (pyttsx3 voice; no Piper, no speech recogniser installed) |
| 6 Operator GUI, alert clips, flight record | Done; RTSP video stream needs GStreamer on the Jetson |
| 7 Edge deployment | Laptop benchmark and soak done; not run on a Jetson, power not measured |
| 8 Second procedure, failure reel, demo script | Done; rehearsal is the team's |

Honest limits: the evaluation data is synthetic; flight deployment requires
fine-tuning on microgravity parabolic flight footage; power measurement (≤ 25 W)
requires a Jetson Orin with a hardware power meter.

## Repository

| Path | What |
|---|---|
| `parikshak/` | The on-board software. No module in it can open a network socket — enforced by `tests/test_offline.py` |
| `parikshak/engine/` | Predicates, step tracker, deviations, alert policy, run log |
| `parikshak/perception/` | Rack lock from AprilTags, marker-based objects, belief frames |
| `parikshak/io/` | Camera, speech, voice commands, clips, downlink packaging |
| `parikshak/gui/` | Desktop operator window |
| `demo/` | The browser demo (outside `parikshak/` because it runs a web server) |
| `procedures/` | CSP-1 and CRX-2, the experiments as files |
| `traces/` | Golden runs and the degraded evaluation corpus |
| `tools/` | Checks, evaluation, benchmark, reel, tag sheets, demo page |
| `deploy/jetson/` | Edge bring-up package — not yet run on hardware |
| `docs/` | Demo script, licence statement, failure reel segments |
| `PLAN.md` | The full build plan |
