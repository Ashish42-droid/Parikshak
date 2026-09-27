# PARIKSHAK — 4-minute demo script

SIH 2026 · PS 26174 · Space Technology

This script shows only what the software on this laptop actually does today. Every
command below has been run. Where PLAN.md §15's original script depends on hardware
or models we do not have (Jetson, trained detector, hand tracking, GStreamer, Piper,
RTSP), the beat is either replaced or said out loud as a limit. **Rehearse it ten
times; keep the synthetic-camera fallback and the browser demo open.**

---

## What is live and what is replayed — say this once, early

| Shown live, from the webcam | Shown by replaying recorded belief |
|---|---|
| Rack lock from 4 printed AprilTags, overlay axes | A skipped step, spoken with its reason |
| Props seen through their tags: position and state (latch closed/open, unit idle) | Picking the wrong vial |
| A spoken crew phrase completing a step ("unit idle") | A hazard (latch open while the unit runs) |
| Blocked camera → "cannot verify", no alarm | The failure-mode reel |
| Rack rotated → lock and axes follow | |

Why the split: with only a webcam and tagged props there is **no hand-tracking or
motion model on this machine**, so any step that asks "is it held?" or "was it
pressed?" reads **not verifiable here** — and the run cannot get past the first such
step. Measured: a live CRX-2 run completes S03 on the crew's phrase, then waits at
S02 ("retrieve vial B", needs hands). It raises **zero alarms** doing so. That
restraint is the point; the deviations themselves are shown from replays, which run
through the identical engine.

---

## Before the judges arrive (T–30 min)

1. **Print** `runs/tags/tags.pdf` at 100% / actual size. Measure one black square: **60.0 mm**. If not, reprint — a scaled sheet puts every distance a few percent wrong silently.
2. **Rack board:** tape tags 101–104 at the corners of a 1.16 m × 0.81 m rectangle, all upright, as on the placement-guide page (centres at x = ±0.55 m, y = +0.30 / −0.45 m).
3. **Props:** tag 203 on "vial B", 211 (closed) on the latch flap, 231 (idle) on the processing unit. Only one state tag of each pair may be visible at a time.
4. **Camera:** ~1.5 m from the board, square-on, all four tags in frame. Even lighting, no glare on the tags.
5. **Terminal 1 (live):**
   ```
   python -m parikshak.run --procedure procedures/crx2_colloid_resuspension.yaml --props racks/props_crx2.json --camera 0
   ```
   Confirm: the status line says RACK LOCKED; vial B, latch [closed] and unit [idle] are drawn on the overlay.
6. **Terminal 2 (fallback, ready but not started):**
   ```
   python -m parikshak.run --camera synthetic --procedure procedures/crx2_colloid_resuspension.yaml --props racks/props_crx2.json
   ```
7. **Terminal 3 (replays + log), in `D:\projects\sih`.** Have `runs/reel/reel.mp4` open in a video player, paused on its first frame.
8. **Browser (backup for every replay beat):** `python -m demo`, or double-click `runs/demo/parikshak_demo.html` if Python misbehaves. It opens on the skipped-latch alert with its reason.
9. `python tools/check.py` → **ALL GATES GREEN**. If not, do not demo the thing that failed.

---

## The script

| Time | Do | Say | Proves |
|---|---|---|---|
| **0:00** | Unplug the network cable / switch Wi-Fi off. | "Everything you are about to see runs offline. No module in the on-board software can open a network connection — a test fails the build if one appears." | Offline, no API |
| **0:20** | Terminal 1 window. Point at the overlay axes and the three props. | "The camera finds four tags on the rack and computes where the rack is. Every position you see is in rack coordinates, not pixels. The vial, the latch and the processing unit are seen through their tags — including the latch's state." | Rack frame, marker perception |
| **0:45** | Click **unit idle** (or say it). Checklist: S03 ticks. | "The procedure asks the crew to confirm the unit is idle. It checks the unit's state and the latch as well as the words — S03 completes." | Crew confirmation + state check |
| **1:00** | Point at S01/S02 rows marked *not verifiable here*. | "Steps that need hand tracking say so. This laptop has no hand model, so it will not guess — a missing model is 'unknown', never an accusation." | Restraint by design |
| **1:15** | Put a hand over the lens for ~5 s. | "Block the camera: no rack lock, 'cannot verify'. No alarm. An occluded camera must never manufacture a deviation." | UNVERIFIED ≠ SKIPPED |
| **1:35** | Rotate the board ~90°. | "Rotate the rack. The lock and the axes follow it — up is the rack's up, not gravity's. The numbers the engine reasons with do not change." | Orientation-agnostic |
| **2:00** | Terminal 3: `python -m parikshak.replay traces/golden/skip_S08_latch.jsonl --speak --view` (or the browser demo, which opens on this alert) | "Now the deviations. This replays a recorded run through the identical engine. The crew skips closing the latch —" read the `[voice]` line and the reason "— it says which step, and why: latch never closed." | Skip, with reason |
| **2:25** | `python -m parikshak.replay traces/golden/wrong_object_vial_b.jsonl --speak` | "Wrong vial: it tells vial A from vial B by what the procedure says they look like." | Semantics, not sequence |
| **2:40** | `python -m parikshak.replay traces/golden/hazard_latch_open_while_running.jsonl --speak` | "A hazard no sequence checker can see: nothing was skipped, the latch was opened while the unit ran." | Invariants |
| **2:55** | `python tools/onboard.py procedures/crx2_colloid_resuspension.yaml` | "CRX-2 is a second experiment written as a file. Same model build, same printed tags — zero retraining. It is the procedure running live in terminal 1." | Zero retraining |
| **3:15** | `python tools/verify_log.py runs/live/<latest>.log.jsonl` and open `runs/downlink_sample/README.txt` | "Every run writes a hash-chained log — edit one byte and it says BROKEN and where. This 140-second run is 8 KB; the same video is about 70 MB. The log goes down first." | Flight record, downlink |
| **3:35** | Play 15–20 s of `runs/reel/reel.mp4` (failure 1). | "And here is where it fails. On 130 degraded CSP-1 runs: 98.7% step accuracy, 97.8% of deviations caught — and 0.85 false alarms per 45 minutes, inside our target of 1. It was 3.0; we found and fixed three causes by replaying failures frame by frame. This clip is one of the four that remain, and the reel says exactly why. CRX-2 meets every target too." | Honesty — "believed about everything else" |
| **4:00** | Stop. | — | |

**Timing rule:** if you are behind at 2:00, drop the wrong-vial replay (2:25). Never drop the honest close.

---

## If something fails on stage

| Failure | Do | Say |
|---|---|---|
| Rack will not lock | Check all 4 tags in frame and not glaring; else switch to Terminal 2 (synthetic) | "Same engine, rendered camera — this is our stage failover." |
| Webcam will not open | Terminal 2 | same |
| Qt window will not open | Add `--headless` | "Same display, as text — it is how the box runs over a serial console." |
| No audio | Nothing — voice lines are printed in replays | "The speech queue decides what to say; the voice is just the output." |
| A replay command errors | Switch to the browser demo, same scenario | "Same engine, in the browser." |
| A false alarm appears live | Leave it on screen | "That is the 0.85-per-45-minutes. It is in our numbers and in the reel." |

---

## Do NOT say

- That it runs on a Jetson, at 25 W, or at a measured edge FPS — **not measured** (Phase 7 is hardware-gated; laptop CPU numbers only).
- That it uses a trained object detector, hand tracking or a motion model — **objects are seen through AprilTag stickers**; no weights exist yet.
- That it streams RTSP — **GStreamer is not installed** here. (It does write a ±10 s video clip around each alert into `runs/live/clips`, with OpenCV.)
- That it never raises a false alarm — **CSP-1 is at 0.85 per 45 min**, measured on synthetic data. Say the number, not "zero".
- Any number from live camera accuracy — **marker precision is measured on rendered frames only**; a real webcam will be worse.
- That the evaluation used real astronaut footage — **it is a synthetic, degraded corpus**.

---

## The five attacks — answers with today's numbers

**1. "Microgravity looks nothing like your data."** Correct, and our data is synthetic. The design keeps learned parts away from gravity: geometry is in rack coordinates from fiducials, which is why rotating the rack changes nothing. The rotation beat shows it; the precision of tag-based positions is measured on renders (60 mm tags: lateral error p95 9 mm, depth p95 34 mm) and must be re-measured on a real camera.

**2. "What is your false-positive rate?"** Honest scoring on 130 degraded CSP-1 runs: **0.85 false alarms per 45 min — within our target of 1**, down from 2.98 after fixing three causes found by frame-by-frame replay. Zero on clean runs, zero on legal reorders, zero on occlusion runs. On CRX-2: **0.00**. The one remaining cause is named in the reel: a vial genuinely moving while it is attached, with the crew moving on 0.2 s inside that step's minimum. Alerts need 0.85 accumulated evidence and 1.5 s persistence; occlusion produces UNVERIFIED, never a skip.

**3. "Why not a checklist tablet?"** A tablet needs the crew to tick every step and cannot catch a mistake they did not notice. We read the same procedure and verify independently, with an automatic hash-chained record.

**4. "This is just if-else on detections."** Decisions are accumulated evidence over time (a CUSUM per step), with duration priors, unordered groups, branches and invariants. "Skipped" is refuted evidence plus a recent contrary sighting — which is why a blocked camera and a missing model both read "unknown" instead of "skipped".

**5. "How would this be certified?"** It is advisory and never commands hardware. No module in the on-board package can open a network connection. Every decision carries a human-readable reason; every run replays deterministically from its trace; the log is hash-chained and names the procedure hash and operating point.

---

## Numbers on the slide, and where they come from

| Claim | Value | Source |
|---|---|---|
| CSP-1 step accuracy (130 degraded runs) | 98.7% | `runs/eval.json` |
| CSP-1 deviation recall | 97.8% (SKIP, HAZARD, WRONG_OBJECT 100%; DURATION 90%) | `runs/eval.json` |
| CSP-1 false alarms / 45 min | **0.85** (target ≤ 1 — met; was 2.98) | `runs/eval.json` |
| Legal-reorder false alarms | 0 (both procedures) | `runs/eval.json`, `runs/eval_crx2.json` |
| CRX-2 (70 runs) | 95.6% accuracy, 97.5% recall, 0.00 FA | `runs/eval_crx2.json` |
| Alert latency | p50 1.6 s, p95 1.8 s | `runs/eval.json` |
| Log vs video | 8,275 B for 140.8 s vs ~70.4 MB (≈ 8,500×) | `runs/downlink_sample/manifest.json` |
| Tag position precision (renders) | 60 mm: lateral p95 9 mm, depth p95 34 mm | `parikshak/perception/markers.py` |
| Laptop CPU speed | 13.4 FPS at the slowest 5% of frames | `runs/bench.json` |
| Live camera-only run | completes S03, waits at the first hand-dependent step, 0 alarms | simulated with `--camera synthetic` |

Regenerate before the event:

```
python tools/eval_report.py --json runs/eval.json
python tools/eval_report.py --traces traces/eval_crx2 --procedure procedures/crx2_colloid_resuspension.yaml --json runs/eval_crx2.json
python tools/make_reel.py docs/reel/segments.json
python tools/make_downlink_sample.py
python tools/make_demo_page.py
```

---

## Rehearsal log

| # | Date | Who ran it | Time | What broke | Fixed? |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |
