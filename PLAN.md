# PARIKSHAK — Build Plan

**On-board procedure witness for crewed missions**

SIH 2026 · Space Technology · Software
Problem statement: AI-based Human Activity Recognition for validating scientific
experiment sequences on-board (BAS / lunar missions), issued by ISRO / Dept. of Space.

> **PS ID: `26174`**

---

## Table of contents

1. [What we are actually building](#1-what-we-are-actually-building)
2. [Problem decode](#2-problem-decode)
3. [Architecture](#3-architecture)
4. [Complete tech stack](#4-complete-tech-stack)
5. [Repo layout and module contracts](#5-repo-layout-and-module-contracts)
6. [Current state](#6-current-state)
7. [The three decisions that determine whether this ships](#7-the-three-decisions-that-determine-whether-this-ships)
8. [Critical path](#8-critical-path)
9. [Week-by-week plan](#9-week-by-week-plan)
10. [Hardware and BOM](#10-hardware-and-bom)
11. [Datasets](#11-datasets)
12. [Data collection campaign](#12-data-collection-campaign)
13. [Team split](#13-team-split)
14. [Evaluation protocol](#14-evaluation-protocol)
15. [The 36 hours](#15-the-36-hours)
16. [Descope ladder](#16-descope-ladder)
17. [Risk register](#17-risk-register)
18. [Research references](#18-research-references)
19. [Deliverables checklist](#19-deliverables-checklist)
20. [Appendix: commands](#20-appendix-commands)

---

## 1. What we are actually building

Not a human-activity classifier. An **on-board procedure-compliance witness** that
replaces the ground controller who cannot be in the loop, and produces a
downlinkable flight record of what the crew actually did.

Three cameras watch a payload rack. A 25 W box verifies every procedure step,
prompts the next one by voice, catches skips and wrong objects within two seconds,
and writes a 40 KB tamper-evident log instead of 1.35 GB of video.

### The one line we repeat to every judge

> **The neural network never learns your experiment — it learns hands and objects.
> The experiment is a file. So a new payload procedure is onboarded in minutes,
> with zero retraining.**

### Why ground support genuinely cannot do this

| Destination | Round-trip light time |
|---|---|
| Earth–Moon (384,400 km) | **~2.6 s** |
| Earth–Mars | **6–44 min** |

Before processing and human reaction. A controller cannot say *"stop, you skipped
the latch"* in time. This is a physics argument, not a convenience argument. Lead
with it.

---

## 2. Problem decode

### Real users vs. who signs off

- **User:** the crew member at the rack — hands busy, eyes on the work, no keyboard.
- **Second user:** the ground PI who reads the log weeks later and must decide
  whether the science is valid.
- **Signs off on adoption:** Human Space Flight Centre payload-ops management and
  the software-assurance group. They care about **false-alarm rate, explainability
  and certifiability** — not our mAP.

### What "solved" means, measurably

| Metric | Target |
|---|---|
| Step recognition accuracy (held-out runs) | ≥ 95 % |
| Skip / out-of-order detection recall | ≥ 90 % |
| **False alarms** | **≤ 1 per 45 min of monitored operation** |
| Alert latency after deviation | ≤ 2 s |
| Throughput on edge SoC | ≥ 10 FPS @ ≤ 25 W |
| Log size vs. raw video | ~40 KB vs ~1.35 GB (**~34,000×**) |
| New experiment onboarding | minutes, **zero retraining** |

Bandwidth arithmetic: 45 min at 1080p30 H.265 ≈ 4 Mbps → 4 × 2700 = 10,800 Mb =
**1.35 GB per camera**. A 24-step structured log ≈ **40 KB**. With event clips
(±10 s per deviation, ~2 per run ≈ 20 MB) it is still **~68×** less while keeping
visual evidence.

### The hidden requirement 80 % of teams will miss

> **The system must onboard a brand-new experiment without retraining the network.**

ISRO does not have one experiment. BAS will fly a catalogue of payload procedures
and new ones every increment. Every other team will train an N-class action
classifier where N = the steps of their one demo experiment. The moment a judge
says *"now do a different experiment,"* that submission is dead.

**Perception is learned. Procedure is data.**

Runners-up that also get missed:
- **False-alarm economics** — one spurious alert in real ops and the speaker gets
  taped over.
- **You cannot train or evaluate a deviation detector on nominal runs only.**

### What everyone else will submit

YOLOv8 + MediaPipe + an LSTM over clips → 8-class step classifier → Streamlit
dashboard → `pyttsx3` says "step skipped" → demo plays a pre-recorded MP4.

We are not doing that.

---

## 3. Architecture

Three solution directions were considered:

| | Core bet | Main risk |
|---|---|---|
| **A** End-to-end video action recognition (VideoMAE/X3D) | A strong backbone learns step semantics directly | Data-hungry, opaque, weak on fine-grained hand-tool actions, **new experiment = retrain** |
| **B** Perception → symbolic procedure engine | Lab steps decompose into verifiable predicates | Primitive errors cascade; procedures authored by hand |
| **C** Digital-twin, synthetic-first, full 3D HMR | The microgravity gap is *the* problem and only simulation touches it | Sim-to-real gap; HMR at 10 FPS on 25 W is genuinely hard |

**Chosen: B as the architecture, C as the data engine, a thin slice of A** for
genuinely motion-defined steps (a small TCN, not a video transformer).

B is the only one that answers the hidden requirement. It also gives a
human-readable justification for every alert, which is what makes it certifiable,
and it turns the log from an afterthought into a byproduct of the reasoning.

### Data flow

```
[2-3 fixed rack cameras · global shutter · 1080p30]
      |
      v
[GStreamer capture + tee] --+--> [NVENC H.265 -> splitmuxsink -> local NVMe ring buffer]
      |                     +--> [RTSP/SRT -> configured ground IP:port]
      v
[Frame scheduler: 10 Hz -> perception, 30 Hz -> recorder]
      |
      +--> [RT-DETR-R18 INT8]  ---> boxes, classes --> [ByteTrack] --> track IDs
      +--> [MediaPipe Hands]   ---> 21 kpts x2 -----> [Contact MLP] --> grasp/manipulate
      +--> [RTMPose-m -> 3D lifter] --> 3D joints
      +--> [AprilTag 36h11 -> PnP]   --> camera->rack [R|t]
      |
      v
[RACK-FRAME CANONICALISER]   <-- gravity is never used; "up" = rack +Z
      |
      v
[WORLD STATE BELIEF]  { object_i: {pos_rack, state, held_by, conf},
                        hand_L/R: {contact_with, grasp_type, conf},
                        body: {joints_rack, conf}, occlusion_mask }
      |         ^
      |         +--- [Motion-step TCN, 2 s window]  (insert / rotate_seal / press)
      v
[PROCEDURE ENGINE]  <---- procedure.yaml { steps, preconditions, effects,
                                           duration priors, hazards, next[] }
   - HSMM over step index with duration priors
   - predicate evaluation -> IN_PROGRESS / COMPLETE / UNVERIFIED
   - deviation classifier -> SKIP | OUT_OF_ORDER | WRONG_OBJECT | REPEAT | DURATION | HAZARD
      |
      +--> [ALERT POLICY: conf gate p>=0.85, persistence >=1.5 s, 3-tier escalation]
      |         +--> [Piper TTS -> speaker]
      +--> [NEXT-STEP PROMPTER] --> GUI + TTS
      +--> [HASH-CHAINED JSONL LOGGER] --> run_log.jsonl + .txt --> CFDP package
      +--> [EVENT CLIP EXPORTER] --> deviation_00x.mp4 (+/-10 s)
      |
      v
[PySide6 GUI: live overlay | checklist | timeline | deviation feed | replay | export]
```

### The orientation-agnostic trick

We do **not** train a gravity-free 3D human mesh recovery model. Instead:

1. AprilTag 36h11 fiducials are bonded to the rack face.
2. PnP recovers camera→rack extrinsics **every frame**.
3. All joints, object centroids and zone tests are expressed in **rack
   coordinates**.

Gravity never enters the pipeline. Rotate the rack, the camera or the crew member
and every number is unchanged. Orientation-agnosticism becomes a property of the
coordinate system, not of a learned model. This is ~200 lines, not a research
project.

Frame convention: origin at the fiducial cluster centroid; `+X` across the rack
face; `+Y` along it (named "up" **for readability only** — no gravitational
meaning); `+Z` out of the face toward the crew.

---

## 4. Complete tech stack

Every learned component is a published, pretrained model with permissive weights.
**There is no novel architecture to invent.**

### Perception

| Component | Model | Size / params | Notes |
|---|---|---|---|
| Object detection | **RT-DETR-R18** | ~20 M | Apache-2.0. Train fast with Ultralytics YOLO11s if you like, but **Ultralytics is AGPL-3.0** — swap before delivery and say so on the slide. PSU judges notice licensing. |
| Tracking | **ByteTrack** | — | Persistent IDs across occlusion |
| Hands | **MediaPipe Hands** (BlazePalm + landmark) | ~2 MB | 21 keypoints × 2 |
| Hand-object contact | Custom 3-layer MLP | ~50 k | Input: hand-kpt→box distances, IoU, relative velocity. Output: `{none, reach, grasp, manipulate, release}` |
| Body pose 2D | **RTMPose-m** (MMPose) | Apache-2.0 | 26-kpt Halpe |
| 3D lift | **MotionBERT-lite** or **VideoPose3D** | — | 27-frame window → 3D joints |
| Rack frame | **AprilTag 36h11** + PnP | — | ≥ 2 tags for lock |
| Motion steps | **1D-TCN** (MS-TCN++ style) | ~0.4 M | 2 s window over hand kpts + contact + object states |
| *(optional)* VLM adjudicator | **Florence-2-base** (0.23 B) or **SmolVLM-500M** | — | Only when HSMM posterior is ambiguous. Not in the hot loop. SHOULD BUILD. |

### Reasoning

| Component | Implementation |
|---|---|
| Procedure definition | **PDL v1.0** — YAML, closed predicate vocabulary, validated at load |
| Sequence tracking | **Hidden semi-Markov model** over the procedure graph with duration priors (~300 LOC numpy) |
| Predicate evaluation | Probabilistic. `all`→min, `any`→max, `not`→1−p, `hold_for`→min over window. **`null` under occlusion propagates to UNVERIFIED** |
| Deviation taxonomy | `SKIP` · `OUT_OF_ORDER` · `WRONG_OBJECT` · `REPEAT` · `DURATION` · `HAZARD` |
| Alert policy | Confidence gate ≥ 0.85, persistence ≥ 1.5 s, 3-tier escalation, cooldown, max 2 alerts/step |

### I/O

| Component | Tool | Notes |
|---|---|---|
| Capture / record / stream | **GStreamer** | `tee` → `nvv4l2h265enc` → `splitmuxsink` (60 s segments) + `rtspclientsink`/SRT |
| TTS | **Piper** (rhasspy) | ONNX, ~60 MB voice, real-time on CPU, fully offline |
| ASR *(SHOULD BUILD)* | **Vosk small-en** or **whisper.cpp base.en INT8** | ~20-command grammar: `mark done`, `repeat`, `override` |
| Logging | JSONL + **SHA-256 hash chain** | Tamper-evident. Embeds procedure hash + weights hashes |
| Downlink packaging | **CCSDS CFDP** (727.0-B) | Priority: log high, clips medium, video low |
| GUI | **PySide6 / Qt6** | Same process as GStreamer, zero IPC, 30 FPS overlays, **no browser runtime** — defends the "standalone offline system" claim |

### Training and tooling

| Purpose | Tool |
|---|---|
| Synthetic data | **BlenderProc2** (DLR) or **NVIDIA Isaac Sim Replicator** |
| Synthetic humans | **SMPL-X** bodies driven by **AMASS** mocap |
| Labelling | **CVAT** or **Label Studio**, ByteTrack propagation between keyframes |
| Edge deployment | **ONNX** → **TensorRT INT8** |
| Pose framework | **MMPose** (OpenMMLab) |
| Validation | `tools/validate_procedure.py` (built) + `jsonschema` |
| Language | Python 3.11+ (3.14 verified locally) |

### On Bhashini

Crew ops are in English. **Do not bolt Bhashini onto astronauts to score buzzword
points** — a judge will call it out. Use AI4Bharat / Bhashini TTS in the
**terrestrial spin-off** (plant operators, loco maintenance, cleanroom SOPs) and
say exactly that.

---

## 5. Repo layout and module contracts

```
parikshak/
  pdl/            procedure loader + validator                  [DONE]
  belief/         BeliefFrame, trace read/write                 <- FREEZE WEEK 1
  perception/
    detect.py     RT-DETR wrapper          -> Detections
    hands.py      MediaPipe                -> HandLandmarks
    contact.py    contact MLP              -> ContactState
    pose.py       RTMPose + 3D lifter      -> Joints3D
    rackframe.py  AprilTag PnP             -> extrinsics, canonicalise
    motion.py     1D-TCN                   -> MotionBelief
    pipeline.py   composes the above       -> BeliefFrame stream
  engine/
    predicates.py evaluate a predicate tree against a BeliefFrame
    hsmm.py       step-index tracking with duration priors
    deviations.py SKIP | OUT_OF_ORDER | WRONG_OBJECT | DURATION | HAZARD
    alerts.py     confidence gate, persistence, escalation
    logger.py     hash-chained JSONL
  io/
    capture.py    GStreamer tee: local H.265 + RTSP/SRT
    tts.py        Piper
    asr.py        Vosk command grammar
  gui/app.py      PySide6
  eval/
    harness.py    metrics, ROC, confusion, per-occlusion strata
    replay.py     trace replay
  data/
    synth/        BlenderProc scene + randomisation config
    label/        CVAT project config, class map
    fetch/        dataset download + subset scripts
procedures/       CSP-1 + template                              [DONE]
schema/           PREDICATES.md + procedure.schema.json         [DONE]
tools/            validate_procedure.py, test_validator.py      [DONE]
```

### The hard rule

> `perception/` may not import `engine/`.
> `engine/` may not import `perception/`.
> Both import `belief/`.

If anyone breaks that, the parallelism collapses and six people end up watching
one person debug.

---

## 6. Current state

| File | Lines | Status |
|---|---|---|
| `schema/PREDICATES.md` | 141 | Closed predicate vocabulary — the perception↔symbolic contract |
| `schema/procedure.schema.json` | 457 | JSON Schema 2020-12, editor autocomplete + structural gate |
| `procedures/csp1_colloid_sample_processing.yaml` | ~614 | Worked 14-step procedure + 2 fault-branch steps |
| `procedures/_template.yaml` | 216 | Authoring skeleton |
| `tools/validate_procedure.py` | ~640 | Load-time semantic validator |
| `tools/test_validator.py` | 150 | 15 negative tests |

**Verified:** semantic validator PASS (0 errors, 0 warnings under `--strict`),
JSON Schema PASS (0 structural errors), negative suite **15/15 caught**.

Starting with the PDL looks strange — no models, no cameras. It is not. The
procedure file is the interface every other module is written against. Most teams
start with `yolo train` and discover in week 9 that nobody agreed what the model's
output feeds into.

### Three real defects the tooling already caught

1. Header declared `estimated_duration_s: 1640`; the steps sum to **381**. A 4×
   error in a number a judge would recheck. Now a hard validator error.
2. Six steps had `objects:` lists drifting from their verification logic — the
   list drives GUI highlight and evidence capture.
3. **`states: [off, amber, ...]` — YAML 1.1 parses bare `off` as boolean `False`.**
   Every `state_is(indicator, off)` would have failed at run time, on the rack,
   with no error anywhere.

That third one is the argument for having a load-time gate at all. Say it out loud
to a judge.

---

## 7. The three decisions that determine whether this ships

### 7.1 Freeze the `BeliefFrame` in week 1 and never break it

```python
@dataclass
class BeliefFrame:
    t_mono: float                      # monotonic, seconds since run start
    t_utc: str
    frame_lock: bool                   # rack extrinsics valid this frame?
    objects: dict[str, ObjectBelief]   # pos_rack, state+conf, track_id, visible, conf
    hands: dict[str, HandBelief]       # left/right: contact_with, grasp_type, conf
    body: BodyBelief | None            # joints in rack coords, conf
    motion: MotionBelief               # TCN class + conf over trailing window
    occlusion: dict[str, float]        # zone name -> fraction occluded
```

Perception **writes** it. The engine **reads** it. It serialises to JSONL, one line
per frame.

The consequence is the whole parallelism plan: the engine can be built, tested and
tuned against hand-written JSONL traces with **no camera, no GPU and no model**.
Two people work on the engine from week 1 while four are still building a cardboard
rack.

### 7.2 Build the trace replay harness in week 1, before anything else

```bash
python -m parikshak.replay traces/run_017_skip_S08.jsonl \
    --procedure procedures/csp1_colloid_sample_processing.yaml
```

Every recorded run produces a trace. Every trace is a regression test. When the
alert policy changes in week 10, re-run 60 traces in 40 seconds and see exactly
which runs changed verdict. Without this you are tuning thresholds by re-watching
video, and you will run out of weeks.

It is also finale insurance: if a camera fails on stage, replay a trace through the
identical engine and the GUI looks the same.

### 7.3 Capture step ground truth at record time, for free

Put a **footswitch** (or a spare keyboard) next to the performer. They tap at each
step boundary. That timestamp stream *is* the step segmentation label — and the
intended sequence is known because we handed them the script.

This collapses labelling from "annotate 1.6 M frames with step labels" to
"annotate ~8 k keyframes with object boxes." It is the difference between a
feasible campaign and a dead project.

---

## 8. Critical path

```
    HARDWARE ORDER ────────────────────────► (3-5 wk lead, START DAY 1)
                                                    │
    MOCK RACK ──► RECORD CAMPAIGN 1 ──► LABEL ──► DETECTOR v1 ──┐
         │              (W3)            (W4)        (W4)        │
         └──► SYNTHETIC PIPELINE ──► PRETRAIN ──────────────────┤
                    (W1-2)              (W2)                    │
                                                                ▼
    BELIEFFRAME FROZEN ──► ENGINE + HSMM ──► GOLDEN TRACES ──► FULL PIPELINE
         (W1)                 (W2-3)            (W3)             (W6)
                                                                 │
                                              ┌──────────────────┤
                                              ▼                  ▼
                                    RECORD CAMPAIGN 2      EDGE DEPLOY
                                    (error runs, W7)         (W7)
                                              │                  │
                                              └────► EVAL + ROC TUNING (W8)
                                                            │
                                    GUI/TTS/GSTREAMER (W9) ─┤
                                                            ▼
                              INTEGRATION (W10) ──► HARDEN (W11) ──► REHEARSE (W12)
```

**The long pole is data and it is not close.** Everything else has slack. Protect
weeks 3, 4 and 7.

---

## 9. Week-by-week plan

Budget: 6 people × 13 weeks × ~15 h/week ≈ **1,170 person-hours**. Everything below
fits inside that or gets cut.

Every gate is a number or a demo. "We worked on it" is not a gate.

| Wk | Dates | Work | **Exit gate** |
|---|---|---|---|
| **0** | Sep 8–14 | **Order hardware today.** Fill in PS ID. Freeze CSP-1. Repo scaffold, CI running validator + negative tests. Assign the 6 roles. Start Ego-Exo4D licence request. | Jetson + cameras ordered with tracking numbers. CI green on push. |
| **1** | Sep 15–21 | `BeliefFrame` frozen. Trace replay harness. Mock rack built, AprilTags bonded, zones measured with a jig. BlenderProc scene started. Download MicroG-4M + AMASS. | `replay.py` runs a hand-written trace end to end. Rack zone coordinates committed. |
| **2** | Sep 22–28 | Synthetic renders (target 60 k frames, background job). Detector pretrain on synthetic. Engine: predicates + HSMM on golden traces. **Draft the SIH deck.** | Engine correctly flags SKIP/OUT_OF_ORDER on 12 hand-written traces. Detector has a synthetic-only mAP number. |
| **3** | Sep 29–Oct 5 | **Recording campaign 1: 30 nominal runs.** Footswitch ground truth. 3 cameras, rotated-rack variants. | 30 runs on disk with step boundaries. ~15 camera-hours. |
| **4** | Oct 6–12 | Label ~8 k keyframes (CVAT + ByteTrack propagation). Detector v1 = synthetic pretrain + real finetune. **Deck submitted.** | Detector v1 mAP@50 on held-out real ≥ 0.80. Deck in. |
| **5** | Oct 13–19 | Contact MLP. RTMPose + 3D lifter. `rackframe.py` — AprilTag PnP and canonicalisation. | **Rotated-rack test: canonicalised joint coords match within 3 cm at 0°, 90°, 180°.** The orientation-agnostic proof. |
| **6** | Oct 20–26 | `pipeline.py` — first light. Video in, BeliefFrame trace out, engine consumes it. Motion TCN trained. | **End-to-end on one recorded nominal run: 14/14 steps COMPLETE.** |
| **7** | Oct 27–Nov 2 | **Recording campaign 2: 30 runs with injected errors**, taxonomised. Edge deploy: ONNX → TensorRT INT8 on Jetson. | ≥ 10 FPS at ≤ 25 W measured with a power meter. 30 error runs on disk. |
| **8** | Nov 3–9 | Eval harness. Leave-one-performer-out + leave-one-camera-pose-out. **ROC sweep, pick operating point.** | **Step acc ≥ 95 %, deviation recall ≥ 90 %, ≤ 1 FA / 45 min — with the curve plotted.** |
| **9** | Nov 10–16 | PySide6 GUI. Piper TTS. GStreamer tee (record + RTSP). Event clip export. Hash-chained logger. | Live run with voice prompts. RTSP plays on a second laptop. Log verifies its own chain. |
| **10** | Nov 17–23 | Integration hardening. Zero-shot mistake-detection F1 on Assembly101. ASR commands + logged override. | Assembly101 number in hand, whatever it is. 5 clean live runs back to back. |
| **11** | Nov 24–30 | **Second procedure YAML, same detector classes, zero retraining.** Failure-mode reel. Occlusion/UNVERIFIED tuning. | **A never-trained procedure runs live.** The differentiator — rehearsed, not improvised. |
| **12** | Dec 1–7 | Freeze code. Rehearse the 4-min script 10×. Build the spare box. Pack. | Two identical boxes, both cold-booted and demoed successfully. |
| **13** | Finale | 36 h: integrate, verify, present. | — |

### Note on the PPT screening

The internal hackathon / idea screening lands **before** any prototype is needed.
The deck is written from design (weeks 1–2), not from working code. That de-risks
the whole schedule — if the build slips, the submission is already in.

---

## 10. Hardware and BOM

**Order on day one.** The Jetson is the risk item.

| Item | ₹ (est.) | Lead time | Note |
|---|---|---|---|
| NVIDIA Jetson Orin Nano 8 GB Super | 40,000 | **3–5 weeks** | 67 TOPS INT8 in Super mode, 25 W |
| 2× IMX296 global-shutter CSI camera | 16,000 | 2–3 wk | **Global shutter matters** — rolling shutter smears fast hands and poisons the contact head |
| NVMe 512 GB + heatsink/fan | 5,000 | 1 wk | |
| USB mic array + speaker | 4,000 | 1 wk | |
| Mock rack: Al extrusion, props, AprilTags | 10,000 | build in W1 | |
| Footswitch / spare keyboard | 1,500 | 1 wk | The ground-truth trick |
| **Total** | **~76,500** | | |

**Fallback if the Jetson slips:** develop on a laptop with RTX 3050/4060 and *also*
publish ONNX-Runtime CPU-only latency. You lose the "25 W box" stage moment but
keep every claim defensible. **Decide by W6, not W12.**

**Honesty on the slide:** flight hardware would be a rad-tolerant SBC. Our
contribution is the software plus a measured latency/power envelope. Say this
before a judge says it.

---

## 11. Datasets

All links verified. **None of these are your training set** — see §11.4.

### 11.1 Tier 1 — get these four

#### MicroG-4M — the only real microgravity data that exists

| | |
|---|---|
| Repo | https://github.com/lei-qi-233/MicroG-4M |
| Data | Hugging Face `LEI-QI-233/MicroG-4M` |
| Paper | https://arxiv.org/abs/2506.02845 |
| Contents | 4,759 three-second clips, 50 microgravity action classes, **390 k+ bounding boxes**, 13 k multi-label action annotations, 1,238 captions, 7 k QA pairs |
| Source | Real ISS, Tiangong and EVA footage **plus cinematic simulations** |
| Licence | **MIT** |

**Use for:** domain-adaptation fine-tuning of the detector, and the honest answer to
*"you trained on Earth video."*

> **Caveat to state before a judge does:** part of it is movie footage. Report
> numbers split by real-mission vs cinematic subsets. 3-second clips give visual
> domain, not sequence structure.

#### Assembly101 — the fixed-camera mistake-detection benchmark

| | |
|---|---|
| Site | https://assembly-101.github.io/ |
| Data | Hugging Face `cvml-nus/assembly101` (migrated May 2026) |
| Scripts | https://github.com/assembly-101/assembly101-download-scripts |
| Contents | 4,321 videos, **8 static + 4 egocentric synchronised views**, 100 k+ coarse and ~1 M fine-grained segments, 18 M 3D hand poses, **mistake-detection annotations** |
| Licence | CC BY-NC 4.0 |

**Use for:** the credibility number. Zero-shot mistake-detection F1 on data we never
collected. The 8-static-camera rig is the closest public analogue to fixed
payload-rack cameras that exists.

#### IndustReal — procedure step recognition, cleanly licensed

| | |
|---|---|
| Repo | https://github.com/TimSchoonbeek/IndustReal |
| Data | https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 |
| Paper | https://arxiv.org/abs/2310.17323 (WACV 2024) |
| Contents | 84 videos, 27 participants, HoloLens 2. Annotations for **procedure step recognition, assembly state detection, action recognition**. Contains both *procedural* errors (omissions) and *execution* errors |
| Licence | **Apache 2.0** |

**Why it matters most:** errors are deliberately held out — a chunk appear only in
val/test, measuring robustness to *unseen* mistakes. Exactly our generalisation
claim. Apache-2.0 means it can touch delivered weights.

#### CaptainCook4D — the error taxonomy

| | |
|---|---|
| Paper | https://arxiv.org/abs/2312.14556 |
| Review | https://openreview.net/forum?id=YFUp7zMrM9 |
| Contents | 384 recordings, 94.5 h, 5.3 k step annotations, 10 k fine-grained actions |
| Error types | order, timing, temperature, preparation, **missing step**, measurement, technique |

**Use for:** steal the error taxonomy for recording campaign 2. Map our injected
errors onto theirs so we cite prior art rather than inventing categories.

### 11.2 Tier 2 — pretraining and supporting

| Dataset | What it gives | Link | Licence |
|---|---|---|---|
| **Ego-Exo4D** | 1,286 h, 740 participants, Aria ego + **4–5 stationary GoPros**, keystep + proficiency benchmarks | https://ego-exo4d-data.org/ | Licence signing, ~2 day approval |
| **HoloAssist** | 169 h, 350 instructor-performer pairs, **mistake detection + intervention prediction** | https://holoassist.github.io/ | CDLAv2 (permissive) |
| **EgoPER** | 28 h, 386 videos, 5 tasks. Errors: Omission, Correction, Modification, Slip, Addition. Trains on **error-free video only** | https://github.com/robert80203/EgoPER_official | see repo |
| **IKEA-ASM** | **3 fixed RGB views** + depth, 33 action classes, pose + instance seg + tracking | https://ikeaasm.github.io/ | CC BY-NC 4.0 |
| **HOI4D** | 2.4 M RGB-D frames, 800 objects / 16 categories, 3D hand pose, object pose tracking | https://hoi4d.github.io/ | CC BY-NC 4.0 |
| **MECCANO** | 20 videos, industrial-like assembly, 64 k active-object boxes, 89 k hand annotations, 20 object + 61 action classes | https://iplab.dmi.unict.it/MECCANO/ | unstated — ask before delivery |
| **DexYCB** | 582 k RGB-D frames, 8 cameras, full grasp approach→contact→hold | https://dex-ycb.github.io/ | CC BY-NC 4.0 |
| **AMASS** | 40+ h, 300+ subjects, 11 k+ motions, SMPL | https://amass.is.tue.mpg.de/ | registration; research use |

**AMASS cannot be skipped** — it drives the SMPL bodies in the BlenderProc pipeline.
Without it the synthetic humans are static mannequins.

**DexYCB** is the right contact-MLP pretrain specifically because it captures the
*whole* grasp trajectory, which is what `grasped`/`released` need to discriminate.

**EgoPER's method is worth reading even if we skip the data** — training on
error-free video only and detecting deviation as anomaly is a legitimate fallback
if error-run collection slips. We still need error runs to *evaluate*.

### 11.3 Tier 3 — microgravity reference footage

**NASA Image and Video Library.** API tested, **no key required**:

```bash
curl -s "https://images-api.nasa.gov/search?q=Expedition%20crew%20science&media_type=video&page_size=50"
```

Measured hit counts: `Expedition crew science` → 99 videos,
`ISS astronaut experiment` → 21, `microgravity science glovebox` → 2.

**Hundreds of videos, not thousands.** A qualitative domain reference and demo
B-roll — not a training set. Public domain.

Also: **PhysAstro-Pose** — physics-inspired semi-supervised pose estimation in
microgravity, https://doi.org/10.3390/s26113406. Directly relevant prior art for the
rack-frame approach.

### 11.4 What none of these give us

**The actual training set is the 60 runs we record.** No public dataset contains our
rack, cartridge, latch, vials, zone geometry or procedure. The detector class map in
`csp1_colloid_sample_processing.yaml` — `sample_cartridge`, `glovebox_latch`,
`cartridge_holder`, `status_indicator` — exists in none of them.

| Need | Source |
|---|---|
| Detector backbone init | Synthetic renders + COCO-pretrained weights |
| Hand-object contact prior | DexYCB, HOI4D |
| Body pose for synthetic humans | AMASS |
| Microgravity visual domain | **MicroG-4M** |
| A benchmark number we didn't collect | **Assembly101** |
| Error taxonomy for campaign 2 | **CaptainCook4D**, IndustReal, EgoPER |
| Demo B-roll / sanity check | NASA image library |

### 11.5 The licence problem — handle it before a PSU judge raises it

| Licence | Datasets | Can it touch delivered weights? |
|---|---|---|
| **MIT** | MicroG-4M | Yes |
| **Apache 2.0** | IndustReal | Yes |
| **CDLAv2** | HoloAssist | Yes |
| **CC BY-NC 4.0** | Assembly101, IKEA-ASM, HOI4D, DexYCB | **Non-commercial only** |
| **Unstated** | MECCANO | Ask the authors |

A fielded ISRO system is not obviously "non-commercial." Do not hand-wave it. Put
this sentence on a slide:

> Delivered weights are trained on our own recorded data, MicroG-4M (MIT),
> IndustReal (Apache-2.0) and synthetic renders we generated. CC BY-NC datasets are
> used only for benchmark reporting in the evaluation section, never in the
> delivered model.

That will land with a procurement-aware judge harder than any accuracy number.

### 11.6 Download order and disk budget

**Do not download everything.**

| Priority | Dataset | Approx. size | When |
|---|---|---|---|
| 1 | MicroG-4M | small (3 s clips) | Week 1 |
| 2 | AMASS | ~10 GB | Week 1 — blocks synthetic pipeline |
| 3 | IndustReal | moderate | Week 2 |
| 4 | Assembly101 — **static views only, skip the 4 ego streams** | large; subset it | Week 3 |
| 5 | DexYCB *or* HOI4D — one, not both | large | Week 5, only if contact head underperforms |
| ✋ | Ego-Exo4D, HoloAssist depth (560 GB) | very large | **Skip unless a specific need appears** |

**Budget ~500 GB, not 3 TB.** A team that spends week 3 downloading instead of
recording has already lost the critical path. Start the Ego-Exo4D licence request in
week 1 anyway — approval takes ~2 days and costs nothing to have sitting there.

---

## 12. Data collection campaign

This is where the project lives or dies.

### Campaign 1 (W3) — 30 nominal runs

6 performers × 5 runs. Vary: performer, camera pose, lighting, **rack rotation
(0°/90°/180°)**, performer orientation (standing, lying, inverted on a mat).
Footswitch at every step boundary.

Target: ~15 camera-hours across 3 cameras, ~1.6 M frames.

### Campaign 2 (W7) — 30 error runs

Not "do it wrong." Scripted, taxonomised injections:

| Deviation | Runs | How |
|---|---|---|
| SKIP | 8 | Omit S08 (latch), S13 (seal), S05 |
| OUT_OF_ORDER | 6 | S09 before S08; S12 before S11 |
| WRONG_OBJECT | 6 | Grab vial B instead of vial A |
| DURATION | 4 | Rush S06 below `min_s`; stall S13 past `max_s` |
| HAZARD | 3 | Open latch while unit runs |
| **Legal reorder (must NOT alert)** | **3** | **Swap S02/S03 — the false-alarm test** |

That last row is the one nobody else records, and it is the one that proves the
unordered-group design.

### Labelling budget

~8 k keyframes × ~25 s each ≈ **55 person-hours**. Split across 6 people in W4 ≈
9 h each. Tight but real. ByteTrack propagation between keyframes roughly halves it
if detector v0 is decent.

### Synthetic pipeline

BlenderProc2, SMPL-X bodies from AMASS, domain randomisation over lighting,
materials, camera pose, motion blur, **plus zero-g free-floating object
trajectories**. Target 60 k frames, auto-labelled. Renders overnight through W1–W2 —
costs render time, not person-time.

---

## 13. Team split

| # | Role | Owns | Peak weeks |
|---|---|---|---|
| 1 | **Data lead** | Recording protocol, labelling ops, CVAT, synthetic pipeline | W1–4, W7 |
| 2 | **Detection** | RT-DETR, ByteTrack, contact MLP, class map | W2, W4–6 |
| 3 | **Geometry** | RTMPose, 3D lift, AprilTag rack frame, canonicalisation | W5–6 |
| 4 | **Engine** | Predicates, HSMM, deviations, alerts, logger | W1–3, W8 |
| 5 | **Systems** | GStreamer, TensorRT/INT8, Jetson, TTS/ASR, power | W7, W9 |
| 6 | **GUI + eval** | PySide6, eval harness, ROC, demo direction, pitch | W8–12 |

Roles 1 and 4 start immediately and never touch each other's code. **Role 6 owns the
pitch from W2** — the person who builds the eval harness is the person who can
answer the judge's hardest question.

---

## 14. Evaluation protocol

If you evaluate on a random frame split you have measured nothing.

### Splits

- **Leave-one-performer-out** — does it generalise to a body it never saw?
- **Leave-one-camera-pose-out** — does it generalise to a viewpoint it never saw?
- **Rotated-rack held-out** — 0° train, 90°/180° test. The orientation-agnostic proof.
- **Zero-shot Assembly101** — mistake-detection F1 on data we never collected.

### Metrics to report

| Metric | Why |
|---|---|
| Step recognition accuracy | The headline |
| Deviation recall **per deviation type** | An aggregate hides that WRONG_OBJECT is at 40 % |
| **False alarms per 45 min, with full ROC** | The number that decides adoption |
| Accuracy stratified by occlusion fraction | Proves the UNVERIFIED design works |
| UNVERIFIED rate | Too high = useless; too low = it's guessing |
| Latency (p50 / p95) and FPS at measured watts | The edge claim |
| Legal-reorder false-alarm rate | Must be **zero**. Proves the unordered group |

### The failure-mode reel

Build a short video of the cases where it **doesn't** work. Judges have spent all
day being told things work perfectly. A team that opens with *"here are four cases
where we fail, and why"* gets believed about everything else.

---

## 15. The 36 hours

Arrive with a working, frozen system. The 36 hours are **integration on the venue's
power and lighting, plus rehearsal.** Not development.

| Hours | Do |
|---|---|
| 0–3 | Unpack, assemble rack, mount cameras, **re-measure zones under venue lighting**, re-verify AprilTag lock |
| 3–6 | Cold-boot both boxes, run 3 nominal runs, confirm 14/14 COMPLETE |
| 6–10 | Run every deviation type once. Fix only what is broken. **No new features.** |
| 10–14 | RTSP to the "ground station" laptop. Verify log hash chain. Export a deviation clip. |
| 14–18 | Sleep in shifts. Two people minimum awake. |
| 18–24 | Second-procedure rehearsal. Occlusion behaviour. Rotated-rack beat. |
| 24–30 | Full 4-min script × 6 runs, timed. Assign who physically touches what. |
| 30–34 | Freeze. Tag the commit. Mirror the spare box. Print the log. |
| 34–36 | Rest, present. |

> **No commit after hour 30.** Teams lose on stage to a change made at hour 34.

### MVP scope

**MUST BUILD** — live camera → detector + hands + contact → belief state; procedure
engine + HSMM on one 14-step procedure; SKIP / OUT_OF_ORDER / WRONG_OBJECT with the
alert policy; Piper voice next-step + deviation alerts; hash-chained JSONL log;
GStreamer local record **and** RTSP to a specified IP; PySide6 GUI with live overlay,
checklist, timeline, deviation feed; runs with the network cable unplugged.

**SHOULD BUILD** — second-camera belief fusion; rack-frame 3D canonicalisation demo;
ASR commands + logged override; event clip export; live procedure-swap flow.

**WILL NOT BUILD** *(say this out loud — scoping discipline scores)* — runtime
SMPL-X mesh recovery; VLM in the hot loop; anything cloud or internet-dependent;
multi-crew re-identification; any training during the 36 hours; a web dashboard.

### 4-minute demo script

| Time | Action | Purpose |
|---|---|---|
| 0:00 | **Unplug the ethernet cable.** "Everything runs on this 25 W box." | Kills the "is it calling an API?" doubt in five seconds |
| 0:20 | Load procedure. Perform steps S01–S04 correctly. Voice prompts each next step; checklist ticks; timeline fills. | Establishes the nominal loop |
| 1:10 | **Deliberately skip S08.** Within ~1.5 s: chime + *"Step eight, close the glovebox latch, was not completed."* GUI shows `latch_closed never true`. | **Wow #1** — the *reason* is what lands |
| 1:50 | **Pick vial B instead of vial A.** *"Wrong vial. You are holding vial B, the orange band."* | Semantics, not just sequence |
| 2:20 | **Block the camera.** System says **UNVERIFIED — workspace occluded**. No false alarm. | **Wow #2** — judges notice restraint more than accuracy |
| 2:50 | **Rotate the rack 90°; performer works inverted.** Still tracks. Show rack-frame skeleton. | **Wow #3** — the PS's "optional" hard part, delivered |
| 3:20 | Open the log: hash-chained, ~38 KB. Play the ±10 s deviation clip. Show RTSP on the "ground station" laptop. | Closes three PS requirements at once |
| 3:45 | **Swap in a `procedure.yaml` the model has never seen. Run two steps.** *"Zero retraining."* | **The closer** |

Rehearse ten times. Keep a second identical box on the table, powered and warm.

### The 5 attacks and the answers

**1. "You trained on Earth video. Microgravity looks nothing like this."**
Correct, and it is our top listed risk. The defence is architectural: learned
components never touch gravity cues — they detect objects, hand keypoints and
contact. Body pose is in rack coordinates via AprilTag extrinsics, so "up" is the
rack. We fine-tune on MicroG-4M (real ISS/Tiangong footage), prove it by physically
rotating the rig and by leave-one-camera-pose-out, and report zero-shot on
Assembly101. Residual gap is free-floating objects and tethers; modelled in Isaac
Sim with zero-g dynamics, and flagged as a gap rather than hidden.

**2. "What's your false-positive rate? An assistant that cries wolf is worse than none."**
*[Quote the held-out number, show the ROC.]* Alerts require posterior ≥ 0.85
sustained ≥ 1.5 s on calibrated confidences with a reject option. Occlusion produces
UNVERIFIED, not a guess. One voice command overrides, and the override is logged.
Alerts are advisory — never blocking.

**3. "ISS already has electronic procedures. Why not a checklist tablet?"**
IPV requires the crew to tick each step manually — that costs exactly the attention
we are returning, and self-reporting structurally cannot catch an error the crew
didn't notice. We don't replace the procedure viewer; we consume the same procedure
and add **independent verification** plus an automatic record.

**4. "This is just YOLO plus if-else."**
The perception output is a calibrated belief state; the sequence tracker is a hidden
semi-Markov model with duration priors, which makes "skipped" a probabilistic
inference that survives three-second occlusions and yields a confidence you can set
policy on. The declarative layer being authorable by a payload engineer rather than
an ML engineer is the *feature* — it is what makes it certifiable and reusable
across the payload catalogue.

**5. "How does this get flight-certified? What about crew privacy?"**
No network path exists in the inference code. Every decision emits a human-readable
justification and supports deterministic replay from stored frames. Weights hash and
procedure hash are in every log record. Video stays on board; only the KB-scale log
is nominally downlinked, and the streamed copy supports identity blurring.
DPDP-aligned by construction: local processing, purpose limitation, configurable
retention. Certifiable as advisory, non-critical software — it never commands
hardware.

---

## 16. Descope ladder

Under pressure people cut randomly. Pre-commit to this order:

1. **ASR voice commands** → GUI button. *(costs nothing on stage)*
2. **Third camera** → run two. *(occlusion stats worsen; say so)*
3. **3D body lift** → 2D pose + rack-frame object geometry only. *(keeps
   orientation-agnosticism via zones; loses the skeleton overlay)*
4. **Motion TCN** → replace `motion_is` clauses with geometry-only verification.
   *(a PDL edit, not a code change)*
5. **Jetson** → laptop GPU + published CPU numbers.
6. **Second procedure demo** → ✋ **never cut this.** It is the differentiator.

Items 3 and 4 are *procedure file edits*, not code changes. That is the architecture
paying rent under time pressure.

---

## 17. Risk register

| # | Risk | Early warning | Mitigation / kill switch |
|---|---|---|---|
| 1 | **Microgravity domain gap** — genuinely unsolved. Lab rack ≠ flight rack; free-floating tools have no terrestrial analogue | Detector collapses on MicroG-4M real-mission subset | Architecture limits exposure (objects + contact transfer far better than "step 7"). Fine-tune on MicroG-4M. Quantify the gap; **state it honestly on the slide** |
| 2 | **False alarms destroy adoption** | W8 ROC shows > 3 FA / 45 min at 90 % recall | Raise `persistence_s`, widen `unverified_after_s`, accept lower recall — and *publish the tradeoff*. A tuned-down system with a curve beats an untuned one with a claim |
| 3 | **Occlusion by the crew's own body** (~30 % of the time) | W6 shows > 25 % UNVERIFIED on nominal runs | 2–3 cameras fused at belief-state level; HSMM duration prior carries state through blackouts; UNVERIFIED never becomes a false COMPLETE |
| 4 | **Certification / data governance** | — | Offline by construction; per-decision justification; deterministic replay; weights + procedure hashes in every record; identity blurring on the stream; advisory-only |
| 5 | **Labelling stalls** | W4 ends with < 5 k frames labelled | Cut to 1 camera view for labelling; lean harder on synthetic pretrain |
| 6 | **Hardware doesn't arrive** | No tracking number by Sep 14 | Commit to laptop path at W6; publish CPU latency |
| 7 | **Contact detection unreliable** | W5 contact-head F1 < 0.75 | Rewrite affected steps to geometry-only predicates (`inside` + `stable`) — a YAML edit |
| 8 | **Team burnout / exam collision** | Two consecutive missed gates | Descope ladder immediately. Do not "catch up next week" — nobody ever does |

> Check the academic calendar **now**. If weeks 3–4 or 7 are compromised, move the
> recording campaigns *earlier*, not later. Campaign 1 needs only the mock rack and a
> footswitch — no models, no Jetson. It can happen in week 2 if pushed.

---

## 18. Research references

### Datasets and benchmarks
- Assembly101 (CVPR 2022) — https://assembly-101.github.io/
- IndustReal (WACV 2024) — https://arxiv.org/abs/2310.17323
- CaptainCook4D — https://arxiv.org/abs/2312.14556
- MicroG-4M — https://arxiv.org/abs/2506.02845
- EgoPER, *Error Detection in Egocentric Procedural Task Videos* (CVPR 2024) — https://github.com/robert80203/EgoPER_official
- PREGO, *Online Mistake Detection in Procedural Egocentric Videos* (CVPR 2024) — https://openaccess.thecvf.com/content/CVPR2024/papers/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.pdf
- IKEA-ASM (WACV 2021) — https://ikeaasm.github.io/
- HoloAssist (ICCV 2023) — https://holoassist.github.io/
- Ego-Exo4D — https://ego-exo4d-data.org/
- HOI4D (CVPR 2022) — https://hoi4d.github.io/
- MECCANO — https://iplab.dmi.unict.it/MECCANO/
- DexYCB (CVPR 2021) — https://dex-ycb.github.io/
- AMASS — https://amass.is.tue.mpg.de/
- NASA Image and Video Library — https://images.nasa.gov · API https://images-api.nasa.gov

### Models and methods
- RT-DETR, *DETRs Beat YOLOs on Real-time Object Detection* — arXiv:2304.08069
- RTMPose / MMPose — https://github.com/open-mmlab/mmpose
- MediaPipe Hands — https://ai.google.dev/edge/mediapipe
- ByteTrack (ECCV 2022) — arXiv:2110.06864
- MS-TCN++ (TPAMI 2020) — temporal action segmentation
- 4D-Humans / HMR2.0 (ICCV 2023) — https://shubham-goel.github.io/4dhumans/ *(cited as the heavyweight alternative we deliberately did not use)*
- AprilTag 3 (IROS 2019) — https://april.eecs.umich.edu/software/apriltag
- PhysAstro-Pose — https://doi.org/10.3390/s26113406
- *Vision-Based Mistake Analysis in Procedural Activities: A Review* — https://arxiv.org/html/2510.19292v2

### Tools
- BlenderProc2 — https://github.com/DLR-RM/BlenderProc
- NVIDIA Isaac Sim — https://developer.nvidia.com/isaac/sim
- Piper TTS — https://github.com/rhasspy/piper
- whisper.cpp — https://github.com/ggml-org/whisper.cpp

### Standards, policy, mission context
- NASA-STD-3001, Space Flight Human-System Standard — https://standards.nasa.gov
- CCSDS File Delivery Protocol (CFDP), CCSDS 727.0-B — https://public.ccsds.org
- ISRO — Gaganyaan / Bharatiya Antariksh Station — https://www.isro.gov.in
- Digital Personal Data Protection Act, 2023 (MeitY) — https://www.meity.gov.in
- Bhashini — https://bhashini.gov.in

### ⚠ Flagged as UNVERIFIED — check before citing on a slide
- NASA's **ISS crew-time hourly price** in the commercial LEO pricing policy. A
  figure on the order of $10⁵/hour is often quoted; **I am not confident of the
  exact number.** Verify on nasa.gov or drop it.
- NASA's **International Procedure Viewer (IPV)** and **Procedure Representation
  Language (PRL) / PRIDE** work (Kortenkamp, Bonasso et al.). Probably the right
  prior art to cite, but confirm the citations — this is exactly the reference a
  judge will know.

---

## 19. Deliverables checklist

- [x] PS ID filled into `PLAN.md` and `procedure.yaml` — **26174**
- [ ] Idea PPT, 6 slides, official SIH format
- [ ] Trained weights + model card + weight hashes
- [x] `procedures/` — CSP-1 **plus a second procedure** using the same detector classes — **CRX-2**, verified zero-retraining against `builds/parikshak-perception-1.2.0.json` (`python tools/onboard.py`)
- [ ] Eval report: accuracy, deviation recall per type, **ROC + chosen operating point**, occlusion strata, leave-one-out splits, Assembly101 zero-shot
- [ ] Power / latency bench on target hardware, measured with a meter
- [ ] Sample `run_log.jsonl` + `.txt` + a verified hash chain
- [ ] Failure-mode reel — the clips where it *doesn't* work
- [ ] Licence statement for delivered weights (§11.5)
- [ ] 4-minute demo script, rehearsed 10×
- [ ] Two identical, cold-boot-tested boxes

---

## 20. Appendix: commands

### Validate a procedure

```bash
python tools/validate_procedure.py procedures/csp1_colloid_sample_processing.yaml --strict
```

### Run the validator's negative tests

```bash
python tools/test_validator.py
```

### Validate against the JSON Schema

```bash
python -c "import json,yaml; from jsonschema import Draft202012Validator; s=json.load(open('schema/procedure.schema.json')); d=yaml.safe_load(open('procedures/csp1_colloid_sample_processing.yaml')); e=list(Draft202012Validator(s).iter_errors(d)); print(f'{len(e)} structural errors')"
```

### Search NASA ISS footage (no API key needed)

```bash
curl -s "https://images-api.nasa.gov/search?q=Expedition%20crew%20science&media_type=video&page_size=50"
```

### Author a new procedure

```bash
cp procedures/_template.yaml procedures/my_experiment.yaml
python tools/validate_procedure.py procedures/my_experiment.yaml --strict
```

> The unmodified template **fails validation by design** — it ships with empty
> placeholders (`prompt_tts: ""`, `detector_class: ""`). That is the tool working,
> not a bug. Fill the blanks until it passes.

The validator will not let you reference a thing the deployed perception build
cannot see. Everything inside the existing detector vocabulary costs an afternoon;
everything outside it costs a labelling campaign. Knowing which side of that line
you are on is the point.

---

## Honest assessment

~1,170 person-hours is enough **if data starts in week 1 and nobody gold-plates the
GUI.** The realistic failure mode is not technical — it is that weeks 3, 4 and 7
collide with mid-semester exams, the recording campaign slips two weeks, and that
cascades into an untuned system at the finale.

The microgravity domain gap is genuinely not solvable by a student team. The honest
position is *"we make the architecture minimise learned gravity-dependence, we
fine-tune on the only real microgravity data that exists, and we quantify the
remaining gap"* — never *"we solved it."*

The value of this submission is concentrated in three places: the
**procedure-as-data split**, the **deviation-first dataset**, and the **UNVERIFIED
state**. If anything gets cut under time pressure, cut the 3D lift and the VLM.
Never those three.
