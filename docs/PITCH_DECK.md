# PARIKSHAK — pitch deck content

SIH 2026 · PS 26174 · Space Technology. Slide-by-slide content for the
screening deck, 12 slides, ~6 minutes spoken. Every number is taken from a file
in `runs/` measured on 22 September 2026 — the source is named on each slide so
it can be re-checked the morning of the presentation.

Speaker notes are under each slide. The live demo is `python -m demo`; the
shareable copy is `runs/demo/parikshak_demo.html`.

---

## 1 · Title

**PARIKSHAK**
An on-board witness that checks every step of an experiment procedure — and
says so when it cannot see.

Team name · college · PS 26174 · Space Technology

> Notes: One sentence only: "When an astronaut runs a science procedure alone,
> nobody is watching the steps. PARIKSHAK is."

---

## 2 · The problem

- Crew time is the scarcest resource on a station, and one missed step can void
  a whole experiment run.
- Procedures are followed from a tablet the crew ticks by hand. Self-reporting
  cannot catch a mistake the crew did not notice.
- Ground support sees the procedure afterwards — through a link with delays and
  gaps — not while it matters.

Visual: a 14-step procedure with step 8 ("close the glovebox latch") highlighted
as the one that gets skipped.

> Notes: Do not quote a price for crew time - no verified figure is on file.
> The point is that the crew's attention is the scarce resource.

---

## 3 · What it does, in one picture

Camera → Perception → Belief frame → Procedure engine → Voice, checklist, log

- Watches the rack, finds objects through markers, and places them in **rack
  coordinates** — up is the rack, not gravity.
- Checks each step's conditions over time, not frame by frame.
- Speaks the next step; alerts only with strong evidence — **with the reason**.
- Writes a hash-chained log of every decision, sent down before any video.

Visual: the five-box flow from the demo's "How it works" tab.

---

## 4 · Demo: a skipped step, caught with its reason

Screenshot of the demo on "Latch left open before the run starts":

- Alert at 47.8 s: *"Step eight, close the glovebox latch, was not completed."*
- Why: `state_is(latch, closed)` never satisfied between 43.0 s and 46.2 s.
- 0.6 s later the hazard it causes — latch open while the unit runs — is
  flagged critical.

> Notes: Run it live if the room allows: `python -m demo`, the page opens on
> this alert. Press "Replay from start" and turn on "Speak".

---

## 5 · Beyond sequence: what a checklist cannot see

Three cards, each a demo scenario:

| Wrong object | Safety rule | Hardware fault |
|---|---|---|
| Vial B picked instead of vial A — the procedure declares them confusable | Latch opened while the unit runs — no step was skipped | Red indicator routes to the fault branch — the crew is not blamed |

---

## 6 · Restraint: unknown is not wrong

- Crew's body blocks the camera → **"cannot verify"**, never "skipped".
- Rack markers lost for 8 s → every position is unknown; nothing is guessed from
  a stale view.
- Steps done in a permitted different order → silent.

Big number: **0 false alarms** on nominal, occlusion and permitted-reorder runs,
both procedures. Source: `runs/eval.json`, `runs/eval_crx2.json`.

> Notes: "Judges notice restraint more than accuracy." Say it.

---

## 7 · A new experiment is a file — zero retraining

- CRX-2 (colloid resuspension, 8 steps) was added as a procedure file.
- Same software, same markers, no code change, no retraining.
- It catches its own skip, wrong vial and a geometric safety rule (vial within
  12 cm of the powered unit).

Visual: the CRX-2 procedure file next to the demo showing its hazard alert.

---

## 8 · Measured results

Degraded synthetic runs — noise, dropped frames, occlusion — each scored against
the error injected into it.

| Measure | Target | CSP-1 · 130 runs | CRX-2 · 70 runs |
|---|---|---|---|
| Step accuracy | ≥ 95 % | 98.7 % | 95.6 % |
| Deviations caught | ≥ 90 % | 97.8 % | 97.5 % |
| False alarms / 45 min | ≤ 1 | **0.85** | 0.00 |
| Alert delay (median / 95th pct) | ≤ 2 s | 1.6 / 1.6 s | 1.6 / 1.8 s |

Sources: `runs/eval.json`, `runs/eval_crx2.json`.

> Notes: Say how it got there: "CSP-1 started at 2.98 false alarms per 45
> minutes. We replayed every failure frame by frame, fixed three causes, and it
> is 0.85 now - with accuracy and recall unchanged. The reel shows the four
> that remain."

---

## 9 · Where it fails — and why

Two or three frames from the failure-mode reel (`runs/reel/reel.mp4`), each with
its one-line cause, taken from `docs/reel/segments.json`.

> Notes: "A team that opens with where it fails gets believed about everything
> else."

---

## 10 · Built to be certified

- **Offline by construction:** no module in the on-board package can open a
  network socket — a test fails the build if one appears.
- **Every decision has a reason**, and every run replays deterministically.
- **Tamper-evident log:** change one byte and verification names the record.
- **~8 KB log for a 141 s run vs ~70 MB of video** — the log goes down first.
  Source: `runs/downlink_sample/manifest.json`.
- Advisory only: it never commands hardware.

---

## 11 · Status and honest limits

| Done | Not yet |
|---|---|
| Engine, two procedures, evaluation, operator window, voice, log, clips, browser demo | Trained object and hand-tracking models (markers stand in today) |
| Laptop speed: 13.4 FPS at the slowest 5 % (target ≥ 10) | Jetson run and power measurement (≤ 25 W) |
| 45-minute run: memory +7.4 MB/hour | Real footage; evaluation data is synthetic |

Licences: every runtime library permissive (Apache-2.0, MIT, BSD, MPL-2.0),
PySide6 LGPL dynamically linked; no dataset or weights shipped. When models are
trained, CC BY-NC datasets are used for benchmark reporting only. Source:
`docs/LICENCE_STATEMENT.md`.

---

## 12 · Ask / close

**PARIKSHAK gives the crew back their attention — and never cries wolf when it
cannot see.**

Next: hand tracking, a real recording campaign, and the Jetson power
measurement.

Demo link / QR to the shareable page.
