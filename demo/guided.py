"""Perform an experiment: the engine guiding a run as it happens.

The replay page shows recorded runs. This drives one interactively. The page
asks for the next step, the person acts - correctly, or deliberately not - and
that action is performed in a simulated rack (`parikshak.eval.synth.RunBuilder`,
the same world model that generated every golden and evaluation run). The belief
frames it emits go straight into a real `ProcedureEngine`, which decides what to
say next.

Nothing about the guidance is scripted. Skip a step and the engine notices by
itself, when a later step verifies while this one never did, and says so in its
own words. Block the camera and it says "cannot verify" instead. The page only
chooses what the crew does; every verdict is the engine's.

**What is simulated and what is not.** The camera is: this laptop has no rack in
front of it, so the world model stands in for one. Everything downstream is the
real thing - the belief frames, the predicate evaluation, evidence accumulation,
the deviation detector, the alert policy with its persistence, and the
hash-chained log.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from demo.scenarios import ROOT, _frame_view, _procedure_view, _rack_view

#: How long "just wait" waits, and how long the crew is assumed to take moving
#: on after a step they skipped.
WAIT_S = 5.0
SKIP_S = 1.5
#: Sessions kept in memory at once. Each holds one run's frames and log.
MAX_SESSIONS = 24


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    detail: str
    #: "correct" | "mistake" | "condition" - the page groups buttons by this.
    kind: str


@dataclass(frozen=True)
class Experiment:
    key: str
    procedure: str
    title: str
    blurb: str
    steps: int
    minutes: float


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("csp1", "procedures/csp1_colloid_sample_processing.yaml",
               "CSP-1 · Colloidal suspension sample processing",
               "Fourteen steps: fetch the cartridge and vial A, check the label, load the "
               "glovebox, run the processing cycle, then seal and stow the sample.",
               14, 2.5),
    Experiment("crx2", "procedures/crx2_colloid_resuspension.yaml",
               "CRX-2 · Colloid resuspension",
               "Eight steps: fetch vial B from cold stowage, agitate it, let it settle, "
               "confirm the suspension, and return it. A different experiment, same software.",
               8, 1.5),
)


def experiments() -> list[dict[str, Any]]:
    return [{"key": e.key, "title": e.title, "blurb": e.blurb, "steps": e.steps,
             "minutes": e.minutes} for e in EXPERIMENTS]


def _experiment(key: str) -> Experiment:
    for e in EXPERIMENTS:
        if e.key == key:
            return e
    raise KeyError(key)


# ----------------------------------------------------------------- mistakes
def _csp1_take_wrong_vial(b) -> None:
    """Vial B instead of vial A - the same object class, told apart only by the
    colour band the procedure declares confusable."""
    b.motion("reach").hold(2)
    b.grasp("vial_b", "right").at("vial_b", tuple(v + 0.02 for v in b.pos["crew_hand"]))
    b.motion("idle").hold(4)


def _csp1_open_latch_while_running(b) -> None:
    """The hazard no sequence check can see: every step is done, in order, and
    the glovebox is opened while the unit is running."""
    b.set_state("latch", "open").motion("reach").hold(4)
    b.set_state("latch", "closed").motion("idle").hold(2)


def _crx2_agitate_near_unit(b) -> None:
    from parikshak.eval import crx2
    crx2.s04_near_unit(b)


def _crx2_take_wrong_vial(b) -> None:
    """Vial A out of the tray instead of vial B out of cold stowage."""
    from parikshak.eval import crx2
    crx2.s02_take_vial_a(b)


#: step -> (label, what it does). Only where the procedure itself declares two
#: objects confusable, which is what makes the mistake a named, expected one.
WRONG_ITEM: dict[tuple[str, str], tuple[str, Callable]] = {
    ("csp1", "S03"): ("Take vial B by mistake", _csp1_take_wrong_vial),
    ("crx2", "S02"): ("Take vial A by mistake", _crx2_take_wrong_vial),
}

HAZARD: dict[tuple[str, str], tuple[str, Callable]] = {
    ("csp1", "S10"): ("Open the glovebox while it runs", _csp1_open_latch_while_running),
    ("crx2", "S04"): ("Agitate it beside the powered unit", _crx2_agitate_near_unit),
}


def _steps_table(key: str) -> dict[str, Callable]:
    from parikshak.eval import crx2, synth
    return dict(crx2.NOMINAL if key == "crx2" else synth.NOMINAL)


# --------------------------------------------------------------------- run
class GuidedRun:
    """One interactive run: a simulated rack, a real engine, and the actions the
    page may ask for next."""

    def __init__(self, experiment: str, run_id: str | None = None) -> None:
        from parikshak.engine.runner import ProcedureEngine
        from parikshak.eval import crx2, synth
        from parikshak.pdl import load_procedure

        self.experiment = _experiment(experiment)
        path = ROOT / self.experiment.procedure
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.proc = load_procedure(path)
        self.builder = (crx2.builder(doc) if experiment == "crx2"
                        else synth.RunBuilder(doc))
        self._dir = tempfile.TemporaryDirectory(prefix="parikshak-guided-")
        self.run_id = run_id or f"guided-{experiment}-{int(time.time())}"
        self.engine = ProcedureEngine(self.proc, run_id=self.run_id,
                                      log_path=Path(self._dir.name) / "run.log.jsonl")
        self._steps = _steps_table(experiment)
        self._codes: dict[str, str] = {}
        self._seen = 0
        self.events: list[dict[str, Any]] = []
        self.done: list[str] = []          # steps the crew has acted on, in order
        self.finished = False
        self.summary: dict[str, Any] | None = None
        self.touched = time.monotonic()
        self._last_prompt: str | None = None
        self._consume(settle=2.0)          # the rack before anyone touches it

    # -- plumbing ------------------------------------------------------
    def _code(self, status: str) -> str:
        if status not in self._codes:
            self._codes[status] = chr(ord("a") + len(self._codes))
        return self._codes[status]

    def _consume(self, settle: float = 0.0) -> None:
        """Feed every frame the builder has produced since last time."""
        from parikshak.gui.state import DisplayState

        if settle:
            self.builder.hold(settle)
        frames = self.builder.frames[self._seen:]
        self._seen = len(self.builder.frames)
        for frame in frames:
            result = self.engine.step(frame)
            if result.active_step and result.active_step != self._last_prompt:
                self.events.append({"t": round(result.t, 2), "type": "prompt",
                                    "step": result.active_step, "text": result.prompt})
                self._last_prompt = result.active_step
            for tr in result.transitions:
                if tr.kind != "entered":
                    self.events.append({"t": round(tr.t, 2), "type": tr.kind,
                                        "step": tr.step_id, "text": tr.detail})
            for a in result.alerts:
                self.events.append({"t": round(a.t, 2), "type": "alert", "step": a.step_id,
                                    "kind": a.kind.value, "severity": a.severity.label,
                                    "text": a.text, "reason": a.reason,
                                    "channels": list(a.channels)})
            for n in result.notices:
                self.events.append({"t": round(n.t, 2), "type": "notice",
                                    "step": n.step_id, "text": n.text})
        self.state = DisplayState.from_engine(self.engine, frame=self.builder.frames[-1],
                                              source="guided")
        self.frame = _frame_view(self.builder.frames[-1], self.state, self._code)

    # -- what the crew may do next -------------------------------------
    @property
    def active(self) -> str | None:
        return self.engine.active_step

    def actions(self) -> list[Action]:
        if self.finished:
            return []
        sid = self.active
        if sid is None:
            return [Action("finish", "End the run", "Close the log and see the summary",
                           "correct")]
        step = self.proc.step(sid)
        out = [Action("do", f"Do step {sid}: {step.name}",
                      step.prompt_tts or step.name, "correct")]
        key = (self.experiment.key, sid)
        if key in WRONG_ITEM:
            label, _fn = WRONG_ITEM[key]
            out.append(Action("wrong", label, "The confusable object, on purpose", "mistake"))
        if key in HAZARD:
            label, _fn = HAZARD[key]
            out.append(Action("hazard", label, "Every step still done, in order", "mistake"))
        nxt = self._next_step(sid)
        if nxt is not None:
            out.append(Action("skip", f"Skip {sid} - go straight to {nxt}",
                              f"Leave {sid} undone and perform {nxt} instead", "mistake"))
        out.append(Action("blocked", f"Do {sid} with the camera blocked",
                          "Performed correctly, but out of sight", "condition"))
        out.append(Action("wait", "Wait, doing nothing",
                          f"{WAIT_S:.0f} seconds pass", "condition"))
        return out

    def _next_step(self, sid: str) -> str | None:
        """The step after this one that the simulated crew can perform."""
        order = [s for s in self.proc.order if s in self._steps]
        if sid not in order:
            return None
        i = order.index(sid)
        return order[i + 1] if i + 1 < len(order) else None

    def perform(self, action_id: str) -> None:
        """Carry out one action, then let the engine see what happened."""
        if self.finished:
            raise ValueError("this run has already ended")
        sid = self.active
        if action_id == "finish" or sid is None:
            self.finish()
            return
        key = (self.experiment.key, sid)
        step_fn = self._steps.get(sid)
        if action_id == "do":
            if step_fn is None:
                raise ValueError(f"no simulated action for {sid}")
            step_fn(self.builder)
            self.done.append(sid)
        elif action_id == "wrong" and key in WRONG_ITEM:
            WRONG_ITEM[key][1](self.builder)
            self.done.append(sid)
        elif action_id == "hazard" and key in HAZARD:
            HAZARD[key][1](self.builder)
            self.done.append(sid)
        elif action_id == "skip":
            nxt = self._next_step(sid)
            if nxt is None:
                raise ValueError(f"{sid} is the last step - there is nothing to skip to")
            # The crew simply moves on. Nobody tells the engine that anything was
            # skipped: it has to notice for itself that a later step verified
            # while this one never did.
            self.builder.motion("idle").hold(SKIP_S)
            self._steps[nxt](self.builder)
            self.done.append(nxt)
        elif action_id == "blocked":
            if step_fn is None:
                raise ValueError(f"no simulated action for {sid}")
            hidden = [e for e in self.proc.step(sid).objects if e in self.builder.entities]
            self.builder.occlude(*hidden, zone="workspace")
            step_fn(self.builder)
            self.builder.clear_occlusion().hold(1.0)
            self.done.append(sid)
        elif action_id == "wait":
            self.builder.motion("idle").hold(WAIT_S)
        else:
            raise ValueError(f"unknown action {action_id!r} for step {sid}")
        self.touched = time.monotonic()
        self._consume()

    def finish(self) -> dict[str, Any]:
        from parikshak.engine.logger import verify_chain

        if not self.finished:
            self.builder.hold(1.0)
            self._consume()
            summary = self.engine.finish()
            records = list(self.engine.log.records) if self.engine.log else []
            ok, message = verify_chain(records)
            self.summary = {
                "duration_s": round(summary.duration_s, 1),
                "frames": summary.frames,
                "status": summary.status,
                "complete": len(summary.complete),
                "skipped": list(summary.skipped),
                "unverified": list(summary.unverified),
                "alerts": [{"t": round(a.t, 2), "step": a.step_id, "kind": a.kind.value,
                            "severity": a.severity.label, "text": a.text, "reason": a.reason}
                           for a in summary.alerts],
                "log": {"records": len(records), "verified": ok, "message": message},
            }
            self.finished = True
            self._dir.cleanup()
        return self.summary or {}

    # -- what the page draws -------------------------------------------
    def static(self) -> dict[str, Any]:
        return {
            "experiment": {"key": self.experiment.key, "title": self.experiment.title,
                           "blurb": self.experiment.blurb},
            "procedure": _procedure_view(self.proc),
            "rack": _rack_view(),
            "run_id": self.run_id,
        }

    def view(self) -> dict[str, Any]:
        return {
            "t": round(self.builder.t, 2),
            "frame": self.frame,
            "status_codes": {c: s for s, c in self._codes.items()},
            "events": self.events,
            "active": self.active,
            "prompt": self.engine.prompt,
            "actions": [a.__dict__ for a in self.actions()],
            "finished": self.finished,
            "summary": self.summary,
            "log_bytes": self.engine.log.size_bytes if self.engine.log else 0,
        }


# ------------------------------------------------------------------ sessions
_SESSIONS: dict[str, GuidedRun] = {}


def start(experiment: str) -> tuple[str, GuidedRun]:
    if len(_SESSIONS) >= MAX_SESSIONS:       # drop the least recently used
        oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k].touched)
        _SESSIONS.pop(oldest, None)
    run = GuidedRun(experiment)
    _SESSIONS[run.run_id] = run
    return run.run_id, run


def get(session_id: str) -> GuidedRun:
    run = _SESSIONS[session_id]
    run.touched = time.monotonic()
    return run
