"""A live run: camera -> perception -> engine -> operator display and voice.

The engine, the display and the speech queue are the ones replay and the eval
harness use. What is new here is only the front half - frames from a camera
instead of a trace, and perception assembled from what this machine actually
has. That is the stage failover argument turned around: the live path is the
replay path with a camera attached.

**What a camera-and-markers setup can see.** No trained weights exist yet, so
objects are seen through AprilTags stuck to them (perception/markers.py): their
rack-frame position, orientation and state are real measurements. Hand contact,
motion classes and crew pose need models this setup does not have. Predicates
that depend on them read UNKNOWN - never false - and `capability_gaps` names the
steps affected before the run starts, so a checklist full of "cannot verify" is
understood as what it is: the system declining to guess.

**What it records.** Every BeliefFrame, written as a trace when the run ends, so
a live run becomes a regression case exactly like a synthetic one; and the
hash-chained log, when a log path is given.

`ReplaySession` drives a recorded trace through the same interface, so the
operator window cannot tell a replay from a live run - PLAN.md's Phase 6 gate.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parikshak.belief.frame import BeliefFrame
from parikshak.belief.trace import TraceWriter, read_trace, write_trace
from parikshak.engine.runner import FrameResult, ProcedureEngine, RunSummary
from parikshak.gui.overlay import OverlayScene, build_overlay
from parikshak.gui.state import DisplayState
from parikshak.io.asr import Command, CommandGrammar, Recognised
from parikshak.io.camera import FrameSource
from parikshak.io.clipbuffer import ClipBuffer
from parikshak.io.tts import Priority, SpeechQueue
from parikshak.pdl.loader import Procedure
from parikshak.pdl.validator import VOCAB, Report, walk_predicates
from parikshak.perception.backends import AprilTagDetector
from parikshak.perception.markers import MarkerDetector, PropMarkers, SharedTags
from parikshak.perception.pipeline import EntityBinding, PerceptionPipeline, PipelineConfig
from parikshak.perception.rackframe import CameraIntrinsics, RackFrameEstimator, TagLayout

#: What a camera with fiducial-marked props can answer. `object_pose_6dof` is
#: honest: a tag's pose is a full 6-DoF measurement.
MARKER_CAPABILITIES = frozenset({
    "object_detection", "object_state", "object_pose_6dof",
    "rack_frame_extrinsics", "crew_confirm",
})
#: Added when a hand tracker is available (MediaPipe).
HAND_CAPABILITIES = frozenset({"hand_landmarks", "hand_object_contact"})


@dataclass(frozen=True)
class StepGap:
    """A step whose verification needs something this setup cannot provide."""

    step_id: str
    name: str
    needs: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.step_id} {self.name}: {'; '.join(self.needs)}"


def capability_gaps(procedure: Procedure, capabilities: frozenset[str]) -> list[StepGap]:
    """Every step that will read UNVERIFIED for want of a capability, and why.

    Walks the same vocabulary table the validator uses, so the list cannot drift
    from what the predicates actually require.
    """
    out: list[StepGap] = []
    for sid in procedure.order:
        step = procedure.step(sid)
        needs: dict[str, None] = {}
        for name, _args, _path in walk_predicates(step.verification,
                                                  f"{sid}.verification", Report()):
            lacking = [c for c in VOCAB[name]["caps"] if c not in capabilities]
            if lacking:
                needs[f"{name} needs {', '.join(lacking)}"] = None
        if needs:
            out.append(StepGap(sid, step.name, tuple(needs)))
    return out


def _frame_loss_hold_s(procedure: Procedure, default: float = 3.0) -> float:
    """The procedure's own `frame.on_frame_loss.max_hold_s`, if it declares one."""
    policy = getattr(procedure, "frame", None)
    for candidate in (getattr(policy, "on_frame_loss", None), policy):
        value = getattr(candidate, "max_hold_s", None)
        if isinstance(value, (int, float)):
            return float(value)
    return default


def announce(speech: SpeechQueue | None, result: FrameResult, last_prompt: str | None,
             *, pump: bool = True) -> str | None:
    """The announcements one frame produces - prompt on a new step, every alert,
    every UNVERIFIED notice. Shared by live and replay so they sound the same.
    Returns the step whose prompt was last announced."""
    if speech is None:
        return last_prompt
    if result.active_step is not None and result.active_step != last_prompt:
        speech.announce_prompt(result.prompt, result.active_step, t=result.t)
        last_prompt = result.active_step
    for alert in result.alerts:
        speech.announce_alert(alert.text, alert.severity.label, alert.step_id,
                              alert.kind.value, t=result.t)
    for notice in result.notices:
        speech.announce_prompt(notice.text, f"unverified:{notice.step_id}", t=result.t)
    if pump:
        speech.pump(t=result.t)
    return last_prompt


@dataclass
class Tick:
    """One frame, and everything the display and log need from it."""

    image: Any
    frame: BeliefFrame
    result: FrameResult
    state: DisplayState
    scene: OverlayScene


class LiveSession:
    """One live run. Drive it with `tick()`; talk to it with `submit()`.

    Owns no loop and no clock: the command-line runner and the Qt window each
    call `tick()` at their own pace, and the frame source says what time it is.
    """

    source_label = "live"

    def __init__(self, procedure: Procedure, source: FrameSource, layout: TagLayout, *,
                 props: PropMarkers | None, intrinsics: CameraIntrinsics | None = None,
                 hfov_deg: float = 70.0, run_id: str = "live",
                 log_path: str | Path | None = None, speech: SpeechQueue | None = None,
                 pump_speech: bool = True, detector=None, hands=None, pose=None, motion=None,
                 contact=None, fps: float = 10.0,
                 record: bool = True, tag_detector=None,
                 trace_path: str | Path | None = None,
                 clips: ClipBuffer | None = None,
                 use_deep_learning: bool = False) -> None:
        #: Evidence clips around each alert, cut from recent camera frames
        #: (io/clipbuffer.py - the laptop path; the flight path cuts them from
        #: recorded segments). None records no video at all.
        self.clips = clips
        if props is not None:
            props.check(procedure, layout)
        #: When set, frames stream to this trace as they arrive (TraceWriter)
        #: instead of being kept in `frames`, so memory stays flat however long
        #: the run. The writer opens on the first frame: its header needs the
        #: capabilities worked out below, and a run with no frames writes no trace.
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self._writer: TraceWriter | None = None
        self.proc = procedure
        self.source = source
        self.props = props
        self.fps = fps
        self.intrinsics = intrinsics or CameraIntrinsics.from_fov(
            source.width, source.height, hfov_deg)
        self.rack = RackFrameEstimator(layout, self.intrinsics,
                                       max_hold_s=_frame_loss_hold_s(procedure))
        tags = SharedTags(tag_detector or AprilTagDetector())

        bindings = EntityBinding.from_procedure(procedure)
        if props is not None:
            # An entity with no tag is one this setup cannot see at all. Left in
            # the frame it would read "not visible" - and `visible` false,
            # `absent` true - which is evidence nobody has. Left out, the engine
            # treats it as unreported: UNKNOWN.
            tagged = {t.entity for t in props.tags.values()}
            bindings = {name: b for name, b in bindings.items() if name in tagged}
        self.unobserved = tuple(sorted(set(procedure.entities) - set(bindings)))

        if detector is None and props is not None:
            detector = MarkerDetector(props, procedure, self.rack, tags)

        if use_deep_learning:
            from parikshak.perception.backends import (
                YoloDetector,
                YoloHands,
                YoloPoseEstimator,
                TcnMotionClassifier,
            )
            from parikshak.perception.contact import ContactHead, ContactMLP
            from parikshak.perception.motion import MotionTCN

            if detector is None:
                try:
                    detector = YoloDetector()
                except Exception:
                    pass
            if pose is None:
                try:
                    pose = YoloPoseEstimator()
                except Exception:
                    pass
            if hands is None and pose is not None:
                try:
                    hands = YoloHands(pose_model=pose)
                except Exception:
                    pass
            if contact is None:
                cw = Path("builds/contact_mlp.npz")
                if cw.exists():
                    try:
                        contact = ContactHead(model=ContactMLP.load(cw))
                    except Exception:
                        pass
            if motion is None:
                mw = Path("builds/motion_tcn.npz")
                if mw.exists():
                    try:
                        motion = TcnMotionClassifier(model=MotionTCN.load(mw))
                    except Exception:
                        pass

        self.pipeline = PerceptionPipeline(
            bindings, self.rack, detector=detector, hands=hands, pose=pose, motion=motion,
            contact=contact, tags=tags, config=PipelineConfig(fps=fps)
        )
        self.engine = ProcedureEngine(procedure, run_id=run_id, log_path=log_path,
                                      model_versions=self.pipeline.model_versions())

        caps = set(MARKER_CAPABILITIES) if props is not None else {
            "rack_frame_extrinsics", "crew_confirm"}
        if detector is not None:
            caps |= {"object_detection", "object_state"}
        if hands is not None:
            caps |= HAND_CAPABILITIES
        if pose is not None:
            caps |= {"body_pose"}
        if motion is not None:
            caps |= {"motion_tcn"}
        if contact is not None and getattr(contact, "is_learned", False):
            caps |= {"hand_object_contact"}
        self.capabilities = frozenset(caps)
        self.gaps = capability_gaps(procedure, self.capabilities)

        self.grammar = CommandGrammar.for_procedure(procedure)
        self.speech = speech
        self.pump_speech = pump_speech
        self.record = record
        self.frames: list[BeliefFrame] = []
        self.last: Tick | None = None
        self._pending: list[str] = []
        self._last_prompt: str | None = None

    # ------------------------------------------------------------------
    def submit(self, text: str, confidence: float = 1.0) -> Recognised | None:
        """Something the crew said, typed or clicked. Returns what it meant.

        Goes through the same closed grammar the recogniser would, so a typed
        command and a spoken one cannot mean different things. A crew token and
        an override token reach the engine as confirmations on the next frame,
        where they are logged.
        """
        heard = self.grammar.match(text, confidence)
        if heard is None:
            return None
        if heard.command in (Command.CONFIRM, Command.OVERRIDE) and heard.token:
            self._pending.append(heard.token)
        elif heard.command is Command.REPEAT and self.speech is not None and self.engine.prompt:
            self.speech.enqueue(self.engine.prompt, Priority.PROMPT,
                                dedupe_key=f"repeat:{self.engine.frames}", droppable=True,
                                t=self.engine.t)
        return heard

    def tick(self) -> Tick | None:
        """Read one frame and run it through. None when the source has ended."""
        ok, image = self.source.read()
        if not ok or image is None:
            return None
        t = self.source.now()
        confirmations, self._pending = tuple(self._pending), []
        frame = self.pipeline.step(image, t, confirmations=confirmations)
        result = self.engine.step(frame)
        if self.clips is not None:
            # The frame goes in before the request, so a clip includes the moment
            # its alert was raised.
            for clip in self.clips.push(t, image):
                self._log_clip(clip)
            for alert in result.alerts:
                self.clips.request(self.engine.run_id, alert.step_id, alert.kind.value, alert.t)
        if self.trace_path is not None:
            if self._writer is None:
                self._writer = TraceWriter(self.trace_path, **self._trace_meta())
            self._writer.append(frame)
        elif self.record:
            self.frames.append(frame)
        self._last_prompt = announce(self.speech, result, self._last_prompt,
                                     pump=self.pump_speech)

        active = self.engine.active_step
        scene = build_overlay(self.proc, frame, self.rack.extrinsics, self.intrinsics,
                              active_objects=self.proc.step(active).objects if active else ())
        state = DisplayState.from_engine(
            self.engine, frame=frame, source=self.source_label,
            unverifiable={g.step_id: "; ".join(g.needs) for g in self.gaps})
        self.last = Tick(image, frame, result, state, scene)
        return self.last

    def _trace_meta(self) -> dict[str, Any]:
        """The trace header for this run - the same whether streamed or buffered."""
        return {
            "procedure_id": self.proc.id, "run_id": self.engine.run_id, "source": "recorded",
            "notes": "live run: fiducial-marked props" if self.props else "live run",
            "extra": {"capabilities": sorted(self.capabilities),
                      "unobserved_entities": list(self.unobserved),
                      "intrinsics": {"fx": self.intrinsics.fx, "fy": self.intrinsics.fy,
                                     "cx": self.intrinsics.cx, "cy": self.intrinsics.cy}},
        }

    def _log_clip(self, clip) -> None:
        """A clip - or the fact that one could not be made - is part of the run's
        record. A reviewer needs to know evidence is missing, not just absent."""
        if self.engine.log is not None:
            self.engine.log.append("evidence_clip", clip.as_dict(), self.engine.t)

    @property
    def recorded_frames(self) -> int:
        """Frames recorded so far, streamed or held."""
        return self._writer.n_frames if self._writer is not None else len(self.frames)

    def finish(self, trace_path: str | Path | None = None) -> RunSummary:
        """End the run and write its trace.

        A session streaming to a trace closes it. Passing that same path again
        is fine; a different one is an error, raised before anything is
        consumed - the frames are already on disk under the first name.
        Otherwise the held frames are written to `trace_path` when one is given.
        """
        if (trace_path is not None and self.trace_path is not None
                and Path(trace_path).resolve() != self.trace_path.resolve()):
            raise ValueError(f"this session streams its trace to {self.trace_path}, "
                             f"not {trace_path}")
        if self.clips is not None:
            # Before the engine closes its log: a clip whose post-roll never came
            # is still evidence, and the log should say what it covers.
            for clip in self.clips.flush():
                self._log_clip(clip)
        summary = self.engine.finish()
        self.source.close()
        if self._writer is not None:
            self._writer.close()
        elif trace_path is not None and self.frames:
            write_trace(trace_path, self.frames, **self._trace_meta())
        return summary


class ReplaySession:
    """A recorded trace, driven exactly like a live session.

    Same `tick()` / `submit()` / `finish()` shape, same engine, same display
    state, same announcements - so the operator window cannot tell the two
    apart. Adds what only a recording can do: `seek` to any frame.

    A trace carries belief, not pixels, so the picture is the rack schematic
    drawn from each frame. When the trace came from a live run, the
    capabilities that run had are in its header, and the same "not verifiable
    here" rows come back.
    """

    source_label = "replay"

    def __init__(self, procedure: Procedure, trace_path: str | Path, *,
                 speech: SpeechQueue | None = None, pump_speech: bool = True,
                 render: bool = True, run_id: str | None = None) -> None:
        self.proc = procedure
        self.trace_path = Path(trace_path)
        self.header, self._frames = read_trace(self.trace_path)
        self.run_id = run_id or self.header.run_id or self.trace_path.stem
        self.speech = speech
        self.pump_speech = pump_speech
        self.render = render
        steps = [b.t_mono - a.t_mono for a, b in zip(self._frames, self._frames[1:])
                 if b.t_mono > a.t_mono]
        self.fps = 1.0 / statistics.median(steps) if steps else 10.0
        caps = self.header.extra.get("capabilities")
        self.gaps = capability_gaps(procedure, frozenset(caps)) if caps else []
        self.unobserved = tuple(self.header.extra.get("unobserved_entities", ()))
        self._reset()

    def _reset(self) -> None:
        self.engine = ProcedureEngine(self.proc, run_id=self.run_id)
        self.position = 0
        self.last: Tick | None = None
        self._last_prompt: str | None = None

    @property
    def length(self) -> int:
        return len(self._frames)

    @property
    def duration_s(self) -> float:
        return self._frames[-1].t_mono if self._frames else 0.0

    def submit(self, text: str, confidence: float = 1.0) -> Recognised | None:
        """A recording already contains everything the crew said. Nothing to add."""
        return None

    def tick(self) -> Tick | None:
        if self.position >= self.length:
            return None
        frame = self._frames[self.position]
        self.position += 1
        result = self.engine.step(frame)
        self._last_prompt = announce(self.speech, result, self._last_prompt,
                                     pump=self.pump_speech)
        image = None
        if self.render:
            try:
                from parikshak.eval.schematic import render_schematic
                active = self.engine.active_step
                image = render_schematic(self.proc, frame, 960, 540,
                                         active_objects=self.proc.step(active).objects
                                         if active else ())
            except ImportError:
                image = None
        state = DisplayState.from_engine(
            self.engine, frame=frame, source=self.source_label,
            unverifiable={g.step_id: "; ".join(g.needs) for g in self.gaps})
        self.last = Tick(image, frame, result, state, OverlayScene())
        return self.last

    def seek(self, index: int) -> None:
        """Move to frame `index`. The engine is deterministic, so going back is a
        silent re-run from the start - the only honest way to get the state at
        an earlier moment, because nothing in the engine can be un-stepped."""
        index = max(0, min(int(index), self.length))
        if index < self.position:
            self._reset()
        while self.position < index:
            self.engine.step(self._frames[self.position])
            self.position += 1
        if self.engine.active_step is not None:
            self._last_prompt = self.engine.active_step   # no burst of stale prompts

    def finish(self, trace_path: str | Path | None = None) -> RunSummary:
        """The recording already exists; `trace_path` is accepted and ignored."""
        return self.engine.finish()
