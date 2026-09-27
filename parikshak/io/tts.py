"""Speech output: a queue with a policy, and a Piper adapter behind it.

The engine already decided WHAT to say and how urgently. This decides what
happens when two things need saying at once, which is a different problem and
the one that determines whether the crew keeps the speaker switched on.

Four rules, all of them about not being annoying:

  **Critical pre-empts.** "Stop, the latch is open while the unit is running"
  does not wait behind a step prompt. It interrupts.

  **Same message, once.** A condition that persists produces the same sentence
  every frame. Saying it every frame is how a system gets muted. Repeats are
  allowed only after the tier's `repeat_s`.

  **Prompts are droppable, alerts are not.** If speech is falling behind, the
  next-step prompt is stale by the time it plays and is worth dropping. An alert
  is never stale - it describes something that happened.

  **Nothing is ever spoken over.** One utterance at a time, because two voices
  at once is worse than either alone.

No engine import: this takes plain text and a priority, so the queue can be
tested without constructing an Alert and so io/ stays free of engine types.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Callable, Protocol, runtime_checkable


class Priority(IntEnum):
    """Lower sorts first out of the queue."""

    CRITICAL = 0
    CAUTION = 1
    ADVISORY = 2
    PROMPT = 3

    @classmethod
    def from_severity(cls, severity: str) -> Priority:
        return {
            "critical": cls.CRITICAL,
            "caution": cls.CAUTION,
            "advisory": cls.ADVISORY,
            "info": cls.PROMPT,
        }.get(severity.lower(), cls.ADVISORY)


@dataclass(order=True)
class Utterance:
    priority: Priority
    seq: int
    text: str = field(compare=False)
    #: Messages sharing a key are the same thing being said again.
    dedupe_key: str = field(compare=False, default="")
    #: Droppable when the queue is backed up. Prompts are; alerts are not.
    droppable: bool = field(compare=False, default=False)
    t: float = field(compare=False, default=0.0)


@runtime_checkable
class Voice(Protocol):
    def say(self, text: str) -> None: ...


class NullVoice:
    """Records what would have been said. The default, so a run never depends on
    an audio device existing - and so tests can assert on speech."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)


class SpeechQueue:
    """Priority queue with dedupe and pre-emption. Synchronous by default.

    `pump()` returns what was actually spoken, so the whole policy is testable
    without threads, audio hardware, or timing. `run_in_background()` exists for
    the live system, and is a thin wrapper over the same `pump`.
    """

    def __init__(self, voice: Voice | None = None, *,
                 repeat_s: float = 10.0, max_pending: int = 8) -> None:
        self.voice = voice or NullVoice()
        self.repeat_s = repeat_s
        self.max_pending = max_pending

        self._q: PriorityQueue[Utterance] = PriorityQueue()
        self._seq = 0
        self._last_said: dict[str, float] = {}
        self._last_text: dict[str, float] = {}
        self._pending_keys: set[str] = set()
        self._stop = threading.Event()
        self.dropped = 0
        self.suppressed = 0

    # ------------------------------------------------------------------
    def enqueue(self, text: str, priority: Priority = Priority.ADVISORY, *,
                dedupe_key: str = "", droppable: bool = False, t: float = 0.0) -> bool:
        """Returns True if it was accepted for speaking."""
        if not text.strip():
            return False
        key = dedupe_key or text

        last = self._last_said.get(key)
        if last is not None and (t - last) < self.repeat_s:
            self.suppressed += 1
            return False
        if key in self._pending_keys:
            self.suppressed += 1
            return False

        # Same SENTENCE, different key. The latch-open invariant is declared on
        # both S09 and S10, so a single latch left open produces two alerts with
        # different dedupe keys and identical wording - and the crew hears
        # "stop, the latch is open" twice in a fifth of a second. Whatever the
        # engine's bookkeeping says, saying the same words twice is one message.
        spoken_at = self._last_text.get(text)
        if spoken_at is not None and (t - spoken_at) < self.repeat_s:
            self.suppressed += 1
            return False

        if self._q.qsize() >= self.max_pending:
            if droppable:
                self.dropped += 1
                return False
            self._drop_one_droppable()

        self._seq += 1
        self._q.put(Utterance(priority, self._seq, text, key, droppable, t))
        self._pending_keys.add(key)
        return True

    def _drop_one_droppable(self) -> None:
        """Make room by discarding the lowest-priority droppable item.

        A stale next-step prompt is worth less than an alert about something
        that already happened, so the prompt goes.
        """
        held: list[Utterance] = []
        victim: Utterance | None = None
        while True:
            try:
                item = self._q.get_nowait()
            except Empty:
                break
            if item.droppable and (victim is None or item.priority > victim.priority):
                if victim is not None:
                    held.append(victim)
                victim = item
            else:
                held.append(item)
        for item in held:
            self._q.put(item)
        if victim is not None:
            self._pending_keys.discard(victim.dedupe_key)
            self.dropped += 1

    # ------------------------------------------------------------------
    def pump(self, t: float = 0.0, limit: int | None = None) -> list[str]:
        """Speak what is queued, highest priority first. Returns what was said."""
        said: list[str] = []
        while limit is None or len(said) < limit:
            try:
                item = self._q.get_nowait()
            except Empty:
                break
            self._pending_keys.discard(item.dedupe_key)
            self.voice.say(item.text)
            self._last_said[item.dedupe_key] = max(t, item.t)
            self._last_text[item.text] = max(t, item.t)
            said.append(item.text)
        return said

    @property
    def pending(self) -> int:
        return self._q.qsize()

    # ------------------------------------------------------------------
    def announce_alert(self, text: str, severity: str, step_id: str,
                       kind: str, t: float = 0.0) -> bool:
        """An alert. Never droppable - it describes something that happened."""
        return self.enqueue(text, Priority.from_severity(severity),
                            dedupe_key=f"alert:{step_id}:{kind}", droppable=False, t=t)

    def announce_prompt(self, text: str, step_id: str, t: float = 0.0) -> bool:
        """The next step. Droppable, because a stale prompt is worse than none."""
        return self.enqueue(text, Priority.PROMPT,
                            dedupe_key=f"prompt:{step_id}", droppable=True, t=t)

    # ------------------------------------------------------------------
    def run_in_background(self, poll_s: float = 0.05) -> threading.Thread:  # pragma: no cover
        def loop():
            while not self._stop.is_set():
                if not self.pump():
                    self._stop.wait(poll_s)
        thread = threading.Thread(target=loop, name="parikshak-tts", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:  # pragma: no cover
        self._stop.set()


# --------------------------------------------------------------------------
class PiperVoice:
    """Piper TTS: ONNX, ~60 MB voice, real time on CPU, fully offline.

    Offline is the requirement, not a preference. A cloud voice would put a
    network dependency in the crew's alert path, which is exactly the claim the
    demo opens by disproving with an unplugged ethernet cable.
    """

    def __init__(self, model_path: str | Path, *, binary: str = "piper",
                 player: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.binary = binary
        self.player = player
        if not self.model_path.exists():
            raise FileNotFoundError(f"piper voice model not found: {self.model_path}")

    def say(self, text: str) -> None:  # pragma: no cover - needs piper installed
        cmd = [self.binary, "--model", str(self.model_path), "--output-raw"]
        speak = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        if self.player:
            play = subprocess.Popen(self.player.split(), stdin=speak.stdout)
            speak.communicate(text.encode("utf-8"))
            play.wait()
        else:
            speak.communicate(text.encode("utf-8"))


class Pyttsx3Voice:
    """The operating system's own offline voice, through pyttsx3.

    SAPI5 on Windows, NSSpeechSynthesizer on macOS, eSpeak on Linux - no model
    file, no network. The development-laptop stand-in for Piper, and a
    perfectly good one: the queue's policy is what makes speech bearable, and
    that is identical whichever voice is behind it.

    The engine is created on the first `say`, not in the constructor. The queue
    speaks from its own thread, and SAPI's COM objects must be used on the thread
    that created them; building the engine in the GUI thread and speaking from
    the queue's makes the first alert hang.
    """

    def __init__(self, *, rate: int | None = None) -> None:
        try:
            import pyttsx3  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("Pyttsx3Voice needs pyttsx3: pip install pyttsx3") from exc
        self.rate = rate
        self._engine = None

    def say(self, text: str) -> None:  # pragma: no cover - needs an audio device
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            if self.rate:
                self._engine.setProperty("rate", self.rate)
        self._engine.say(text)
        self._engine.runAndWait()


class CallbackVoice:
    """Routes speech to any callable - a GUI caption bar, a test, a log."""

    def __init__(self, fn: Callable[[str], None]) -> None:
        self.fn = fn

    def say(self, text: str) -> None:
        self.fn(text)
