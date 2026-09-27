"""Frame sources for a live run: a real camera, or a scripted synthetic one.

The GStreamer graph in capture.py is the flight path - record, perceive and
stream from one tee. It needs GStreamer, which a development laptop usually does
not have, so `parikshak run` reads frames through this thinner interface
instead: OpenCV on a webcam, or a synthetic scene with known ground truth. The
live session does not know which it has, exactly as the engine does not know
whether a BeliefFrame came from a camera or a trace.

Two sources, one protocol:

    read()  -> (ok, BGR image or None)
    now()   -> seconds since the source opened, on the source's own clock
    close()

A synthetic source keeps a deterministic clock - frame count over frame rate -
so a smoke test of the live loop gives the same verdicts on a loaded CI machine
as on an idle one. A real camera uses wall-clock monotonic time, because the
crew does not slow down when the laptop does.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


class CameraError(RuntimeError):
    """The frame source could not be opened or has stopped delivering."""


@runtime_checkable
class FrameSource(Protocol):
    width: int
    height: int

    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def now(self) -> float: ...
    def close(self) -> None: ...


class OpenCVCamera:
    """A webcam through OpenCV.

    DirectShow on Windows: the default Media Foundation backend takes several
    seconds to open some USB cameras and ignores resolution requests on others.
    The resolution actually delivered is read back, because intrinsics computed
    for 1280x720 on a camera quietly giving 640x480 put every coordinate in the
    wrong place.
    """

    def __init__(self, index: int = 0, *, width: int = 1280, height: int = 720,
                 api: str = "auto") -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional extra
            raise CameraError("a live camera needs OpenCV: pip install opencv-python") from exc
        self._cv2 = cv2
        backend = cv2.CAP_ANY
        if api == "dshow" or (api == "auto" and sys.platform.startswith("win")):
            backend = cv2.CAP_DSHOW
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise CameraError(f"camera {index} could not be opened - is it connected, and "
                              f"not in use by another application?")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # One buffered frame. Perception reads at 10 Hz from a 30 Hz camera; with
        # the default buffer every read returns a frame from a third of a second
        # ago, and the crew watches the checklist lag their hands.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise CameraError(f"camera {index} opened but delivered no frame")
        self.height, self.width = frame.shape[:2]
        self.index = index
        self._t0 = time.monotonic()

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        return (bool(ok and frame is not None), frame if ok else None)

    def now(self) -> float:
        return time.monotonic() - self._t0

    def close(self) -> None:
        self._cap.release()


@dataclass(frozen=True)
class ScriptedProp:
    """A prop tag present in the synthetic scene between t_start and t_end."""

    tag_id: int
    t_start: float
    t_end: float
    centre: tuple[float, float, float]


@dataclass
class SyntheticCamera:
    """A TagScene played on a deterministic clock.

    The rack tags are always in view; props come and go on a schedule; during a
    blackout the whole frame is a flat grey, which is what a hand over the lens
    looks like to a detector. Enough to watch the live loop lock, see a prop,
    lose the lock and recover - with no camera and no printed tags.
    """

    scene: object                       # parikshak.eval.scene.TagScene
    layout: object                      # parikshak.perception.rackframe.TagLayout
    props: list[ScriptedProp] = field(default_factory=list)
    tag_size_m: float = 0.06
    fps: float = 10.0
    blackouts: list[tuple[float, float]] = field(default_factory=list)
    duration_s: float | None = None
    _frames: int = 0

    @property
    def width(self) -> int:
        return self.scene.width

    @property
    def height(self) -> int:
        return self.scene.height

    def now(self) -> float:
        return self._frames / self.fps

    def read(self) -> tuple[bool, np.ndarray | None]:
        t = self.now()
        if self.duration_s is not None and t > self.duration_s:
            return False, None
        self._frames += 1
        if any(a <= t < b for a, b in self.blackouts):
            return True, np.full((self.height, self.width, 3), 90, dtype=np.uint8)
        tags = list(self.layout.corners.items())
        tags += [(p.tag_id, self.scene.square(p.centre, self.tag_size_m))
                 for p in self.props if p.t_start <= t < p.t_end]
        return True, self.scene.render(tags)

    def close(self) -> None:
        pass
