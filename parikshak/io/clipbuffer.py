"""Deviation clips on a laptop: the last seconds of video, held in memory.

`clips.py` cuts evidence out of recorded segment files with ffmpeg - the flight
path, where GStreamer is recording to disk anyway. A laptop running
`parikshak run` has neither a recorder nor ffmpeg. This keeps the most recent
frames in memory as JPEG and, when a deviation is alerted at time t, writes a
clip covering [t - pre_s, t + post_s] with OpenCV once the post-roll has been
seen.

The rules `clips.py` sets still hold. A clip is evidence, so a clip that could
not be made in full is still written and carries the reason - "partial
coverage: have [3.0, 11.0] of [-2.0, 11.0]" - and a clip that could not be made
at all is a `ClipResult` with `available=False`, never a silent omission.

Memory is bounded: frames older than the pre-roll are dropped unless a pending
clip still needs them. At 10 Hz, 960 px wide JPEG frames are ~40-60 KB, so a
10 s pre-roll plus a 10 s post-roll is ~10 MB.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from parikshak.io.clips import DEFAULT_POST_S, DEFAULT_PRE_S, ClipRequest, ClipResult

#: Slack when judging coverage: one frame late is not a gap worth reporting.
_FRAME_SLACK = 1.5


class ClipBuffer:
    """Keep recent frames; cut a clip around each alert."""

    def __init__(self, out_dir: str | Path, *, pre_s: float = DEFAULT_PRE_S,
                 post_s: float = DEFAULT_POST_S, fps: float = 10.0,
                 max_width: int = 960, jpeg_quality: int = 80) -> None:
        self.out_dir = Path(out_dir)
        self.pre_s, self.post_s, self.fps = pre_s, post_s, fps
        self.max_width, self.jpeg_quality = max_width, jpeg_quality
        self._frames: deque[tuple[float, bytes]] = deque()
        self._pending: list[ClipRequest] = []
        self.results: list[ClipResult] = []
        self.t = 0.0

    # ------------------------------------------------------------------
    def push(self, t: float, image: Any) -> list[ClipResult]:
        """Add one camera frame. Returns the clips completed by it."""
        import cv2

        self.t = t
        h, w = image.shape[:2]
        if w > self.max_width:
            image = cv2.resize(image, (self.max_width, int(h * self.max_width / w)),
                               interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            self._frames.append((t, jpeg.tobytes()))

        done = [r for r in self._pending if t >= r.window[1]]
        out = [self._write(r) for r in done]
        self._pending = [r for r in self._pending if r not in done]
        self._trim()
        return out

    def request(self, run_id: str, step_id: str, kind: str, t_event: float) -> ClipRequest:
        """Ask for a clip around an alert. It is written once the post-roll is in."""
        req = ClipRequest(run_id=run_id, step_id=step_id, kind=kind, t_event=t_event,
                          pre_s=self.pre_s, post_s=self.post_s)
        if not any(p.basename == req.basename for p in self._pending):
            self._pending.append(req)
        return req

    def flush(self) -> list[ClipResult]:
        """End of run: write every pending clip with what there is."""
        out = [self._write(r) for r in self._pending]
        self._pending = []
        return out

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    @property
    def unavailable(self) -> list[ClipResult]:
        return [r for r in self.results if not r.available]

    # ------------------------------------------------------------------
    def _trim(self) -> None:
        keep_from = self.t - self.pre_s
        for req in self._pending:
            keep_from = min(keep_from, req.window[0])
        while self._frames and self._frames[0][0] < keep_from:
            self._frames.popleft()

    def _write(self, req: ClipRequest) -> ClipResult:
        import cv2
        import numpy as np

        lo, hi = req.t_event - req.pre_s, req.t_event + req.post_s
        frames = [(t, b) for t, b in self._frames if lo <= t <= hi]
        if not frames:
            result = ClipResult(req, None, [], False,
                                f"no frames buffered for [{lo:.1f}, {hi:.1f}]")
            self.results.append(result)
            return result

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{req.basename}.mp4"
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                 (width, height))
        try:
            if not writer.isOpened():
                result = ClipResult(req, None, [], False, "the video writer could not be opened")
                self.results.append(result)
                return result
            for _t, data in frames:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                writer.write(img)
        finally:
            writer.release()

        slack = _FRAME_SLACK / self.fps
        have = (frames[0][0], frames[-1][0])
        reason = ""
        if have[0] > lo + slack or have[1] < hi - slack:
            reason = (f"partial coverage: have [{have[0]:.1f}, {have[1]:.1f}] "
                      f"of [{lo:.1f}, {hi:.1f}]")
        result = ClipResult(req, path, [], True, reason)
        self.results.append(result)
        return result
