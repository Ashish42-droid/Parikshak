"""Event clips: the visual evidence for a deviation.

A deviation at t produces a clip covering [t - 10, t + 10]. Two clips on a
typical run is about 20 MB - still roughly seventy times less than downlinking
the whole recording, while keeping exactly the part a reviewer wants to look at.

The part worth testing is not the encoding. It is deciding WHICH recorded
segments cover the window, and what to do when they do not:

  - The window straddles a segment boundary, so the clip spans two files.
  - The deviation happened less than ten seconds into the run, so the pre-roll
    is short. That is a shorter clip, not an error.
  - The ring buffer already trimmed the segment. Then there is no clip, and the
    log must say the evidence is unavailable rather than silently omitting it.

That last case is why `export` returns a result object instead of a path. A
missing clip is a fact about the run, and a flight record that quietly drops it
is worse than one that records the gap.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from parikshak.io.capture import Segment, SegmentIndex

#: Seconds either side of the event. Ten is enough to see the approach and the
#: consequence without turning the clip into the recording.
DEFAULT_PRE_S = 10.0
DEFAULT_POST_S = 10.0


@dataclass(frozen=True, slots=True)
class ClipRequest:
    run_id: str
    step_id: str
    kind: str
    t_event: float
    pre_s: float = DEFAULT_PRE_S
    post_s: float = DEFAULT_POST_S

    @property
    def window(self) -> tuple[float, float]:
        return (max(0.0, self.t_event - self.pre_s), self.t_event + self.post_s)

    @property
    def basename(self) -> str:
        return f"{self.run_id}_{self.step_id}_{self.kind.lower()}_{self.t_event:07.1f}"


@dataclass
class ClipResult:
    """What happened. A clip that could not be made is recorded, not hidden."""

    request: ClipRequest
    path: Path | None = None
    segments: list[Segment] = field(default_factory=list)
    available: bool = True
    reason: str = ""

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path and self.path.exists() else 0

    def as_dict(self) -> dict:
        return {
            "step_id": self.request.step_id,
            "kind": self.request.kind,
            "t_event": round(self.request.t_event, 3),
            "window": [round(v, 3) for v in self.request.window],
            "available": self.available,
            "reason": self.reason,
            "path": self.path.name if self.path else None,
            "segments": [s.path.name for s in self.segments],
        }


class ClipExporter:
    """Selects segments for a window and cuts a clip from them.

    Selection is pure and tested. Cutting shells out to ffmpeg, because writing
    a container remuxer is not this project's contribution and a wrong one loses
    evidence silently.
    """

    def __init__(self, index: SegmentIndex, out_dir: str | Path, *,
                 ffmpeg: str = "ffmpeg", ring=None) -> None:
        self.index = index
        self.out_dir = Path(out_dir)
        self.ffmpeg = ffmpeg
        self.ring = ring
        self.results: list[ClipResult] = []

    # ------------------------------------------------------------------
    def plan(self, request: ClipRequest) -> ClipResult:
        """Decide what the clip needs, without cutting it.

        Called the moment a deviation is detected so the ring buffer can pin
        those segments before the encoder rolls past them. Evidence outranks
        capacity, and the window is only reachable for as long as it is on disk.
        """
        lo, hi = request.window
        segments = self.index.covering(lo, hi)
        if not segments:
            return ClipResult(request, None, [], False,
                              f"no recorded segment covers [{lo:.1f}, {hi:.1f}] - "
                              f"already trimmed, or recording was not enabled")
        if self.ring is not None:
            self.ring.protect(segments)
        covered_from = min(s.t_start for s in segments)
        covered_to = max(s.t_end for s in segments)
        reason = ""
        if covered_from > lo or covered_to < hi:
            # Partial coverage is still evidence, but the log should say so
            # rather than implying the clip shows the full window.
            reason = (f"partial coverage: have [{covered_from:.1f}, {covered_to:.1f}] "
                      f"of [{lo:.1f}, {hi:.1f}]")
        return ClipResult(request, None, list(segments), True, reason)

    # ------------------------------------------------------------------
    def export(self, request: ClipRequest, *, dry_run: bool = False) -> ClipResult:
        result = self.plan(request)
        self.results.append(result)
        if not result.available:
            return result

        self.out_dir.mkdir(parents=True, exist_ok=True)
        result.path = self.out_dir / f"{request.basename}.mkv"
        if dry_run:
            return result

        lo, _hi = request.window
        first = result.segments[0]
        try:
            self._cut(result, offset=max(0.0, lo - first.t_start))
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            result.available = False
            result.reason = f"clip export failed: {exc}"
            result.path = None
        finally:
            if self.ring is not None:
                self.ring.release(result.segments)
        return result

    def _cut(self, result: ClipResult, offset: float) -> None:  # pragma: no cover
        """Stream-copy the window out of the covering segments.

        `-c copy` rather than re-encoding: a deviation clip is evidence, and
        re-encoding it discards detail for no benefit while costing time on a
        box that has inference to do.
        """
        request = result.request
        duration = request.pre_s + request.post_s
        concat = self.out_dir / f"{request.basename}.txt"
        concat.write_text(
            "\n".join(f"file '{s.path.as_posix()}'" for s in result.segments) + "\n",
            encoding="utf-8", newline="\n")
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "concat", "-safe", "0", "-i", str(concat),
               "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
               "-c", "copy", str(result.path)]
        subprocess.run(cmd, check=True, timeout=120)
        concat.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    @property
    def unavailable(self) -> list[ClipResult]:
        """Deviations with no visual evidence. Belongs in the run summary - a
        reviewer needs to know a clip is missing, not just that it is absent."""
        return [r for r in self.results if not r.available]
