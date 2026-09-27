"""Trace I/O - BeliefFrame streams on disk as JSONL.

    line 1  : header  {"_header": "1.0", "run_id": ..., "procedure_id": ..., ...}
    line 2+ : one BeliefFrame each

Why JSONL and not a binary format: every recorded run becomes a regression test,
and a regression corpus you cannot read with `head` is a corpus nobody inspects
when a test starts failing at 2 a.m. in week 11. Size is not the constraint here -
a 143 s run is ~700 lines.

Determinism matters more than it looks. These files are fixtures: the same frames
must produce the same bytes on a Windows dev box and on the Jetson, or "did the
trace change?" stops being answerable with a diff. Hence explicit newline="\\n"
and fixed separators.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parikshak.belief.frame import BELIEF_SCHEMA_VERSION, BeliefFrame, TraceFormatError

#: Frames use compact separators (they dominate the file), the header does not
#: (it is read by humans). This asymmetry is inherited from the existing corpus.
_FRAME_SEPARATORS = (",", ":")

_HEADER_KEY = "_header"
_HEADER_FIELDS = ("_header", "run_id", "procedure_id", "source", "notes",
                  "n_frames", "duration_s")


@dataclass(frozen=True, slots=True)
class TraceHeader:
    """Line 1 of a trace. Says what this run was and what it should prove.

    `extra` carries the golden corpus's `expect` block - the assertions the
    engine must satisfy for this run. Keeping expectations in the trace rather
    than in test code means a new trace is a data file, not a code change.
    """

    schema: str
    run_id: str
    procedure_id: str
    source: str
    notes: str = ""
    n_frames: int = 0
    duration_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def expect(self) -> dict[str, Any]:
        """Expected verdicts, if this is a golden trace. Empty for recorded runs."""
        return self.extra.get("expect", {})

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            _HEADER_KEY: self.schema,
            "run_id": self.run_id,
            "procedure_id": self.procedure_id,
            "source": self.source,
            "notes": self.notes,
            "n_frames": self.n_frames,
            "duration_s": self.duration_s,
        }
        d.update(self.extra)
        return d

    @classmethod
    def from_json(cls, d: Any, where: str = "header") -> TraceHeader:
        if not isinstance(d, dict):
            raise TraceFormatError(f"{where}: expected an object, got {type(d).__name__}")
        if _HEADER_KEY not in d:
            raise TraceFormatError(
                f"{where}: line 1 has no {_HEADER_KEY!r} key - this file is missing its "
                f"header, or the frames start on line 1")
        schema = d[_HEADER_KEY]
        if schema != BELIEF_SCHEMA_VERSION:
            raise TraceFormatError(
                f"{where}: trace is BeliefFrame schema v{schema}, this build reads "
                f"v{BELIEF_SCHEMA_VERSION}. Regenerate the trace or check out the "
                f"matching engine revision.")
        return cls(
            schema=schema,
            run_id=str(d.get("run_id", "")),
            procedure_id=str(d.get("procedure_id", "")),
            source=str(d.get("source", "unknown")),
            notes=str(d.get("notes", "")),
            n_frames=int(d.get("n_frames", 0)),
            duration_s=float(d.get("duration_s", 0.0)),
            extra={k: v for k, v in d.items() if k not in _HEADER_FIELDS},
        )


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------
def write_trace(
    path: str | Path,
    frames: Iterable[BeliefFrame],
    *,
    procedure_id: str,
    run_id: str,
    source: str = "unknown",
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> TraceHeader:
    """Write frames to `path` as JSONL, header first. Returns the header written.

    `source` is free text but conventionally one of: synthetic (generated),
    recorded (real camera), replay (re-emitted). It ends up in the run log, so a
    reviewer can always tell which numbers came from real hardware.
    """
    path = Path(path)
    frames = list(frames)
    header = TraceHeader(
        schema=BELIEF_SCHEMA_VERSION,
        run_id=run_id,
        procedure_id=procedure_id,
        source=source,
        notes=notes,
        n_frames=len(frames),
        duration_s=round(frames[-1].t_mono, 3) if frames else 0.0,
        extra=dict(extra or {}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(header.to_json()) + "\n")
        for frame in frames:
            fh.write(json.dumps(frame.to_json(), separators=_FRAME_SEPARATORS) + "\n")
    return header


class TraceWriter:
    """Write a trace while the run happens, holding no frames in memory.

    `write_trace` needs every frame in a list, which for a 45-minute live run at
    10 FPS is ~100 MB that exists only to be written at the end - and is lost
    if the process dies first. This streams instead:

        during the run   frames go to `<path>.partial`, whose header declares
                         n_frames 0 ("count unknown"), flushed every FLUSH_EVERY
                         frames - so a crash still leaves a readable trace, short
                         at most the unflushed tail (drop a torn last line)
        on close()       the final file is written with the true n_frames and
                         duration, the frame lines copied across one at a time,
                         and the partial removed

    The closed file is byte-identical to what `write_trace` would have produced
    from the same frames, so the truncation check in `read_trace` and "did the
    trace change?" diffs both keep working.
    """

    FLUSH_EVERY = 50

    def __init__(self, path: str | Path, *, procedure_id: str, run_id: str,
                 source: str = "unknown", notes: str = "",
                 extra: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self._fields = {"run_id": run_id, "procedure_id": procedure_id,
                        "source": source, "notes": notes}
        self._extra = dict(extra or {})
        self.n_frames = 0
        self._last_t = 0.0
        self.header: TraceHeader | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.partial.open("w", encoding="utf-8", newline="\n")
        self._fh.write(json.dumps(self._header(0, 0.0).to_json()) + "\n")
        self._fh.flush()

    def _header(self, n_frames: int, duration_s: float) -> TraceHeader:
        return TraceHeader(schema=BELIEF_SCHEMA_VERSION, n_frames=n_frames,
                           duration_s=duration_s, extra=dict(self._extra), **self._fields)

    def append(self, frame: BeliefFrame) -> None:
        if self._fh is None:
            raise TraceFormatError(f"{self.path}: frame appended after the trace was closed")
        self._fh.write(json.dumps(frame.to_json(), separators=_FRAME_SEPARATORS) + "\n")
        self.n_frames += 1
        self._last_t = frame.t_mono
        if self.n_frames % self.FLUSH_EVERY == 0:
            self._fh.flush()

    def close(self) -> TraceHeader:
        """Finalise the trace. Safe to call twice; returns the header written."""
        if self.header is not None:
            return self.header
        self._fh.close()
        self._fh = None
        header = self._header(self.n_frames, round(self._last_t, 3) if self.n_frames else 0.0)
        staging = self.path.with_name(self.path.name + ".tmp")
        with self.partial.open("r", encoding="utf-8", newline="") as src, \
                staging.open("w", encoding="utf-8", newline="\n") as dst:
            src.readline()   # the provisional header
            dst.write(json.dumps(header.to_json()) + "\n")
            for line in src:
                dst.write(line)
        staging.replace(self.path)
        self.partial.unlink()
        self.header = header
        return header

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------
def read_trace_header(path: str | Path) -> TraceHeader:
    """Read only line 1. Cheap enough to scan a whole corpus."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
    if not first.strip():
        raise TraceFormatError(f"{path}: file is empty")
    return TraceHeader.from_json(_loads(first, path, 1), f"{path}:1")


def iter_trace(path: str | Path) -> Iterator[BeliefFrame]:
    """Stream frames without holding the run in memory.

    Used by replay and by the eval harness, which sweeps thresholds over dozens
    of runs and has no reason to materialise all of them at once.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno == 1:
                TraceHeader.from_json(_loads(line, path, 1), f"{path}:1")
                continue
            if not line.strip():
                continue
            yield BeliefFrame.from_json(_loads(line, path, lineno), f"{path}:{lineno}")


def read_trace(path: str | Path) -> tuple[TraceHeader, list[BeliefFrame]]:
    """Read a whole trace. Validates the header's frame count against reality -
    a truncated trace that silently replays short is a wasted debugging evening."""
    path = Path(path)
    header = read_trace_header(path)
    frames = list(iter_trace(path))
    if header.n_frames and header.n_frames != len(frames):
        raise TraceFormatError(
            f"{path}: header declares {header.n_frames} frames, file has {len(frames)} - "
            f"truncated or appended-to")
    return header, frames


def _loads(line: str, path: Path, lineno: int) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise TraceFormatError(f"{path}:{lineno}: not valid JSON - {exc.msg}") from exc
