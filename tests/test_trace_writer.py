"""A streamed trace is the same file a buffered one would have been.

TraceWriter exists so a 45-minute live run does not hold ~100 MB of frames to
write them at the end. Replacing the buffered writer is only safe if nothing
downstream can tell - golden-trace diffs, the truncation check, replay.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parikshak.belief.frame import TraceFormatError
from parikshak.belief.trace import TraceWriter, iter_trace, read_trace, write_trace

ROOT = Path(__file__).resolve().parent.parent
META = {"procedure_id": "CSP-1", "run_id": "stream", "source": "recorded", "notes": "live run",
        "extra": {"capabilities": ["crew_confirm", "object_state"],
                  "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0}}}


@pytest.fixture(scope="module")
def frames():
    return read_trace(ROOT / "traces" / "golden" / "nominal.jsonl")[1]


def test_a_streamed_trace_is_byte_identical_to_a_buffered_one(frames, tmp_path):
    buffered = tmp_path / "buffered.jsonl"
    streamed = tmp_path / "streamed.jsonl"
    expected = write_trace(buffered, frames, **META)
    with TraceWriter(streamed, **META) as writer:
        for f in frames:
            writer.append(f)
    assert streamed.read_bytes() == buffered.read_bytes()
    assert writer.header == expected
    assert not writer.partial.exists()
    header, again = read_trace(streamed)
    assert header.n_frames == len(frames) == len(again)


def test_a_run_that_dies_mid_way_leaves_a_readable_partial(frames, tmp_path):
    writer = TraceWriter(tmp_path / "crash.jsonl", **META)
    for f in frames[:2 * TraceWriter.FLUSH_EVERY]:     # ends exactly on a flush
        writer.append(f)
    recovered = list(iter_trace(writer.partial))
    assert len(recovered) == 2 * TraceWriter.FLUSH_EVERY
    assert read_trace(writer.partial)[0].n_frames == 0, "a partial must not claim a count"
    assert not writer.path.exists(), "the final trace appears only on close"
    writer.close()


def test_close_is_final(frames, tmp_path):
    writer = TraceWriter(tmp_path / "t.jsonl", **META)
    writer.append(frames[0])
    first = writer.close()
    assert writer.close() is first
    with pytest.raises(TraceFormatError):
        writer.append(frames[1])


def test_an_empty_run_matches_too(tmp_path):
    write_trace(tmp_path / "a.jsonl", [], **META)
    TraceWriter(tmp_path / "b.jsonl", **META).close()
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
