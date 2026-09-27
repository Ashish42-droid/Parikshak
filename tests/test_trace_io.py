"""Trace I/O, and the properties that make the golden corpus usable as fixtures.

The corpus is only a regression suite if regenerating it is a no-op. The
`hash(name)` defect these tests now guard against made every regeneration
produce different track IDs, which would have turned every trace diff into
noise nobody reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parikshak.belief.frame import BELIEF_SCHEMA_VERSION, TraceFormatError, empty_frame
from parikshak.belief.trace import iter_trace, read_trace, read_trace_header, write_trace

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "traces" / "golden"

GOLDEN_TRACES = sorted(GOLDEN.glob("*.jsonl"))


def some_frames(n: int = 3):
    return [empty_frame(t_mono=round(i * 0.2, 3)) for i in range(n)]


# -- write/read round trip -------------------------------------------------
def test_write_then_read_round_trips(tmp_path):
    frames = some_frames(5)
    write_trace(tmp_path / "t.jsonl", frames, procedure_id="CSP-1",
                run_id="unit", source="synthetic", notes="hello")
    header, back = read_trace(tmp_path / "t.jsonl")
    assert back == frames
    assert (header.run_id, header.procedure_id, header.source) == ("unit", "CSP-1", "synthetic")
    assert header.n_frames == 5


def test_header_records_duration_from_last_frame(tmp_path):
    write_trace(tmp_path / "t.jsonl", some_frames(5), procedure_id="CSP-1", run_id="unit")
    assert read_trace_header(tmp_path / "t.jsonl").duration_s == pytest.approx(0.8)


def test_extra_block_survives_the_round_trip(tmp_path):
    """`expect` lives in the trace so that adding a golden case is a data change,
    not a code change."""
    write_trace(tmp_path / "t.jsonl", some_frames(2), procedure_id="CSP-1",
                run_id="unit", extra={"expect": {"complete": ["S01"]}})
    assert read_trace_header(tmp_path / "t.jsonl").expect == {"complete": ["S01"]}


def test_line_endings_are_lf_on_every_platform(tmp_path):
    """Fixtures must hash the same on a Windows dev box and on the Jetson."""
    p = tmp_path / "t.jsonl"
    write_trace(p, some_frames(3), procedure_id="CSP-1", run_id="unit")
    assert b"\r\n" not in p.read_bytes()


def test_writing_the_same_frames_twice_gives_identical_bytes(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    frames = some_frames(4)
    for p in (a, b):
        write_trace(p, frames, procedure_id="CSP-1", run_id="unit")
    assert a.read_bytes() == b.read_bytes()


# -- malformed input -------------------------------------------------------
def test_empty_file_is_rejected(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(TraceFormatError, match="empty"):
        read_trace_header(p)


def test_missing_header_is_rejected(tmp_path):
    """A file whose frames start on line 1 must fail loudly, not silently drop
    its first frame."""
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(empty_frame().to_json()) + "\n", encoding="utf-8")
    with pytest.raises(TraceFormatError, match="no '_header'"):
        read_trace_header(p)


def test_future_schema_version_is_refused(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"_header": "2.0", "run_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(TraceFormatError, match="schema"):
        read_trace_header(p)


def test_truncated_trace_is_detected(tmp_path):
    """Replaying a short run and wondering why the last three steps never
    completed is an evening nobody gets back."""
    p = tmp_path / "t.jsonl"
    write_trace(p, some_frames(5), procedure_id="CSP-1", run_id="unit")
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(TraceFormatError, match="truncated"):
        read_trace(p)


def test_bad_json_names_the_line(tmp_path):
    p = tmp_path / "t.jsonl"
    write_trace(p, some_frames(3), procedure_id="CSP-1", run_id="unit")
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2][:-5]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(TraceFormatError, match=r":3:"):
        list(iter_trace(p))


# -- the golden corpus -----------------------------------------------------
def test_the_corpus_exists():
    assert GOLDEN_TRACES, "no golden traces - run: python tools/make_golden_traces.py"


@pytest.mark.parametrize("path", GOLDEN_TRACES, ids=lambda p: p.stem)
def test_every_golden_trace_parses_under_the_frozen_schema(path):
    header, frames = read_trace(path)
    assert header.schema == BELIEF_SCHEMA_VERSION
    assert frames, f"{path.name} has no frames"
    assert header.procedure_id == "CSP-1"


@pytest.mark.parametrize("path", GOLDEN_TRACES, ids=lambda p: p.stem)
def test_every_golden_trace_declares_what_it_proves(path):
    """A trace with no expectations is a file that can never fail, which is
    worse than not having it."""
    assert read_trace_header(path).expect, f"{path.name} has an empty expect block"


@pytest.mark.parametrize("path", GOLDEN_TRACES, ids=lambda p: p.stem)
def test_time_advances_monotonically(path):
    _, frames = read_trace(path)
    times = [f.t_mono for f in frames]
    assert times == sorted(times), f"{path.name}: t_mono goes backwards"


@pytest.mark.parametrize("path", GOLDEN_TRACES, ids=lambda p: p.stem)
def test_track_ids_are_stable_across_processes(path):
    """Guards the `hash(name)` defect: Python randomises string hashing per
    process, so the first generator emitted different IDs on every run."""
    _, frames = read_trace(path)
    ids = {name: ob.track_id for name, ob in frames[0].objects.items()}
    for f in frames:
        for name, ob in f.objects.items():
            assert ob.track_id == ids[name], f"{path.name}: {name} changed track_id mid-run"


def test_manifest_agrees_with_the_traces():
    manifest = json.loads((GOLDEN / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == len(GOLDEN_TRACES)
    for entry in manifest:
        header, frames = read_trace(GOLDEN / entry["trace"])
        assert len(frames) == entry["frames"]
        assert header.expect == entry["expect"]
