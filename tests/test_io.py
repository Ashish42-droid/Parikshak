"""I/O: capture graph, speech policy, command grammar, clips, downlink.

None of GStreamer, Piper or Vosk is installed here, and that is the point -
everything below is checkable without them. What gets tested is the logic that
actually goes wrong: the shape of the capture graph, what happens when two
things need saying at once, which segments a clip needs, and what survives when
the link budget runs out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parikshak.io.asr import CONSEQUENTIAL, Command, CommandGrammar, normalise
from parikshak.io.capture import (
    CaptureConfig,
    CaptureError,
    RingBuffer,
    Segment,
    SegmentIndex,
    build_pipeline,
)
from parikshak.io.clips import ClipExporter, ClipRequest
from parikshak.io.downlink import DownlinkPackage, Priority, compression_ratio
from parikshak.io.tts import NullVoice, Priority as SpeechPriority, SpeechQueue


# ==========================================================================
# capture graph
# ==========================================================================
def cfg(**kw) -> CaptureConfig:
    kw.setdefault("record_dir", Path("/data/rec"))
    return CaptureConfig(**kw)


def test_every_tee_branch_has_a_queue():
    """Without a queue per branch the tee runs in one thread and the slowest
    consumer sets the pace for the camera."""
    p = build_pipeline(cfg(stream_url="rtsp://host/rack"))
    branches = [b for b in p.split("t. ! ") if b.strip()][1:]
    assert len(branches) == 3
    assert all(b.lstrip().startswith("queue") for b in branches)


def test_the_recorder_branch_is_never_leaky():
    """Dropping frames from the flight record to keep a preview smooth is the
    wrong trade. The recorder must apply backpressure, not discard."""
    p = build_pipeline(cfg())
    recorder = next(b for b in p.split("t. ! ") if "splitmuxsink" in b)
    assert "leaky" not in recorder


def test_the_stream_branch_is_leaky():
    """A ground link that goes away must not apply backpressure all the way to
    the camera and take the local recording down with it."""
    p = build_pipeline(cfg(stream_url="rtsp://host/rack"))
    stream = next(b for b in p.split("t. ! ") if "rtspclientsink" in b)
    assert "leaky=downstream" in stream


def test_perception_is_decimated_not_the_camera():
    """The recorder keeps full frame rate regardless of what inference
    sustains, so the flight record is not degraded by the SoC."""
    p = build_pipeline(cfg(fps=30, perception_fps=10))
    assert "framerate=30/1" in p            # camera
    assert "framerate=10/1" in p            # perception only
    assert "videorate" in p


def test_csi_source_uses_the_nvmm_converter():
    """Frames from nvarguscamerasrc live in GPU memory and only nvvidconv can
    touch them. videoconvert fails caps negotiation at pipeline start with an
    error that names neither element."""
    p = build_pipeline(cfg(source="csi"))
    assert "nvvidconv" in p
    assert "nvv4l2h265enc" in p
    recorder = next(b for b in p.split("t. ! ") if "splitmuxsink" in b)
    assert "videoconvert" not in recorder


def test_laptop_source_uses_the_software_encoder():
    p = build_pipeline(cfg(source="v4l2"))
    assert "x265enc" in p and "nvv4l2h265enc" not in p


def test_encoder_can_be_forced():
    assert "nvv4l2h265enc" in build_pipeline(cfg(source="v4l2", encoder="nvh265"))


def test_record_location_uses_posix_separators():
    """GStreamer treats a backslash as an escape, so a Windows path silently
    produces a location nothing can be written to."""
    p = build_pipeline(cfg(record_dir=Path("D:/runs/today")))
    location = next(t for t in p.split() if t.startswith("location="))
    assert "\\" not in location


def test_segment_duration_is_in_nanoseconds():
    p = build_pipeline(cfg(segment_s=60))
    assert "max-size-time=60000000000" in p


def test_srt_and_rtsp_use_their_own_sinks():
    assert "srtsink uri=" in build_pipeline(cfg(stream_url="srt://host:9000"))
    assert "rtspclientsink location=" in build_pipeline(cfg(stream_url="rtsp://host/x"))


def test_a_pipeline_that_keeps_nothing_is_refused():
    with pytest.raises(CaptureError, match="discard every frame"):
        build_pipeline(CaptureConfig(record_dir=None, stream_url=None))


def test_perception_cannot_outrun_the_camera():
    with pytest.raises(CaptureError, match="cannot be invented"):
        build_pipeline(cfg(fps=10, perception_fps=30))


def test_nothing_leaves_the_box_by_default():
    """The demo opens by unplugging the ethernet cable. That has to be true."""
    p = build_pipeline(cfg())
    assert "rtsp" not in p and "srt" not in p and "://" not in p


# ==========================================================================
# segment index and ring buffer
# ==========================================================================
def index_with(*spans) -> SegmentIndex:
    idx = SegmentIndex()
    for i, (lo, hi) in enumerate(spans):
        idx.segments.append(Segment(Path(f"cam0_{i:05d}.mkv"), lo, hi, 1000))
    return idx


def test_segments_covering_a_window():
    idx = index_with((0, 60), (60, 120), (120, 180))
    assert [s.path.name for s in idx.covering(50, 70)] == ["cam0_00000.mkv",
                                                           "cam0_00001.mkv"]
    assert [s.path.name for s in idx.covering(130, 140)] == ["cam0_00002.mkv"]


def test_a_window_outside_the_recording_returns_nothing():
    """Not an error. The ring buffer may already have trimmed it, and that is a
    fact about the run rather than a failure."""
    assert index_with((120, 180)).covering(0, 30) == []


def test_open_segment_closes_the_previous_one():
    idx = SegmentIndex()
    idx.open_segment("cam0_00000.mkv", 0.0)
    idx.open_segment("cam0_00001.mkv", 60.0)
    assert idx.segments[0].t_end == 60.0


def test_segment_index_parses_the_counter():
    assert SegmentIndex.index_of("cam0_00042.mkv") == 42
    assert SegmentIndex.index_of("not-a-segment.mkv") is None


def test_ring_buffer_trims_oldest_first():
    idx = index_with((0, 60), (60, 120), (120, 180), (180, 240))
    ring = RingBuffer(idx, max_bytes=2500, min_segments=1)
    removed = ring.trim(delete=False)
    assert [s.path.name for s in removed] == ["cam0_00000.mkv", "cam0_00001.mkv"]
    assert idx.total_bytes <= 2500


def test_ring_buffer_will_not_delete_evidence():
    """A segment a deviation clip still needs outranks disk capacity. If
    everything left is evidence, trimming stops rather than destroying it."""
    idx = index_with((0, 60), (60, 120), (120, 180))
    ring = RingBuffer(idx, max_bytes=500, min_segments=1)
    ring.protect(idx.segments[:2])
    ring.trim(delete=False)
    names = {s.path.name for s in idx.segments}
    assert {"cam0_00000.mkv", "cam0_00001.mkv"} <= names


def test_ring_buffer_keeps_a_minimum_number_of_segments():
    idx = index_with((0, 60), (60, 120))
    ring = RingBuffer(idx, max_bytes=1, min_segments=2)
    assert ring.trim(delete=False) == []


def test_ring_buffer_actually_deletes(tmp_path):
    idx = SegmentIndex()
    for i in range(3):
        p = tmp_path / f"cam0_{i:05d}.mkv"
        p.write_bytes(b"x" * 1000)
        idx.segments.append(Segment(p, i * 60.0, (i + 1) * 60.0, 1000))
    RingBuffer(idx, max_bytes=1500, min_segments=1).trim()
    assert not (tmp_path / "cam0_00000.mkv").exists()
    assert (tmp_path / "cam0_00002.mkv").exists()


# ==========================================================================
# speech policy
# ==========================================================================
def test_critical_speech_pre_empts_a_prompt():
    """"Stop, the latch is open" does not wait behind "retrieve the vial"."""
    q = SpeechQueue()
    q.announce_prompt("Retrieve vial A.", "S03")
    q.announce_alert("Stop. The latch is open.", "critical", "S10", "HAZARD")
    assert q.pump()[0].startswith("Stop.")


def test_the_same_message_is_not_repeated_immediately():
    """A persisting condition produces the same sentence every frame. Saying it
    every frame is how a speaker gets taped over."""
    q = SpeechQueue(repeat_s=10.0)
    assert q.announce_alert("Latch open.", "critical", "S10", "HAZARD", t=0.0)
    q.pump(t=0.0)
    assert not q.announce_alert("Latch open.", "critical", "S10", "HAZARD", t=2.0)
    assert q.suppressed == 1


def test_a_message_repeats_after_the_interval():
    q = SpeechQueue(repeat_s=10.0)
    q.announce_alert("Latch open.", "critical", "S10", "HAZARD", t=0.0)
    q.pump(t=0.0)
    assert q.announce_alert("Latch open.", "critical", "S10", "HAZARD", t=11.0)


def test_duplicates_already_queued_are_collapsed():
    q = SpeechQueue()
    assert q.announce_alert("Latch open.", "critical", "S10", "HAZARD")
    assert not q.announce_alert("Latch open.", "critical", "S10", "HAZARD")
    assert q.pending == 1


def test_prompts_are_dropped_before_alerts_when_backed_up():
    """A stale next-step prompt is worth less than an alert about something that
    already happened."""
    q = SpeechQueue(max_pending=3)
    for i in range(3):
        q.announce_prompt(f"Step {i}.", f"S0{i}")
    assert q.announce_alert("Step eight was not completed.", "caution", "S08", "SKIP")
    spoken = q.pump()
    assert any("not completed" in s for s in spoken)
    assert q.dropped == 1


def test_an_alert_is_never_dropped_for_capacity():
    q = SpeechQueue(max_pending=2)
    for i in range(4):
        q.announce_alert(f"Deviation {i}.", "caution", f"S{i:02d}", "SKIP")
    assert len(q.pump()) == 4


def test_speech_reaches_the_voice():
    voice = NullVoice()
    q = SpeechQueue(voice)
    q.announce_alert("Wrong vial.", "caution", "S03", "WRONG_OBJECT")
    q.pump()
    assert voice.spoken == ["Wrong vial."]


def test_empty_text_is_never_queued():
    q = SpeechQueue()
    assert not q.announce_prompt("   ", "S01")
    assert q.pending == 0


def test_severity_maps_to_priority():
    assert SpeechPriority.from_severity("critical") < SpeechPriority.from_severity("caution")
    assert SpeechPriority.from_severity("nonsense") is SpeechPriority.ADVISORY


# ==========================================================================
# voice commands
# ==========================================================================
def test_the_grammar_is_closed_and_enumerable():
    """You can write down everything the system will accept. That is what makes
    it certifiable, and it is why this is not open dictation."""
    g = CommandGrammar()
    assert len(g.phrases) >= 15
    assert all(isinstance(c, Command) for c in g.phrases.values())
    assert "override" in g.vocabulary


def test_filler_words_do_not_defeat_a_command():
    """Crew speech has filler. A grammar demanding exact strings fails on real
    people."""
    g = CommandGrammar()
    assert g.match("uh, mark done please").command is Command.MARK_DONE


def test_longer_phrases_win():
    g = CommandGrammar()
    assert g.match("mark step done").phrase == "mark step done"


def test_unrecognised_speech_is_not_a_command():
    assert CommandGrammar().match("pass me the blue one") is None


def test_low_confidence_speech_is_rejected_outright():
    assert CommandGrammar(min_confidence=0.5).match("override", confidence=0.2) is None


def test_a_consequential_command_heard_indistinctly_is_confirmed_not_obeyed():
    """Mishearing "mark done" from a fan and a cough would write a false
    completion into a tamper-evident record."""
    g = CommandGrammar()
    assert g.match("mark done", confidence=0.6).needs_confirmation
    assert not g.match("mark done", confidence=0.95).needs_confirmation
    assert not g.match("status", confidence=0.6).needs_confirmation


def test_consequential_commands_are_the_ones_that_change_the_record():
    assert Command.OVERRIDE in CONSEQUENTIAL
    assert Command.MARK_DONE in CONSEQUENTIAL
    assert Command.STATUS not in CONSEQUENTIAL


def test_override_tokens_come_from_the_procedure_not_this_file():
    """The procedure author decides which words count as an override."""
    g = CommandGrammar()
    tokens = ("override", "mark done", "continue anyway")
    assert g.matches_override_token("continue anyway then", tokens) == "continue anyway"
    assert g.matches_override_token("what is next", tokens) is None


def test_normalise_strips_punctuation_and_case():
    assert normalise("  Mark, DONE!  ") == "mark done"


# ==========================================================================
# event clips
# ==========================================================================
def test_a_clip_spanning_a_segment_boundary_pulls_both(tmp_path):
    idx = index_with((0, 60), (60, 120))
    ex = ClipExporter(idx, tmp_path)
    result = ex.export(ClipRequest("run1", "S08", "SKIP", t_event=58.0), dry_run=True)
    assert result.available
    assert len(result.segments) == 2


def test_an_early_deviation_gives_a_shorter_clip_not_an_error(tmp_path):
    """A skip four seconds into the run has four seconds of pre-roll."""
    ex = ClipExporter(index_with((0, 60)), tmp_path)
    result = ex.export(ClipRequest("run1", "S01", "SKIP", t_event=4.0), dry_run=True)
    assert result.available
    assert result.request.window[0] == 0.0


def test_a_trimmed_window_is_recorded_as_unavailable(tmp_path):
    """A flight record that quietly omits a missing clip is worse than one that
    records the gap."""
    ex = ClipExporter(index_with((600, 660)), tmp_path)
    result = ex.export(ClipRequest("run1", "S08", "SKIP", t_event=30.0), dry_run=True)
    assert not result.available
    assert "trimmed" in result.reason
    assert ex.unavailable == [result]


def test_partial_coverage_is_flagged(tmp_path):
    ex = ClipExporter(index_with((0, 15)), tmp_path)
    result = ex.export(ClipRequest("run1", "S02", "SKIP", t_event=12.0), dry_run=True)
    assert result.available
    assert "partial coverage" in result.reason


def test_planning_a_clip_protects_its_segments_from_the_ring_buffer(tmp_path):
    """Called the moment a deviation is detected, so the encoder cannot roll
    past the evidence before it is cut."""
    idx = index_with((0, 60), (60, 120))
    ring = RingBuffer(idx, max_bytes=1, min_segments=1)
    ex = ClipExporter(idx, tmp_path, ring=ring)
    ex.plan(ClipRequest("run1", "S08", "SKIP", t_event=58.0))
    ring.trim(delete=False)
    assert len(idx.segments) == 2


def test_clip_result_serialises_for_the_log(tmp_path):
    ex = ClipExporter(index_with((0, 60)), tmp_path)
    d = ex.export(ClipRequest("r", "S08", "SKIP", t_event=30.0), dry_run=True).as_dict()
    assert d["step_id"] == "S08" and d["available"] is True
    assert d["window"] == [20.0, 40.0]


# ==========================================================================
# downlink
# ==========================================================================
def package(tmp_path) -> DownlinkPackage:
    pkg = DownlinkPackage("run1", "CSP-1")
    (tmp_path / "log.jsonl").write_bytes(b"l" * 5_000)
    (tmp_path / "clip.mkv").write_bytes(b"c" * 500_000)
    (tmp_path / "seg.mkv").write_bytes(b"v" * 5_000_000)
    pkg.add(tmp_path / "seg.mkv", "video_segment", t_start=0.0, t_end=60.0)
    pkg.add(tmp_path / "clip.mkv", "deviation_clip", t_start=20.0, t_end=40.0)
    pkg.add(tmp_path / "log.jsonl", "run_log")
    return pkg


def test_the_log_goes_first_and_the_video_last(tmp_path):
    """When the pass runs out, what survives must be what a reviewing PI needs."""
    kinds = [a.kind for a in package(tmp_path).ordered]
    assert kinds == ["run_log", "deviation_clip", "video_segment"]


def test_a_short_pass_sends_the_log_and_the_evidence(tmp_path):
    fits = package(tmp_path).within_budget(1_000_000)
    assert [a.kind for a in fits] == ["run_log", "deviation_clip"]


def test_budgeting_does_not_repack_to_squeeze_video_in(tmp_path):
    """A pass that sends video because it happened to fit, instead of the next
    clip, is optimising the wrong quantity."""
    pkg = package(tmp_path)
    assert pkg.within_budget(4_000) == []      # not even the log fits


def test_every_artifact_is_checksummed(tmp_path):
    for art in package(tmp_path).artifacts:
        assert len(art.sha256) == 64
        assert art.size_bytes > 0


def test_verification_catches_a_file_changed_after_packaging(tmp_path):
    """Separate from the log's own hash chain: the chain proves the log was not
    edited, this proves the file was not truncated in transit."""
    pkg = package(tmp_path)
    assert pkg.verify() == []
    (tmp_path / "clip.mkv").write_bytes(b"tampered")
    assert any("clip.mkv" in p for p in pkg.verify())


def test_a_missing_file_cannot_be_packaged(tmp_path):
    with pytest.raises(FileNotFoundError):
        DownlinkPackage("r", "CSP-1").add(tmp_path / "nope.mkv", "run_log")


def test_manifest_names_its_transport_and_groups_by_priority(tmp_path):
    m = package(tmp_path).manifest()
    assert "CFDP" in m["transport"]
    assert m["bytes_by_priority"]["log"] == 5_000
    assert m["bytes_by_priority"]["video"] == 5_000_000


def test_manifest_is_written_deterministically(tmp_path):
    pkg = package(tmp_path)
    p = pkg.write_manifest(tmp_path / "manifest.json")
    assert b"\r\n" not in p.read_bytes()


def test_compression_ratio_shows_its_arithmetic(tmp_path):
    """"34,000x" invites "compared to what?", and the answer belongs on the same
    slide as the claim."""
    r = compression_ratio(log_bytes=40_000, duration_s=2700, cameras=1, bitrate_kbps=4000)
    assert r["video_bytes"] == 1_350_000_000          # 1.35 GB, as PLAN.md states
    assert r["ratio"] > 30_000
    assert r["bitrate_kbps"] == 4000 and r["cameras"] == 1


def test_identical_wording_is_spoken_once_even_from_different_steps():
    """CSP-1 declares the same latch-open invariant on S09 and S10, so one open
    latch produces two alerts with different dedupe keys and identical wording.
    Whatever the engine's bookkeeping says, the crew hears one message."""
    q = SpeechQueue(repeat_s=10.0)
    text = "Stop. The glovebox latch is open while the processing unit is running."
    assert q.announce_alert(text, "critical", "S09", "HAZARD", t=0.0)
    q.pump(t=0.0)
    assert not q.announce_alert(text, "critical", "S10", "HAZARD", t=0.2)
    assert q.suppressed == 1


def test_different_wording_from_different_steps_is_still_spoken():
    q = SpeechQueue(repeat_s=10.0)
    assert q.announce_alert("Latch open.", "critical", "S09", "HAZARD", t=0.0)
    q.pump(t=0.0)
    assert q.announce_alert("Vial not attached.", "caution", "S10", "SKIP", t=0.2)


# ==========================================================================
# procedure tokens in the voice grammar
# ==========================================================================
PROCEDURES_DIR = Path(__file__).resolve().parent.parent / "procedures"


def test_without_procedure_tokens_a_confirmation_cannot_be_heard():
    """The bug this section fixes. The grammar used to be built from generic
    commands only, so "label verified" - the phrase CSP-1 S04 waits for - could
    never be recognised by voice, and the step could only ever be clicked."""
    assert CommandGrammar().match("label verified") is None


def test_every_crew_token_in_every_procedure_is_recognisable():
    from parikshak.pdl import load_procedure
    for path in sorted(PROCEDURES_DIR.glob("[!_]*.yaml")):
        proc = load_procedure(path)
        grammar = CommandGrammar.for_procedure(proc)
        assert proc.crew_tokens, f"{path.name} declares no crew tokens"
        for token in proc.crew_tokens:
            heard = grammar.match(f"uh, {token}, over")
            assert heard is not None, f"{path.name}: '{token}' is not recognisable"
            assert heard.command is Command.CONFIRM and heard.token == token
        for token in proc.alert_policy.override_tokens:
            assert grammar.match(token).command is Command.OVERRIDE


def test_a_procedure_token_wins_a_tie_with_a_generic_phrase():
    """CSP-1 waits for "acknowledged", which is also a generic command. Inside
    that procedure, the token is what the crew meant."""
    heard = CommandGrammar(tokens=["acknowledged"]).match("acknowledged")
    assert heard.command is Command.CONFIRM and heard.token == "acknowledged"


def test_a_longer_generic_phrase_is_not_swallowed_by_a_shorter_token():
    heard = CommandGrammar(tokens=["done"]).match("mark step done")
    assert heard.command is Command.MARK_DONE


def test_an_indistinct_confirmation_is_asked_back():
    """A confirmation writes a completion into a tamper-evident record."""
    heard = CommandGrammar(tokens=["unit idle"]).match("unit idle", confidence=0.6)
    assert heard.needs_confirmation


def test_the_recogniser_vocabulary_includes_procedure_words():
    grammar = CommandGrammar(tokens=["suspension uniform"])
    assert {"suspension", "uniform"} <= set(grammar.vocabulary)
