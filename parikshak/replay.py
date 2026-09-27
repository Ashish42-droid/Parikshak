"""Replay a recorded trace through the engine.

    python -m parikshak.replay traces/golden/skip_S08_latch.jsonl \\
        --procedure procedures/csp1_colloid_sample_processing.yaml

Every recorded run produces a trace; every trace is a regression test. When the
alert policy changes in week 10, re-run sixty traces in forty seconds and see
exactly which runs changed verdict. The alternative is tuning thresholds by
re-watching video, and there are not enough weeks for that.

It is also the insurance for the demo: if a camera fails on stage, a trace
replays through the identical engine and the GUI looks the same, because
nothing below this file knows where the frames came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parikshak.belief.trace import iter_trace, read_trace_header
from parikshak.gui.state import DisplayState
from parikshak.io.tts import CallbackVoice, SpeechQueue
from parikshak.engine.runner import ProcedureEngine, RunSummary
from parikshak.pdl.loader import Procedure, ProcedureError

DEFAULT_PROCEDURE = Path("procedures/csp1_colloid_sample_processing.yaml")


def _speech_queue(procedure: Procedure) -> SpeechQueue:
    """A speech queue that prints instead of speaking.

    Piper is swapped in by passing a PiperVoice; nothing else changes, which is
    the point of keeping the policy separate from the synthesiser.
    """
    spoken: list[str] = []

    def emit(text: str) -> None:
        spoken.append(text)
        print(f"    [voice] {text}")

    repeat_s = float(procedure.alert_policy.severity_tiers
                     .get("critical", {}).get("repeat_s", 10.0))
    queue = SpeechQueue(CallbackVoice(emit), repeat_s=repeat_s)
    queue.voice.spoken = spoken  # type: ignore[attr-defined]
    return queue


def replay(trace_path: Path, procedure: Procedure, *,
           log_path: Path | None = None, live: bool = False,
           speak: bool = False, view: bool = False) -> RunSummary:
    header = read_trace_header(trace_path)
    if header.procedure_id != procedure.id:
        print(f"  ! trace was recorded against {header.procedure_id}, "
              f"replaying against {procedure.id}", file=sys.stderr)

    engine = ProcedureEngine(procedure, run_id=header.run_id or trace_path.stem,
                             log_path=log_path)
    # The voice path, exercised end to end. The engine decides WHAT to say; the
    # speech queue decides what happens when two things need saying at once,
    # and that policy is where a demo either sounds calm or sounds like a
    # smoke alarm. Running it in replay means it is rehearsed before the rack.
    speech = _speech_queue(procedure) if speak else None
    last_prompt: str | None = None

    for frame in iter_trace(trace_path):
        result = engine.step(frame)

        if speech is not None:
            if result.active_step is not None and result.active_step != last_prompt:
                speech.announce_prompt(result.prompt, result.active_step, t=result.t)
                last_prompt = result.active_step
            for alert in result.alerts:
                speech.announce_alert(alert.text, alert.severity.label,
                                      alert.step_id, alert.kind.value, t=result.t)
            for notice in result.notices:
                speech.announce_prompt(notice.text, f"unverified:{notice.step_id}",
                                       t=result.t)
            speech.pump(t=result.t)

        if not live:
            continue
        for tr in result.transitions:
            if tr.kind != "entered":
                print(f"  {tr.t:8.1f}s  {tr.kind.upper():<11} {tr.step_id:<10} {tr.detail}")
        for notice in result.notices:
            print(f"  {notice.t:8.1f}s  UNVERIFIED  {notice.step_id:<10} {notice.text}")
        for alert in result.alerts:
            print(f"  {alert}")
    if view:
        # The operator display, built from the same engine state a live run
        # would produce. Rendering it here is the stage failover rehearsed: if a
        # camera fails, this is what the screen shows, from a file.
        from parikshak.gui.app import render_text
        print(render_text(DisplayState.from_engine(engine, frame=frame,
                                                   source="replay")))

    summary = engine.finish()
    if speech is not None:
        speech.pump(t=summary.duration_s)
        print(f"  spoken: {len(speech.voice.spoken)} utterance(s), "
              f"{speech.suppressed} suppressed as repeats, "
              f"{speech.dropped} stale prompt(s) dropped")
    return summary


def check_expectations(summary: RunSummary, expect: dict) -> list[str]:
    """Compare against the trace's own `expect` block.

    Expectations travel in the trace rather than in test code, so adding a
    golden case is a data change. See traces/golden/manifest.json.
    """
    problems: list[str] = []
    kinds = {d.kind.value for d in summary.deviations}
    by_kind: dict[str, list[str]] = {}
    for d in summary.deviations:
        by_kind.setdefault(d.kind.value, []).append(d.step_id)

    for sid in expect.get("complete", []):
        if sid not in summary.complete:
            problems.append(f"{sid} expected COMPLETE, got {summary.status.get(sid)}")
    if expect.get("no_alert") and summary.deviations:
        problems.append(f"expected no deviations, got {sorted(kinds)}")
    for key, kind in (("skipped", "SKIP"), ("wrong_object", "WRONG_OBJECT"),
                      ("hazard", "HAZARD")):
        for sid in expect.get(key, []):
            if sid not in by_kind.get(kind, []):
                problems.append(f"{sid} expected {kind}, got {by_kind.get(kind) or 'none'}")
    for sid in expect.get("unverified_at_least", []):
        if sid not in summary.unverified:
            problems.append(f"{sid} expected UNVERIFIED, got {list(summary.unverified) or 'none'}")
    return problems


def report(summary: RunSummary, expect: dict | None = None, verbose: bool = False) -> bool:
    print(f"\n=== {summary.run_id} ===")
    print(f"  {summary.frames} frames, {summary.duration_s:.1f}s")
    print(f"  COMPLETE {len(summary.complete)}   SKIPPED {list(summary.skipped) or '-'}   "
          f"UNVERIFIED {list(summary.unverified) or '-'}")
    print(f"  {len(summary.deviations)} deviation(s), {len(summary.alerts)} alert(s) "
          f"= {summary.alerts_per_45min:.1f} per 45 min")

    if verbose or summary.deviations:
        for dev in summary.deviations:
            print(f"    {dev}")
    for alert in summary.alerts:
        print(f"    {alert}")
    for notice in summary.notices:
        print(f"    [{notice.t:7.1f}s] UNVERIFIED    {notice.step_id:<10} {notice.text}")
    if summary.log_path:
        print(f"  log: {summary.log_path} ({summary.log_path.stat().st_size / 1024:.1f} KB)")

    if expect:
        problems = check_expectations(summary, expect)
        if problems:
            print("  EXPECTATION MISMATCH:")
            for p in problems:
                print(f"    !! {p}")
            return False
        print("  expectations: MET")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay a BeliefFrame trace through the procedure engine.")
    ap.add_argument("traces", nargs="+", type=Path, help="trace .jsonl file(s)")
    ap.add_argument("--procedure", type=Path, default=DEFAULT_PROCEDURE)
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="write a hash-chained run log per trace into this directory")
    ap.add_argument("--live", action="store_true",
                    help="print transitions as they happen, as the GUI would show them")
    ap.add_argument("--speak", action="store_true",
                    help="run the speech queue and print what the crew would hear")
    ap.add_argument("--view", action="store_true",
                    help="render the operator display as text at the end of the run")
    ap.add_argument("--check", action="store_true",
                    help="fail if a trace does not meet its own expect block")
    ap.add_argument("--window", action="store_true",
                    help="open the first trace in the operator window, with replay controls")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        procedure = Procedure.load(args.procedure)
    except ProcedureError as exc:
        print(f"cannot load procedure: {exc}", file=sys.stderr)
        return 2

    if args.window:
        # The same window a live run uses, driven by a ReplaySession: the stage
        # failover, and PLAN.md's Phase 6 gate, as one command.
        from parikshak.gui.app import GuiUnavailable, run_session_window
        from parikshak.live import ReplaySession
        trace = args.traces[0]
        if not trace.exists():
            print(f"no such trace: {trace}", file=sys.stderr)
            return 2
        speech = _speech_queue(procedure) if args.speak else None
        try:
            return run_session_window(ReplaySession(procedure, trace, speech=speech),
                                      speech=speech)
        except GuiUnavailable as exc:
            print(f"cannot open the window: {exc}", file=sys.stderr)
            return 2

    ok = True
    for trace_path in args.traces:
        if not trace_path.exists():
            print(f"no such trace: {trace_path}", file=sys.stderr)
            ok = False
            continue
        log_path = (args.log_dir / f"{trace_path.stem}.jsonl") if args.log_dir else None
        header = read_trace_header(trace_path)
        summary = replay(trace_path, procedure, log_path=log_path, live=args.live,
                         speak=args.speak, view=args.view)
        ok &= report(summary, header.expect if args.check else None, args.verbose)

    if args.check:
        print("\n" + ("ALL TRACES MET THEIR EXPECTATIONS" if ok else "EXPECTATION MISMATCHES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
