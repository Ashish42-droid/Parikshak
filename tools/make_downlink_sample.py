#!/usr/bin/env python3
"""Build a sample downlink package: what goes to the ground after a run.

    python tools/make_downlink_sample.py --out runs/downlink_sample

Replays a golden run (a skipped latch, so there is something to report) through
the unchanged engine with its hash-chained log on, then packages the flight
record in CFDP transmission order - the log first, evidence next, video last -
with a checksummed manifest, the log-versus-video arithmetic, and a README a
reviewer can follow with nothing but Python.

Honest about what is missing: on a machine without ffmpeg and a video recorder
there are no deviation clips or video segments, so the package does not
pretend to have them. The manifest and the README say so.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parikshak.belief.trace import iter_trace, read_trace_header  # noqa: E402
from parikshak.engine.logger import verify_log_file  # noqa: E402
from parikshak.engine.runner import ProcedureEngine  # noqa: E402
from parikshak.io.downlink import DownlinkPackage, compression_ratio  # noqa: E402
from parikshak.pdl import load_procedure  # noqa: E402

DEFAULT_TRACE = ROOT / "traces" / "golden" / "skip_S08_latch.jsonl"
DEFAULT_PROCEDURE = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
#: One short pass over a slow link, to show what the priority order buys.
PASS_BUDGET_BYTES = 64 * 1024


def build(out_dir: Path, *, trace: Path = DEFAULT_TRACE,
          procedure_path: Path = DEFAULT_PROCEDURE,
          eval_report: Path | None = ROOT / "runs" / "eval.json") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = read_trace_header(trace)
    procedure = load_procedure(procedure_path)
    run_id = header.run_id or trace.stem

    log_path = out_dir / f"{run_id}.log.jsonl"
    engine = ProcedureEngine(procedure, run_id=run_id, log_path=log_path)
    for frame in iter_trace(trace):
        engine.step(frame)
    summary = engine.finish()
    text_path = log_path.with_suffix(".txt")

    procedure_copy = out_dir / procedure_path.name
    shutil.copyfile(procedure_path, procedure_copy)

    package = DownlinkPackage(run_id=run_id, procedure_id=procedure.id)
    package.add(log_path, "run_log", t_start=0.0, t_end=summary.duration_s)
    if text_path.exists():
        package.add(text_path, "run_log_text", t_start=0.0, t_end=summary.duration_s)
    package.add(procedure_copy, "procedure")
    if eval_report is not None and eval_report.exists():
        report_copy = out_dir / "eval_report.json"
        shutil.copyfile(eval_report, report_copy)
        package.add(report_copy, "eval_report")

    manifest = package.manifest()
    manifest["verified_chain"] = verify_log_file(log_path)[1]
    manifest["not_included"] = {
        "deviation_clips": "not produced: ffmpeg and a video recorder are not installed here",
        "video_segments": "not produced: no GStreamer recording on this machine",
    }
    ratio = compression_ratio(log_path.stat().st_size, summary.duration_s)
    manifest["log_vs_video"] = ratio
    first_pass = package.within_budget(PASS_BUDGET_BYTES)
    manifest["first_pass"] = {"budget_bytes": PASS_BUDGET_BYTES,
                              "sends": [a.path.name for a in first_pass]}
    manifest["run_summary"] = summary.as_dict()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8",
                                           newline="\n")

    readme = f"""PARIKSHAK sample downlink package - {run_id} ({procedure.id})

What this is
  The flight record of one run, packaged in the order it would go to the ground
  over CCSDS CFDP (727.0-B): the hash-chained log first, evidence second, video
  last. The run is a replay of a golden synthetic trace - a skipped latch - not a
  camera recording.

Contents, in transmission order
""" + "".join(f"  {a.path.name:<45} {a.kind:<14} {a.size_bytes:>9} bytes  sha256 {a.sha256[:16]}\n"
              for a in package.ordered) + f"""
Not included, and why
  deviation clips  ffmpeg and a video recorder are not installed on this machine
  video segments   no GStreamer recording on this machine

Log versus video
  This log is {ratio['log_bytes']} bytes for {ratio['duration_s']} s of operation.
  The same run as 1 camera of H.265 at {ratio['bitrate_kbps']} kbps would be about
  {ratio['video_bytes']} bytes: {ratio['ratio']}x larger.

A {PASS_BUDGET_BYTES // 1024} KB pass sends
  {', '.join(a.path.name for a in first_pass) or 'nothing'}

How to check it on the ground
  1. Every file: recompute SHA-256 and compare with manifest.json.
  2. The log itself: python tools/verify_log.py {log_path.name}
     Result when packaged: {manifest['verified_chain']}
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8", newline="\n")
    manifest["_problems"] = package.verify()
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a sample downlink package.")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "downlink_sample")
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--procedure", type=Path, default=DEFAULT_PROCEDURE)
    args = ap.parse_args(argv)
    manifest = build(args.out, trace=args.trace, procedure_path=args.procedure)
    if manifest["_problems"]:
        print("package does not verify:", manifest["_problems"])
        return 1
    ratio = manifest["log_vs_video"]
    print(f"{len(manifest['artifacts'])} artefact(s), {manifest['total_bytes']} bytes -> {args.out}")
    print(f"  log {ratio['log_bytes']} bytes vs ~{ratio['video_bytes']} bytes of video "
          f"({ratio['ratio']}x); {manifest['verified_chain']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
