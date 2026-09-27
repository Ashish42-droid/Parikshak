#!/usr/bin/env python3
"""Generate the evaluation corpus: scenarios x degradation profiles x seeds.

    python tools/make_eval_corpus.py                      # CSP-1, 3 seeds
    python tools/make_eval_corpus.py --seeds 5
    python tools/make_eval_corpus.py --procedure crx2     # -> traces/eval_crx2

Writes traces/eval*/*.jsonl plus a manifest. Each trace carries the ground truth
that was injected into it, which is what makes false-alarm rate a checkable
number rather than an assertion.

This corpus is NOT the training set and NOT a substitute for the sixty recorded
runs. It exists so the eval harness, the ROC sweep and the regression suite are
working in week 2 instead of week 8 - so that when real data lands it flows into
a harness that already produces numbers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parikshak.belief.trace import write_trace  # noqa: E402
from parikshak.eval.corpus import build_corpus  # noqa: E402
from parikshak.eval.perturb import PROFILES  # noqa: E402

#: name -> (procedure file, default output directory)
PROCEDURES = {
    "csp1": (ROOT / "procedures" / "csp1_colloid_sample_processing.yaml",
             ROOT / "traces" / "eval"),
    "crx2": (ROOT / "procedures" / "crx2_colloid_resuspension.yaml",
             ROOT / "traces" / "eval_crx2"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procedure", choices=sorted(PROCEDURES), default="csp1")
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds per degraded profile (clean is generated once)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--clean-first", action="store_true",
                    help="delete the output directory before writing")
    args = ap.parse_args()

    proc_path, default_out = PROCEDURES[args.procedure]
    out_dir = args.out or default_out
    doc = yaml.safe_load(proc_path.read_text(encoding="utf-8"))
    if args.clean_first and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_list = None
    if args.procedure == "crx2":
        from parikshak.eval.crx2 import corpus_scenarios
        scenario_list = corpus_scenarios()
    entries = build_corpus(doc, seeds=args.seeds, scenario_list=scenario_list)
    manifest = []
    total_frames = total_kb = 0.0

    print(f"{len(entries)} traces -> {out_dir}\n")
    for entry in entries:
        path = out_dir / f"{entry.run_id}.jsonl"
        write_trace(path, entry.frames,
                    procedure_id=doc["procedure"]["id"],
                    run_id=entry.run_id, source="synthetic",
                    notes=entry.notes, extra=entry.header_extra())
        kb = path.stat().st_size / 1024
        total_frames += len(entry.frames)
        total_kb += kb
        manifest.append({
            "trace": path.name, "scenario": entry.scenario,
            "profile": entry.profile, "seed": entry.seed,
            "frames": len(entry.frames),
            "injected": [i.as_dict() for i in entry.injected],
            "expect": entry.expect,
        })

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    by_profile: dict[str, int] = {}
    for e in entries:
        by_profile[e.profile] = by_profile.get(e.profile, 0) + 1
    for name in (p.name for p in PROFILES):
        print(f"  {name:<10} {by_profile.get(name, 0):>4} traces")
    injected = sum(len(e.injected) for e in entries)
    clean_runs = sum(1 for e in entries if not e.injected)
    print(f"\n  {int(total_frames)} frames, {total_kb / 1024:.1f} MB")
    print(f"  {injected} injected deviations across {len(entries) - clean_runs} runs; "
          f"{clean_runs} runs with nothing injected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
