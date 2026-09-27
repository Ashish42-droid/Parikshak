#!/usr/bin/env python3
"""Evaluate the engine over a trace corpus and print the section-14 numbers.

    python tools/eval_report.py                          # traces/eval
    python tools/eval_report.py --traces traces/golden
    python tools/eval_report.py --sweep                  # ROC over the alert gate
    python tools/eval_report.py --json runs/eval.json

The ROC sweep is what turns "our false-alarm rate is low" into a curve with a
chosen operating point on it. PLAN.md risk 2: a tuned-down system with a
published tradeoff beats an untuned one with a claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parikshak.eval.harness import (  # noqa: E402
    EvalReport, choose_operating_point, evaluate, sweep,
)
from parikshak.pdl.loader import Procedure  # noqa: E402

PROC_PATH = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"


def bar(value: float, width: int = 28, lo: float = 0.0, hi: float = 1.0) -> str:
    if hi <= lo:
        return ""
    n = int(round(width * max(0.0, min(1.0, (value - lo) / (hi - lo)))))
    return "#" * n + "." * (width - n)


def print_report(report: EvalReport) -> None:
    d = report.as_dict()
    print("\n" + "=" * 72)
    print(f"  {d['runs']} runs, {d['monitored_s'] / 60:.1f} minutes monitored")
    print("=" * 72)

    print(f"\n  step accuracy            {d['step_accuracy']:>7.1%}   "
          f"{bar(d['step_accuracy'])}   target >= 95%")
    print(f"  deviation recall         {d['deviation_recall']:>7.1%}   "
          f"{bar(d['deviation_recall'])}   target >= 90%")
    fa = d["false_alarms_per_45min"]
    print(f"  false alarms / 45 min    {fa:>7.2f}   "
          f"{bar(min(fa, 4.0), lo=0, hi=4.0)}   target <= 1.00")
    print(f"    (over {d['clean_runs']} runs with nothing injected)")
    print(f"  cascade alerts / dev run {d['cascade_per_deviation_run']:>7.2f}   "
          f"correct-but-downstream, not charged to the budget")
    print(f"  UNVERIFIED rate          {d['unverified_rate']:>7.1%}")
    print(f"  alert latency p50 / p95  {d['latency_p50_s']:>7.2f} / "
          f"{d['latency_p95_s']:.2f} s   target <= 2.0 s")

    print("\n  recall by deviation type   (an aggregate hides a type at 40%)")
    for kind, value in sorted(d["recall_by_type"].items()):
        print(f"    {kind:<14} {value:>6.1%}   {bar(value)}")

    print("\n  the runs that must be silent")
    for label, value in (("nominal", d["nominal_false_alarms"]),
                         ("legal reorder", d["legal_reorder_false_alarms"]),
                         ("occlusion / lock loss", d["occlusion_false_alarms"])):
        mark = "ok" if value == 0 else "!!"
        print(f"    {mark}  {label:<24} {value} false alarm(s)")

    print("\n  by degradation profile")
    print(f"    {'profile':<10} {'runs':>5} {'step acc':>9} {'recall':>8} {'FA/45m':>8} {'UNVER':>7}")
    for name in ("clean", "mild", "moderate", "harsh"):
        sub = report.by_profile().get(name)
        if sub is None:
            continue
        s = sub.as_dict()
        print(f"    {name:<10} {s['runs']:>5} {s['step_accuracy']:>8.1%} "
              f"{s['deviation_recall']:>8.1%} {s['false_alarms_per_45min']:>8.2f} "
              f"{s['unverified_rate']:>7.1%}")

    print("\n  by occlusion fraction   (proves the UNVERIFIED design works)")
    print(f"    {'stratum':<10} {'runs':>5} {'step acc':>9} {'recall':>8} {'FA/45m':>8} {'UNVER':>7}")
    for name, sub in report.by_occlusion_stratum().items():
        s = sub.as_dict()
        print(f"    {name:<10} {s['runs']:>5} {s['step_accuracy']:>8.1%} "
              f"{s['deviation_recall']:>8.1%} {s['false_alarms_per_45min']:>8.2f} "
              f"{s['unverified_rate']:>7.1%}")

    print("\n  what the engine called each injected deviation")
    for kind, counts in sorted(report.kind_confusion.items()):
        called = ", ".join(f"{k}x{n}" for k, n in counts.most_common())
        print(f"    {kind:<14} -> {called}")


def print_sweep(points, chosen) -> None:
    print("\n" + "=" * 72)
    print("  ROC over the alert gate")
    print("=" * 72)
    print(f"\n    {'complete':>9} {'deviation':>10} {'step acc':>9} {'recall':>8} "
          f"{'FA/45m':>8} {'reorder':>8} {'UNVER':>7}  {'targets':>7}")
    last_ct = None
    for p in points:
        if last_ct is not None and p.complete_threshold != last_ct:
            print()
        last_ct = p.complete_threshold
        mark = "MEETS" if p.meets_targets() else ""
        star = " <-- chosen" if chosen is not None and p == chosen else ""
        print(f"    {p.complete_threshold:>9.2f} {p.deviation_threshold:>10.2f} "
              f"{p.step_accuracy:>8.1%} {p.recall:>8.1%} "
              f"{p.false_alarms_per_45min:>8.2f} {p.legal_reorder_false_alarms:>8} "
              f"{p.unverified_rate:>7.1%}  {mark:>7}{star}")
    if chosen is None:
        print("\n  No operating point meets every target on this corpus.")
        print("  That is a result, not a failure - publish the tradeoff (PLAN.md risk 2).")
    else:
        print(f"\n  Chosen: complete_threshold={chosen.complete_threshold}, "
              f"deviation_threshold={chosen.deviation_threshold}, "
              f"persistence_s={chosen.persistence_s}")
        print(f"          -> step accuracy {chosen.step_accuracy:.1%}, "
              f"recall {chosen.recall:.1%}, "
              f"{chosen.false_alarms_per_45min:.2f} false alarms / 45 min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=ROOT / "traces" / "eval")
    ap.add_argument("--procedure", type=Path, default=PROC_PATH)
    ap.add_argument("--sweep", action="store_true", help="run the ROC sweep")
    ap.add_argument("--json", type=Path, default=None, help="write metrics as JSON")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N traces")
    args = ap.parse_args()

    traces = sorted(args.traces.glob("*.jsonl"))
    if args.limit:
        traces = traces[: args.limit]
    if not traces:
        print(f"no traces in {args.traces} - run tools/make_eval_corpus.py first",
              file=sys.stderr)
        return 2

    procedure = Procedure.load(args.procedure)
    report = evaluate(procedure, traces)
    print_report(report)

    payload = {"metrics": report.as_dict()}
    if args.sweep:
        points = sweep(procedure, traces, progress=True)
        chosen = choose_operating_point(points)
        print_sweep(points, chosen)
        payload["roc"] = [dataclasses.asdict(p) for p in points]
        payload["chosen"] = dataclasses.asdict(chosen) if chosen else None

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        print(f"\n  metrics written to {args.json}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
