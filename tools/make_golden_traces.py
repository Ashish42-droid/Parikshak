#!/usr/bin/env python3
"""
Generate the golden traces for every procedure that has them.

    python tools/make_golden_traces.py                    # all procedures
    python tools/make_golden_traces.py --procedure csp1   # traces/golden/
    python tools/make_golden_traces.py --procedure crx2   # traces/golden_crx2/

These are what the engine is built and tuned against with no camera, no GPU and
no trained model. Each trace answers: "what would the perception layer report if
the crew did X?"

They are synthesised rather than typed by hand only because files of two
thousand JSON lines each are not a thing a person should type. The content is
authored - every deviation below is a deliberate statement about what the engine
must conclude, and the scripts read as procedures, not as fixtures.

The world model lives in parikshak/eval/synth.py. CSP-1's scenarios are below;
CRX-2's live beside its scripts in parikshak/eval/crx2.py. Both are written by
the same code into the same trace format, checked by the same replay harness,
and run through the same engine - which is itself part of what CRX-2 proves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parikshak.belief.trace import write_trace  # noqa: E402
from parikshak.eval.synth import (  # noqa: E402
    NOMINAL, RunBuilder, run, s10_wait,
)

PROC_PATH = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"
OUT_DIR = ROOT / "traces" / "golden"



# ------------------------------------------------------------------ traces

def build_all(proc: dict) -> list[dict]:
    specs: list[dict] = []

    def add(name, builder, expect, notes):
        specs.append({"name": name, "b": builder, "expect": expect, "notes": notes})

    add("nominal", run(proc),
        {"complete": [s for s, _ in NOMINAL]},
        "All 14 steps performed correctly and in order.")

    add("legal_reorder_S03_S02",
        run(proc, order=["S01", "S03", "S02"] + [s for s, _ in NOMINAL[3:]]),
        {"complete": [s for s, _ in NOMINAL], "no_alert": True},
        "Vial retrieved before cartridge. Both are in unordered group G_GATHER, "
        "so this is LEGAL and must produce no alert. The false-alarm test.")

    add("skip_S08_latch", run(proc, skip={"S08"}),
        {"skipped": ["S08"]},
        "Glovebox latch never closed before START. The primary demo beat.")

    add("skip_S13_seal", run(proc, skip={"S13"}),
        {"skipped": ["S13"]},
        "Cartridge stowed without sealing the sample bag. Critical step.")

    def s03_take_wrong_vial(b: RunBuilder):
        # Reaches past the blue-banded vial A and takes the orange-banded
        # vial B. Vial A is never touched, so S03 can never verify.
        b.motion("reach").hold(2)
        b.grasp("vial_b", "right")
        b.at("vial_b", tuple(v + 0.02 for v in b.pos["crew_hand"]))
        b.motion("idle").hold(6)

    add("wrong_object_vial_b",
        run(proc, order=["S01", "S02", "S03", "S04"],
            replace={"S03": s03_take_wrong_vial}),
        {"wrong_object": ["S03"]},
        "Vial B (orange band) picked instead of vial A (blue). Visually near "
        "identical; the error a fatigued crew member actually makes.")

    def occlude_s06(b: RunBuilder, sid: str):
        # mutate() runs AFTER step `sid`, so arming on S05 puts the occlusion in
        # place for the whole of S06 and clearing on S06 lifts it afterwards.
        # Occluding and clearing both on S05 - the first version - left S06
        # running in clear view and the trace proved nothing.
        if sid == "S05":
            b.occlude("cartridge_sc_a", "holder_h1", zone="glovebox_interior")
        if sid == "S06":
            b.clear_occlusion()

    add("occlusion_S06", run(proc, mutate=occlude_s06),
        {"unverified_at_least": ["S06"]},
        "Crew torso blocks the workspace during insertion. Must report "
        "UNVERIFIED, never SKIPPED. An occluded camera must not manufacture "
        "a deviation.")

    def lose_lock(b: RunBuilder, sid: str):
        # AprilTag lock is down for the whole of S07, not for a gap before it.
        if sid == "S06":
            b.lose_lock()
        if sid == "S07":
            b.regain_lock()

    add("frame_loss_S07", run(proc, mutate=lose_lock),
        {"unverified_at_least": ["S07"]},
        "AprilTag lock lost for 8 s. All rack-frame geometry becomes unknown; "
        "on_frame_loss.then is UNVERIFIED, never a guess from a stale extrinsic.")

    def hazard(b: RunBuilder, sid: str):
        # The latch must be open FOR THE DURATION of S10, which is when the unit
        # is running and the invariant is live. Opening and closing it in the gap
        # between S09 and S10 violates nothing.
        if sid == "S09":
            b.set_state("latch", "open")
        if sid == "S10":
            b.set_state("latch", "closed")

    add("hazard_latch_open_while_running", run(proc, mutate=hazard),
        {"hazard": ["S10"]},
        "Latch opened while the processing unit runs. No step was skipped, so "
        "sequence checking alone would never catch this. Invariant fires at "
        "CRITICAL immediately.")

    add("fault_branch_red_indicator",
        RunBuilder(proc), {"branch": "F01"},
        "Hardware fault: red indicator. Must route to F01, NOT blame the crew. "
        "Misattributing a hardware fault to the operator loses trust fastest.")
    b = specs[-1]["b"]
    b.hold(2)
    for sid, fn in NOMINAL[:9]:
        fn(b)
    s10_wait(b, fault=True)
    b.hold(4)

    return specs


#: Which expectation keys correspond to a deliberately injected deviation.
#: `branch` and `unverified_at_least` are NOT deviations - one is legal routing
#: around a hardware fault, the other is the system declining to guess - and
#: scoring either as a missed detection would punish correct behaviour.
INJECTED_FROM_EXPECT = {
    "skipped": "SKIP",
    "wrong_object": "WRONG_OBJECT",
    "hazard": "HAZARD",
}


def injected_from(expect: dict) -> list[dict]:
    """Ground truth for the eval harness, derived from the trace's own expect
    block so the two can never drift apart."""
    return [{"kind": kind, "step_id": sid}
            for key, kind in INJECTED_FROM_EXPECT.items()
            for sid in expect.get(key, [])]


# ------------------------------------------------------------------ targets

#: Every procedure with a golden corpus. Adding one is a line here and a
#: build_golden() beside its scripts.
TARGETS = {
    "csp1": ("procedures/csp1_colloid_sample_processing.yaml", "traces/golden"),
    "crx2": ("procedures/crx2_colloid_resuspension.yaml", "traces/golden_crx2"),
}


def _scenarios(target: str):
    if target == "csp1":
        return build_all
    from parikshak.eval import crx2
    return crx2.build_golden


def generate(target: str) -> int:
    proc_rel, out_rel = TARGETS[target]
    proc = yaml.safe_load((ROOT / proc_rel).read_text(encoding="utf-8"))
    out_dir = ROOT / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = proc["procedure"]

    manifest = []
    print(f"{meta['id']} rev {meta['revision']} -> {out_dir}")
    print()
    for spec in _scenarios(target)(proc):
        frames = spec["b"].frames
        path = out_dir / f"{spec['name']}.jsonl"
        # The header contract the eval harness reads: the verdict the engine
        # must reach, the ground truth it is scored against, and which
        # scenario and degradation profile produced it. Golden traces are the
        # clean profile of their scenario.
        write_trace(path, frames,
                    procedure_id=meta["id"],
                    run_id=spec["name"], source="synthetic",
                    notes=spec["notes"],
                    extra={"expect": spec["expect"],
                           "injected": injected_from(spec["expect"]),
                           "scenario": spec["name"],
                           "profile": "clean"})
        kb = path.stat().st_size / 1024
        print(f"  {spec['name']:<34} {len(frames):>5} frames  "
              f"{frames[-1].t_mono:>6.1f}s  {kb:>7.1f} KB")
        manifest.append({"trace": path.name, "expect": spec["expect"],
                         "notes": spec["notes"], "frames": len(frames)})

    # LF and a trailing newline on every platform, like the traces themselves.
    # The first version wrote this in Windows text mode, so the manifest was
    # the one CRLF file in an otherwise byte-stable corpus.
    text = json.dumps(manifest, indent=2) + chr(10)
    (out_dir / "manifest.json").write_text(text, encoding="utf-8", newline=chr(10))
    print()
    print(f"  {len(manifest)} traces + manifest.json")
    return len(manifest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate golden traces.")
    ap.add_argument("--procedure", choices=[*TARGETS, "all"], default="all")
    args = ap.parse_args(argv)
    targets = list(TARGETS) if args.procedure == "all" else [args.procedure]
    for target in targets:
        generate(target)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
