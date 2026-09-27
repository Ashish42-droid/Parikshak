"""Demo scenarios: real engine replays, captured frame by frame for a browser.

Nothing here is canned. A scenario is a recorded BeliefFrame trace run through
the same ProcedureEngine that `parikshak run` and the desktop window use. For
every frame the browser gets what the engine concluded - the operator display
(`DisplayState`, exactly what the desktop GUI draws) and where perception put
each object - plus every prompt, alert and "cannot verify" notice with the
reason it gives. The run log is hash-chained and checked with the same
`verify_chain` as tools/verify_log.py, and the outcome is checked against the
expectations the trace carries, so a scenario that stops doing what its title
says is reported as such on screen rather than shown anyway.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RACK_LAYOUT = ROOT / "racks" / "msg_a_fiducials.json"
CSP1 = "procedures/csp1_colloid_sample_processing.yaml"
CRX2 = "procedures/crx2_colloid_resuspension.yaml"


@dataclass(frozen=True)
class Scenario:
    id: str
    procedure: str
    trace: str
    title: str
    #: One sentence: what a first-time viewer should watch for.
    watch_for: str
    #: "Correct runs", "Mistakes it catches", "Knowing what it cannot see",
    #: "A second experiment" - the picker groups by this.
    group: str


CORRECT = "Correct runs"
CATCHES = "Mistakes it catches"
RESTRAINT = "Knowing what it cannot see"
SECOND = "A second experiment, zero retraining"

SCENARIOS: tuple[Scenario, ...] = (
    Scenario("csp1-skip-latch", CSP1, "traces/golden/skip_S08_latch.jsonl",
             "Latch left open before the run starts",
             "The crew starts the processing unit without closing the glovebox latch. "
             "The alert names the step and the reason.", CATCHES),
    Scenario("csp1-wrong-vial", CSP1, "traces/golden/wrong_object_vial_b.jsonl",
             "The wrong vial is picked up",
             "Vial B (orange band) is taken instead of vial A (blue band). The procedure "
             "declares the two confusable, so the engine watches for exactly this.", CATCHES),
    Scenario("csp1-hazard", CSP1, "traces/golden/hazard_latch_open_while_running.jsonl",
             "Latch opened while the unit is running",
             "Nothing was skipped, so a checklist would stay silent. A safety invariant "
             "fires as critical.", CATCHES),
    Scenario("csp1-skip-seal", CSP1, "traces/golden/skip_S13_seal.jsonl",
             "Sample bag stowed without sealing",
             "A critical step missed near the end of the run.", CATCHES),
    Scenario("csp1-nominal", CSP1, "traces/golden/nominal.jsonl",
             "All 14 steps done correctly",
             "Each step is prompted and ticked in turn, with no alarms.", CORRECT),
    Scenario("csp1-nominal-noisy", CSP1, "traces/eval/nominal__harsh_0.jsonl",
             "A correct run under heavy sensor noise",
             "Dropped frames, jitter and occlusion. The crew does everything right, and "
             "the system must not cry wolf.", CORRECT),
    Scenario("csp1-reorder", CSP1, "traces/golden/legal_reorder_S03_S02.jsonl",
             "Two steps done in a permitted order",
             "The vial is fetched before the cartridge. The procedure allows either "
             "order, so there is no alarm.", CORRECT),
    Scenario("csp1-occlusion", CSP1, "traces/golden/occlusion_S06.jsonl",
             "The crew's body blocks the camera",
             "The insertion happens out of view. It says 'cannot verify', never "
             "'skipped'.", RESTRAINT),
    Scenario("csp1-frame-loss", CSP1, "traces/golden/frame_loss_S07.jsonl",
             "Rack markers lost for 8 seconds",
             "With no rack lock every position is unknown, so nothing is guessed from "
             "a stale view.", RESTRAINT),
    Scenario("csp1-fault", CSP1, "traces/golden/fault_branch_red_indicator.jsonl",
             "A hardware fault, not a crew error",
             "A red indicator routes the run to the fault branch instead of blaming "
             "the crew.", RESTRAINT),
    Scenario("crx2-nominal", CRX2, "traces/golden_crx2/nominal.jsonl",
             "A different experiment, done correctly",
             "CRX-2 is a new procedure written as a file. Same software, same models, "
             "no retraining.", SECOND),
    Scenario("crx2-skip-agitate", CRX2, "traces/golden_crx2/skip_S04_agitate.jsonl",
             "Vial set down without being shaken",
             "Only a count of shaking motions can tell - the vial ends up in the "
             "right place either way.", SECOND),
    Scenario("crx2-wrong-vial", CRX2, "traces/golden_crx2/wrong_object_vial_a.jsonl",
             "The wrong vial, in the second experiment",
             "Vial A taken instead of vial B: CSP-1's mistake inverted, with only the "
             "procedure file changed.", SECOND),
    Scenario("crx2-hazard", CRX2, "traces/golden_crx2/hazard_vial_near_unit.jsonl",
             "Vial shaken too close to the powered unit",
             "A geometric safety rule: within 12 cm of the running unit.", SECOND),
)

#: What the picker shows first: the clearest single demonstration.
DEFAULT = "csp1-skip-latch"
GROUP_ORDER = (CATCHES, CORRECT, RESTRAINT, SECOND)


def catalogue() -> list[dict[str, str]]:
    return [asdict(s) for s in SCENARIOS]


def find(scenario_id: str) -> Scenario:
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    raise KeyError(scenario_id)


def _r(v: Any, digits: int = 3) -> float | None:
    return None if v is None else round(float(v), digits)


def _main_path(proc) -> set[str]:
    """Steps on the procedure's normal sequence: `next` followed from the entry,
    an unordered group taken whole. Fault branches lie off it - the page hides
    them unless a run actually takes one, so a correct run's checklist does not
    end with a fault response still "waiting"."""
    groups = {g.id: g for g in proc.groups}
    main: set[str] = set()
    node = proc.entry
    while node is not None and node not in main:
        group = groups.get(node) or (proc.group_of(node) if node in proc.steps else None)
        if group is not None:
            main.update(group.members)
            node = group.successor if group.successor not in main else None
            continue
        main.add(node)
        step = proc.steps.get(node)
        node = step.next[0] if step is not None and step.next else None
    return main


def _procedure_view(proc) -> dict[str, Any]:
    zones = []
    for z in proc.zones.values():
        if z.kind == "box":
            zones.append({"name": z.name, "kind": "box",
                          "lo": [_r(c) for c in z.lo], "hi": [_r(c) for c in z.hi]})
        else:
            zones.append({"name": z.name, "kind": "sphere",
                          "centre": [_r(c) for c in z.centre], "radius": _r(z.radius)})
    main = _main_path(proc)
    steps = []
    for sid in proc.order:
        step = proc.step(sid)
        steps.append({"id": sid, "name": step.name, "critical": step.critical,
                      "prompt": step.prompt_tts, "objects": list(step.objects),
                      "branch": sid not in main})
    return {"id": proc.id, "title": proc.title, "steps": steps, "zones": zones,
            "entities": {name: e.spoken for name, e in proc.entities.items()}}


def _rack_view() -> list[dict[str, Any]]:
    layout = json.loads(RACK_LAYOUT.read_text(encoding="utf-8"))
    return [{"id": int(tid), "corners": [[_r(c[0]), _r(c[1])] for c in tag["corners"]]}
            for tid, tag in layout["tags"].items()]


def _frame_view(frame, state, code) -> dict[str, Any]:
    objects = {}
    for name, ob in frame.objects.items():
        flags = (1 if ob.visible else 0) | (2 if ob.occluded else 0) | (4 if ob.held_by else 0)
        pos = None if ob.pos_rack is None else [_r(ob.pos_rack[0]), _r(ob.pos_rack[1])]
        objects[name] = [pos, ob.state or "", flags]
    hands = [[_r(h.wrist_rack[0]), _r(h.wrist_rack[1]), h.contact_with or ""]
             for h in frame.hands.values() if h.present and h.wrist_rack is not None]
    return {
        "t": _r(frame.t_mono, 2),
        "lock": bool(frame.frame_lock),
        "active": state.active_step,
        "st": "".join(code(row.status) for row in state.steps),
        "obj": objects,
        "hands": hands,
        "motion": [frame.motion.cls, _r(frame.motion.conf, 2)],
        "occ": list(state.occluded_zones),
        "notice": state.unverified_notice,
    }


def _log_sample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The header, the records about deviations and alerts, and the last one -
    what a reviewer opens a flight record to find."""
    if not records:
        return []
    picked = [records[0]]
    picked += [r for r in records[1:-1]
                if any(word in str(r.get("kind", "")) for word in ("alert", "deviation"))][:3]
    if len(records) > 1:
        picked.append(records[-1])
    return picked


def replay(scenario_id: str) -> dict[str, Any]:
    """Run one scenario through the engine and capture everything the page shows."""
    from parikshak.belief.trace import read_trace
    from parikshak.engine.logger import verify_chain
    from parikshak.engine.runner import ProcedureEngine
    from parikshak.gui.state import DisplayState
    from parikshak.pdl import load_procedure
    from parikshak.replay import check_expectations

    sc = find(scenario_id)
    proc = load_procedure(ROOT / sc.procedure)
    header, frames = read_trace(ROOT / sc.trace)

    codes: dict[str, str] = {}

    def code(status: str) -> str:
        if status not in codes:
            codes[status] = chr(ord("a") + len(codes))
        return codes[status]

    views: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    said: set[tuple[float, str, str]] = set()
    last_prompt: str | None = None

    def alert_event(a) -> None:
        key = (round(a.t, 2), a.step_id, a.kind.value)
        if key in said:
            return
        said.add(key)
        events.append({"t": _r(a.t, 2), "type": "alert", "step": a.step_id,
                       "kind": a.kind.value, "severity": a.severity.label,
                       "text": a.text, "reason": a.reason, "channels": list(a.channels),
                       "confidence": _r(a.confidence, 2)})

    with tempfile.TemporaryDirectory() as tmp:
        engine = ProcedureEngine(proc, run_id=sc.id, log_path=Path(tmp) / "run.log.jsonl")
        for f in frames:
            result = engine.step(f)
            state = DisplayState.from_engine(engine, frame=f, source="replay")
            if result.active_step and result.active_step != last_prompt:
                events.append({"t": _r(result.t, 2), "type": "prompt",
                               "step": result.active_step, "text": result.prompt})
                last_prompt = result.active_step
            for tr in result.transitions:
                if tr.kind != "entered":
                    events.append({"t": _r(tr.t, 2), "type": tr.kind, "step": tr.step_id,
                                   "text": tr.detail})
            for a in result.alerts:
                alert_event(a)
            for n in result.notices:
                events.append({"t": _r(n.t, 2), "type": "notice", "step": n.step_id,
                               "text": n.text})
            views.append(_frame_view(f, state, code))
        summary = engine.finish()
        for a in summary.alerts:          # anything decided only at the end of the run
            alert_event(a)
        records = list(engine.log.records) if engine.log else []

    events.sort(key=lambda e: e["t"])
    ok, message = verify_chain(records)
    problems = check_expectations(summary, header.expect)
    return {
        "scenario": asdict(sc),
        "procedure": _procedure_view(proc),
        "rack": _rack_view(),
        "status_codes": {c: s for s, c in codes.items()},
        "frames": views,
        "events": events,
        "summary": {"duration_s": _r(summary.duration_s, 1), "frames": summary.frames,
                    "status": summary.status, "alerts": len(summary.alerts),
                    "deviations": [str(d) for d in summary.deviations]},
        "expectation": {"expected": header.expect, "met": not problems, "problems": problems},
        "log": {"records": len(records),
                "bytes": sum(len(json.dumps(r, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False)) + 1 for r in records),
                "verified": ok, "message": message, "sample": _log_sample(records)},
    }


@lru_cache(maxsize=len(SCENARIOS))
def replay_json(scenario_id: str) -> str:
    """`replay` serialised once. A scenario is deterministic, so the server
    replays each one at most once per process."""
    return json.dumps(replay(scenario_id), separators=(",", ":"))


def results() -> dict[str, Any]:
    """The measured numbers the results page shows, straight from runs/."""
    def load(name: str) -> Any:
        path = ROOT / "runs" / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return {"csp1": load("eval.json"), "crx2": load("eval_crx2.json"),
            "bench": load("bench.json"), "soak": load("soak_stream.json")}
