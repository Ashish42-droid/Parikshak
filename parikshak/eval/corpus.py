"""The evaluation corpus: scenarios x degradation profiles x seeds.

Scenarios follow the campaign-2 taxonomy in PLAN.md section 12, which is itself
borrowed from CaptainCook4D / IndustReal / EgoPER rather than invented. The
counts mirror the planned recording campaign, so the synthetic corpus and the
real one are directly comparable when the real one arrives.

Each trace carries GROUND TRUTH in its header - not just what the engine should
conclude, but what was actually injected:

    expect.injected = [{"kind": "SKIP", "step_id": "S08"}, ...]

That list is what makes a false alarm definable. An alert matching an injected
deviation is a true positive; an alert that matches nothing injected is a false
positive; an injected deviation with no alert is a miss. Without ground truth in
the file, "false alarms per 45 minutes" is a number nobody can check.

The legal-reorder scenarios are the ones that matter most and the ones nobody
else records: they inject NOTHING and must produce NOTHING. A system that
alerts on a correct-but-reordered run is worse than no system.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

from parikshak.belief.frame import BeliefFrame
from parikshak.eval import synth
from parikshak.eval.perturb import PROFILES, Perturbation, apply
from parikshak.pdl.validator import min_satisfaction_time

#: Recognition time the corpus allows after the final step's own minimum
#: satisfaction time: evidence latency (p95 1.8 s on the degraded corpus) plus
#: the 2 s settle inside `released`.
RECOGNITION_ALLOWANCE_S = 4.0


def terminal_verification(proc: dict, nominal=None) -> Any:
    """The verification tree of the last step a scripted run performs.
    `nominal` is the run's step script; CSP-1's when not given."""
    last = (synth.NOMINAL if nominal is None else nominal)[-1][0]

    def walk(nodes):
        for node in nodes or ():
            if not isinstance(node, dict):
                continue
            if node.get("id") == last:
                return node.get("verification")
            for key in ("steps", "unordered"):
                found = walk(node.get(key)) if isinstance(node.get(key), list) else None
                if found is not None:
                    return found
        return None

    flow = proc["flow"]
    return walk(flow["steps"] if isinstance(flow, dict) else flow)


def session_tail_s(proc: dict, nominal=None) -> float:
    """How long a corpus session keeps recording after the last scripted action.

    A real session does not stop the instant the last object is stowed - the
    crew shuts the locker and closes out. A trace that does stop scores
    truncation, not perception: CSP-1 S14 cannot verify in under 6 s (hold_for
    3 s of stable 3 s), the corpus used to end 2 s after it, and on the degraded
    corpus that turned ~12 correctly performed final steps into UNVERIFIED and
    left four skipped S13s never passed. Derived from the procedure, not tuned.
    """
    return max(2.0, min_satisfaction_time(terminal_verification(proc, nominal))
               + RECOGNITION_ALLOWANCE_S)


def _run(proc: dict, **kw) -> synth.RunBuilder:
    return synth.run(proc, tail_s=session_tail_s(proc), **kw)


@dataclass(frozen=True, slots=True)
class Injected:
    """One deviation deliberately introduced. This is the ground truth."""

    kind: str
    step_id: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "step_id": self.step_id}


@dataclass
class Scenario:
    name: str
    build: Callable[[dict], synth.RunBuilder]
    injected: tuple[Injected, ...] = ()
    expect: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_nominal(self) -> bool:
        return not self.injected


# --------------------------------------------------------------------------
# scenario constructors
# --------------------------------------------------------------------------
def _nominal(proc: dict) -> synth.RunBuilder:
    return _run(proc)


def _skip(step_id: str):
    def build(proc: dict) -> synth.RunBuilder:
        return _run(proc, skip={step_id})
    return build


def _reorder(a: str, b: str):
    """Swap two steps in the performed order."""
    def build(proc: dict) -> synth.RunBuilder:
        order = [sid for sid, _ in synth.NOMINAL]
        ia, ib = order.index(a), order.index(b)
        order[ia], order[ib] = order[ib], order[ia]
        return _run(proc, order=order)
    return build


def _wrong_object(proc: dict) -> synth.RunBuilder:
    """Vial B picked instead of vial A. Physically adjacent, visually near
    identical apart from the colour band - the error a fatigued crew member
    actually makes, and the one a checklist tablet structurally cannot catch."""
    def s03_wrong(b: synth.RunBuilder):
        b.motion("reach").hold(2)
        b.grasp("vial_b", "right").move("vial_b", "workspace").motion("idle").hold(6)
    return _run(proc, replace={"S03": s03_wrong})


def _rush(step_id: str, fn):
    """Perform a step far below its min_s."""
    def build(proc: dict) -> synth.RunBuilder:
        def quick(b: synth.RunBuilder):
            fn(b, fast=True)
        return _run(proc, replace={step_id: quick})
    return build


def _stall(step_id: str, seconds: float):
    """Dwell past max_s before completing."""
    def build(proc: dict) -> synth.RunBuilder:
        original = dict(synth.NOMINAL)[step_id]

        def slow(b: synth.RunBuilder):
            b.motion("idle").hold(seconds)
            original(b)
        return _run(proc, replace={step_id: slow})
    return build


def _hazard(proc: dict) -> synth.RunBuilder:
    def mutate(b: synth.RunBuilder, sid: str):
        if sid == "S09":
            b.set_state("latch", "open")
        if sid == "S10":
            b.set_state("latch", "closed")
    return _run(proc, mutate=mutate)


def _occlude(step_before: str, step_during: str):
    def build(proc: dict) -> synth.RunBuilder:
        def mutate(b: synth.RunBuilder, sid: str):
            if sid == step_before:
                b.occlude("cartridge_sc_a", "holder_h1", zone="glovebox_interior")
            if sid == step_during:
                b.clear_occlusion()
        return _run(proc, mutate=mutate)
    return build


def _lock_loss(step_before: str, step_during: str):
    def build(proc: dict) -> synth.RunBuilder:
        def mutate(b: synth.RunBuilder, sid: str):
            if sid == step_before:
                b.lose_lock()
            if sid == step_during:
                b.regain_lock()
        return _run(proc, mutate=mutate)
    return build


def _s12_fast(b: synth.RunBuilder, fast: bool = True):
    """Remove the cartridge far below S12's min_s.

    S12 is used rather than S06 because S06's verification contains
    hold_for(1.5, stable(1.5)) and therefore cannot be satisfied before 3.0 s -
    its min_s of 3 s is unreachable from below, so no rush of S06 is detectable
    at all. The validator now warns about exactly that.
    """
    b.motion("withdraw").hold(0.2)
    b.grasp("cartridge_sc_a", "left").move("cartridge_sc_a", "workspace")
    b.set_state("holder_h1", "empty").motion("idle").hold(0.2)


# --------------------------------------------------------------------------
# the corpus definition - counts mirror PLAN.md section 12 campaign 2
# --------------------------------------------------------------------------
def scenarios() -> list[Scenario]:
    """Every scenario, once. Multiplied by profiles and seeds in build_corpus."""
    out: list[Scenario] = [
        Scenario("nominal", _nominal, (), {"complete": [s for s, _ in synth.NOMINAL]},
                 "Correct run. Must be silent."),
    ]

    # SKIP - the omissions that matter, including both critical steps.
    for sid in ("S05", "S08", "S13"):
        out.append(Scenario(
            f"skip_{sid}", _skip(sid), (Injected("SKIP", sid),),
            {"skipped": [sid]},
            f"{sid} omitted entirely."))

    # OUT_OF_ORDER - performed, but in the wrong sequence.
    for a, b in (("S08", "S09"), ("S11", "S12")):
        out.append(Scenario(
            f"out_of_order_{b}_before_{a}", _reorder(a, b),
            (Injected("SKIP", a),), {},
            f"{b} performed before {a}."))

    # LEGAL REORDER - injects nothing. Must produce nothing.
    # The row nobody else records, and the one that proves the unordered group.
    out.append(Scenario(
        "legal_reorder_S03_S02", _reorder("S02", "S03"), (),
        {"complete": [s for s, _ in synth.NOMINAL], "no_alert": True},
        "S02 and S03 are an unordered group. Either order is correct."))

    out.append(Scenario(
        "wrong_object_vial_b", _wrong_object,
        (Injected("WRONG_OBJECT", "S03"),), {"wrong_object": ["S03"]},
        "Vial B taken instead of vial A."))

    out.append(Scenario(
        "duration_rush_S12", _rush("S12", _s12_fast),
        (Injected("DURATION", "S12"),), {},
        "Cartridge removal rushed below min_s."))

    # S11 max_s is 90 s. Dwelling 110 s puts it unambiguously past the prior.
    out.append(Scenario(
        "duration_stall_S11", _stall("S11", 110.0),
        (Injected("DURATION", "S11"),), {},
        "Latch opening stalled past max_s."))

    out.append(Scenario(
        "hazard_latch_open", _hazard,
        (Injected("HAZARD", "S10"),), {"hazard": ["S10"]},
        "Latch opened while the processing unit runs."))

    # Occlusion and lock loss inject NO deviation. They test restraint: the
    # engine must report UNVERIFIED and stay quiet, not invent a skip.
    out.append(Scenario(
        "occlusion_S06", _occlude("S05", "S06"), (),
        {"unverified_at_least": ["S06"]},
        "Crew torso blocks the workspace. UNVERIFIED, never SKIPPED."))
    out.append(Scenario(
        "frame_loss_S07", _lock_loss("S06", "S07"), (),
        {"unverified_at_least": ["S07"]},
        "AprilTag lock lost. Geometry unknown, not false."))

    return out


# --------------------------------------------------------------------------
@dataclass
class CorpusEntry:
    run_id: str
    scenario: str
    profile: str
    seed: int
    frames: list[BeliefFrame]
    injected: tuple[Injected, ...]
    expect: dict[str, Any]
    notes: str

    def header_extra(self) -> dict[str, Any]:
        """What lands in the trace header. `injected` is the ground truth the
        harness scores against; `expect` is the verdict the engine must reach."""
        return {
            "expect": self.expect,
            "injected": [i.as_dict() for i in self.injected],
            "scenario": self.scenario,
            "profile": self.profile,
            "seed": self.seed,
        }


def build_corpus(proc_doc: dict, *, profiles: tuple[Perturbation, ...] = PROFILES,
                 seeds: int = 3,
                 scenario_list: list[Scenario] | None = None) -> list[CorpusEntry]:
    """Every scenario, under every profile, at `seeds` seeds.

    `scenario_list` defaults to CSP-1's scenarios(); another procedure passes its
    own (crx2.corpus_scenarios) - the degradation and scoring are shared.

    The clean profile is generated once per scenario rather than once per seed:
    with no noise the seeds would produce identical traces, and a corpus padded
    with duplicates makes every metric look better than it is.
    """
    out: list[CorpusEntry] = []
    for sc in (scenarios() if scenario_list is None else scenario_list):
        ideal = sc.build(proc_doc).frames
        for profile in profiles:
            n = 1 if profile.name == "clean" else seeds
            for seed in range(n):
                # crc32, not hash(): Python randomises string hashing per
                # process, and a corpus that changes every time it is built
                # cannot be a regression corpus.
                p = profile.with_seed(seed * 1000 + zlib.crc32(sc.name.encode()) % 997)
                frames = apply(ideal, p)
                suffix = profile.name if n == 1 else f"{profile.name}_{seed}"
                out.append(CorpusEntry(
                    run_id=f"{sc.name}__{suffix}",
                    scenario=sc.name, profile=profile.name, seed=p.seed,
                    frames=frames, injected=sc.injected,
                    expect=sc.expect, notes=sc.notes,
                ))
    return out
