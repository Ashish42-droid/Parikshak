"""Trace synthesis - what perception WOULD have reported, for a scripted run.

Two jobs, one world model:

  1. The nine golden traces. Hand-authored scenarios, each a deliberate statement
     about what the crew did and what the engine must therefore conclude.
  2. The eval corpus. The same scenarios replayed under sensor degradation -
     confidence loss, dropped frames, position noise, occlusion, lost rack lock -
     so the ROC sweep has something to sweep over before a single real frame
     exists.

The second is why this lives in the package rather than in tools/. A corpus you
can only regenerate by editing a script is a corpus that stops being regenerated.

None of this is a substitute for the sixty recorded runs. It is what makes the
eval harness, the threshold sweep and the regression suite exist in week 2
instead of week 8, so that when real data arrives it flows into a harness that
already works.
"""

from __future__ import annotations

import zlib
from datetime import datetime, timedelta, timezone

from parikshak.belief.frame import (
    BeliefFrame, BodyBelief, HandBelief, MotionBelief, ObjectBelief, contact_key,
)

FPS = 5.0                    # 0.2 s resolution: ample for 1-3 s hold_for windows
T0 = datetime(2026, 12, 11, 4, 30, 0, tzinfo=timezone.utc)

# Confidences for a clean, unoccluded observation. Deliberately not 1.0 - a
# trace that hands the engine perfect certainty tests nothing about how it
# behaves on real perception output.
#
# These sit just above alert_policy.complete_threshold (0.85), and that is not
# an accident. `all` combines by MIN, so a step's confidence is the confidence
# of its WEAKEST sensor: a four-clause step gated at 0.85 requires every one of
# detection, state, contact and motion to clear 0.85 independently. The first
# draft set motion at 0.82 and every motion-bearing step became permanently
# unverifiable while all four of its clauses read as true.
#
# This is a real constraint on W8 threshold tuning, not a fixture detail. If the
# deployed TCN peaks around 0.80 on clean insertions, complete_threshold cannot
# stay at 0.85 - and it is far cheaper to know that now than in November.
CONF = 0.93          # RT-DETR on a clean, unoccluded object
STATE_CONF = 0.91    # object-state head
MOTION_CONF = 0.88   # 1D-TCN on an unambiguous motion
HAND_CONF = 0.90     # MediaPipe + contact MLP
CONTACT_CONF = 0.89  # object-object contact head
BODY_CONF = 0.88     # RTMPose + rack-frame lift


def stable_track_id(name: str) -> int:
    """A deterministic stand-in for the ID ByteTrack would have assigned.

    NOT `hash(name)`: Python randomises string hashing per process, so the first
    version of this file produced a different track_id on every run. The traces
    are regression fixtures - regenerating one must change nothing unless the
    scripted behaviour changed, or every diff is noise and nobody reads them.

    Non-sequential on purpose: engine code must never assume track IDs are
    ordered or contiguous, because real ones are neither.
    """
    return 100 + zlib.crc32(name.encode("utf-8")) % 900


def zone_centre(z: dict) -> tuple[float, float, float]:
    if z["type"] == "box":
        return tuple((a + b) / 2 for a, b in zip(z["min"], z["max"]))  # type: ignore[return-value]
    return tuple(z["centre"])  # type: ignore[return-value]


#: CSP-1's starting world. The default, so every existing call - and every byte
#: of the committed golden corpus - is unchanged by making the world a parameter.
CSP1_HOME = {
    "cartridge_sc_a": "locker_l2_ambient",
    "vial_a": "tray_t1",
    "vial_b": "tray_t1",
    "sample_bag_sb01": "workspace",
    "holder_h1": "holder_h1_seat",
    "process_unit": "glovebox_interior",
    "indicator": "glovebox_interior",
    "latch": "glovebox_interior",
    "locker_l1": "locker_l1_cold",
    "locker_l2": "locker_l2_ambient",
    "restraint": "restraint_zone",
    "crew_hand": "workspace",
}
CSP1_STATES = {
    "latch": "closed", "holder_h1": "empty",
    "process_unit": "idle", "indicator": "off",
    "sample_bag_sb01": "open",
}
#: Nudge siblings apart so vial_a and vial_b are not co-located in tray T1.
CSP1_OFFSETS = {"vial_b": 0.05}


class RunBuilder:
    """Simulates the world so we can emit what perception would have seen.

    The starting world is a parameter because a second procedure on the same
    rack starts from a different arrangement - CRX-2 keeps vial B in the cold
    locker - and a synthesiser that could only describe CSP-1's bench would be
    the one piece of the system that was secretly procedure-specific.
    """

    def __init__(self, proc: dict, fps: float = FPS, *,
                 home: dict[str, str] | None = None,
                 states: dict[str, str] | None = None,
                 offsets: dict[str, float] | None = None):
        self.proc = proc
        self.zones = proc["zones"]
        self.entities = proc["entities"]
        self.dt = 1.0 / fps
        self.t = 0.0
        self.frames: list[BeliefFrame] = []

        self.pos: dict[str, tuple] = {}
        self.state: dict[str, str] = {}
        self.visible: dict[str, bool] = {}
        self.occluded: set[str] = set()
        self.held: dict[str, str] = {}       # entity -> "left"|"right"
        self.motion_cls = "idle"
        self.pending_confirm: list[str] = []
        self.occl_zones: dict[str, float] = {}
        self.contacts: dict[str, float] = {}
        self.quat: dict[str, tuple] = {}
        self.restrained = False   # the crew member starts NOT in the foot restraint
        self.frame_lock = True

        self._home(CSP1_HOME if home is None else home,
                   CSP1_STATES if states is None else states,
                   CSP1_OFFSETS if offsets is None else offsets)

    # -- world setup -----------------------------------------------------

    def _home(self, home: dict[str, str], states: dict[str, str],
              offsets: dict[str, float]) -> None:
        for e, z in home.items():
            self.pos[e] = zone_centre(self.zones[z])
            self.visible[e] = True
        for e, delta in offsets.items():
            self.pos[e] = tuple(v + delta for v in self.pos[e])
        self.state.update(states)

    # -- mutators --------------------------------------------------------

    def move(self, entity: str, zone: str, jitter: float = 0.0):
        c = zone_centre(self.zones[zone])
        self.pos[entity] = tuple(v + jitter for v in c)
        return self

    def at(self, entity: str, xyz: tuple):
        self.pos[entity] = xyz
        return self

    def set_state(self, entity: str, state: str):
        self.state[entity] = state
        return self

    def grasp(self, entity: str, hand: str = "left"):
        self.held[entity] = hand
        return self

    def release(self, entity: str):
        self.held.pop(entity, None)
        return self

    def motion(self, cls: str):
        self.motion_cls = cls
        return self

    def attach(self, a: str, b: str, conf: float = CONTACT_CONF, aligned: bool = True):
        """Seat one object against another: contact asserted, axes co-aligned.

        S07 attaches the vial to the cartridge and verifies with `contacting`
        AND `aligned`. Both need perception fields that BeliefFrame v1.0 did not
        have, which is why it has them now.
        """
        self.contacts[contact_key(a, b)] = conf
        q = (1.0, 0.0, 0.0, 0.0) if aligned else (0.906, 0.0, 0.423, 0.0)  # ~50 deg off
        self.quat[a] = q
        self.quat[b] = (1.0, 0.0, 0.0, 0.0)
        return self

    def detach(self, a: str, b: str):
        self.contacts.pop(contact_key(a, b), None)
        return self

    def restrain(self, value: bool = True):
        """Crew member gets into (or out of) the foot restraint.

        The first version had the body inside the restraint zone from frame
        zero, so S01 verified 2 s into a run whose min_s is 5 - a DURATION
        deviation on the cleanest possible trace. A nominal run that raises an
        alert is a fixture that would have taught us to distrust the alert.
        """
        self.restrained = value
        return self

    def confirm(self, token: str):
        """Crew says it aloud or clicks it. Emitted on the next frame only."""
        self.pending_confirm.append(token)
        return self

    def occlude(self, *entities: str, zone: str | None = None, frac: float = 0.9):
        self.occluded.update(entities)
        if zone:
            self.occl_zones[zone] = frac
        return self

    def clear_occlusion(self):
        self.occluded.clear()
        self.occl_zones.clear()
        return self

    def lose_lock(self):
        self.frame_lock = False
        return self

    def regain_lock(self):
        self.frame_lock = True
        return self

    # -- emit ------------------------------------------------------------

    def hold(self, seconds: float):
        n = max(1, int(round(seconds / self.dt)))
        for _ in range(n):
            self.frames.append(self._snapshot())
            self.pending_confirm.clear()
            self.t += self.dt
        return self

    def _snapshot(self) -> BeliefFrame:
        objects: dict[str, ObjectBelief] = {}
        for name in self.entities:
            occ = name in self.occluded
            vis = self.visible.get(name, True) and not occ
            objects[name] = ObjectBelief(
                visible=vis,
                conf=CONF if vis else 0.0,
                pos_rack=self.pos.get(name) if not occ else None,
                quat_rack=self.quat.get(name) if not occ else None,
                track_id=stable_track_id(name),
                state=self.state.get(name) if name in self.state else None,
                state_conf=STATE_CONF if (name in self.state and not occ) else 0.0,
                held_by=self.held.get(name),
                occluded=occ,
            )

        hands: dict[str, HandBelief] = {}
        for side in ("left", "right"):
            carried = next((e for e, h in self.held.items() if h == side), None)
            wrist = self.pos.get(carried) if carried else self.pos["crew_hand"]
            hands[side] = HandBelief(
                present=True, conf=HAND_CONF, wrist_rack=wrist,
                contact_with=carried,
                grasp_type="grasp" if carried else "none",
                grasp_conf=HAND_CONF if carried else 0.0,
            )

        rz = zone_centre(self.zones["restraint_zone"])
        if not self.restrained:
            # Standing clear of the restraint: same posture, translated out of
            # the zone along the rack face.
            rz = (rz[0], rz[1] - 0.9, rz[2])
        body = BodyBelief(conf=BODY_CONF, joints_rack={
            "ankle_l": (rz[0] - 0.1, rz[1], rz[2]),
            "ankle_r": (rz[0] + 0.1, rz[1], rz[2]),
            "pelvis": (rz[0], rz[1] + 0.2, rz[2]),
        })

        return BeliefFrame(
            t_mono=round(self.t, 3),
            t_utc=(T0 + timedelta(seconds=self.t)).isoformat().replace("+00:00", "Z"),
            frame_lock=self.frame_lock,
            objects=objects,
            hands=hands,
            body=body,
            motion=MotionBelief(cls=self.motion_cls, conf=MOTION_CONF),
            occlusion=dict(self.occl_zones),
            contacts={k: v for k, v in self.contacts.items()
                      if not any(e in self.occluded for e in k.split("|"))},
            confirmations=tuple(self.pending_confirm),
        )


# ------------------------------------------------------------------ scripts
# Each function below is a statement about crew behaviour. Read them as
# procedures being performed, not as test fixtures being assembled.

def s01_restrain(b: RunBuilder):
    b.motion("reach").hold(3)
    b.restrain().motion("idle").hold(4)


def s02_get_cartridge(b: RunBuilder):
    b.motion("reach").hold(2)
    b.grasp("cartridge_sc_a", "left").move("cartridge_sc_a", "workspace").motion("idle").hold(4)


def s03_get_vial(b: RunBuilder):
    b.motion("reach").hold(2)
    b.grasp("vial_a", "right").at("vial_a", tuple(v + 0.02 for v in b.pos["crew_hand"]))
    b.motion("idle").hold(4)


def s04_verify_label(b: RunBuilder):
    # Held up to read the barcode, then confirmed aloud. The confirmation is
    # the evidence: merely holding the cartridge proves nothing about whether
    # anyone looked at the label.
    b.at("crew_hand", b.pos["cartridge_sc_a"]).hold(3)
    b.confirm("label verified").hold(2)


def s05_open_latch(b: RunBuilder):
    b.motion("reach").hold(1)
    b.set_state("latch", "open").motion("idle").hold(3)


def s06_insert_cartridge(b: RunBuilder):
    b.motion("insert").hold(2)
    b.move("cartridge_sc_a", "holder_h1_seat").hold(2)
    b.release("cartridge_sc_a").set_state("holder_h1", "loaded").motion("idle").hold(3)


def s07_attach_vial(b: RunBuilder):
    b.motion("rotate_seal").hold(2)
    b.at("vial_a", tuple(v + 0.03 for v in b.pos["cartridge_sc_a"]))
    b.attach("vial_a", "cartridge_sc_a").hold(4)
    b.motion("idle").hold(1)


def s08_close_latch(b: RunBuilder):
    b.motion("reach").hold(1)
    b.set_state("latch", "closed").motion("idle").hold(3)


def s09_press_start(b: RunBuilder):
    b.at("crew_hand", zone_centre(b.zones["start_button"])).motion("press").hold(2)
    b.set_state("process_unit", "running").set_state("indicator", "amber")
    b.motion("idle").hold(3)


def s10_wait(b: RunBuilder, fault: bool = False):
    b.hold(62)
    if fault:
        b.set_state("process_unit", "fault").set_state("indicator", "red").hold(6)
    else:
        b.set_state("process_unit", "complete").set_state("indicator", "green").hold(6)


def s11_open_latch(b: RunBuilder):
    b.motion("reach").hold(1)
    b.set_state("latch", "open").motion("idle").hold(3)


def s12_remove_cartridge(b: RunBuilder):
    b.motion("withdraw").hold(2)
    b.grasp("cartridge_sc_a", "left").move("cartridge_sc_a", "workspace")
    b.set_state("holder_h1", "empty").motion("idle").hold(3)


def s13_seal_bag(b: RunBuilder):
    b.motion("reach").hold(2)
    b.at("cartridge_sc_a", b.pos["sample_bag_sb01"]).release("cartridge_sc_a")
    b.set_state("sample_bag_sb01", "sealed").motion("idle").hold(4)


def s14_stow(b: RunBuilder):
    # A sealed bag carries what is sealed in it. Before this, stowing moved the
    # bag and left the cartridge floating at the workspace, so S13's `inside`
    # went false the moment S14 began - invisible whenever S13 was recognised
    # first, and a false SKIP of S13 whenever occlusion delayed it. A cartridge
    # that was never sealed in (skip_S13) stays where it is. Sealing is the
    # test, not position: S12 already leaves the cartridge at the workspace
    # centre, which is where the open bag sits.
    bagged = b.state.get("sample_bag_sb01") == "sealed"
    b.grasp("sample_bag_sb01", "left").motion("reach").hold(2)
    b.move("sample_bag_sb01", "locker_l1_cold").release("sample_bag_sb01")
    if bagged:
        b.at("cartridge_sc_a", b.pos["sample_bag_sb01"])
    b.motion("idle").hold(5)


NOMINAL = [
    ("S01", s01_restrain), ("S02", s02_get_cartridge), ("S03", s03_get_vial),
    ("S04", s04_verify_label), ("S05", s05_open_latch), ("S06", s06_insert_cartridge),
    ("S07", s07_attach_vial), ("S08", s08_close_latch), ("S09", s09_press_start),
    ("S10", s10_wait), ("S11", s11_open_latch), ("S12", s12_remove_cartridge),
    ("S13", s13_seal_bag), ("S14", s14_stow),
]


def run(proc: dict, skip: set[str] | None = None, order: list[str] | None = None,
        mutate=None, replace: dict | None = None, *, nominal=None,
        home: dict[str, str] | None = None, states: dict[str, str] | None = None,
        offsets: dict[str, float] | None = None, tail_s: float = 2.0) -> RunBuilder:
    """Script a whole run. `tail_s` is how long recording continues after the
    last action; the golden traces were generated with the 2 s default, and the
    evaluation corpus sets it from the procedure (see corpus.session_tail_s)."""
    nominal = NOMINAL if nominal is None else nominal
    b = RunBuilder(proc, home=home, states=states, offsets=offsets)
    skip = skip or set()
    steps = dict(nominal)
    steps.update(replace or {})
    seq = order or [sid for sid, _ in nominal]
    b.hold(2)                                  # settle before the run starts
    for sid in seq:
        if sid in skip:
            continue
        steps[sid](b)
        if mutate:
            mutate(b, sid)
    b.hold(tail_s)
    return b

