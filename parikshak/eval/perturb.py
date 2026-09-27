"""Sensor degradation applied to an ideal run.

The scripted traces describe what the crew did. Real perception does not report
that cleanly, and an engine tuned only against clean traces is tuned against a
world that does not exist. So the same scenario is replayed under measured
degradation and the operating point is chosen against the degraded version.

What this models, and why each one is here:

  conf_scale / conf_jitter  Detector confidence is not 0.93 on a rack under
                            venue lighting. `all` combines by MIN, so a step is
                            only as confident as its weakest clause - lowering
                            confidence is the single most effective way to make
                            steps stop verifying, which is exactly the failure
                            we need the ROC to price.

  pos_noise_m               Centroid jitter. Feeds `stable`, `in_zone`,
                            `inside`, `near` - the predicates whose thresholds
                            are in centimetres.

  drop_rate                 Frames lost to a busy encoder or a dropped CSI
                            packet. Matters because `hold_for` windows are
                            measured in wall time, not in frames.

  occlusion                 The crew's own torso, ~30% of the time per PLAN.md
                            risk 3. Must produce UNVERIFIED, never a deviation.

  lock_loss                 AprilTag PnP failing. All geometry becomes unknown.

What this deliberately does NOT model: the microgravity domain gap. No amount of
Gaussian noise on Earth-recorded confidences tells you how a free-floating tool
behaves. That gap is measured against MicroG-4M and reported, not simulated.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from parikshak.belief.frame import BeliefFrame, HandBelief, ObjectBelief


@dataclass(frozen=True, slots=True)
class Perturbation:
    """One named sensor-degradation profile.

    Deterministic given `seed`: the same profile and seed must reproduce the
    same trace byte for byte, or the eval corpus is not a regression corpus.
    """

    name: str = "clean"
    conf_scale: float = 1.0
    conf_jitter: float = 0.0
    pos_noise_m: float = 0.0
    drop_rate: float = 0.0
    occlusion_rate: float = 0.0
    occlusion_zone: str = "glovebox_interior"
    lock_loss_rate: float = 0.0
    seed: int = 0

    def with_seed(self, seed: int) -> Perturbation:
        return replace(self, seed=seed)


#: The sweep. `clean` is the control; `harsh` is roughly what a first-pass
#: detector on real data looks like before any tuning.
PROFILES: tuple[Perturbation, ...] = (
    Perturbation("clean"),
    Perturbation("mild", conf_scale=0.97, conf_jitter=0.02, pos_noise_m=0.004,
                 drop_rate=0.02, occlusion_rate=0.05, lock_loss_rate=0.01),
    Perturbation("moderate", conf_scale=0.93, conf_jitter=0.04, pos_noise_m=0.008,
                 drop_rate=0.05, occlusion_rate=0.15, lock_loss_rate=0.03),
    Perturbation("harsh", conf_scale=0.88, conf_jitter=0.06, pos_noise_m=0.015,
                 drop_rate=0.10, occlusion_rate=0.30, lock_loss_rate=0.06),
)

PROFILES_BY_NAME = {p.name: p for p in PROFILES}


def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _jitter_conf(rng: random.Random, conf: float, p: Perturbation) -> float:
    if conf <= 0.0:
        return 0.0  # a zero confidence means "not detected"; noise must not revive it
    return _clamp(conf * p.conf_scale + rng.gauss(0.0, p.conf_jitter))


def _jitter_pos(rng: random.Random, pos, sigma: float):
    if pos is None or sigma <= 0.0:
        return pos
    return tuple(v + rng.gauss(0.0, sigma) for v in pos)


def apply(frames: Sequence[BeliefFrame], p: Perturbation) -> list[BeliefFrame]:
    """Degrade a clean run. Returns a new frame list; inputs are untouched.

    Occlusion and lock loss are applied in RUNS, not per independent frame. A
    torso blocks the view for two seconds, not for one frame in seven - and the
    difference matters enormously, because `hold_for` and `stable` care about
    contiguous windows. Independent per-frame dropout would produce a system
    that looks far more robust than it is.
    """
    if p.name == "clean":
        return list(frames)

    rng = random.Random(p.seed)
    out: list[BeliefFrame] = []

    occluded_until = -1.0
    lock_lost_until = -1.0
    # A dropped VIDEO frame does not drop what the crew said. Confirmations
    # arrive from the voice pipeline as events; losing a CSI packet cannot unsay
    # "run complete". Dropping them with the frame made CRX-2's spoken steps
    # vanish under harsh degradation - a false SKIP of S03, and a tracker that
    # stalled before S04 or never passed S07 - which measured the simulator, not
    # the engine. They ride on the next frame that survives.
    carried: tuple = ()

    for frame in frames:
        if p.drop_rate and rng.random() < p.drop_rate:
            carried += tuple(frame.confirmations)
            continue

        t = frame.t_mono
        if t > occluded_until and p.occlusion_rate and rng.random() < p.occlusion_rate * 0.1:
            occluded_until = t + rng.uniform(1.0, 3.0)
        if t > lock_lost_until and p.lock_loss_rate and rng.random() < p.lock_loss_rate * 0.1:
            lock_lost_until = t + rng.uniform(0.5, 2.0)

        occluding = t <= occluded_until
        lock = frame.frame_lock and t > lock_lost_until

        objects: dict[str, ObjectBelief] = {}
        for name, ob in frame.objects.items():
            hidden = ob.occluded or (occluding and _in_zone_of_interest(name, p))
            if hidden:
                objects[name] = replace(ob, visible=False, occluded=True, conf=0.0,
                                        pos_rack=None, quat_rack=None, state_conf=0.0)
                continue
            objects[name] = replace(
                ob,
                conf=_jitter_conf(rng, ob.conf, p),
                state_conf=_jitter_conf(rng, ob.state_conf, p),
                pos_rack=_jitter_pos(rng, ob.pos_rack, p.pos_noise_m),
            )

        hands: dict[str, HandBelief] = {
            side: replace(h,
                          conf=_jitter_conf(rng, h.conf, p),
                          grasp_conf=_jitter_conf(rng, h.grasp_conf, p),
                          wrist_rack=_jitter_pos(rng, h.wrist_rack, p.pos_noise_m))
            for side, h in frame.hands.items()
        }

        body = frame.body
        if body is not None:
            body = replace(
                body,
                conf=_jitter_conf(rng, body.conf, p),
                joints_rack={k: _jitter_pos(rng, v, p.pos_noise_m)
                             for k, v in body.joints_rack.items()},
            )

        occlusion = dict(frame.occlusion)
        if occluding:
            occlusion[p.occlusion_zone] = max(occlusion.get(p.occlusion_zone, 0.0), 0.85)

        contacts = {k: _jitter_conf(rng, v, p) for k, v in frame.contacts.items()
                    if not (occluding and any(_in_zone_of_interest(e, p)
                                              for e in k.split("|")))}

        out.append(replace(
            frame,
            frame_lock=lock,
            objects=objects,
            hands=hands,
            body=body,
            motion=replace(frame.motion, conf=_jitter_conf(rng, frame.motion.conf, p)),
            occlusion=occlusion,
            contacts=contacts,
            confirmations=carried + tuple(frame.confirmations),
        ))
        carried = ()
    return out


#: Entities plausibly hidden when the crew's torso blocks the glovebox. Kept as
#: a name list rather than a geometric test because the synthesiser has no
#: camera model - and pretending it does would be dressing a guess up as physics.
_OCCLUDABLE = frozenset({
    "cartridge_sc_a", "vial_a", "vial_b", "holder_h1", "latch",
    "process_unit", "indicator", "sample_bag_sb01",
})


def _in_zone_of_interest(entity: str, _p: Perturbation) -> bool:
    return entity in _OCCLUDABLE
