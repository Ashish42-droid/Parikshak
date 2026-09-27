"""
PARIKSHAK procedure validator - PDL v1.0

Runs at load time, before a single frame is read. Its job is to prove that the
deployed perception build can answer every question this procedure file asks.

That is the whole "zero retraining" claim, made checkable:
  - authoring inside the trained detector class vocabulary is free,
  - authoring outside it fails loudly, here, with a named missing class,
  - and nothing in between silently degrades into a model that guesses.

Usage:
    python tools/validate_procedure.py procedures/csp1_colloid_sample_processing.yaml
    python tools/validate_procedure.py procedures/*.yaml --strict

The CLI in tools/ is a shim over this module. The logic lives in the package so
that parikshak.pdl.loader can refuse to hand the engine a procedure that has not
passed the same gate the command line applies - one validator, not two.

Exit codes: 0 clean (warnings allowed), 1 errors found, 2 could not load.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  pip install pyyaml")

SUPPORTED_PDL = {"1.0"}

# --------------------------------------------------------------------------
# The closed predicate vocabulary. Mirrors schema/PREDICATES.md exactly.
# `args`  : arg name -> (required?, ref-kind or None)
# `caps`  : capabilities this predicate forces the deployed build to provide
# --------------------------------------------------------------------------
DETECT = ["object_detection"]
HANDS = ["hand_landmarks", "hand_object_contact"]
GEOM = ["rack_frame_extrinsics"]
BODY = ["body_pose", "rack_frame_extrinsics"]

VOCAB: dict[str, dict] = {
    # presence
    "visible":        {"args": {"entity": (True, "entity"), "min_conf": (False, None)}, "caps": DETECT},
    "absent":         {"args": {"entity": (True, "entity")}, "caps": DETECT},
    "count_of":       {"args": {"entity": (True, "entity"), "n": (True, None)}, "caps": DETECT},
    # manipulation
    "grasped":        {"args": {"entity": (True, "entity"), "hand": (False, "hand")}, "caps": HANDS},
    "released":       {"args": {"entity": (True, "entity")}, "caps": HANDS},
    "contacting":     {"args": {"a": (True, "entity"), "b": (True, "entity")}, "caps": HANDS},
    "hand_in_zone":   {"args": {"hand": (True, "hand"), "zone": (True, "zone")}, "caps": HANDS + GEOM},
    # geometry
    "inside":         {"args": {"a": (True, "entity"), "container": (True, "zone_or_entity"),
                                "margin_m": (False, None)}, "caps": GEOM},
    "near":           {"args": {"a": (True, "entity"), "b": (True, "entity"), "dist_m": (True, None)}, "caps": GEOM},
    "aligned":        {"args": {"a": (True, "entity"), "b": (True, "entity"), "tol_deg": (True, None)},
                       "caps": GEOM + ["object_pose_6dof"]},
    "in_zone":        {"args": {"entity": (True, "entity"), "zone": (True, "zone")}, "caps": GEOM},
    "stable":         {"args": {"entity": (True, "entity"), "seconds": (True, None), "tol_m": (False, None)},
                       "caps": GEOM},
    # object state
    "state_is":       {"args": {"entity": (True, "entity"), "state": (True, "state")}, "caps": ["object_state"]},
    # motion
    "motion_is":      {"args": {"motion_class": (True, "motion"), "min_conf": (False, None)}, "caps": ["motion_tcn"]},
    "motion_count":   {"args": {"motion_class": (True, "motion"), "n": (True, None), "window_s": (False, None)},
                       "caps": ["motion_tcn"]},
    # crew posture
    "body_restrained": {"args": {"zone": (True, "zone")}, "caps": BODY},
    "body_in_zone":    {"args": {"zone": (True, "zone")}, "caps": BODY},
    # engine
    "elapsed_since":  {"args": {"step_id": (True, "step"), "seconds": (True, None)}, "caps": []},
    "crew_confirmed": {"args": {"token": (True, None)}, "caps": ["crew_confirm"]},
    "occluded":       {"args": {"zone": (True, "zone")}, "caps": []},
    "assert_true":    {"args": {"flag": (True, "flag")}, "caps": [], "scalar_arg": "flag"},
}

COMBINATORS = {"all", "any", "not", "hold_for", "occurred"}

#: A geometric tolerance below this many standard deviations of the detector's
#: centroid noise is not measurable. Empirical: P(stable) stays above ~95% only
#: when tol_m >= ~5 sigma over a 1.5 s window at 5 fps.
MIN_TOLERANCE_SIGMAS = 5.0


#: Sampling rate assumed when sizing a `stable` window, until the deployed
#: build publishes a measured one. Deliberately low: fewer samples in the window
#: means a higher noise floor, so an assumption here errs toward warning.
ASSUMED_PERCEPTION_HZ = 5.0

#: 95th percentile of the chi distribution with three degrees of freedom - the
#: length of a 3-D vector of unit-variance Gaussian errors.
CHI3_P95 = 2.795


def stable_noise_floor(seconds: float, sigma: float,
                       hz: float = ASSUMED_PERCEPTION_HZ) -> float:
    """The smallest `tol_m` at which `stable` holds on >= 95% of frames for an
    object that is genuinely not moving, given per-axis centroid noise `sigma`.

    `stable` compares the mean positions of the window's two halves (h samples
    each). Per axis that difference is Gaussian with variance 2 sigma^2 / h, so
    its 3-D length is sigma * sqrt(2 / h) * chi_3. Verified by Monte Carlo to
    two decimal places for 4 to 30 samples.

    For contrast, the earlier estimator - the largest distance of any sample
    from the first - needed about 5 sigma regardless of window length, which is
    where MIN_TOLERANCE_SIGMAS came from. A longer window now lowers the floor.
    """
    half = max(1, int(round(seconds * hz)) // 2)
    return math.sqrt(2.0 / half) * CHI3_P95 * sigma


def min_satisfaction_time(node) -> float:
    """Earliest a verification tree can possibly become true, in seconds.

    Temporal windows COMPOSE rather than overlap: `hold_for(1.5, stable(1.5))`
    needs 1.5 s of history before `stable` can be true at all, and then 1.5 s of
    `stable` staying true - 3.0 s, not 1.5. A step whose `duration.min_s` is
    below this can never be completed "too fast", so its DURATION check is dead
    code and a rushed performance goes unreported.
    """
    if not isinstance(node, dict) or len(node) != 1:
        return 0.0
    key, val = next(iter(node.items()))
    if key == "hold_for" and isinstance(val, dict):
        return float(val.get("seconds", 0)) + min_satisfaction_time(val.get("expr"))
    if key == "stable" and isinstance(val, dict):
        return float(val.get("seconds", 0))
    if key == "all" and isinstance(val, list):
        return max((min_satisfaction_time(c) for c in val), default=0.0)
    if key == "any" and isinstance(val, list):
        return min((min_satisfaction_time(c) for c in val), default=0.0)
    if key in ("not", "occurred"):
        return min_satisfaction_time(val)
    return 0.0

#: Which argument carries the tolerance, per predicate.
TOLERANCE_ARG = {"stable": "tol_m", "inside": "margin_m", "near": "dist_m"}

# PyYAML implements YAML 1.1, where bare off/on/yes/no/y/n are BOOLEANS.
# An indicator state written as `off` silently becomes False and every
# state_is(indicator, off) comparison then fails at run time, on the rack,
# with no error anywhere. Caught here instead.
YAML11_TRAP = (
    "parsed as {value!r} ({kind}), not a string - YAML 1.1 treats bare "
    "off/on/yes/no/true/false as booleans. Quote it: \"off\""
)


def normalise_token(token) -> str:
    """A token as a recogniser hears it.

    Case, punctuation and spacing do not survive speech, so two tokens that
    differ only in those are one token - and must be declared once.
    """
    return " ".join(re.findall(r"[a-z0-9]+", str(token).lower()))


def not_a_string(where: str, value, rep: "Report") -> bool:
    if isinstance(value, str):
        return False
    rep.err(where, YAML11_TRAP.format(value=value, kind=type(value).__name__))
    return True
VALID_HANDS = {"left", "right", "any", "both"}
VALID_SEVERITY = {"info", "advisory", "caution", "critical"}
VALID_STEP_TYPES = {"discrete", "motion", "wait", "crew_confirm"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")


# --------------------------------------------------------------------------
# Predicate tree walking
# --------------------------------------------------------------------------
def walk_predicates(node, path: str, rep: Report):
    """Yield (name, args_dict, path) for every leaf predicate in a tree."""
    if node is None:
        return
    if not isinstance(node, dict):
        rep.err(path, f"expected a predicate mapping, got {type(node).__name__}: {node!r}")
        return
    if len(node) != 1:
        rep.err(path, f"a predicate node must have exactly one key, found {sorted(node)}")
        return

    key, val = next(iter(node.items()))

    if key in ("all", "any"):
        if not isinstance(val, list):
            rep.err(f"{path}.{key}", "must be a list of predicates")
            return
        for i, child in enumerate(val):
            yield from walk_predicates(child, f"{path}.{key}[{i}]", rep)
        return

    if key == "not":
        yield from walk_predicates(val, f"{path}.not", rep)
        return

    if key == "hold_for":
        if not isinstance(val, dict) or "seconds" not in val or "expr" not in val:
            rep.err(f"{path}.hold_for", "requires keys 'seconds' and 'expr'")
            return
        yield from walk_predicates(val["expr"], f"{path}.hold_for.expr", rep)
        return

    if key == "occurred":
        # Scopes a clause to the whole step rather than the current frame.
        yield from walk_predicates(val, f"{path}.occurred", rep)
        return

    if key not in VOCAB:
        near = [v for v in VOCAB if v.startswith(key[:4])]
        hint = f" (did you mean {near[0]}?)" if near else ""
        rep.err(path, f"unknown predicate '{key}'{hint} - vocabulary is closed, see schema/PREDICATES.md")
        return

    spec = VOCAB[key]
    if not isinstance(val, dict):
        scalar = spec.get("scalar_arg")
        if scalar:
            val = {scalar: val}
        else:
            rep.err(f"{path}.{key}", f"expected an argument mapping, got {val!r}")
            return

    for arg, (required, _kind) in spec["args"].items():
        if required and arg not in val:
            rep.err(f"{path}.{key}", f"missing required argument '{arg}'")
    for arg in val:
        if arg not in spec["args"]:
            rep.err(f"{path}.{key}", f"unknown argument '{arg}' (accepts {sorted(spec['args'])})")

    yield key, val, f"{path}.{key}"


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check(doc: dict, rep: Report) -> dict:
    ver = str(doc.get("pdl_version", ""))
    if ver not in SUPPORTED_PDL:
        rep.err("pdl_version", f"'{ver}' unsupported (this validator handles {sorted(SUPPORTED_PDL)})")

    for key in ("procedure", "requires", "frame", "zones", "entities", "alert_policy", "logging", "flow"):
        if key not in doc:
            rep.err("root", f"missing required top-level block '{key}'")
    if rep.errors:
        return {}

    requires = doc["requires"]
    declared_caps = set(requires.get("capabilities", []))
    declared_classes = set(requires.get("detector_classes", []))
    entities = doc["entities"] or {}
    zones = doc["zones"] or {}
    motion_reg = set((doc.get("motion_classes") or {}).get("classes", []))
    flow = doc["flow"]
    steps = {s["id"]: s for s in flow.get("steps", [])}
    groups = {g["id"]: g for g in flow.get("groups", [])}

    # -- entities -----------------------------------------------------------
    for name, e in entities.items():
        cls = e.get("detector_class")
        if not cls:
            rep.err(f"entities.{name}", "missing detector_class")
        elif cls not in declared_classes:
            rep.err(f"entities.{name}",
                    f"detector_class '{cls}' not in requires.detector_classes - "
                    f"this needs a NEW detection dataset, not a procedure edit")
        for sib in e.get("confusable_with", []):
            if sib not in entities:
                rep.err(f"entities.{name}.confusable_with", f"undeclared entity '{sib}'")
        if "states" in e and "unknown" not in e["states"]:
            rep.warn(f"entities.{name}.states",
                     "no 'unknown' state - a state classifier with no abstain option "
                     "will fabricate a value under occlusion")
        for i, st in enumerate(e.get("states") or []):
            not_a_string(f"entities.{name}.states[{i}]", st, rep)

    # Same YAML 1.1 trap applies to every other bare-token list in the file.
    for i, c in enumerate((doc.get("motion_classes") or {}).get("classes", [])):
        not_a_string(f"motion_classes.classes[{i}]", c, rep)
    for i, c in enumerate(requires.get("detector_classes", [])):
        not_a_string(f"requires.detector_classes[{i}]", c, rep)
    for i, c in enumerate(requires.get("capabilities", [])):
        not_a_string(f"requires.capabilities[{i}]", c, rep)

    # -- zones --------------------------------------------------------------
    for name, z in zones.items():
        t = z.get("type")
        if t == "box":
            lo, hi = z.get("min"), z.get("max")
            if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) == 3 and len(hi) == 3):
                rep.err(f"zones.{name}", "box needs 3-element min and max")
            elif any(a >= b for a, b in zip(lo, hi)):
                rep.err(f"zones.{name}", f"degenerate box: min {lo} not strictly less than max {hi}")
        elif t == "sphere":
            if len(z.get("centre", [])) != 3 or z.get("radius", 0) <= 0:
                rep.err(f"zones.{name}", "sphere needs a 3-element centre and positive radius")
        else:
            rep.err(f"zones.{name}", f"unknown zone type '{t}' (box | sphere)")

    # -- alert policy -------------------------------------------------------
    ap = doc["alert_policy"]
    for k in ("complete_threshold", "deviation_threshold"):
        v = ap.get(k)
        if not isinstance(v, (int, float)) or not 0.0 < v <= 1.0:
            rep.err(f"alert_policy.{k}", f"must be in (0, 1], got {v!r}")
    if "evidence_rate_hz" in ap:
        r = ap["evidence_rate_hz"]
        if not isinstance(r, (int, float)) or isinstance(r, bool) or not 0.0 < r <= 30.0:
            rep.err("alert_policy.evidence_rate_hz",
                    f"must be in (0, 30] observations per second, got {r!r}")
    if "hold_for_tolerance" in ap:
        q = ap["hold_for_tolerance"]
        if not isinstance(q, (int, float)) or isinstance(q, bool) or not 0.0 <= q < 0.5:
            rep.err("alert_policy.hold_for_tolerance",
                    f"must be in [0, 0.5) - a share of the window - got {q!r}")
    if ap.get("persistence_s", 0) <= 0:
        rep.warn("alert_policy.persistence_s",
                 "no persistence window - single-frame noise will reach the crew as speech")
    if not ap.get("advisory_only", False):
        rep.warn("alert_policy.advisory_only",
                 "not advisory-only; a system that can block crew action needs a far heavier "
                 "certification case than this design argues for")

    # -- flags asserted by effects -----------------------------------------
    asserted: set[str] = set()
    for sid, s in steps.items():
        for eff in s.get("effects") or []:
            if "assert" in eff:
                asserted.add(eff["assert"])

    # -- predicates ---------------------------------------------------------
    used_caps: set[str] = set()
    used_motions: set[str] = set()
    used_tokens: dict[str, list[str]] = {}
    step_entity_refs: dict[str, set[str]] = {}

    def resolve(kind, value, where):
        if kind == "entity":
            if value not in entities:
                rep.err(where, f"undeclared entity '{value}'")
        elif kind == "zone":
            if value not in zones:
                rep.err(where, f"undeclared zone '{value}'")
        elif kind == "zone_or_entity":
            if value not in zones and value not in entities:
                rep.err(where, f"'{value}' is neither a declared zone nor a declared entity")
        elif kind == "hand":
            if value not in VALID_HANDS:
                rep.err(where, f"hand must be one of {sorted(VALID_HANDS)}, got '{value}'")
        elif kind == "motion":
            used_motions.add(value)
            if motion_reg and value not in motion_reg:
                rep.err(where, f"motion class '{value}' not emitted by the deployed TCN "
                               f"(registry: {sorted(motion_reg)})")
        elif kind == "step":
            if value not in steps:
                rep.err(where, f"references unknown step '{value}'")
        elif kind == "flag":
            if value not in asserted:
                rep.err(where, f"flag '{value}' is never established by any step's effects")

    # Per-axis centroid noise of the deployed detector, if it has been measured.
    # None means "not calibrated yet" and the precision check stays dormant.
    sigma = requires.get("position_sigma_m")
    tolerance_floor = (MIN_TOLERANCE_SIGMAS * float(sigma)
                       if isinstance(sigma, (int, float)) and sigma > 0 else None)

    def check_tolerance(name, args, path):
        """Is this tolerance loose enough to be measurable by the deployed build?

        The vocabulary check proves the build can answer the question. This
        checks it can answer it precisely enough - a tolerance tighter than the
        detector's own noise makes the clause fail on correct runs, and the
        engine then reports a skip that never happened.
        """
        if tolerance_floor is None:
            return
        arg = TOLERANCE_ARG.get(name)
        if arg is None:
            return
        tol = args.get(arg, {"stable": 0.02}.get(name))
        if not isinstance(tol, (int, float)) or tol <= 0:
            return
        if name == "stable":
            seconds = args.get("seconds")
            if not isinstance(seconds, (int, float)) or seconds <= 0:
                return
            floor = stable_noise_floor(float(seconds), float(sigma))
            if tol < floor:
                rep.warn(path,
                         f"tol_m={tol} m is under the noise floor for a {seconds:g} s window "
                         f"({floor:.3f} m at position_sigma_m {sigma} m, assuming "
                         f"{ASSUMED_PERCEPTION_HZ:g} Hz), so a stationary object reads as "
                         f"drifting on more than 5% of frames and the step will stall. "
                         f"Loosen to >= {floor:.3f} m, lengthen the window, or improve "
                         f"the detector.")
            return
        # `inside` and `near` compare one frame's centroid against a boundary,
        # not a windowed average, so they keep the conservative single-sample
        # multiple.
        if tol < tolerance_floor:
            rep.warn(path,
                     f"{arg}={tol} m is under {MIN_TOLERANCE_SIGMAS:g}x the declared "
                     f"position_sigma_m ({sigma} m), so the clause holds well under 95% "
                     f"of the time on a correct run and the step will read as skipped. "
                     f"Loosen to >= {tolerance_floor:.3f} m, or improve the detector.")

    def scan(node, where, owner=None):
        for name, args, path in walk_predicates(node, where, rep):
            used_caps.update(VOCAB[name]["caps"])
            check_tolerance(name, args, path)
            if name == "motion_count" and ".verification" in path and ".occurred" not in path:
                # A count is an achievement at an instant, and it decays out of
                # its window. Unlatched, a step that also checks a state - "while
                # holding it" - is only verifiable at the moment the last count
                # registers, and the next step changing that state reads as
                # evidence the count never happened. CRX-2 S04 did exactly this.
                rep.warn(path,
                         "motion_count in a verification is not latched with occurred - the "
                         "step is only verifiable at the instant the count completes, and a "
                         "later step changing any state it is combined with reads as a skip. "
                         "Wrap the clause (or the whole tree) in occurred.")
            if name == "crew_confirmed":
                used_tokens.setdefault(str(args.get("token")), []).append(path)
            for arg, val in args.items():
                kind = VOCAB[name]["args"].get(arg, (False, None))[1]
                if kind:
                    resolve(kind, val, path)
                if owner and kind in ("entity", "zone_or_entity") and val in entities:
                    step_entity_refs.setdefault(owner, set()).add(val)
            if name == "state_is":
                ent = entities.get(args.get("entity"))
                if ent is not None:
                    valid = ent.get("states")
                    if not valid:
                        rep.err(path, f"entity '{args['entity']}' declares no states but state_is is used on it")
                    elif args.get("state") not in valid:
                        rep.err(path, f"state '{args.get('state')}' not in {valid}")

    # -- min_s must be reachable -------------------------------------------
    for sid, st in steps.items():
        dur = st.get("duration") or {}
        floor = min_satisfaction_time(st.get("verification"))
        min_s = dur.get("min_s")
        if isinstance(min_s, (int, float)) and floor > 0 and min_s <= floor:
            rep.warn(f"flow.steps[{sid}].duration",
                     f"min_s={min_s}s is at or below the verification tree's own minimum "
                     f"satisfaction time ({floor:g}s from composed hold_for/stable "
                     f"windows), so this step can never complete 'too fast' and its "
                     f"DURATION check can never fire. Raise min_s above {floor:g}s or "
                     f"shorten the windows.")

    # -- steps --------------------------------------------------------------
    for sid, s in steps.items():
        w = f"flow.steps[{sid}]"
        if s.get("type") not in VALID_STEP_TYPES:
            rep.err(w, f"type must be one of {sorted(VALID_STEP_TYPES)}, got {s.get('type')!r}")
        if not s.get("prompt_tts"):
            rep.err(w, "missing prompt_tts - every step must be speakable, that is the interface")
        if "verification" not in s:
            rep.err(w, "missing verification - a step the engine cannot verify cannot be logged")

        scan(s.get("preconditions"), f"{w}.preconditions", owner=sid)
        scan(s.get("verification"), f"{w}.verification", owner=sid)
        for i, inv in enumerate(s.get("invariants") or []):
            # Hazard invariants deliberately reference entities outside the step's
            # own objects list (the latch during a press), so they are not owned.
            scan(inv.get("expr"), f"{w}.invariants[{i}]")
            if inv.get("severity") not in VALID_SEVERITY:
                rep.err(f"{w}.invariants[{i}]", f"severity must be one of {sorted(VALID_SEVERITY)}")
            if not inv.get("tts"):
                rep.warn(f"{w}.invariants[{i}]", "hazard invariant with no spoken text")
        for i, br in enumerate(s.get("branch") or []):
            scan(br.get("when"), f"{w}.branch[{i}].when")
            if br.get("goto") not in steps and br.get("goto") not in groups:
                rep.err(f"{w}.branch[{i}]", f"goto target '{br.get('goto')}' does not exist")

        for i, eff in enumerate(s.get("effects") or []):
            if "set_state" in eff:
                tgt = eff["set_state"]
                ent = entities.get(tgt.get("entity"))
                if ent is None:
                    rep.err(f"{w}.effects[{i}]", f"undeclared entity '{tgt.get('entity')}'")
                elif tgt.get("state") not in (ent.get("states") or []):
                    rep.err(f"{w}.effects[{i}]", f"state '{tgt.get('state')}' not declared on '{tgt['entity']}'")

        for obj in s.get("objects") or []:
            if obj not in entities:
                rep.err(f"{w}.objects", f"undeclared entity '{obj}'")

        d = s.get("duration") or {}
        mn, nm, mx = d.get("min_s"), d.get("nominal_s"), d.get("max_s")
        if None in (mn, nm, mx):
            rep.err(f"{w}.duration", "needs min_s, nominal_s and max_s - the HSMM has no prior without them")
        elif not mn <= nm <= mx:
            rep.err(f"{w}.duration", f"expected min_s <= nominal_s <= max_s, got {mn} / {nm} / {mx}")
        if d.get("sigma_s", 0) <= 0:
            rep.warn(f"{w}.duration", "sigma_s must be positive; a zero-variance prior makes the HSMM brittle")

        if s.get("critical") and not s.get("on_skip"):
            rep.warn(w, "critical step with no on_skip block - a silent critical skip is the "
                        "exact failure this system exists to prevent")

        wo = s.get("on_wrong_object")
        if wo:
            for c in wo.get("confusable", []):
                if c not in entities:
                    rep.err(f"{w}.on_wrong_object", f"undeclared entity '{c}'")
            if wo.get("severity") not in VALID_SEVERITY:
                rep.err(f"{w}.on_wrong_object", f"severity must be one of {sorted(VALID_SEVERITY)}")

        if s.get("terminal") and s.get("next"):
            rep.err(w, "terminal step must have an empty next list")
        if not s.get("terminal") and not s.get("next") and not s.get("branch"):
            rep.err(w, "non-terminal step has no successor")

    # -- groups -------------------------------------------------------------
    for gid, g in groups.items():
        w = f"flow.groups[{gid}]"
        if g.get("type") != "unordered":
            rep.err(w, f"unknown group type '{g.get('type')}'")
        members = g.get("members") or []
        if len(members) < 2:
            rep.err(w, "an unordered group needs at least two members")
        for m in members:
            if m not in steps:
                rep.err(w, f"member '{m}' is not a declared step")
            elif steps[m].get("next") != [gid]:
                rep.err(f"flow.steps[{m}].next",
                        f"member of unordered group {gid} must declare next: [{gid}]")
        if g.get("successor") not in steps and g.get("successor") not in groups:
            rep.err(w, f"successor '{g.get('successor')}' does not exist")

    # -- graph: reachability and cycles ------------------------------------
    member_of = {m: gid for gid, g in groups.items() for m in (g.get("members") or [])}

    def collapse(nid):
        return member_of.get(nid, nid)

    adj: dict[str, set[str]] = {}
    for sid, s in steps.items():
        src = collapse(sid)
        out = adj.setdefault(src, set())
        for tgt in (s.get("next") or []):
            t = collapse(tgt)
            if t != src:
                out.add(t)
        for br in (s.get("branch") or []):
            t = collapse(br.get("goto"))
            if t and t != src:
                out.add(t)
    for gid, g in groups.items():
        adj.setdefault(gid, set()).add(collapse(g.get("successor")))

    entry = collapse(flow.get("entry"))
    if flow.get("entry") not in steps:
        rep.err("flow.entry", f"'{flow.get('entry')}' is not a declared step")

    seen: set[str] = set()
    stack = [entry]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))

    for sid in steps:
        if collapse(sid) not in seen:
            rep.err(f"flow.steps[{sid}]", "unreachable from flow.entry")

    # The nominal path follows `next` and group successors only. Anything
    # reachable exclusively through a `branch` goto is an off-nominal response
    # and must not be counted in the expected duration or the crew's estimate.
    nominal_adj: dict[str, set[str]] = {}
    for sid, s in steps.items():
        src = collapse(sid)
        out = nominal_adj.setdefault(src, set())
        for tgt in (s.get("next") or []):
            t = collapse(tgt)
            if t != src:
                out.add(t)
    for gid, g in groups.items():
        nominal_adj.setdefault(gid, set()).add(collapse(g.get("successor")))

    nominal_seen: set[str] = set()
    stack = [entry]
    while stack:
        n = stack.pop()
        if n in nominal_seen:
            continue
        nominal_seen.add(n)
        stack.extend(nominal_adj.get(n, ()))
    off_nominal = {sid for sid in steps if collapse(sid) not in nominal_seen}

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}

    def dfs(n, trail):
        colour[n] = GREY
        for m in sorted(adj.get(n, ())):
            if colour.get(m, WHITE) == GREY:
                rep.err("flow", f"cycle in procedure graph: {' -> '.join(trail + [n, m])}")
            elif colour.get(m, WHITE) == WHITE:
                dfs(m, trail + [n])
        colour[n] = BLACK

    dfs(entry, [])

    terminals = {sid for sid, s in steps.items() if s.get("terminal")}
    declared_exit = set(flow.get("exit") or [])
    if terminals != declared_exit:
        rep.err("flow.exit",
                f"declared {sorted(declared_exit)} but terminal steps are {sorted(terminals)}")

    # -- crew confirmation tokens ------------------------------------------
    # Closed, for the same reason the predicate vocabulary is. The voice grammar
    # is BUILT from this list, so a token a step waits for but the file does not
    # declare is a step no microphone can ever satisfy. A one-letter typo -
    # "label verifed" - is otherwise invisible until the crew says the right
    # words at the rack and nothing happens.
    declared_tokens: dict[str, str] = {}
    for i, tok in enumerate(doc.get("crew_tokens") or []):
        if not_a_string(f"crew_tokens[{i}]", tok, rep):
            continue
        key = normalise_token(tok)
        if not key:
            rep.err(f"crew_tokens[{i}]", "empty token - there is nothing to say")
        elif key in declared_tokens:
            rep.err(f"crew_tokens[{i}]",
                    f"'{tok}' is the same spoken phrase as '{declared_tokens[key]}' - "
                    f"a recogniser cannot tell them apart")
        else:
            declared_tokens[key] = tok
    for tok, paths in sorted(used_tokens.items()):
        if normalise_token(tok) not in declared_tokens:
            rep.err(paths[0],
                    f"crew_confirmed token '{tok}' is not declared in crew_tokens - the "
                    f"voice grammar is built from that list, so this step could never be "
                    f"confirmed by speech")
    spoken_used = {normalise_token(t) for t in used_tokens}
    for key, tok in declared_tokens.items():
        if key not in spoken_used:
            rep.warn("crew_tokens", f"'{tok}' is declared but no crew_confirmed waits for it")
    override_cfg = (doc.get("alert_policy") or {}).get("crew_override") or {}
    for otok in override_cfg.get("tokens") or []:
        if isinstance(otok, str) and normalise_token(otok) in declared_tokens:
            rep.err("crew_tokens",
                    f"'{declared_tokens[normalise_token(otok)]}' collides with "
                    f"alert_policy.crew_override token '{otok}' - the same words cannot "
                    f"both confirm a step and silence an alert")

    # -- the capability contract, the whole point --------------------------
    missing = used_caps - declared_caps
    if missing:
        rep.err("requires.capabilities",
                f"procedure uses predicates needing {sorted(missing)}, which the file does not declare")
    unused = declared_caps - used_caps
    if unused:
        rep.warn("requires.capabilities",
                 f"declared but never used: {sorted(unused)} - loading perception components "
                 f"nothing asks for costs frame budget on a 25 W part")

    used_classes = {entities[n]["detector_class"] for n in entities if "detector_class" in entities[n]}
    dead = declared_classes - used_classes
    if dead:
        rep.warn("requires.detector_classes", f"declared but no entity uses: {sorted(dead)}")

    # -- declared estimate vs the arithmetic -------------------------------
    # Caught a 4x discrepancy in the first file this validator ever ran on.
    # A header number nobody re-derives is a number that will be wrong on stage.
    nominal_path = sum((s.get("duration") or {}).get("nominal_s", 0)
                       for sid, s in steps.items() if sid not in off_nominal)
    est = doc["procedure"].get("estimated_duration_s")
    if est and nominal_path and abs(est - nominal_path) / nominal_path > 0.25:
        rep.err("procedure.estimated_duration_s",
                f"declares {est} s but the nominal path sums to {nominal_path} s "
                f"({est / nominal_path:.1f}x) - one of the two is wrong")

    # -- objects list vs predicates actually used --------------------------
    # The objects list drives GUI highlighting and evidence capture. If it
    # drifts from the verification logic the operator is shown the wrong thing.
    for sid, s in steps.items():
        listed = set(s.get("objects") or [])
        referenced = step_entity_refs.get(sid, set())
        for missing_obj in sorted(referenced - listed):
            rep.warn(f"flow.steps[{sid}].objects",
                     f"'{missing_obj}' is used in this step's logic but not listed - "
                     f"it will not be highlighted or captured as evidence")

    return {
        "steps": steps, "groups": groups, "entities": entities, "zones": zones,
        "used_caps": used_caps, "used_motions": used_motions, "terminals": terminals,
        "off_nominal": off_nominal, "nominal_path_s": nominal_path,
    }


# --------------------------------------------------------------------------
def summarise(doc: dict, ctx: dict) -> None:
    p = doc["procedure"]
    steps = ctx["steps"]
    print(f"\n  {p['id']} rev {p.get('revision')} - {p['title']}")
    print(f"  {len(steps)} steps, {len(ctx['groups'])} unordered group(s), "
          f"{len(ctx['entities'])} entities, {len(ctx['zones'])} zones")

    nominal = ctx["nominal_path_s"]
    print(f"  nominal path duration: {nominal} s ({nominal / 60:.1f} min), "
          f"declared estimate {p.get('estimated_duration_s')} s")

    print("\n  HSMM duration priors (Gaussian, seconds)")
    print("  " + "-" * 66)
    print(f"  {'step':<10} {'type':<13} {'min':>5} {'nom':>5} {'sigma':>6} {'max':>5}  crit  path")
    for sid, s in steps.items():
        d = s.get("duration") or {}
        path = "off-nom" if sid in ctx["off_nominal"] else ""
        print(f"  {sid:<10} {s.get('type', ''):<13} {d.get('min_s', 0):>5} {d.get('nominal_s', 0):>5} "
              f"{d.get('sigma_s', 0):>6} {d.get('max_s', 0):>5}  "
              f"{'*' if s.get('critical') else ' ':<4}  {path}")

    print(f"\n  capabilities actually exercised: {sorted(ctx['used_caps'])}")
    print(f"  motion classes referenced:       {sorted(ctx['used_motions'])}")

    hazards = sum(len(s.get("invariants") or []) for s in steps.values())
    branches = sum(len(s.get("branch") or []) for s in steps.values())
    wrongobj = sum(1 for s in steps.values() if s.get("on_wrong_object"))
    print(f"  hazard invariants: {hazards}   branches: {branches}   "
          f"wrong-object guards: {wrongobj}   terminals: {sorted(ctx['terminals'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a PARIKSHAK PDL procedure file.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--quiet", action="store_true", help="suppress the summary block")
    ap.add_argument("--build", type=Path, default=None,
                    help="also check every requirement against this perception build "
                         "manifest - the zero-retraining check")
    args = ap.parse_args()

    build = None
    if args.build is not None:
        from parikshak.pdl.build import BuildError, PerceptionBuild
        try:
            build = PerceptionBuild.load(args.build)
        except BuildError as exc:
            print(f"  ERROR: {exc}")
            return 2

    worst = 0
    for path in args.files:
        print(f"\n=== {path} ===")
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print("  ERROR: file not found")
            worst = max(worst, 2)
            continue
        except yaml.YAMLError as exc:
            print(f"  ERROR: YAML will not parse\n{exc}")
            worst = max(worst, 2)
            continue

        rep = Report()
        ctx = check(doc, rep)
        if build is not None and ctx:
            from parikshak.pdl.build import check_build
            check_build(doc, build, rep)

        for e in rep.errors:
            print(f"  ERROR   {e}")
        for w in rep.warnings:
            print(f"  WARN    {w}")

        if rep.errors:
            print(f"\n  FAILED - {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
            worst = max(worst, 1)
        else:
            if ctx and not args.quiet:
                summarise(doc, ctx)
            verdict = "PASS"
            if rep.warnings and args.strict:
                verdict = "FAILED (strict)"
                worst = max(worst, 1)
            print(f"\n  {verdict} - 0 errors, {len(rep.warnings)} warning(s)")

    return worst
