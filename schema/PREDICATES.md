# PARIKSHAK Predicate Vocabulary — PDL v1.0

This is **the contract between the learned layer and the symbolic layer.**

A procedure author may use *only* the predicates below. The loader rejects anything
else before a single frame is processed. This is what makes "author a new procedure
without retraining" an engineering guarantee rather than a slogan: the vocabulary is
closed, every entry is computed by a deployed perception component, and the validator
proves at load time that the deployed model can answer every question the file asks.

`tests/test_predicates_doc.py` checks this document against the validator's own
vocabulary table. If the code gains a predicate, a combinator or a capability
dependency that is not written down here, the gate fails. The contract cannot quietly
drift behind the implementation.

---

## 1. Evaluation semantics

Every predicate returns a **probability in [0, 1]**, never a bool. The World State
Belief is probabilistic, so the procedure logic must be too.

Combinators (chosen for calibration stability, not elegance):

| Form | Result | Why |
|---|---|---|
| `all: [a, b, c]` | `min(p_a, p_b, p_c)` | Product collapses toward 0 with clause count; min does not. A 6-clause step should not be penalised for being specific. |
| `any: [a, b]` | `max(p_a, p_b)` | Noisy-or over-counts correlated evidence from one camera. |
| `not: a` | `1 - p_a` | |
| `hold_for: {seconds: N, expr: a}` | `p_a` over the trailing N-second window, taken at the `alert_policy.hold_for_tolerance` quantile (default 0.2) rather than the strict minimum | This is the persistence window, expressed declaratively instead of buried in engine code. The quantile tolerates isolated frames of detector noise: a strict minimum makes a clause true on 93% of frames hold a seven-frame window only 60% of the time. A genuine change of state disagrees on every frame after it, so it is still caught, a frame or two later. |
| `occurred: a` | `max` of `p_a` over the current step so far | A momentary action is over by the time its consequence is visible. "Press start" is satisfied by a moment, not a state; without this every momentary action would have to be caught in its single frame. Scoped to the step, so a motion seen during step 4 cannot satisfy step 9. |

A predicate whose supporting evidence is **occluded** returns `null`, not `0`.
`null` propagates: any step whose verification tree evaluates to `null` is marked
`UNVERIFIED`, never `SKIPPED`. **This is the single most important rule in the system.**
An occluded camera must never manufacture a deviation.

### 1.1 How `null` is implemented

The engine represents a truth value as an interval `[lo, hi]` over the probability
(`parikshak/engine/truth.py`). A known probability `p` is `[p, p]`; `null` is `[0, 1]`.
The combinators above apply to both ends. On fully known input this is exactly the
table above — `lo == hi == min` — so it is a generalisation, not a redefinition.

It exists because strict null-propagation has a pathology. In
`all: [state_is(latch, closed) = 0.02, in_zone(vial, glovebox) = null]` the first
clause is confidently false, so the conjunction is confidently false whatever the
occluded clause says — but strict propagation reports it `UNVERIFIED`, inflating the
unverified rate with cases that were actually decided. With intervals it is
`[0.0, 0.02]`: refuted.

The rule this document calls the most important still holds, and is tested directly:
widening an interval can only move a value *out of* "satisfied" and *into*
`UNVERIFIED`, never into "refuted". Losing information cannot create a deviation.

Thresholds are set in `alert_policy`, never inline in a step.

---

## 2. Reference frame

All geometry is evaluated in the **rack frame**, recovered per-frame from AprilTag
36h11 fiducials on the rack face. Gravity is never referenced, sampled, or assumed.

- origin — centroid of the rack fiducial cluster
- `+X` — along the rack face, to the crew's right when facing the rack
- `+Y` — along the rack face, orthogonal to X (named "up" **for readability only**;
  it carries no gravitational meaning)
- `+Z` — out of the rack face, toward the crew

Units: metres, seconds, degrees. Angles are unsigned unless stated.

Because zones and distances are defined in this frame, rotating the rack, the camera,
or the crew member changes nothing numerically. Orientation-agnosticism is a property
of the coordinate system, not of a learned model.

---

## 3. The vocabulary

### 3.1 Presence — requires capability `object_detection`

| Predicate | Signature | Meaning |
|---|---|---|
| `visible` | `(entity, min_conf=0.5)` | Entity detected and tracked this frame. A detection below `min_conf` is **unknown**, not false: it is not a sighting the author accepts, and it is not an absence either. Use `absent` to require that something is not there. |
| `absent` | `(entity)` | Entity confidently not present in any camera. Distinct from occluded. |
| `count_of` | `(entity, n)` | Exactly `n` instances tracked. Used for consumable checks. |

### 3.2 Manipulation — requires `hand_landmarks` + `hand_object_contact`

| Predicate | Signature | Meaning |
|---|---|---|
| `grasped` | `(entity, hand=any)` | Contact head reports `grasp` or `manipulate`. `hand` ∈ `left`, `right`, `any`, `both`. |
| `released` | `(entity)` | Previously grasped, now `none`, and entity is stable - centroid within 3 cm over the 2 s after it was let go. Until those 2 s have passed it is **undecided**, not false: the object may yet stay put, and "not settled yet" is not evidence it was never released. Never having seen it held counts against it only if hands were tracked at some point in the run; with no hand model it is **unknown**. |
| `contacting` | `(a, b)` | Two entities in contact (not necessarily hand-mediated). |
| `hand_in_zone` | `(hand, zone)` | Wrist landmark inside the named zone. Also needs `rack_frame_extrinsics`: a zone is a rack-frame volume. |

### 3.3 Geometry — requires `rack_frame_extrinsics`

| Predicate | Signature | Meaning |
|---|---|---|
| `inside` | `(a, container, margin_m=0.0)` | Centroid of `a` within the container. A container may be a zone, or an entity that declares `extent_m`. |
| `near` | `(a, b, dist_m)` | Centroid separation ≤ `dist_m`. |
| `aligned` | `(a, b, tol_deg)` | Principal axes within `tol_deg`. Needs 6-DoF pose for both. |
| `in_zone` | `(entity, zone)` | Entity centroid inside a declared zone. |
| `stable` | `(entity, seconds, tol_m=0.02)` | Centroid moved < `tol_m` over the window, measured as the distance between the mean positions of the window's first and second halves - so per-frame detector jitter is averaged away rather than maximised, and a longer window gives a more certain answer. If the object moved but was let go inside the window, the answer is **undecided**, not false - the motion may be the crew carrying it there - and becomes a real verdict once a whole window has passed since the let-go. A window with frames hidden inside it can **confirm** stability - each half at least half observed, and the drift within tolerance - but never refute it: a partial window that shows motion is unknown, because losing information must never create a deviation. In microgravity this is how "stowed" is distinguished from "drifting". |

### 3.4 Object state — requires `object_state`

| Predicate | Signature | Meaning |
|---|---|---|
| `state_is` | `(entity, state)` | State classifier head output. `state` must be declared in the entity's `states` list, and the deployed build's state head for that detector class must emit it (see §7). |

### 3.5 Motion — requires `motion_tcn`

| Predicate | Signature | Meaning |
|---|---|---|
| `motion_is` | `(motion_class, min_conf=0.6)` | TCN's active class this window. False when the TCN names a different class at or above `min_conf`; **unknown** below `min_conf`, whichever class it names - a classifier that is not confident is not claiming anything, and a run with no motion model at all reads unknown rather than false. |
| `motion_count` | `(motion_class, n, window_s=60)` | Class fired `n` times in the window. For "shake ten times". Counts transitions into the class, not frames spent in it. Short of `n` is false only if some frame in the window carries a confident motion label (confidence >= 0.5); otherwise **unknown**. |

### 3.6 Crew posture — requires `body_pose`

| Predicate | Signature | Meaning |
|---|---|---|
| `body_restrained` | `(zone)` | Both ankles inside the restraint zone, in rack coordinates. Works inverted. |
| `body_in_zone` | `(zone)` | Pelvis root inside zone. |

### 3.7 Engine — no perception capability

| Predicate | Signature | Meaning |
|---|---|---|
| `elapsed_since` | `(step_id, seconds)` | ≥ `seconds` since that step completed. |
| `crew_confirmed` | `(token)` | Crew said or clicked the confirmation token. Logged with source. Requires capability `crew_confirm`, and `token` must be declared in `crew_tokens` (§6). |
| `occluded` | `(zone)` | Occlusion mask covers the zone. Used to gate, not to alert. |
| `assert_true` | `(flag)` | A flag set by an earlier step's `effects: [assert: flag]` and not since retracted. Flags come only from steps the engine accepted — perception can never assert one. Accepts the scalar form `{assert_true: flag}`. |

---

## 4. Capability implication table

The validator derives required capabilities from the predicates actually used and
checks them against `requires.capabilities`. Authors cannot silently depend on a
perception component the deployed build does not ship. A predicate listed in more than
one row needs the union of those capabilities.

| Predicate | Implies capability |
|---|---|
| `visible`, `absent`, `count_of` | `object_detection` |
| `grasped`, `released`, `contacting` | `hand_landmarks`, `hand_object_contact` |
| `hand_in_zone` | `hand_landmarks`, `hand_object_contact`, `rack_frame_extrinsics` |
| `inside`, `near`, `aligned`, `in_zone`, `stable` | `rack_frame_extrinsics` |
| `aligned` | `object_pose_6dof` |
| `state_is` | `object_state` |
| `motion_is`, `motion_count` | `motion_tcn` |
| `body_restrained`, `body_in_zone` | `body_pose`, `rack_frame_extrinsics` |
| `crew_confirmed` | `crew_confirm` |
| `elapsed_since`, `occluded`, `assert_true` | — |

---

## 5. Adding a new predicate

A new predicate is a **perception change**, not a procedure change, and follows the
model release cycle: implement the estimator, publish calibration on the held-out set,
bump `min_model_version`, add the capability to the build manifest (§7), add the row
above. Procedure authors never do this.

This boundary is the whole design. Everything on the authoring side of it costs an
afternoon. Everything on the perception side of it costs a data collection campaign.
Keep the line visible.

---

## 6. Crew tokens

Every phrase a `crew_confirmed` predicate waits for is declared once, at the top level:

```yaml
crew_tokens: ["label verified", "fault logged", "acknowledged"]
```

The list is closed for the same reason the predicate vocabulary is. The voice grammar
is **built from it**, so a token a step waits for but the file does not declare is a
step no microphone can ever satisfy. The validator rejects:

- a `crew_confirmed` token that is not declared — the one-letter typo that would
  otherwise be discovered at the rack, when the crew says the right words and nothing
  happens;
- two declared tokens that are the same spoken phrase once case, punctuation and
  spacing are removed, because a recogniser cannot tell them apart;
- a token that collides with an `alert_policy.crew_override` token, because the same
  words cannot both confirm a step and silence an alert.

A declared token nothing waits for is a warning.

---

## 7. The deployed build

A procedure names the perception build it targets in `requires.min_model_version`. That
build is a file, `builds/<name>.json`, listing exactly what one set of trained weights
provides: detector classes, capabilities, the motion registry, and — per detector
class — the states its state classifier emits.

`python tools/validate_procedure.py <file> --strict --build builds/<name>.json` checks
every requirement against it. `python tools/onboard.py <file>` prints the verdict.

What changes without retraining, and what does not:

| Free — a procedure edit | Costs a data campaign — a model release |
|---|---|
| steps, order, unordered groups, branches | a new detector class |
| predicates over things the build already sees | a new state for an existing class |
| zones, durations, thresholds, alert wording | a new motion class |
| crew tokens | a new capability (e.g. 6-DoF pose) |

State vocabularies belong in the right-hand column and are easy to miss. Asking
`state_is(latch, half_open)` needs a retrained state head exactly as surely as a new
object needs a retrained detector, which is why the build lists them per class.
