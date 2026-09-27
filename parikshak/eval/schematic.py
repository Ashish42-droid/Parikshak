"""A to-scale drawing of the rack face, from one BeliefFrame.

Traces carry belief, not pixels, so a replay has no camera image to show. This
draws what the engine was told instead: the rack face seen from the crew's side,
the procedure's zones by their X-Y footprint, every object where perception put
it - with its state, and filled when a hand holds it - the wrists, zones that
were occluded, and a banner when the rack frame was lost.

Fixtures, not runtime: used by the failure-mode reel and by anyone explaining a
replay. It never draws what the frame does not contain. An object with no rack
position is listed as unlocated, not drawn at its last known place, for the same
reason the live overlay refuses to draw over a lost rack lock.
"""

from __future__ import annotations

import numpy as np

#: Rack-face extent drawn, in metres: generous enough for every CSP-1 zone.
DEFAULT_EXTENT = ((-0.72, 0.72), (-0.62, 0.47))

_INK = (40, 40, 40)
_ZONE = (150, 150, 150)
_ACTIVE = (40, 120, 230)
_HELD = (30, 140, 60)
_LOST = (40, 60, 200)
_HAND = (170, 90, 20)
_PAPER = (250, 250, 247)


def render_schematic(procedure, frame, width: int = 960, height: int = 540, *,
                     active_objects=(), extent=DEFAULT_EXTENT) -> np.ndarray:
    """A BGR image of what `frame` says about the rack."""
    import cv2

    img = np.full((height, width, 3), _PAPER, dtype=np.uint8)
    (x0, x1), (y0, y1) = extent
    margin = 28
    scale = min((width - 2 * margin) / (x1 - x0), (height - 2 * margin - 24) / (y1 - y0))

    def px(x: float, y: float) -> tuple[int, int]:
        return (int(margin + (x - x0) * scale), int(margin + 24 + (y1 - y) * scale))

    def text(s: str, at, colour=_INK, size=0.45, bold=1) -> None:
        cv2.putText(img, s, at, cv2.FONT_HERSHEY_SIMPLEX, size, colour, bold, cv2.LINE_AA)

    text("RACK FACE (crew view, +X right, +Y up) - drawn from belief, not a camera",
         (margin, 20), _INK, 0.5)

    for name, zone in procedure.zones.items():
        if zone.kind != "box":
            continue
        (zx0, zy0, _), (zx1, zy1, _) = zone.lo, zone.hi
        p, q = px(zx0, zy1), px(zx1, zy0)
        occluded = frame.zone_occlusion(name) > 0.2
        if occluded:
            overlay = img.copy()
            cv2.rectangle(overlay, p, q, (200, 200, 200), -1)
            img[:] = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        cv2.rectangle(img, p, q, _ZONE, 1)
        text(name + (" (occluded)" if occluded else ""), (p[0] + 4, p[1] + 14), _ZONE, 0.38)

    unlocated: list[str] = []
    highlight = set(active_objects)
    for name, ob in frame.objects.items():
        entity = procedure.entities.get(name)
        label = entity.spoken if entity is not None else name
        if ob.state:
            label += f" [{ob.state}]"
        if ob.pos_rack is None:
            if ob.visible or ob.occluded:
                unlocated.append(label + (" hidden" if ob.occluded else ""))
            continue
        c = px(ob.pos_rack[0], ob.pos_rack[1])
        colour = _LOST if ob.occluded or not ob.visible else (_HELD if ob.held_by else _INK)
        cv2.circle(img, c, 7, colour, -1 if ob.held_by else 2, cv2.LINE_AA)
        if name in highlight:
            cv2.circle(img, c, 12, _ACTIVE, 2, cv2.LINE_AA)
        text(label + (f" held ({ob.held_by})" if ob.held_by else ""), (c[0] + 10, c[1] - 6),
             colour, 0.4)

    for side, hand in frame.hands.items():
        if hand.present and hand.wrist_rack is not None:
            c = px(hand.wrist_rack[0], hand.wrist_rack[1])
            cv2.drawMarker(img, c, _HAND, cv2.MARKER_CROSS, 14, 2)
            detail = f"{side} hand" + (f": {hand.grasp_type} {hand.contact_with}"
                                       if hand.contact_with else "")
            text(detail, (c[0] + 10, c[1] + 14), _HAND, 0.38)

    if not frame.frame_lock:
        cv2.rectangle(img, (0, height - 40), (width, height), (60, 60, 200), -1)
        text("NO RACK LOCK - positions unknown", (margin, height - 14), (255, 255, 255), 0.6, 2)
    elif unlocated:
        text("unlocated: " + ", ".join(unlocated), (margin, height - 12), _LOST, 0.4)
    text(f"t = {frame.t_mono:6.1f} s   motion: {frame.motion.cls} {frame.motion.conf:.2f}",
         (width - 290, 20), _INK, 0.45)
    return img
