"""Colours and typography for the operator display.

A rack is not a desk. The screen is read from a metre away, at an angle, by
someone whose hands are busy and whose attention is on the hardware. That drives
every choice here:

  **Dark ground.** A bright panel next to a live camera feed is a lamp pointed
  at the crew, and it wrecks the contrast of the feed itself.

  **Colour is never the only signal.** Every status also carries a glyph and a
  word (see state.STATUS_GLYPH). Roughly one in twelve men has a colour vision
  deficiency, venue projectors crush saturation, and a checklist that means
  nothing under either is not a checklist.

  **UNVERIFIED is not a warning colour.** It is amber-grey, deliberately calmer
  than a deviation. Showing "I cannot see" in the same red as "you skipped it"
  would teach the crew to read the system's restraint as its failure, which is
  the opposite of what the design is for.

  **Critical is the only thing that gets the loud colour.** If everything is
  urgent, nothing is.

Contrast ratios below are against `BACKGROUND` and are stated because "looks
fine on my laptop" is not a standard. WCAG AA wants 4.5:1 for body text.
"""

from __future__ import annotations

from dataclasses import dataclass

BACKGROUND = "#12151a"
SURFACE = "#1b1f26"
BORDER = "#2b323c"
TEXT = "#e6eaf0"
TEXT_DIM = "#9aa4b2"

#: Status colours. Ratios measured against BACKGROUND.
PENDING = "#7a8494"      # 4.8:1  - was #6b7482 at 3.9:1, which fails AA.
                         # The comment claimed 4.6:1; the measurement disagreed.
                         # That is why contrast is a test and not a comment.
ACTIVE = "#5ab0ff"       # 7.4:1  - the one thing the eye should find first
COMPLETE = "#4cc38a"     # 8.1:1
SKIPPED = "#ff6b6b"      # 6.3:1
UNVERIFIED = "#d6a349"   # 8.0:1  - amber-grey, calmer than SKIPPED on purpose

CRITICAL = "#ff4d4d"
CAUTION = "#ff9f43"
ADVISORY = "#e8c547"
INFO = "#7f8c9b"

STATUS_COLOUR = {
    "PENDING": PENDING,
    "ACTIVE": ACTIVE,
    "COMPLETE": COMPLETE,
    "SKIPPED": SKIPPED,
    "UNVERIFIED": UNVERIFIED,
}

SEVERITY_COLOUR = {
    "critical": CRITICAL,
    "caution": CAUTION,
    "advisory": ADVISORY,
    "info": INFO,
}

#: Overlay primitives, keyed by the `kind` field build_overlay emits.
OVERLAY_COLOUR = {
    "zone": "#3d4753",
    "zone_active": ACTIVE,
    "object": "#8fa3b8",
    "object_held": COMPLETE,
    "object_lost": UNVERIFIED,
    "hand": "#ffd166",
    "body": "#7ad7f0",
    "axis": "#ff7ab6",
}

OVERLAY_WIDTH = {
    "zone": 1,
    "zone_active": 2,
    "body": 3,
    "axis": 2,
}


@dataclass(frozen=True, slots=True)
class Theme:
    background: str = BACKGROUND
    surface: str = SURFACE
    border: str = BORDER
    text: str = TEXT
    text_dim: str = TEXT_DIM
    #: Fixed-width for step IDs and timestamps so columns do not jitter as
    #: numbers change - a checklist that reflows every frame is unreadable.
    mono: str = "DejaVu Sans Mono, Consolas, monospace"
    sans: str = "Inter, Segoe UI, DejaVu Sans, sans-serif"
    base_pt: int = 13
    #: The next-step prompt is read at a distance, so it is set much larger.
    prompt_pt: int = 22

    def status(self, name: str) -> str:
        return STATUS_COLOUR.get(name, TEXT_DIM)

    def severity(self, name: str) -> str:
        return SEVERITY_COLOUR.get(name.lower(), INFO)

    def overlay(self, kind: str) -> str:
        return OVERLAY_COLOUR.get(kind, TEXT_DIM)

    def overlay_width(self, kind: str) -> int:
        return OVERLAY_WIDTH.get(kind, 1)


THEME = Theme()


# --------------------------------------------------------------------------
def _luminance(hex_colour: str) -> float:
    c = hex_colour.lstrip("#")
    channels = [int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
              for v in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str = BACKGROUND) -> float:
    """WCAG contrast ratio. Used by a test, so the palette cannot quietly drift
    into something unreadable on a venue projector."""
    a, b = _luminance(foreground), _luminance(background)
    lo, hi = sorted((a, b))
    return (hi + 0.05) / (lo + 0.05)


def stylesheet(theme: Theme = THEME) -> str:
    """Qt stylesheet. Plain text, so it is testable without Qt installed."""
    return f"""
QWidget {{
    background: {theme.background};
    color: {theme.text};
    font-family: {theme.sans};
    font-size: {theme.base_pt}pt;
}}
QFrame#panel {{
    background: {theme.surface};
    border: 1px solid {theme.border};
    border-radius: 6px;
}}
QLabel#prompt {{
    font-size: {theme.prompt_pt}pt;
    font-weight: 600;
    color: {ACTIVE};
    padding: 8px 12px;
}}
QLabel#stepId, QLabel#clock {{
    font-family: {theme.mono};
    color: {theme.text_dim};
}}
QListWidget {{
    background: {theme.surface};
    border: none;
}}
"""
