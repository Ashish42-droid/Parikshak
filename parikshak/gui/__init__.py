"""The operator display.

Qt lives in `app.build_window` and nowhere else. Everything the crew sees is
computed in `state.DisplayState` and `overlay.build_overlay`, both of which are
plain data - so the display can be asserted in a test on a machine with no Qt,
and a replayed trace produces the identical screen to a live camera because the
widgets never learn which one they are drawing.
"""

from parikshak.gui.app import (
    AppConfig,
    GuiUnavailable,
    HeadlessDisplay,
    build_window,
    format_clock,
    format_deviation,
    format_step_row,
    format_timeline,
    render_text,
    require_qt,
    supports_unicode,
)
from parikshak.gui.overlay import Marker, OverlayScene, Polyline, build_overlay
from parikshak.gui.state import DeviationRow, DisplayState, StepRow, TimelineEvent
from parikshak.gui.theme import THEME, Theme, contrast_ratio, stylesheet

__all__ = [
    "AppConfig", "DeviationRow", "DisplayState", "GuiUnavailable", "HeadlessDisplay",
    "Marker", "OverlayScene", "Polyline", "StepRow", "THEME", "Theme", "TimelineEvent",
    "build_overlay", "build_window", "contrast_ratio", "format_clock",
    "format_deviation", "format_step_row", "format_timeline", "render_text",
    "require_qt", "stylesheet", "supports_unicode",
]
