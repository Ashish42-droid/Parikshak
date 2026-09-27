"""The operator display: PySide6 over the view model.

    +---------------------------------+---------------------------+
    |  CAUTION: SKIP S08 - latch ...  |  00:46.0                  |
    |   live view + rack overlay      |  CHECKLIST                |
    |                                 |   ✓ S01 Restrain          |
    |                                 |   ▸ S06 Insert cartridge  |
    +---------------------------------+---------------------------+
    |  NEXT: Close the glovebox latch.                            |
    |  TIMELINE  ====|==x=====|====^                              |
    |  [unit idle] [override] [ say...    ]  [Export] [Ack]       |
    |  [>] [step] [1x] ------o---------- 46.0 / 140.8 s  (replay) |
    +---------------------------------+---------------------------+

Qt in this file and nowhere else. Every widget reads a `DisplayState` and draws
it; none of them compute anything. That is why the panels can be asserted in a
test on a machine with no Qt installed, and why replaying a trace and running a
camera produce identical screens - the widgets cannot tell the difference
because they never see the source. `run_session_window` drives a LiveSession
and a ReplaySession through the same code; only a session that can seek gets
the replay controls.

PySide6 rather than a browser, deliberately. A web dashboard means a browser
runtime, a local server and a port, which undercuts the "standalone offline
box" claim the demo opens by making. Qt runs in the same process as GStreamer
with no IPC.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from parikshak.gui.overlay import OverlayScene
from parikshak.gui.state import DisplayState, StepRow, TimelineEvent
from parikshak.gui.theme import THEME, Theme, stylesheet


class GuiUnavailable(RuntimeError):
    """PySide6 is not installed. The engine, eval and replay paths do not need
    it, so this is a normal condition on a build machine - not a failure."""


def require_qt():
    """Import PySide6 or explain why the GUI cannot start.

    Lazy, so `import parikshak.gui` works everywhere and only opening a window
    needs the dependency.
    """
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - optional extra
        raise GuiUnavailable(
            "PySide6 is not installed: pip install 'parikshak[gui]'. "
            "The engine, replay and eval paths do not need it.") from exc
    return QtCore, QtGui, QtWidgets


# --------------------------------------------------------------------------
# formatting - pure, so the panels can be asserted without Qt
# --------------------------------------------------------------------------
def supports_unicode(stream=None) -> bool:
    """Can this output encode the status glyphs?

    Checked rather than assumed. A cp1252 Windows console and a plain serial
    terminal both raise on U+2713, and a display that dies mid-run with a
    UnicodeEncodeError is worse than one that draws a plus sign.
    """
    import sys
    stream = stream if stream is not None else sys.stdout
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "✓▸✗·".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def format_clock(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}.{int((t % 1) * 10)}"


def format_step_row(row: StepRow, *, ascii_only: bool = False) -> str:
    """One checklist line: glyph, id, name, and the timing that matters.

    The glyph leads so status is legible before any colour is perceived.
    """
    elapsed = f"{row.elapsed:5.1f}s" if row.elapsed is not None else "     -"
    flag = "!" if row.critical else " "
    note = ""
    if row.was_ever_unverified:
        note = " (was blind)"
    elif row.unverifiable and row.status in ("PENDING", "ACTIVE", "UNVERIFIED"):
        note = " (not verifiable here)"
    glyph = row.ascii_glyph if ascii_only else row.glyph
    return f"{glyph} {flag}{row.step_id:<5} {row.name:<38} {elapsed}{note}"


def format_deviation(row) -> str:
    """Feed entry. The reason is on the line, not behind a hover - "deviation
    detected" is useless at a rack."""
    return (f"[{format_clock(row.t)}] {row.severity.upper():<8} {row.kind:<13} "
            f"{row.step_id:<5} {row.reason}")


def deviation_key(row) -> str:
    """How an acknowledgement names a deviation - the same `step:kind` form
    DisplayState.from_engine accepts."""
    return f"{row.step_id}:{row.kind}"


def format_timeline(events, width: int = 60, duration: float = 0.0) -> str:
    """A one-line ASCII timeline, used by the headless view and the tests.

    Deviations are drawn last so they are never overwritten by a step boundary
    landing in the same cell.
    """
    if not events:
        return "." * width
    span = max(duration, max(e.t for e in events), 1e-6)
    cells = ["."] * width
    for event in events:
        i = min(width - 1, int(event.t / span * (width - 1)))
        if not event.is_deviation:
            cells[i] = "|"
    for event in events:
        if event.is_deviation:
            i = min(width - 1, int(event.t / span * (width - 1)))
            cells[i] = "!" if event.severity == "critical" else "x"
    return "".join(cells)


@dataclass(frozen=True, slots=True)
class TimelineMark:
    """One event placed on the timeline strip, in pixels along its width."""

    x: float
    kind: str
    severity: str
    label: str
    is_deviation: bool


def timeline_marks(events: Iterable[TimelineEvent], duration: float,
                   width: float) -> list[TimelineMark]:
    """Every event on ONE scale from 0 to `width`, deviations last.

    The scale runs to the longer of the run's duration and the last event, so
    a mark can never be placed past the end of the strip. Deviations sort last
    so a painter drawing in order puts them over any step tick at the same x -
    the same rule as the text timeline.
    """
    events = list(events)
    span = max(duration, max((e.t for e in events), default=0.0), 1e-6)
    marks = [TimelineMark(min(width, max(0.0, e.t / span * width)), e.kind, e.severity,
                          f"{e.step_id} {e.kind}", e.is_deviation) for e in events]
    return sorted(marks, key=lambda m: m.is_deviation)


def banner_text(state: DisplayState, acknowledged: Iterable[str] = ()) -> tuple[str, str | None]:
    """(text, severity) for the status banner: the worst deviation nobody has
    acknowledged yet. Acknowledging silences the banner, never the record."""
    acked = set(acknowledged)
    live = [d for d in state.deviations
            if not d.acknowledged and deviation_key(d) not in acked]
    if not live:
        return "No open deviations", None
    worst = min(live, key=lambda d: (d.rank, -d.t))
    more = f"   (+{len(live) - 1} more)" if len(live) > 1 else ""
    return (f"{worst.severity.upper()}: {worst.kind} {worst.step_id} - {worst.reason}{more}",
            worst.severity)


def export_state(state: DisplayState, directory: str | Path,
                 acknowledged: Iterable[str] = ()) -> Path:
    """The Export button: what is on screen, as JSON, named by run and time."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = state.as_dict()
    payload["acknowledged"] = sorted(set(acknowledged))
    path = out / f"{state.run_id}_{state.t:07.1f}s.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return path


def letterbox(image_w: int, image_h: int, view_w: int, view_h: int
              ) -> tuple[float, float, float]:
    """(scale, x offset, y offset) that fits an image into a view unstretched.

    The overlay is computed in IMAGE pixels. Drawing the image stretched to the
    widget while drawing the overlay unscaled puts every zone and marker in the
    wrong place the moment the window is resized - so both go through this one
    transform.
    """
    if image_w <= 0 or image_h <= 0 or view_w <= 0 or view_h <= 0:
        return 1.0, 0.0, 0.0
    scale = min(view_w / image_w, view_h / image_h)
    return scale, (view_w - image_w * scale) / 2.0, (view_h - image_h * scale) / 2.0


def render_text(state: DisplayState, *, width: int = 78,
                ascii_only: bool | None = None) -> str:
    """The whole display as text.

    Not a fallback for a broken GUI - it is how the display is asserted in
    tests, and how a run can be watched over a serial console on a box with no
    monitor attached.
    """
    if ascii_only is None:
        ascii_only = not supports_unicode()
    lines = [
        "=" * width,
        f" {state.procedure_id}  {state.procedure_title}",
        f" run {state.run_id}   t={format_clock(state.t)}   "
        f"{'RACK LOCKED' if state.frame_lock else 'NO RACK LOCK'}   [{state.source}]",
        "=" * width,
        "",
        f" NEXT: {state.prompt}" if state.prompt else " NEXT: -",
        "",
    ]
    for row in state.steps:
        lines.append(" " + format_step_row(row, ascii_only=ascii_only))
    lines += ["", f" {state.status_line()}", ""]
    lines.append(" " + format_timeline(state.timeline, width - 2, state.t))

    if state.unverified_notice:
        lines += ["", f" {state.unverified_notice}"]
    feed = state.feed()
    if feed:
        lines += ["", " DEVIATIONS"]
        lines += ["  " + format_deviation(d) for d in feed]
    lines += ["", "=" * width]
    return "\n".join(lines)


# --------------------------------------------------------------------------
@dataclass
class AppConfig:
    theme: Theme = THEME
    window_title: str = "PARIKSHAK"
    show_overlay: bool = True
    #: Refresh interval. The engine runs at 10 Hz; redrawing faster only burns
    #: power that the 25 W budget does not have to spare.
    refresh_ms: int = 100
    export_dir: Path | None = None


class HeadlessDisplay:
    """The display without a display.

    Used by `replay --view`, by anyone on a serial console, and by the tests.
    Same `DisplayState`, same formatting functions, no Qt - which is what makes
    "the screen shows X" an assertable statement.
    """

    def __init__(self, config: AppConfig | None = None,
                 sink: Callable[[str], None] = print) -> None:
        self.config = config or AppConfig()
        self.sink = sink
        self.frames = 0
        self.last: DisplayState | None = None

    def update(self, state: DisplayState, scene: OverlayScene | None = None) -> None:
        self.frames += 1
        self.last = state

    def render(self) -> str:
        return render_text(self.last) if self.last else "(no state yet)"

    def show(self) -> None:
        self.sink(self.render())

    def export(self, path: str | Path) -> Path:
        """The export button, headless. Writes what is on screen as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = self.last.as_dict() if self.last else {}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        return p


# --------------------------------------------------------------------------
# Qt widgets
# --------------------------------------------------------------------------
SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0)


def build_window(config: AppConfig | None = None, *, phrases: Sequence[str] = (),
                 on_submit: Callable[[str], str] | None = None,
                 seekable: bool = False):  # pragma: no cover - needs Qt
    """Construct the main window. Import-time safe; Qt only loads when called.

    `phrases` become one button each - the procedure's crew tokens and its
    override token - and `on_submit` receives whatever was clicked or typed and
    returns a line for the status bar. `seekable` adds the replay controls.
    The window only draws; `run_session_window` wires the controls.
    """
    QtCore, QtGui, QtWidgets = require_qt()
    config = config or AppConfig()
    theme = config.theme

    class OverlayView(QtWidgets.QWidget):
        """Camera frame (or replay schematic) with the rack overlay on top."""

        def __init__(self) -> None:
            super().__init__()
            self._image: QtGui.QImage | None = None
            self._scene = OverlayScene()
            self.setMinimumSize(640, 360)

        def set_frame(self, image: QtGui.QImage | None, scene: OverlayScene) -> None:
            self._image, self._scene = image, scene
            self.update()

        def paintEvent(self, _event) -> None:
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor(theme.background))
            if self._image is not None:
                scale, ox, oy = letterbox(self._image.width(), self._image.height(),
                                          self.width(), self.height())
                painter.save()
                painter.translate(ox, oy)
                painter.scale(scale, scale)
                painter.drawImage(0, 0, self._image)
                self._paint_scene(painter, 1.0 / max(scale, 1e-6))
                painter.restore()
            if self._scene.notice:
                painter.setPen(QtGui.QColor(theme.status("UNVERIFIED")))
                painter.drawText(self.rect().adjusted(12, 12, -12, -12),
                                 QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter,
                                 self._scene.notice)
            painter.end()

        def _paint_scene(self, painter, unscale: float) -> None:
            """Primitives are in image pixels; pen widths and text stay a
            constant size on screen whatever the scale."""
            for line in self._scene.polylines:
                if line.is_empty:
                    continue
                colour = QtGui.QColor(theme.overlay(line.kind))
                painter.setPen(QtGui.QPen(colour, theme.overlay_width(line.kind) * unscale))
                pts = [QtCore.QPointF(x, y) for x, y in line.points]
                painter.drawPolyline(QtGui.QPolygonF(pts))
            font = painter.font()
            font.setPointSizeF(max(1.0, font.pointSizeF() * unscale))
            painter.setFont(font)
            for marker in self._scene.markers:
                colour = QtGui.QColor(theme.overlay(marker.kind))
                painter.setPen(QtGui.QPen(colour, 2 * unscale))
                painter.drawEllipse(QtCore.QPointF(marker.x, marker.y), 5 * unscale, 5 * unscale)
                if marker.label:
                    painter.drawText(QtCore.QPointF(marker.x + 9 * unscale,
                                                    marker.y - 6 * unscale), marker.label)
                if marker.detail:
                    painter.drawText(QtCore.QPointF(marker.x + 9 * unscale,
                                                    marker.y + 9 * unscale), marker.detail)

    class TimelineView(QtWidgets.QWidget):
        """The run on one horizontal scale: step outcomes as short ticks,
        deviations as tall ones in their severity colour, a playhead at now."""

        def __init__(self) -> None:
            super().__init__()
            self._events: list[TimelineEvent] = []
            self._duration = 0.0
            self._t = 0.0
            self.setMinimumHeight(40)

        def set_state(self, events, duration: float, t: float) -> None:
            self._events, self._duration, self._t = list(events), duration, t
            self.update()

        def paintEvent(self, _event) -> None:
            painter = QtGui.QPainter(self)
            w, h = self.width(), self.height()
            pad = 8
            usable = max(1, w - 2 * pad)
            painter.fillRect(self.rect(), QtGui.QColor(theme.surface))
            painter.setPen(QtGui.QPen(QtGui.QColor(theme.border), 1))
            painter.drawLine(pad, h // 2, pad + usable, h // 2)
            span = max(self._duration, self._t)
            colours = {"completed": theme.status("COMPLETE"), "skipped": theme.status("SKIPPED"),
                       "unverified": theme.status("UNVERIFIED"), "hazard": theme.severity("critical")}
            for mark in timeline_marks(self._events, span, usable):
                x = pad + mark.x
                if mark.is_deviation:
                    colour = theme.severity(mark.severity) if mark.kind == "alert" \
                        else colours.get(mark.kind, theme.text_dim)
                    painter.setPen(QtGui.QPen(QtGui.QColor(colour), 3))
                    painter.drawLine(int(x), 4, int(x), h - 4)
                else:
                    painter.setPen(QtGui.QPen(QtGui.QColor(colours.get(mark.kind, theme.text_dim)), 2))
                    painter.drawLine(int(x), h // 2 - 6, int(x), h // 2 + 6)
            if span > 0:
                x = pad + min(usable, self._t / span * usable)
                painter.setPen(QtGui.QPen(QtGui.QColor(theme.status("ACTIVE")), 2))
                painter.drawLine(int(x), 2, int(x), h - 2)
            painter.end()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(config.window_title)
            self.setStyleSheet(stylesheet(theme))
            self.on_close: Callable[[], None] | None = None

            self.banner = QtWidgets.QLabel("No open deviations")
            self.banner.setWordWrap(True)
            self.view = OverlayView()
            self.prompt = QtWidgets.QLabel("-")
            self.prompt.setObjectName("prompt")
            self.prompt.setWordWrap(True)
            self.notice = QtWidgets.QLabel("")
            self.notice.setWordWrap(True)
            self.timeline = TimelineView()
            self.checklist = QtWidgets.QListWidget()
            self.feed = QtWidgets.QListWidget()
            self.clock = QtWidgets.QLabel("00:00.0")
            self.clock.setObjectName("clock")
            self.heard = QtWidgets.QLabel("")

            say = QtWidgets.QHBoxLayout()
            for phrase in phrases:
                button = QtWidgets.QPushButton(phrase)
                button.clicked.connect(lambda _=False, p=phrase: self._submit(p))
                say.addWidget(button)
            self.entry = QtWidgets.QLineEdit()
            self.entry.setPlaceholderText("say a crew phrase, press Enter")
            self.entry.returnPressed.connect(self._submit_entry)
            say.addWidget(self.entry, 1)
            self.export_button = QtWidgets.QPushButton("Export view")
            self.ack_button = QtWidgets.QPushButton("Acknowledge selected")
            say.addWidget(self.export_button)
            say.addWidget(self.ack_button)

            self.play_button = self.step_button = self.speed = self.slider = None
            self.position = None
            replay = None
            if seekable:
                replay = QtWidgets.QHBoxLayout()
                self.play_button = QtWidgets.QPushButton("Pause")
                self.step_button = QtWidgets.QPushButton("Step")
                self.speed = QtWidgets.QComboBox()
                for s in SPEEDS:
                    self.speed.addItem(f"{s:g}x", s)
                self.speed.setCurrentIndex(SPEEDS.index(1.0))
                self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                self.position = QtWidgets.QLabel("0.0 s")
                for widget in (self.play_button, self.step_button, self.speed):
                    replay.addWidget(widget)
                replay.addWidget(self.slider, 1)
                replay.addWidget(self.position)

            left = QtWidgets.QVBoxLayout()
            left.addWidget(self.banner)
            left.addWidget(self.view, 1)
            left.addWidget(self.prompt)
            left.addWidget(self.notice)
            left.addWidget(self.timeline)
            left.addLayout(say)
            if replay is not None:
                left.addLayout(replay)
            left.addWidget(self.heard)
            right = QtWidgets.QVBoxLayout()
            right.addWidget(self.clock)
            right.addWidget(self.checklist, 2)
            right.addWidget(self.feed, 1)

            root = QtWidgets.QHBoxLayout()
            root.addLayout(left, 3)
            root.addLayout(right, 2)
            central = QtWidgets.QWidget()
            central.setLayout(root)
            self.setCentralWidget(central)

        def _submit(self, text: str) -> None:
            if on_submit is not None:
                self.heard.setText(on_submit(text))

        def _submit_entry(self) -> None:
            text = self.entry.text().strip()
            self.entry.clear()
            if text:
                self._submit(text)

        def selected_deviation(self) -> str | None:
            item = self.feed.currentItem()
            return item.data(QtCore.Qt.UserRole) if item is not None else None

        def update_state(self, state: DisplayState,
                         scene: OverlayScene | None = None,
                         image=None, acknowledged: Iterable[str] = ()) -> None:
            acked = set(acknowledged)
            self.clock.setText(format_clock(state.t))
            self.prompt.setText(state.prompt or "-")
            self.notice.setText(state.unverified_notice)
            self.setWindowTitle(f"{config.window_title} - {state.status_line()}")
            text, severity = banner_text(state, acked)
            self.banner.setText(text)
            colour = theme.severity(severity) if severity else theme.status("COMPLETE")
            self.banner.setStyleSheet(f"color: {colour}; font-weight: 600; padding: 4px 8px;")
            self.timeline.set_state(state.timeline, state.t, state.t)

            self.checklist.clear()
            for row in state.steps:
                item = QtWidgets.QListWidgetItem(format_step_row(row))
                item.setForeground(QtGui.QColor(theme.status(row.status)))
                if row.unverifiable:
                    item.setToolTip(row.unverifiable)
                self.checklist.addItem(item)

            selected = self.selected_deviation()
            self.feed.clear()
            for dev in state.feed():
                key = deviation_key(dev)
                done = dev.acknowledged or key in acked
                item = QtWidgets.QListWidgetItem(format_deviation(dev) + ("   (acknowledged)" if done else ""))
                item.setData(QtCore.Qt.UserRole, key)
                item.setForeground(QtGui.QColor(theme.text_dim if done else theme.severity(dev.severity)))
                self.feed.addItem(item)
                if key == selected:
                    self.feed.setCurrentItem(item)

            self.view.set_frame(image, scene or OverlayScene())

        def feed_texts(self) -> list[str]:
            return [self.feed.item(i).text() for i in range(self.feed.count())]

        def checklist_texts(self) -> list[str]:
            return [self.checklist.item(i).text() for i in range(self.checklist.count())]

        def closeEvent(self, event) -> None:
            if self.on_close is not None:
                self.on_close()
            super().closeEvent(event)

    return MainWindow()


def to_qimage(image):  # pragma: no cover - needs Qt
    """A BGR numpy frame as a QImage that owns its own pixels."""
    _QtCore, QtGui, _QtWidgets = require_qt()
    rgb = image[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


def run_session_window(session, *, seconds: float | None = None, speech=None,
                       trace_path: str | Path | None = None,
                       export_dir: str | Path = Path("runs/exports"),
                       close_at_end: bool | None = None,
                       report: Callable[[str], None] = print) -> int:  # pragma: no cover - needs Qt
    """Open the operator window over a live or replay session and run it.

    Identical window, identical wiring, for both: the gate PLAN.md sets for
    Phase 6. A session that can `seek` (a replay) also gets play/pause, step,
    speed and a seek slider, and by default stays open at its last frame.

    Whichever way the run ends - source exhausted, `seconds` elapsed, window
    closed - it is finished exactly once, so the trace and log are always kept.
    """
    QtCore, _QtGui, QtWidgets = require_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    proc = session.proc
    seekable = hasattr(session, "seek")
    if close_at_end is None:
        close_at_end = not seekable
    phrases = [] if seekable else list(proc.crew_tokens) + list(proc.alert_policy.override_tokens)
    acknowledged: set[str] = set()

    def on_submit(text: str) -> str:
        heard = session.submit(text)
        if heard is None:
            return f'"{text}" is not in the grammar - ignored'
        return f"heard: {heard.token or heard.phrase}"

    window = build_window(AppConfig(window_title=f"PARIKSHAK - {proc.id}"),
                          phrases=phrases, on_submit=on_submit, seekable=seekable)
    if speech is not None:
        session.pump_speech = False
        speech.run_in_background()

    fps = float(getattr(session, "fps", None) or 10.0)
    speed = {"value": 1.0}
    started = time.monotonic()
    finished: dict[str, Any] = {}

    def redraw() -> None:
        tick = session.last
        if tick is None:
            return
        image = to_qimage(tick.image) if tick.image is not None else None
        window.update_state(tick.state, tick.scene, image, acknowledged)
        if seekable:
            window.slider.blockSignals(True)
            window.slider.setValue(session.position)
            window.slider.blockSignals(False)
            window.position.setText(f"{tick.state.t:.1f} / {session.duration_s:.1f} s")

    def finish() -> None:
        if finished:
            return
        timer.stop()
        summary = session.finish(trace_path=trace_path)
        finished["summary"] = summary
        if speech is not None:
            speech.stop()
        report(f"{summary.frames} frames, {summary.duration_s:.1f}s, "
               f"{len(summary.alerts)} alert(s); trace {trace_path}")
        app.quit()

    def on_tick() -> None:
        if seconds is not None and time.monotonic() - started > seconds:
            finish()
            return
        tick = session.tick()
        if tick is None:
            if close_at_end:
                finish()
            else:
                timer.stop()
                if window.play_button is not None:
                    window.play_button.setText("Play")
            return
        redraw()

    def set_interval() -> None:
        timer.setInterval(max(1, int(1000.0 / (fps * speed["value"]))))

    def on_export() -> None:
        if session.last is None:
            return
        path = export_state(session.last.state, export_dir, acknowledged)
        window.heard.setText(f"exported {path}")

    def on_ack() -> None:
        key = window.selected_deviation()
        if key:
            acknowledged.add(key)
            window.heard.setText(f"acknowledged {key} - the deviation stays on record")
            redraw()

    timer = QtCore.QTimer()
    set_interval()
    timer.timeout.connect(on_tick)
    window.export_button.clicked.connect(on_export)
    window.ack_button.clicked.connect(on_ack)

    if seekable:
        window.slider.setRange(0, session.length)

        def on_play() -> None:
            if timer.isActive():
                timer.stop()
                window.play_button.setText("Play")
            else:
                if session.position >= session.length:
                    session.seek(0)
                timer.start()
                window.play_button.setText("Pause")

        def on_step() -> None:
            timer.stop()
            window.play_button.setText("Play")
            if session.tick() is not None:
                redraw()

        def on_speed(index: int) -> None:
            speed["value"] = window.speed.itemData(index)
            set_interval()

        def on_seek() -> None:
            session.seek(max(0, window.slider.value() - 1))
            if session.tick() is not None:
                redraw()

        window.play_button.clicked.connect(on_play)
        window.step_button.clicked.connect(on_step)
        window.speed.currentIndexChanged.connect(on_speed)
        window.slider.sliderReleased.connect(on_seek)

    window.on_close = finish
    window.resize(1280, 820)
    window.show()
    timer.start()
    app.exec()
    finish()
    window.close()
    return 0


#: The name `parikshak run` has always called.
run_live_window = run_session_window
