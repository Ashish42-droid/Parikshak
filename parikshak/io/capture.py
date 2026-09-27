"""Capture: one camera, three consumers, built as a GStreamer tee.

    camera --> tee --+--> H.265 --> splitmuxsink --> local segments (ring buffer)
                     +--> videorate 10 Hz --> appsink --> perception
                     +--> H.265 --> rtspclientsink --> ground station

The tee matters more than it looks. Encoding once and splitting the encoded
stream would couple the recorder's frame rate to the perception rate; splitting
raw and encoding twice keeps the flight record at full rate while perception
runs at whatever the box can sustain. PLAN.md section 3 specifies 30 Hz to the
recorder and 10 Hz to perception, and this is where that split is expressed.

The pipeline is built as a STRING and tested as a string. That is deliberate: a
GStreamer pipeline is a small program, its failure mode is a caps negotiation
error thirty seconds into a demo, and the parts that get typed wrong - element
order, queue placement, the leaky queue that stops a slow consumer stalling the
whole graph - are all visible in the text. Nothing here needs GStreamer
installed to be checked.

There is exactly one outbound network path in this file, `stream_url`, and it is
explicit, configured, and off by default. Everything else runs with the cable
unplugged. `tests/test_offline.py` asserts that the inference layers have no
network path at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Segment length. Sixty seconds is a compromise: shorter means more files and
#: more index overhead, longer means a deviation clip has to pull in more video
#: than it needs and the ring buffer is coarser to trim.
DEFAULT_SEGMENT_S = 60

#: Perception runs slower than the recorder on purpose - the flight record
#: should not be degraded to whatever the SoC can infer at.
DEFAULT_PERCEPTION_FPS = 10
DEFAULT_RECORD_FPS = 30


class CaptureError(RuntimeError):
    """The capture graph could not be built or started."""


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Everything the capture graph needs. No defaults that reach the network."""

    device: str = "/dev/video0"
    #: "csi" for the Jetson's IMX296 CSI cameras, "v4l2" for a USB or laptop
    #: camera, "test" for the GStreamer pattern source (demos with no hardware).
    source: str = "v4l2"
    width: int = 1920
    height: int = 1080
    fps: int = DEFAULT_RECORD_FPS
    perception_fps: int = DEFAULT_PERCEPTION_FPS
    bitrate_kbps: int = 4000
    segment_s: int = DEFAULT_SEGMENT_S
    #: "nvh265" uses the Jetson's hardware encoder; "x265" is the laptop path.
    #: "auto" resolves at build time from the source type.
    encoder: str = "auto"
    record_dir: Path | None = None
    #: The single outbound network path. None means nothing leaves the box.
    stream_url: str | None = None
    camera_id: str = "cam0"

    def resolved_encoder(self) -> str:
        if self.encoder != "auto":
            return self.encoder
        return "nvh265" if self.source == "csi" else "x265"


# --------------------------------------------------------------------------
# pipeline construction
# --------------------------------------------------------------------------
def _source_branch(cfg: CaptureConfig) -> str:
    caps = f"width={cfg.width},height={cfg.height},framerate={cfg.fps}/1"
    if cfg.source == "csi":
        # nvarguscamerasrc keeps frames in NVMM memory so the hardware encoder
        # never copies them through the CPU.
        return (f"nvarguscamerasrc sensor-id=0 ! "
                f"video/x-raw(memory:NVMM),{caps},format=NV12")
    if cfg.source == "test":
        return f"videotestsrc is-live=true pattern=smpte ! video/x-raw,{caps}"
    return f"v4l2src device={cfg.device} ! video/x-raw,{caps}"


def _convert(cfg: CaptureConfig, to_bgr: bool = False) -> str:
    """The right converter for where the frames live.

    On the CSI path frames are in NVMM (GPU) memory and only `nvvidconv` can
    touch them - `videoconvert` cannot negotiate those caps and the pipeline
    fails at start with an error that names neither element. Getting out of NVMM
    to BGR for perception takes two hops: nvvidconv to BGRx, then videoconvert.
    """
    if cfg.source == "csi":
        return ("nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
                "video/x-raw,format=BGR" if to_bgr else "nvvidconv")
    return ("videoconvert ! video/x-raw,format=BGR" if to_bgr else "videoconvert")


def _encoder_branch(cfg: CaptureConfig) -> str:
    if cfg.resolved_encoder() == "nvh265":
        return (f"nvv4l2h265enc bitrate={cfg.bitrate_kbps * 1000} "
                f"insert-sps-pps=true iframeinterval={cfg.fps} ! h265parse")
    # x265 on a laptop. zerolatency because a recorder that buffers two seconds
    # makes the deviation clip timestamps wrong by two seconds.
    return (f"x265enc bitrate={cfg.bitrate_kbps} tune=zerolatency "
            f"key-int-max={cfg.fps} ! h265parse")


def _queue(leaky: bool = True) -> str:
    """A queue on every tee branch, and a leaky one at that.

    Without queues the branches run in one thread and the slowest sets the pace
    for all of them. Without `leaky=downstream` a stalled consumer - a ground
    link that has gone away - applies backpressure all the way to the camera and
    takes the local recording down with it. The flight record must survive the
    network, not depend on it.
    """
    base = "queue max-size-buffers=8 max-size-time=0 max-size-bytes=0"
    return f"{base} leaky=downstream" if leaky else base


def build_pipeline(cfg: CaptureConfig) -> str:
    """The full launch string. Pure text - no GStreamer needed to build or test."""
    if cfg.record_dir is None and cfg.stream_url is None:
        raise CaptureError("capture would discard every frame: set record_dir, "
                           "stream_url, or both")
    if cfg.perception_fps > cfg.fps:
        raise CaptureError(
            f"perception_fps ({cfg.perception_fps}) exceeds capture fps ({cfg.fps}); "
            f"frames cannot be invented downstream of the camera")

    parts = [f"{_source_branch(cfg)} ! tee name=t"]

    # Local recording. Never leaky: dropping frames from the flight record to
    # keep a preview smooth is the wrong trade.
    if cfg.record_dir is not None:
        # POSIX separators: GStreamer treats a backslash as an escape, so a
        # Windows path silently produces a location nothing can be written to.
        pattern = (Path(cfg.record_dir) / f"{cfg.camera_id}_%05d.mkv").as_posix()
        parts.append(
            f"t. ! {_queue(leaky=False)} ! {_convert(cfg)} ! {_encoder_branch(cfg)} ! "
            f"splitmuxsink name=recorder location={pattern} "
            f"max-size-time={cfg.segment_s * 1_000_000_000} muxer=matroskamux")

    # Perception. videorate decimates rather than the camera running slow, so
    # the recorder keeps full frame rate regardless of what inference sustains.
    parts.append(
        f"t. ! {_queue()} ! videorate ! video/x-raw,framerate={cfg.perception_fps}/1 ! "
        f"{_convert(cfg, to_bgr=True)} ! "
        f"appsink name=perception emit-signals=true max-buffers=2 drop=true")

    if cfg.stream_url is not None:
        sink = ("srtsink uri=" if cfg.stream_url.startswith("srt")
                else "rtspclientsink location=")
        parts.append(
            f"t. ! {_queue()} ! {_convert(cfg)} ! {_encoder_branch(cfg)} ! "
            f"{sink}{cfg.stream_url}")

    return " \\\n  ".join(parts)


# --------------------------------------------------------------------------
# segment index
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Segment:
    path: Path
    t_start: float
    t_end: float
    size_bytes: int = 0

    def overlaps(self, lo: float, hi: float) -> bool:
        return self.t_start < hi and self.t_end > lo


class SegmentIndex:
    """Which recorded file covers which stretch of run time.

    splitmuxsink names files by counter, not by timestamp, so something has to
    remember when each one started. Doing it here rather than parsing filenames
    means a deviation clip can be cut from run-relative time - the same clock the
    log and the traces use - instead of from wall-clock filenames that drift.
    """

    #: cam0_00042.mkv -> 42
    _NAME = re.compile(r"^(?P<cam>.+)_(?P<index>\d{5})\.[A-Za-z0-9]+$")

    def __init__(self) -> None:
        self.segments: list[Segment] = []

    def open_segment(self, path: str | Path, t_start: float) -> None:
        """Called when splitmuxsink starts a new file."""
        self.close_open(t_start)
        self.segments.append(Segment(Path(path), t_start, t_start))

    def close_open(self, t_end: float) -> None:
        if not self.segments:
            return
        last = self.segments[-1]
        if last.t_end <= last.t_start:
            size = last.path.stat().st_size if last.path.exists() else 0
            self.segments[-1] = Segment(last.path, last.t_start, t_end, size)

    def covering(self, lo: float, hi: float) -> list[Segment]:
        """Segments overlapping [lo, hi], in order. Empty when the window falls
        outside what is still on disk - which is a real answer, not an error:
        the ring buffer may have already trimmed it."""
        if hi < lo:
            lo, hi = hi, lo
        return [s for s in self.segments if s.overlaps(lo, hi)]

    @staticmethod
    def index_of(path: str | Path) -> int | None:
        m = SegmentIndex._NAME.match(Path(path).name)
        return int(m.group("index")) if m else None

    @property
    def total_bytes(self) -> int:
        return sum(s.size_bytes for s in self.segments)

    def drop(self, segment: Segment) -> None:
        self.segments = [s for s in self.segments if s.path != segment.path]


# --------------------------------------------------------------------------
# ring buffer
# --------------------------------------------------------------------------
class RingBuffer:
    """Keeps local recording inside a disk budget by trimming oldest first.

    A 512 GB NVMe holds many hours of 1080p H.265, but "many" is not "unbounded"
    and a recorder that fills the disk takes the log down with it - the one
    artefact that actually has to survive. So the budget is enforced here, and
    segments protected by a pending clip export are never trimmed.
    """

    def __init__(self, index: SegmentIndex, *, max_bytes: int,
                 min_segments: int = 2) -> None:
        self.index = index
        self.max_bytes = max_bytes
        self.min_segments = min_segments
        self.protected: set[Path] = set()

    def protect(self, segments: list[Segment]) -> None:
        """Pin segments a deviation clip still needs. Evidence outranks capacity."""
        self.protected.update(s.path for s in segments)

    def release(self, segments: list[Segment]) -> None:
        for s in segments:
            self.protected.discard(s.path)

    def trim(self, *, delete: bool = True) -> list[Segment]:
        """Drop oldest segments until inside budget. Returns what was removed."""
        removed: list[Segment] = []
        while (self.index.total_bytes > self.max_bytes
               and len(self.index.segments) > self.min_segments):
            candidate = next((s for s in self.index.segments
                              if s.path not in self.protected), None)
            if candidate is None:
                break  # everything left is evidence; stop rather than delete it
            self.index.drop(candidate)
            removed.append(candidate)
            if delete and candidate.path.exists():
                candidate.path.unlink()
        return removed


# --------------------------------------------------------------------------
# the runtime adapter
# --------------------------------------------------------------------------
class GStreamerCapture:
    """Runs the graph and hands frames to a callback.

    Lazily imported, and unexercised until there is a camera. The value in this
    file is `build_pipeline` and the index, both of which are checkable now.
    """

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self.index = SegmentIndex()
        self._pipeline = None

    @property
    def launch_string(self) -> str:
        return build_pipeline(self.cfg)

    def start(self, on_frame=None) -> None:  # pragma: no cover - needs GStreamer
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except (ImportError, ValueError) as exc:
            raise CaptureError(
                "GStreamer python bindings are not available; install gstreamer1.0 "
                "and python3-gi, or run from recorded traces instead") from exc

        Gst.init(None)
        self._pipeline = Gst.parse_launch(self.launch_string)
        recorder = self._pipeline.get_by_name("recorder")
        if recorder is not None:
            recorder.connect("format-location", self._on_new_segment)
        if on_frame is not None:
            sink = self._pipeline.get_by_name("perception")
            sink.connect("new-sample", on_frame)
        self._pipeline.set_state(Gst.State.PLAYING)

    def _on_new_segment(self, _splitmux, fragment_id):  # pragma: no cover
        import time
        path = Path(str(self.cfg.record_dir)) / f"{self.cfg.camera_id}_{fragment_id:05d}.mkv"
        self.index.open_segment(path, time.monotonic())
        return str(path)

    def stop(self) -> None:  # pragma: no cover - needs GStreamer
        if self._pipeline is None:
            return
        from gi.repository import Gst
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
