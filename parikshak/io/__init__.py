"""I/O: capture, speech, voice commands, event clips, downlink packaging.

The boundary between the system and the world. Everything here is optional at
import time - GStreamer, Piper and Vosk are extras - because the engine and the
eval harness must run on a laptop with none of them, and because ASR is first
on the descope ladder.

One outbound network path exists in this package, `CaptureConfig.stream_url`,
and it is explicit and off by default. `tests/test_offline.py` asserts that the
inference layers have none at all.
"""

from parikshak.io.asr import Command, CommandGrammar, Recognised, VoskRecogniser
from parikshak.io.capture import (
    CaptureConfig,
    CaptureError,
    GStreamerCapture,
    RingBuffer,
    Segment,
    SegmentIndex,
    build_pipeline,
)
from parikshak.io.clips import ClipExporter, ClipRequest, ClipResult
from parikshak.io.downlink import (
    Artifact,
    DownlinkPackage,
    Priority as DownlinkPriority,
    compression_ratio,
)
from parikshak.io.tts import (
    CallbackVoice,
    NullVoice,
    PiperVoice,
    Priority as SpeechPriority,
    SpeechQueue,
)

__all__ = [
    "Artifact", "CallbackVoice", "CaptureConfig", "CaptureError", "ClipExporter",
    "ClipRequest", "ClipResult", "Command", "CommandGrammar", "DownlinkPackage",
    "DownlinkPriority", "GStreamerCapture", "NullVoice", "PiperVoice", "Recognised",
    "RingBuffer", "Segment", "SegmentIndex", "SpeechPriority", "SpeechQueue",
    "VoskRecogniser", "build_pipeline", "compression_ratio",
]
