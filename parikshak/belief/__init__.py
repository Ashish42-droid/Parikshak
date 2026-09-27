"""The perception -> engine contract.

`BeliefFrame` is frozen at schema v1.1. Perception writes it, the engine reads
it, and it serialises to one JSONL line per frame. Changing a field is a
breaking change: bump BELIEF_SCHEMA_VERSION and regenerate every golden trace.
"""

from parikshak.belief.frame import (
    BELIEF_SCHEMA_VERSION,
    CONTACT_SEP,
    BeliefFrame,
    BodyBelief,
    HandBelief,
    MotionBelief,
    ObjectBelief,
    Quat,
    TraceFormatError,
    Vec3,
    contact_key,
    empty_frame,
)
from parikshak.belief.trace import (
    TraceHeader,
    read_trace,
    read_trace_header,
    write_trace,
)

__all__ = [
    "BELIEF_SCHEMA_VERSION",
    "CONTACT_SEP",
    "BeliefFrame",
    "BodyBelief",
    "HandBelief",
    "MotionBelief",
    "ObjectBelief",
    "Quat",
    "TraceFormatError",
    "Vec3",
    "contact_key",
    "empty_frame",
    "TraceHeader",
    "read_trace",
    "read_trace_header",
    "write_trace",
]
