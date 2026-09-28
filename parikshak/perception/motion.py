"""Temporal 1D-TCN for motion-step classification.

Maps a sliding temporal window of hand and object kinematics (2.0 s at 10 Hz,
i.e. 20 timesteps of 16-channel feature vectors) to the 7 motion classes
registered in `builds/parikshak-perception-1.2.0.json`:
  - idle
  - reach
  - insert
  - rotate_seal
  - press
  - agitate
  - withdraw

Supports pure NumPy forward inference for deterministic, lightweight execution
on CPU, as well as PyTorch for training.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from parikshak.perception.types import (
    ContactState,
    Detection,
    HandLandmarks,
    MotionState,
)

#: Motion vocabulary matching builds/parikshak-perception-1.2.0.json
MOTION_CLASSES = (
    "idle",
    "reach",
    "insert",
    "rotate_seal",
    "press",
    "agitate",
    "withdraw",
)
N_CLASSES = len(MOTION_CLASSES)

#: Sliding window length: 2.0 s @ 10 Hz = 20 timesteps
WINDOW_TIMESTEPS = 20
N_FEATURES = 16


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _conv1d_same(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int = 1) -> np.ndarray:
    """1D convolution along time dimension (T, C_in) -> (T, C_out).
    weight shape: (C_out, C_in, K). bias shape: (C_out,).
    """
    T, C_in = x.shape
    C_out, _, K = weight.shape
    pad = (K - 1) * dilation // 2
    # Pad symmetrically along time dimension
    x_padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros((T, C_out), dtype=x.dtype)

    for t in range(T):
        # receptive window
        window = x_padded[t:t + K * dilation:dilation]  # (K, C_in)
        # sum_{k, c_in} window[k, c_in] * weight[c_out, c_in, k]
        # np.tensordot: window is (K, C_in), weight is (C_out, C_in, K)
        val = np.einsum("ki,oik->o", window, weight) + bias
        out[t] = val
    return out


# --------------------------------------------------------------------------
# Features extractor for single timestep
# --------------------------------------------------------------------------
def extract_frame_features(
    hand: HandLandmarks | None = None,
    contact: ContactState | None = None,
    target_det: Detection | None = None,
    prev_hand_wrist: tuple[float, float] | None = None,
    prev_obj_centre: tuple[float, float] | None = None,
    dt: float = 0.1,
) -> np.ndarray:
    """Extracts 16-D kinematic feature vector for one frame.

    0..2:  hand wrist position (x, y, span)
    3..5:  hand velocity (vx, vy, speed)
    6:     hand grip aperture
    7..8:  contact probability, contact class code
    9..11: object position (x, y, diag)
    12..13: object velocity (speed, relative closing speed)
    14:    rotation/angular proxy
    15:    hand-object direction alignment
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    dt_safe = max(dt, 1e-4)

    if hand is not None:
        wx, wy = hand.wrist_px
        span = max(hand.span_px, 1.0)
        feat[0] = wx / 640.0
        feat[1] = wy / 480.0
        feat[2] = span / 100.0

        if prev_hand_wrist is not None:
            vx = (wx - prev_hand_wrist[0]) / span / dt_safe
            vy = (wy - prev_hand_wrist[1]) / span / dt_safe
            feat[3] = vx
            feat[4] = vy
            feat[5] = float(np.hypot(vx, vy))

        feat[6] = hand.grip_aperture

    if contact is not None:
        feat[7] = contact.score
        contact_map = {"none": 0.0, "reach": 1.0, "grasp": 2.0, "manipulate": 3.0, "release": 4.0}
        feat[8] = contact_map.get(contact.label, 0.0) / 4.0

    if target_det is not None:
        cx, cy = target_det.centre_px
        x1, y1, x2, y2 = target_det.box_px
        diag = float(np.hypot(x2 - x1, y2 - y1))
        feat[9] = cx / 640.0
        feat[10] = cy / 480.0
        feat[11] = diag / 100.0

        if prev_obj_centre is not None:
            ovx = (cx - prev_obj_centre[0]) / dt_safe
            ovy = (cy - prev_obj_centre[1]) / dt_safe
            feat[12] = float(np.hypot(ovx, ovy)) / 100.0

        if hand is not None:
            dx = cx - hand.wrist_px[0]
            dy = cy - hand.wrist_px[1]
            dist = float(np.hypot(dx, dy)) or 1.0
            if prev_hand_wrist is not None:
                h_vx = hand.wrist_px[0] - prev_hand_wrist[0]
                h_vy = hand.wrist_px[1] - prev_hand_wrist[1]
                # Cosine alignment between hand velocity and object direction
                h_speed = float(np.hypot(h_vx, h_vy))
                if h_speed > 1e-3:
                    feat[15] = float((h_vx * dx + h_vy * dy) / (h_speed * dist))

    return feat


# --------------------------------------------------------------------------
# MotionTCN: 1D Dilated Temporal Convolutional Network
# --------------------------------------------------------------------------
@dataclass
class MotionTCN:
    """Dilated 1D Temporal Convolutional Network forward engine.

    Structure:
      in_conv: Conv1D(16 -> 64, K=3) + ReLU
      res1:    Conv1D(64 -> 64, K=3, dilation=1) + ReLU + Residual
      res2:    Conv1D(64 -> 64, K=3, dilation=2) + ReLU + Residual
      res3:    Conv1D(64 -> 64, K=3, dilation=4) + ReLU + Residual
      pool:    Temporal average pooling -> 64
      fc1:     Linear(64 -> 32) + ReLU
      fc2:     Linear(32 -> 7) + Softmax
    """

    w_in: np.ndarray  # (64, 16, 3)
    b_in: np.ndarray  # (64,)
    w_res1: np.ndarray  # (64, 64, 3)
    b_res1: np.ndarray  # (64,)
    w_res2: np.ndarray  # (64, 64, 3)
    b_res2: np.ndarray  # (64,)
    w_res3: np.ndarray  # (64, 64, 3)
    b_res3: np.ndarray  # (64,)
    w_fc1: np.ndarray  # (64, 32)
    b_fc1: np.ndarray  # (32,)
    w_fc2: np.ndarray  # (32, 7)
    b_fc2: np.ndarray  # (7,)
    sha256: str = ""

    def __post_init__(self) -> None:
        if self.w_in.shape[1] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, got {self.w_in.shape[1]}")
        if self.w_fc2.shape[1] != N_CLASSES:
            raise ValueError(f"expected {N_CLASSES} classes, got {self.w_fc2.shape[1]}")

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """Forward pass. Expects window shape (T, N_FEATURES) or (N_FEATURES,)."""
        x = np.asarray(window, dtype=np.float32)
        if x.ndim == 1:
            # Single feature vector: tile to full window
            x = np.tile(x, (WINDOW_TIMESTEPS, 1))
        elif x.shape[0] < WINDOW_TIMESTEPS:
            # Pad front if window has not filled
            pad_len = WINDOW_TIMESTEPS - x.shape[0]
            x = np.pad(x, ((pad_len, 0), (0, 0)), mode="edge")
        elif x.shape[0] > WINDOW_TIMESTEPS:
            x = x[-WINDOW_TIMESTEPS:]

        # 1. in_conv
        h = _relu(_conv1d_same(x, self.w_in, self.b_in, dilation=1))

        # 2. res1 (dilation 1)
        r1 = _relu(_conv1d_same(h, self.w_res1, self.b_res1, dilation=1))
        h = h + r1

        # 3. res2 (dilation 2)
        r2 = _relu(_conv1d_same(h, self.w_res2, self.b_res2, dilation=2))
        h = h + r2

        # 4. res3 (dilation 4)
        r3 = _relu(_conv1d_same(h, self.w_res3, self.b_res3, dilation=4))
        h = h + r3

        # 5. Temporal Global Pooling
        pooled = np.mean(h, axis=0)  # (64,)

        # 6. Classifier Head
        dense = _relu(pooled @ self.w_fc1 + self.b_fc1)  # (32,)
        logits = dense @ self.w_fc2 + self.b_fc2  # (7,)
        return _softmax(logits)

    @classmethod
    def load(cls, path: str | Path) -> MotionTCN:
        p = Path(path)
        blob = p.read_bytes()
        d = np.load(p) if p.suffix == ".npz" else json.loads(blob.decode("utf-8"))
        return cls(
            w_in=np.asarray(d["w_in"], dtype=np.float32),
            b_in=np.asarray(d["b_in"], dtype=np.float32),
            w_res1=np.asarray(d["w_res1"], dtype=np.float32),
            b_res1=np.asarray(d["b_res1"], dtype=np.float32),
            w_res2=np.asarray(d["w_res2"], dtype=np.float32),
            b_res2=np.asarray(d["b_res2"], dtype=np.float32),
            w_res3=np.asarray(d["w_res3"], dtype=np.float32),
            b_res3=np.asarray(d["b_res3"], dtype=np.float32),
            w_fc1=np.asarray(d["w_fc1"], dtype=np.float32),
            b_fc1=np.asarray(d["b_fc1"], dtype=np.float32),
            w_fc2=np.asarray(d["w_fc2"], dtype=np.float32),
            b_fc2=np.asarray(d["b_fc2"], dtype=np.float32),
            sha256=hashlib.sha256(blob).hexdigest(),
        )


# --------------------------------------------------------------------------
# Classifier adapter conforming to MotionClassifier protocol
# --------------------------------------------------------------------------
class TcnMotionClassifier:
    """Sliding-window temporal classifier conforming to MotionClassifier protocol."""

    def __init__(self, model: MotionTCN | None = None, *, min_conf: float = 0.5) -> None:
        if model is None:
            weights_file = Path("builds/motion_tcn.npz")
            if weights_file.exists():
                try:
                    model = MotionTCN.load(weights_file)
                except Exception:
                    model = None
        self.model = model
        self.min_conf = min_conf
        self._history: deque[np.ndarray] = deque(maxlen=WINDOW_TIMESTEPS)
        self.version = "tcn-motion-1.1.0"

    @property
    def is_learned(self) -> bool:
        return self.model is not None

    @property
    def weights_sha256(self) -> str:
        return self.model.sha256 if self.model else "scripted-motion-stub"

    def push_frame(
        self,
        hand: HandLandmarks | None = None,
        contact: ContactState | None = None,
        target_det: Detection | None = None,
        dt: float = 0.1,
    ) -> None:
        """Pushes current frame telemetry into the sliding window."""
        prev_hand = self._last_hand if hasattr(self, "_last_hand") else None
        prev_obj = self._last_obj if hasattr(self, "_last_obj") else None

        vec = extract_frame_features(
            hand=hand,
            contact=contact,
            target_det=target_det,
            prev_hand_wrist=prev_hand,
            prev_obj_centre=prev_obj,
            dt=dt,
        )
        self._history.append(vec)

        if hand is not None:
            self._last_hand = hand.wrist_px
        if target_det is not None:
            self._last_obj = target_det.centre_px

    def __call__(self, window: np.ndarray | None = None) -> MotionState:
        if self.model is None:
            return MotionState("idle", 0.0)

        if window is not None and window.size > 1:
            arr = np.asarray(window, dtype=np.float32)
        else:
            if not self._history:
                return MotionState("idle", 0.0)
            arr = np.array(self._history, dtype=np.float32)

        probs = self.model(arr)
        best_idx = int(np.argmax(probs))
        score = float(probs[best_idx])
        label = MOTION_CLASSES[best_idx]
        return MotionState(label, score)
