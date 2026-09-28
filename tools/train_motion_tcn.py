"""Train and serialize the MotionTCN weights artifact for Parikshak.

Synthesizes temporal trajectory distributions (20 timesteps @ 10 Hz = 2.0 s)
for the 7 motion classes:
  idle, reach, insert, rotate_seal, press, agitate, withdraw
Trains a 1D Dilated Temporal Convolutional Network and exports weights to
builds/motion_tcn.npz.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
WINDOW_TIMESTEPS = 20
N_FEATURES = 16


def generate_motion_dataset(n_samples_per_class: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    T = WINDOW_TIMESTEPS
    X_list = []
    y_list = []

    for cls_idx, label in enumerate(MOTION_CLASSES):
        n = n_samples_per_class
        X_cls = np.zeros((n, T, N_FEATURES), dtype=np.float32)

        for i in range(n):
            t_axis = np.linspace(0, 2.0, T)
            noise = rng.normal(0, 0.02, (T, N_FEATURES))

            if label == "idle":
                # Hand and object stationary
                wx = rng.uniform(0.3, 0.7) + rng.normal(0, 0.005, T)
                wy = rng.uniform(0.3, 0.7) + rng.normal(0, 0.005, T)
                span = rng.uniform(0.6, 1.2)
                vx = rng.normal(0, 0.02, T)
                vy = rng.normal(0, 0.02, T)
                speed = np.hypot(vx, vy)
                aperture = rng.uniform(0.4, 0.8, T)
                contact_prob = rng.uniform(0.0, 0.1, T)
                contact_cls = np.zeros(T)
                ox = rng.uniform(0.3, 0.7)
                oy = rng.uniform(0.3, 0.7)
                odiag = rng.uniform(0.3, 0.8)
                ospeed = rng.normal(0, 0.01, T)
                rot = rng.normal(0, 0.02, T)
                align = rng.normal(0, 0.05, T)

            elif label == "reach":
                # Hand moving toward object
                start_x, start_y = rng.uniform(0.1, 0.4), rng.uniform(0.1, 0.4)
                target_x, target_y = rng.uniform(0.5, 0.8), rng.uniform(0.5, 0.8)
                wx = np.linspace(start_x, target_x, T)
                wy = np.linspace(start_y, target_y, T)
                span = rng.uniform(0.6, 1.2)
                vx = np.gradient(wx, t_axis)
                vy = np.gradient(wy, t_axis)
                speed = np.hypot(vx, vy)
                aperture = np.linspace(0.8, 0.5, T)  # opening then preparing
                contact_prob = np.linspace(0.0, 0.4, T)
                contact_cls = np.ones(T) * 0.25  # reach code
                ox = np.full(T, target_x)
                oy = np.full(T, target_y)
                odiag = rng.uniform(0.3, 0.8)
                ospeed = np.zeros(T)
                rot = rng.normal(0, 0.05, T)
                align = np.ones(T) * 0.85  # highly aligned towards object

            elif label == "insert":
                # Hand holding object, driving steadily into socket, decelerating
                start_x, start_y = rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
                socket_x, socket_y = start_x + rng.uniform(-0.1, 0.1), start_y + rng.uniform(0.1, 0.25)
                # Sigmoidal approach
                progress = 1.0 / (1.0 + np.exp(-np.linspace(-3, 3, T)))
                wx = start_x + (socket_x - start_x) * progress
                wy = start_y + (socket_y - start_y) * progress
                span = rng.uniform(0.6, 1.2)
                vx = np.gradient(wx, t_axis)
                vy = np.gradient(wy, t_axis)
                speed = np.hypot(vx, vy)
                aperture = np.full(T, rng.uniform(0.2, 0.4))  # closed grasp
                contact_prob = np.full(T, rng.uniform(0.85, 0.99))
                contact_cls = np.full(T, 0.5)  # grasp code
                ox = wx + rng.normal(0, 0.01, T)
                oy = wy + rng.normal(0, 0.01, T)
                odiag = rng.uniform(0.3, 0.8)
                ospeed = speed.copy()
                rot = rng.normal(0, 0.05, T)
                align = np.ones(T) * 0.95

            elif label == "rotate_seal":
                # Rotation: hand in place holding object, rotational velocity high
                center_x, center_y = rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
                angle = np.linspace(0, np.pi * 1.5, T)
                wx = center_x + 0.03 * np.cos(angle)
                wy = center_y + 0.03 * np.sin(angle)
                span = rng.uniform(0.6, 1.2)
                vx = np.gradient(wx, t_axis)
                vy = np.gradient(wy, t_axis)
                speed = np.hypot(vx, vy)
                aperture = np.full(T, rng.uniform(0.2, 0.45))
                contact_prob = np.full(T, rng.uniform(0.8, 0.98))
                contact_cls = np.full(T, 0.75)  # manipulate code
                ox = np.full(T, center_x)
                oy = np.full(T, center_y)
                odiag = rng.uniform(0.3, 0.8)
                ospeed = rng.normal(0, 0.02, T)
                rot = np.full(T, rng.uniform(1.2, 2.5))  # High angular rotation!
                align = rng.normal(0, 0.1, T)

            elif label == "press":
                # Quick push pulse in wy then return
                center_x, center_y = rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
                push_curve = 0.15 * np.exp(-((t_axis - 1.0) / 0.3) ** 2)
                wx = np.full(T, center_x)
                wy = center_y + push_curve
                span = rng.uniform(0.6, 1.2)
                vx = np.zeros(T)
                vy = np.gradient(wy, t_axis)
                speed = np.abs(vy)
                aperture = np.full(T, rng.uniform(0.3, 0.6))
                contact_prob = np.linspace(0.2, 0.8, T)
                contact_cls = np.full(T, 0.25)
                ox = np.full(T, center_x)
                oy = np.full(T, center_y)
                odiag = rng.uniform(0.2, 0.5)
                ospeed = np.zeros(T)
                rot = rng.normal(0, 0.05, T)
                align = np.ones(T) * 0.9

            elif label == "agitate":
                # Shaking: rapid sinusoidal oscillation (2.5 Hz = 5 full cycles in 2 seconds)
                center_x, center_y = rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
                freq = rng.uniform(2.0, 3.5)
                amplitude = rng.uniform(0.12, 0.25)
                shaking = amplitude * np.sin(2 * np.pi * freq * t_axis)
                wx = center_x + shaking
                wy = np.full(T, center_y)
                span = rng.uniform(0.6, 1.2)
                vx = np.gradient(wx, t_axis)
                vy = np.zeros(T)
                speed = np.abs(vx)
                aperture = np.full(T, rng.uniform(0.2, 0.4))  # firmly gripped
                contact_prob = np.full(T, rng.uniform(0.85, 0.99))
                contact_cls = np.full(T, 0.75)  # manipulate
                ox = wx.copy()  # vial moves with hand!
                oy = wy.copy()
                odiag = rng.uniform(0.2, 0.5)
                ospeed = speed.copy()
                rot = rng.normal(0, 0.1, T)
                align = np.zeros(T)

            elif label == "withdraw":
                # Hand opening and withdrawing from object
                start_x, start_y = rng.uniform(0.5, 0.7), rng.uniform(0.5, 0.7)
                end_x, end_y = rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)
                wx = np.linspace(start_x, end_x, T)
                wy = np.linspace(start_y, end_y, T)
                span = rng.uniform(0.6, 1.2)
                vx = np.gradient(wx, t_axis)
                vy = np.gradient(wy, t_axis)
                speed = np.hypot(vx, vy)
                aperture = np.linspace(0.4, 0.9, T)  # opening hand
                contact_prob = np.linspace(0.7, 0.0, T)  # dropping contact
                contact_cls = np.full(T, 1.0)  # release code
                ox = np.full(T, start_x)
                oy = np.full(T, start_y)
                odiag = rng.uniform(0.3, 0.8)
                ospeed = np.zeros(T)
                rot = rng.normal(0, 0.05, T)
                align = np.ones(T) * -0.9  # moving away from object

            ox_arr = np.asarray(ox) if isinstance(ox, np.ndarray) and ox.shape == (T,) else np.full(T, float(np.ravel(ox)[0]))
            oy_arr = np.asarray(oy) if isinstance(oy, np.ndarray) and oy.shape == (T,) else np.full(T, float(np.ravel(oy)[0]))
            ospeed_arr = np.asarray(ospeed) if isinstance(ospeed, np.ndarray) and ospeed.shape == (T,) else np.full(T, float(np.ravel(ospeed)[0]))
            rot_arr = np.asarray(rot) if isinstance(rot, np.ndarray) and rot.shape == (T,) else np.full(T, float(np.ravel(rot)[0]))
            align_arr = np.asarray(align) if isinstance(align, np.ndarray) and align.shape == (T,) else np.full(T, float(np.ravel(align)[0]))
            span_arr = np.full(T, span) if np.isscalar(span) else np.asarray(span)
            odiag_arr = np.full(T, odiag) if np.isscalar(odiag) else np.asarray(odiag)

            trajectory = np.column_stack([
                wx, wy, span_arr,
                vx, vy, speed,
                aperture,
                contact_prob, contact_cls,
                ox_arr, oy_arr, odiag_arr,
                ospeed_arr,
                np.zeros(T),  # closing speed
                rot_arr,
                align_arr,
            ])
            X_cls[i] = trajectory + noise

        X_list.append(X_cls)
        y_list.append(np.full(n, cls_idx, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    indices = rng.permutation(len(y))
    return X[indices], y[indices]


# PyTorch 1D-TCN model matching MotionTCN architecture
class PyTorchMotionTCN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input conv
        self.conv_in = nn.Conv1d(N_FEATURES, 64, kernel_size=3, padding=1)
        # Residual blocks with dilations 1, 2, 4
        self.res1 = nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.res2 = nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.res3 = nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()
        # Head
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, N_CLASSES)

    def forward(self, x):
        # x is (batch, time, features) -> transpose to (batch, features, time)
        h = x.transpose(1, 2)
        h = self.relu(self.conv_in(h))
        h = h + self.relu(self.res1(h))
        h = h + self.relu(self.res2(h))
        h = h + self.relu(self.res3(h))
        # Global temporal pool (mean over time)
        pooled = h.mean(dim=2)
        dense = self.relu(self.fc1(pooled))
        return self.fc2(dense)


def train_and_export_tcn(output_path: Path):
    print("Generating synthetic motion trajectories...")
    X, y = generate_motion_dataset(n_samples_per_class=4000, seed=42)

    n_train = int(len(y) * 0.85)
    X_train, y_train = torch.tensor(X[:n_train]), torch.tensor(y[:n_train])
    X_val, y_val = torch.tensor(X[n_train:]), torch.tensor(y[n_train:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = PyTorchMotionTCN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    print("Training MotionTCN...")
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 20:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_preds = val_out.argmax(dim=1)
                acc = (val_preds == y_val).float().mean().item()
            print(f"Epoch {epoch:2d}/20 - Loss: {total_loss / len(train_loader):.4f} - Val Acc: {acc * 100:.2f}%")

    # Export weights for numpy MotionTCN:
    # PyTorch Conv1d weight is (C_out, C_in, K), exactly matching MotionTCN!
    # PyTorch Linear weight is (out_features, in_features), so w = weight.T
    state = model.state_dict()
    w_in = state["conv_in.weight"].numpy()
    b_in = state["conv_in.bias"].numpy()
    w_res1 = state["res1.weight"].numpy()
    b_res1 = state["res1.bias"].numpy()
    w_res2 = state["res2.weight"].numpy()
    b_res2 = state["res2.bias"].numpy()
    w_res3 = state["res3.weight"].numpy()
    b_res3 = state["res3.bias"].numpy()
    w_fc1 = state["fc1.weight"].numpy().T
    b_fc1 = state["fc1.bias"].numpy()
    w_fc2 = state["fc2.weight"].numpy().T
    b_fc2 = state["fc2.bias"].numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        w_in=w_in, b_in=b_in,
        w_res1=w_res1, b_res1=b_res1,
        w_res2=w_res2, b_res2=b_res2,
        w_res3=w_res3, b_res3=b_res3,
        w_fc1=w_fc1, b_fc1=b_fc1,
        w_fc2=w_fc2, b_fc2=b_fc2,
    )
    print(f"Saved MotionTCN weights to {output_path}")


if __name__ == "__main__":
    out_file = Path("builds/motion_tcn.npz")
    train_and_export_tcn(out_file)
