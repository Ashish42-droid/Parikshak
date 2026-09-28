"""Train and serialize the ContactMLP weights artifact for Parikshak.

Synthesizes physically grounded feature distributions for the 5 grasp states:
  none, reach, grasp, manipulate, release
Trains a 3-layer MLP and exports weights to builds/contact_mlp.npz.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

N_FEATURES = 12
LABELS = ("none", "reach", "grasp", "manipulate", "release")
N_CLASSES = len(LABELS)


def generate_synthetic_contact_dataset(n_samples_per_class: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []

    for cls_idx, label in enumerate(LABELS):
        n = n_samples_per_class

        if label == "none":
            # Hand distant from object
            gap = rng.uniform(0.75, 3.5, n)
            wrist_gap = gap + rng.uniform(0.2, 1.0, n)
            mean_gap = gap + rng.uniform(0.2, 0.8, n)
            aperture = rng.uniform(0.3, 1.3, n)
            inside = np.zeros(n)
            wrist_in = np.zeros(n)
            iou = rng.uniform(0.0, 0.05, n)
            diag = rng.uniform(0.4, 2.5, n)
            closing = rng.normal(0.0, 0.1, n)
            obj_speed = rng.uniform(0.0, 0.1, n)
            det_score = rng.uniform(0.35, 0.95, n)
            hand_score = rng.uniform(0.4, 0.95, n)

        elif label == "reach":
            # Hand approaching object
            gap = rng.uniform(0.18, 0.85, n)
            wrist_gap = gap + rng.uniform(0.15, 0.8, n)
            mean_gap = gap + rng.uniform(0.1, 0.4, n)
            aperture = rng.uniform(0.5, 1.15, n)
            inside = rng.uniform(0.0, 0.2, n)
            wrist_in = np.zeros(n)
            iou = rng.uniform(0.05, 0.35, n)
            diag = rng.uniform(0.4, 2.5, n)
            closing = rng.uniform(0.15, 1.6, n)
            obj_speed = rng.uniform(0.0, 0.15, n)
            det_score = rng.uniform(0.45, 0.98, n)
            hand_score = rng.uniform(0.5, 0.98, n)

        elif label == "grasp":
            # Hand closed around object, resting or steady
            gap = rng.uniform(0.0, 0.14, n)
            wrist_gap = rng.uniform(0.1, 0.55, n)
            mean_gap = gap + rng.uniform(0.01, 0.12, n)
            aperture = rng.uniform(0.12, 0.52, n)
            inside = rng.uniform(0.4, 1.0, n)
            wrist_in = rng.binomial(1, 0.25, n).astype(float)
            iou = rng.uniform(0.25, 0.85, n)
            diag = rng.uniform(0.4, 2.5, n)
            closing = rng.normal(0.0, 0.08, n)
            obj_speed = rng.uniform(0.0, 0.22, n)
            det_score = rng.uniform(0.55, 0.99, n)
            hand_score = rng.uniform(0.6, 0.99, n)

        elif label == "manipulate":
            # Hand closed around object while object is moving
            gap = rng.uniform(0.0, 0.14, n)
            wrist_gap = rng.uniform(0.1, 0.55, n)
            mean_gap = gap + rng.uniform(0.01, 0.12, n)
            aperture = rng.uniform(0.12, 0.52, n)
            inside = rng.uniform(0.4, 1.0, n)
            wrist_in = rng.binomial(1, 0.25, n).astype(float)
            iou = rng.uniform(0.25, 0.85, n)
            diag = rng.uniform(0.4, 2.5, n)
            closing = rng.normal(0.0, 0.1, n)
            obj_speed = rng.uniform(0.35, 2.5, n)  # distinct from static grasp
            det_score = rng.uniform(0.55, 0.99, n)
            hand_score = rng.uniform(0.6, 0.99, n)

        elif label == "release":
            # Hand opening and withdrawing from object
            gap = rng.uniform(0.05, 0.35, n)
            wrist_gap = rng.uniform(0.18, 0.8, n)
            mean_gap = gap + rng.uniform(0.08, 0.3, n)
            aperture = rng.uniform(0.58, 1.25, n)
            inside = rng.uniform(0.0, 0.3, n)
            wrist_in = np.zeros(n)
            iou = rng.uniform(0.08, 0.45, n)
            diag = rng.uniform(0.4, 2.5, n)
            closing = rng.uniform(-1.6, -0.15, n)  # negative closing speed
            obj_speed = rng.uniform(0.0, 0.3, n)
            det_score = rng.uniform(0.45, 0.98, n)
            hand_score = rng.uniform(0.5, 0.98, n)

        X_cls = np.column_stack([
            wrist_gap, gap, mean_gap, aperture, inside, wrist_in,
            iou, diag, closing, obj_speed, det_score, hand_score
        ])
        y_cls = np.full(n, cls_idx, dtype=np.int64)

        X_list.append(X_cls)
        y_list.append(y_cls)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    # Shuffle
    indices = rng.permutation(len(y))
    return X[indices], y[indices]


class PyTorchContactMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        return self.net(x)


def train_and_export(output_path: Path):
    print("Generating synthetic contact dataset...")
    X, y = generate_synthetic_contact_dataset(n_samples_per_class=12000, seed=42)

    n_train = int(len(y) * 0.85)
    X_train, y_train = torch.tensor(X[:n_train]), torch.tensor(y[:n_train])
    X_val, y_val = torch.tensor(X[n_train:]), torch.tensor(y[n_train:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    model = PyTorchContactMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    print("Training ContactMLP...")
    for epoch in range(1, 26):
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

        if epoch % 5 == 0 or epoch == 25:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_preds = val_out.argmax(dim=1)
                acc = (val_preds == y_val).float().mean().item()
            print(f"Epoch {epoch:2d}/25 - Loss: {total_loss / len(train_loader):.4f} - Val Acc: {acc * 100:.2f}%")

    # Export weights
    # In ContactMLP:
    #   h = np.maximum(0.0, x @ self.w1 + self.b1)
    # PyTorch Linear weight is (out_features, in_features), so w = weight.T
    state = model.state_dict()
    w1 = state["net.0.weight"].numpy().T
    b1 = state["net.0.bias"].numpy()
    w2 = state["net.2.weight"].numpy().T
    b2 = state["net.2.bias"].numpy()
    w3 = state["net.4.weight"].numpy().T
    b3 = state["net.4.bias"].numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        w1=w1, b1=b1,
        w2=w2, b2=b2,
        w3=w3, b3=b3,
    )
    print(f"Saved ContactMLP weights to {output_path}")


if __name__ == "__main__":
    out_file = Path("builds/contact_mlp.npz")
    train_and_export(out_file)
