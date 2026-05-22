"""
Test inferenza IK — eseguibile in locale dopo aver copiato:
    ik_training/best_model.ckpt
    ik_training/scaler_y.pkl

Dipendenze locali (pip install):
    torch pytorch-lightning scikit-learn joblib numpy

Uso:
    python test_infer.py                          # usa pose di esempio predefinite
    python test_infer.py --x 0.3 --y 0.1 --z 0.4 --qx 0 --qy 0 --qz 0 --qw 1
    python test_infer.py --ckpt altro/percorso/model.ckpt
"""

import argparse
import os
import sys
import numpy as np
import joblib
import torch
import torch.nn as nn
import pytorch_lightning as pl


# ──────────────────────────────────────────
#  COPIA DELL'ARCHITETTURA  (deve essere identica a train.py)
# ──────────────────────────────────────────

OUTPUT_COLS = [f"joint_{i}" for i in range(7)]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class IKNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, dropout):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class LitIKRegressor(pl.LightningModule):
    def __init__(self, input_dim=7, output_dim=7,
                 hidden_dim=512, n_blocks=4, dropout=0.1,
                 lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = IKNet(input_dim, output_dim, hidden_dim, n_blocks, dropout)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────
#  INFERENZA
# ──────────────────────────────────────────

def load_model(ckpt_path: str):
    print(f"Caricamento checkpoint: {ckpt_path}")
    model = LitIKRegressor.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def predict(model, scaler_y, pose: list) -> np.ndarray:
    """
    pose: [x, y, z, qx, qy, qz, qw]  — coordinate reali del robot
    ritorna: array (7,) con i valori dei giunti in radianti
    """
    x = torch.tensor([pose], dtype=torch.float32)
    with torch.no_grad():
        pred_norm = model(x).numpy()
    return scaler_y.inverse_transform(pred_norm)[0]


def print_result(pose, joints, label=""):
    x, y, z, qx, qy, qz, qw = pose
    if label:
        print(f"\n{'─'*50}")
        print(f"  {label}")
    print(f"  Input  → pos=({x:.3f}, {y:.3f}, {z:.3f})  "
          f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
    print(f"  Output →")
    for name, val in zip(OUTPUT_COLS, joints):
        bar = "█" * int(abs(val) / 5.5 * 20)
        sign = "+" if val >= 0 else "-"
        print(f"    {name}: {val:+7.4f} rad  ({np.degrees(val):+7.2f}°)  {sign}{bar}")


# ──────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────

EXAMPLE_POSES = [
    ([0.0,  0.0,  0.3,  0.0,  0.0,  0.0,  1.0],  "Centro, nessuna rotazione (identità)"),
    ([0.3,  0.0,  0.3,  0.0,  0.0,  0.0,  1.0],  "Destra, nessuna rotazione"),
    ([0.0,  0.3,  0.3,  0.0,  0.0,  0.0,  1.0],  "Avanti, nessuna rotazione"),
    ([0.3,  0.3,  0.5,  0.0,  0.707, 0.0, 0.707], "Angolo alto, rotazione 90° attorno Y"),
    ([-0.2, -0.2, 0.1,  0.0,  0.0,  0.707, 0.707],"Basso sinistra, rotazione 90° attorno Z"),
]


def main():
    parser = argparse.ArgumentParser(description="Test inferenza IK")
    parser.add_argument("--ckpt",    default="ik_training/best_model.ckpt")
    parser.add_argument("--out-dir", default="ik_training")
    parser.add_argument("--x",  type=float, default=None)
    parser.add_argument("--y",  type=float, default=None)
    parser.add_argument("--z",  type=float, default=None)
    parser.add_argument("--qx", type=float, default=None)
    parser.add_argument("--qy", type=float, default=None)
    parser.add_argument("--qz", type=float, default=None)
    parser.add_argument("--qw", type=float, default=None)
    args = parser.parse_args()

    # Verifica file necessari
    missing = [f for f in [args.ckpt, os.path.join(args.out_dir, "scaler_y.pkl")]
               if not os.path.exists(f)]
    if missing:
        print("❌ File mancanti:")
        for f in missing:
            print(f"   {f}")
        print("\nCopia dal server con:")
        print("  scp user@server:/path/ik_training/best_model.ckpt ./ik_training/")
        print("  scp user@server:/path/ik_training/scaler_y.pkl    ./ik_training/")
        sys.exit(1)

    model   = load_model(args.ckpt)
    scaler_y = joblib.load(os.path.join(args.out_dir, "scaler_y.pkl"))

    hp = model.hparams
    print(f"\nModello caricato — hidden={hp.hidden_dim}, blocks={hp.n_blocks}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # Posa singola da CLI
    custom = [args.x, args.y, args.z, args.qx, args.qy, args.qz, args.qw]
    if all(v is not None for v in custom):
        joints = predict(model, scaler_y, custom)
        print_result(custom, joints, label="Posa custom")

    # Pose di esempio predefinite
    else:
        print(f"\n{'═'*50}")
        print("  POSE DI ESEMPIO")
        print(f"{'═'*50}")
        for pose, label in EXAMPLE_POSES:
            joints = predict(model, scaler_y, pose)
            print_result(pose, joints, label=label)

    print(f"\n{'─'*50}")
    print("✓ Inferenza completata.\n")


if __name__ == "__main__":
    main()