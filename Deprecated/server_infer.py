"""
Server HTTP locale per inferenza IK.

Ascolta su http://localhost:5000 e accetta richieste POST con il target
in coordinate Godot. Applica la catena completa di trasformazioni:

    godot_space → pybullet_world → arm_base_frame → rete → joint angles

Uso:
    python ik_server.py
    python ik_server.py --port 5001 --ckpt altro/percorso/model.ckpt

Richiesta (JSON):
    POST /ik
    {
        "pos":  [x, y, z],          # coordinate Godot
        "quat": [qx, qy, qz, qw]   # quaternione Godot
    }

Risposta (JSON):
    {
        "joints": [j0, j1, j2, j3, j4, j5, j6],   # radianti
        "joints_deg": [...]                         # gradi (comodo per debug)
    }

Dipendenze:
    pip install torch pytorch-lightning scikit-learn joblib numpy
    (nessuna dipendenza da pybullet — le trasformazioni sono pure numpy)
"""

import argparse
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import joblib
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


# ──────────────────────────────────────────
#  CONFIGURAZIONE
# ──────────────────────────────────────────

DEFAULT_CKPT      = "ik_training/best_model.ckpt"
DEFAULT_SCALER_DIR = "ik_training"
DEFAULT_PORT      = 5000

# Giunto di montaggio del braccio sul piedistallo (da openarm_right.urdf):
#   <origin rpy="1.5708 0 0" xyz="0.0 -0.031 0.698"/>
_RIGHT_BASE_T   = np.array([0.0, -0.031, 0.698])
_RIGHT_BASE_RPY = [1.5708, 0.0, 0.0]


# ──────────────────────────────────────────
#  ARCHITETTURA  (identica a train.py)
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
#  TRASFORMAZIONI  (pure numpy, no pybullet)
# ──────────────────────────────────────────

def _euler_rpy_to_quat(rpy):
    """Converte RPY (roll, pitch, yaw) in quaternione [qx, qy, qz, qw]."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return np.array([
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    ])

def _quat_to_matrix(q):
    """Quaternione [qx, qy, qz, qw] → matrice di rotazione 3x3."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def _quat_multiply(q1, q2):
    """Prodotto quaternionico q1 ⊗ q2, formato [qx, qy, qz, qw]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


# Pre-calcola la trasformazione del giunto di montaggio una volta sola
_q_mount     = _euler_rpy_to_quat(_RIGHT_BASE_RPY)
_R_mount     = _quat_to_matrix(_q_mount)
_q_mount_inv = np.array([-_q_mount[0], -_q_mount[1], -_q_mount[2], _q_mount[3]])


def godot_to_pybullet(godot_pos, godot_quat):
    """
    Godot (Y su, destrorso) → PyBullet world (Z su, destrorso).

        pb_X =  g_X
        pb_Y = -g_Z
        pb_Z =  g_Y

    Il quaternione segue gli stessi assi.
    """
    gx, gy, gz = godot_pos
    gqx, gqy, gqz, gqw = godot_quat
    return (
        np.array([gx, -gz, gy]),
        np.array([gqx, -gqz, gqy, gqw]),
    )


def world_to_arm_base(pb_pos, pb_quat):
    """
    PyBullet world → frame base del braccio (openarm_right).
    Applica l'inverso della trasformazione del giunto di montaggio.
    """
    arm_pos  = _R_mount.T @ (np.array(pb_pos) - _RIGHT_BASE_T)
    arm_quat = _quat_multiply(_q_mount_inv, np.array(pb_quat))
    return arm_pos, arm_quat


# ──────────────────────────────────────────
#  STATO GLOBALE DEL SERVER
# ──────────────────────────────────────────

_model    = None
_scaler_x = None
_scaler_y = None


def load_model(ckpt_path: str, scaler_dir: str):
    global _model, _scaler_x, _scaler_y
    print(f"Caricamento modello: {ckpt_path}")
    _model = LitIKRegressor.load_from_checkpoint(ckpt_path, map_location="cpu")
    _model.eval()
    _scaler_x = joblib.load(os.path.join(scaler_dir, "scaler_x.pkl"))
    _scaler_y = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))
    hp = _model.hparams
    n  = sum(p.numel() for p in _model.parameters())
    print(f"  hidden={hp.hidden_dim}, blocks={hp.n_blocks}, params={n:,}")
    print("Modello pronto.\n")


def ik_infer(pos_arm, quat_arm) -> np.ndarray:
    """Normalizza l'input, chiama la rete, de-normalizza l'output."""
    pos_norm = _scaler_x.transform([pos_arm.tolist()])[0].tolist()
    pose = pos_norm + quat_arm.tolist()
    x = torch.tensor([pose], dtype=torch.float32)
    with torch.no_grad():
        pred_norm = _model(x).numpy()
    return _scaler_y.inverse_transform(pred_norm)[0]


def run_pipeline(godot_pos, godot_quat) -> np.ndarray:
    """Catena completa: Godot → PyBullet → arm base → rete."""
    pb_pos,  pb_quat  = godot_to_pybullet(godot_pos, godot_quat)
    arm_pos, arm_quat = world_to_arm_base(pb_pos, pb_quat)
    return ik_infer(arm_pos, arm_quat)


# ──────────────────────────────────────────
#  HTTP SERVER
# ──────────────────────────────────────────

class IKHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Stampa solo richieste IK, sopprime i log HTTP di default
        pass

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ik":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"JSON non valido: {e}"})
            return

        # Validazione input
        try:
            pos  = list(map(float, body["pos"]))
            quat = list(map(float, body["quat"]))
            assert len(pos)  == 3, "pos deve avere 3 elementi"
            assert len(quat) == 4, "quat deve avere 4 elementi [qx,qy,qz,qw]"
        except (KeyError, AssertionError, TypeError, ValueError) as e:
            self._send_json(400, {"error": str(e)})
            return

        try:
            joints = run_pipeline(np.array(pos), np.array(quat))
        except Exception as e:
            self._send_json(500, {"error": f"Inferenza fallita: {e}"})
            return

        result = {
            "joints":     joints.tolist(),
            "joints_deg": [math.degrees(j) for j in joints],
        }

        # Log compatto
        print(f"  pos={[round(v,3) for v in pos]}  "
              f"→ joints=[{', '.join(f'{math.degrees(j):+.1f}°' for j in joints)}]")

        self._send_json(200, result)


# ──────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IK inference server")
    parser.add_argument("--ckpt",      default=DEFAULT_CKPT)
    parser.add_argument("--scaler-dir", default=DEFAULT_SCALER_DIR)
    parser.add_argument("--port",      type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    missing = [f for f in [
        args.ckpt,
        os.path.join(args.scaler_dir, "scaler_x.pkl"),
        os.path.join(args.scaler_dir, "scaler_y.pkl"),
    ] if not os.path.exists(f)]

    if missing:
        print("❌ File mancanti:")
        for f in missing:
            print(f"   {f}")
        sys.exit(1)

    load_model(args.ckpt, args.scaler_dir)

    server = HTTPServer(("localhost", args.port), IKHandler)
    print(f"Server in ascolto su http://localhost:{args.port}")
    print(f"  POST /ik      → inferenza")
    print(f"  GET  /health  → health check")
    print(f"  Ctrl+C per fermare\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer fermato.")


if __name__ == "__main__":
    main()