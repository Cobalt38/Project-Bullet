#!/usr/bin/env python3
"""
IK Inference Script – OpenArm Right (7-DOF)

Due modalità operative:
  --mode local   → inferenza diretta nel processo (utile per test PyBullet)
  --mode server  → avvia un server TCP e accetta richieste JSON via socket

Se non si passa --mode, viene chiesto interattivamente.

Formato richiesta socket (JSON su una riga, terminata da '\\n'):
  {"x": 0.3, "y": 0.1, "z": 0.5, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Formato risposta socket (JSON su una riga):
  {"angles_deg": [j1, j2, j3, j4, j5, j6, j7], "angles_rad": [...], "sin_cos": [...]}

Esempio uso locale:
  python ik_inference.py --model_dir ik_model --mode local

Esempio server:
  python ik_inference.py --model_dir ik_model --mode server --host 0.0.0.0 --port 9999
"""

import argparse
import json
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Custom layer/loss (devono corrispondere esattamente a quelli del training)
# ---------------------------------------------------------------------------

class MinMaxNormalization(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self._scale = np.array(scale, dtype=np.float32)

    def build(self, input_shape):
        self.scale = tf.constant(self._scale, dtype=tf.float32, name="scale")
        super().build(input_shape)

    def call(self, x, training=False):
        return x / self.scale

    def get_config(self):
        cfg = super().get_config()
        cfg["scale"] = self._scale.tolist()
        return cfg

    @classmethod
    def from_config(cls, config):
        config["scale"] = np.array(config["scale"], dtype=np.float32)
        return cls(**config)


class QuatTo6D(tf.keras.layers.Layer):
    def call(self, x, training=False):
        pos = x[..., :3]
        q   = x[..., 3:]
        q   = tf.math.l2_normalize(q, axis=-1)
        qx, qy, qz, qw = q[...,0], q[...,1], q[...,2], q[...,3]
        r00 = 1 - 2*(qy**2 + qz**2)
        r10 =     2*(qx*qy + qw*qz)
        r20 =     2*(qx*qz - qw*qy)
        r01 =     2*(qx*qy - qw*qz)
        r11 = 1 - 2*(qx**2 + qz**2)
        r21 =     2*(qy*qz + qw*qx)
        six_d = tf.stack([r00, r10, r20, r01, r11, r21], axis=-1)
        return tf.concat([pos, six_d], axis=-1)

    def get_config(self):
        return super().get_config()


class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, x, training=None):
        if training:
            return x + tf.random.normal(tf.shape(x), stddev=self.stddev)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg["stddev"] = self.stddev
        return cfg


def ik_loss(y_true, y_pred):
    sin_p, cos_p = y_pred[:, 0::2], y_pred[:, 1::2]
    norm = tf.sqrt(sin_p**2 + cos_p**2)
    sin_p_n = sin_p / (norm + 1e-8)
    cos_p_n = cos_p / (norm + 1e-8)
    sin_t, cos_t = y_true[:, 0::2], y_true[:, 1::2]
    angular  = tf.reduce_mean(1.0 - (sin_t*sin_p_n + cos_t*cos_p_n))
    unit_pen = tf.reduce_mean((norm - 1.0)**2)
    return angular + 0.05 * unit_pen


CUSTOM_OBJECTS = {
    "MinMaxNormalization": MinMaxNormalization,
    "QuatTo6D":            QuatTo6D,
    "GaussianNoise":       GaussianNoise,
    "ik_loss":             ik_loss,
}

# ---------------------------------------------------------------------------
# Caricamento modello
# ---------------------------------------------------------------------------

def load_model_and_metadata(model_dir: str):
    model_dir = Path(model_dir)

    # Prova prima il checkpoint migliore, poi il modello finale
    ckpt = model_dir / "checkpoints" / "best_model.keras"
    final = model_dir / "model.keras"

    if ckpt.exists():
        model_path = ckpt
    elif final.exists():
        model_path = final
    else:
        raise FileNotFoundError(
            f"Nessun modello trovato in {model_dir}.\n"
            f"Cercati:\n  {ckpt}\n  {final}"
        )

    print(f"Caricamento modello da: {model_path}")
    model = tf.keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)
    print("Modello caricato.")

    metadata_path = model_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Metadata caricati: {len(metadata.get('output_columns', []))} output cols")
    else:
        print("ATTENZIONE: metadata.json non trovato. I nomi dei giunti non saranno disponibili.")

    return model, metadata


# ---------------------------------------------------------------------------
# Inferenza
# ---------------------------------------------------------------------------

def build_input(x: float, y: float, z: float,
                qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([[x, y, z, qx, qy, qz, qw]], dtype=np.float32)


def run_inference(model, input_array: np.ndarray) -> np.ndarray:
    """Ritorna il vettore sin/cos raw del modello."""
    return model(input_array, training=False).numpy()[0]


def decode_output(raw: np.ndarray, output_cols: List[str]) -> Dict:
    """
    Converte sin/cos → angoli in gradi e radianti.
    raw: vettore [sin_j1, cos_j1, sin_j2, cos_j2, ...]
    """
    sin_vals = raw[0::2]
    cos_vals = raw[1::2]
    angles_rad = np.arctan2(sin_vals, cos_vals)
    angles_deg = np.rad2deg(angles_rad)

    joint_names = []
    if output_cols:
        joint_names = [c[:-4] for c in output_cols if c.endswith("_sin")]

    return {
        "joint_names": joint_names,
        "angles_deg":  angles_deg.tolist(),
        "angles_rad":  angles_rad.tolist(),
        "sin_cos":     raw.tolist(),
    }


# Alias di compatibilità (nel caso ik_inference.py precedente avesse nomi diversi)
load_model = load_model_and_metadata

def pose_from_dict(d: Dict) -> tuple:
    return (
        float(d["x"]),  float(d["y"]),  float(d["z"]),
        float(d["qx"]), float(d["qy"]), float(d["qz"]), float(d["qw"]),
    )


# ---------------------------------------------------------------------------
# Modalità LOCAL – REPL interattivo (+ hook PyBullet)
# ---------------------------------------------------------------------------

def run_local(model, metadata: Dict):
    """
    REPL che legge pose da stdin e stampa gli angoli.
    Puoi importare questa funzione da un altro script PyBullet:

        from ik_inference import load_model_and_metadata, run_inference, decode_output, build_input
        model, meta = load_model_and_metadata("ik_model")
        raw = run_inference(model, build_input(x, y, z, qx, qy, qz, qw))
        result = decode_output(raw, meta.get("output_columns", []))
    """
    output_cols = metadata.get("output_columns", [])
    print("\n=== Modalità LOCAL ===")
    print("Inserisci la posa target come JSON su una riga, oppure digita 'q' per uscire.")
    print('Formato: {"x": 0.3, "y": 0.1, "z": 0.5, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}')
    print()

    while True:
        try:
            line = input("pose> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita.")
            break

        if line.lower() in ("q", "quit", "exit"):
            break
        if not line:
            continue

        try:
            d = json.loads(line)
            x, y, z, qx, qy, qz, qw = pose_from_dict(d)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [ERRORE] JSON non valido: {e}")
            continue

        inp = build_input(x, y, z, qx, qy, qz, qw)
        raw = run_inference(model, inp)
        result = decode_output(raw, output_cols)

        print(f"  Angoli (deg): {[f'{a:.2f}' for a in result['angles_deg']]}")
        if result["joint_names"]:
            for name, deg, rad in zip(result["joint_names"],
                                      result["angles_deg"],
                                      result["angles_rad"]):
                print(f"    {name:30s}  {deg:8.3f}°  ({rad:.4f} rad)")
        print()


# ---------------------------------------------------------------------------
# Modalità SERVER – socket TCP
# ---------------------------------------------------------------------------

def handle_client(conn: socket.socket, addr, model, output_cols: List[str]):
    print(f"[SERVER] Connessione da {addr}")
    buffer = b""
    try:
        with conn:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line.decode("utf-8"))
                        x, y, z, qx, qy, qz, qw = pose_from_dict(d)
                        inp = build_input(x, y, z, qx, qy, qz, qw)
                        raw = run_inference(model, inp)
                        result = decode_output(raw, output_cols)
                        response = json.dumps({
                            "angles_deg": result["angles_deg"],
                            "angles_rad": result["angles_rad"],
                            "sin_cos":    result["sin_cos"],
                        }) + "\n"
                        conn.sendall(response.encode("utf-8"))
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        err = json.dumps({"error": str(e)}) + "\n"
                        conn.sendall(err.encode("utf-8"))
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        print(f"[SERVER] Connessione chiusa da {addr}")


def run_server(model, metadata: Dict, host: str, port: int):
    output_cols = metadata.get("output_columns", [])
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(8)
    print(f"\n=== Modalità SERVER ===")
    print(f"In ascolto su {host}:{port}")
    print("Ctrl+C per fermare.")
    print()
    print("Esempio di richiesta (una riga JSON):")
    print('  {"x":0.3,"y":0.1,"z":0.5,"qx":0.0,"qy":0.0,"qz":0.0,"qw":1.0}')
    print()

    try:
        while True:
            conn, addr = server_sock.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, model, output_cols),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Fermato.")
    finally:
        server_sock.close()


# ---------------------------------------------------------------------------
# Selezione modalità interattiva
# ---------------------------------------------------------------------------

def ask_mode() -> str:
    print("\nSeleziona modalità operativa:")
    print("  [1] local   – inferenza diretta (REPL / test PyBullet)")
    print("  [2] server  – socket TCP (JSON)")
    while True:
        try:
            key = input("Scelta [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if key == "1":
            return "local"
        if key == "2":
            return "server"
        print("  Digita 1 oppure 2.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IK Inference – OpenArm Right",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model_dir", default="ik_model",
        help="Directory del modello (default: ik_model)",
    )
    p.add_argument(
        "--mode", choices=["local", "server"], default=None,
        help="Modalità: 'local' (REPL) o 'server' (socket TCP). "
             "Se omesso, viene chiesto interattivamente.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Host per il server (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=9999,  help="Porta per il server (default: 9999)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model, metadata = load_model_and_metadata(args.model_dir)

    mode = args.mode if args.mode else ask_mode()

    if mode == "local":
        run_local(model, metadata)
    elif mode == "server":
        run_server(model, metadata, args.host, args.port)
    else:
        print(f"Modalità sconosciuta: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()