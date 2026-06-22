#!/usr/bin/env python3
"""
Quantizzazione del modello IK – OpenArm Right

Modalità disponibili:
  float16   → conversione TFLite float16 (safe, ~2x più piccolo)
  int8      → PTQ INT8 con dataset di calibrazione dal Parquet (più veloce su CPU)
  qat       → Quantization-Aware Training (usa se PTQ degrada troppo l'accuratezza)

Uso:
  python quantize_model.py --model_dir ik_model --mode float16
  python quantize_model.py --model_dir ik_model --mode int8 --parquet_file dataset.parquet
  python quantize_model.py --model_dir ik_model --mode qat   --parquet_file dataset.parquet

Output:
  ik_model/quantized/model_float16.tflite
  ik_model/quantized/model_int8.tflite
  ik_model/quantized/model_qat/          (SavedModel per QAT, poi convertibile a TFLite)

Per usare il modello TFLite nell'inferenza:
  from quantize_model import TFLitePredictor
  predictor = TFLitePredictor("ik_model/quantized/model_float16.tflite")
  angles_rad = predictor.predict(x, y, z, qx, qy, qz, qw)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf

# ---------------------------------------------------------------------------
# Import custom objects dallo stesso ik_inference.py
# ---------------------------------------------------------------------------
from train_tf_parquet import (
    MinMaxNormalization, QuatTo6D, GaussianNoise,
    ik_loss, INPUT_COLS, INPUT_SCALE
)

from ik_inference import load_model_and_metadata

CUSTOM_OBJECTS = {
    "MinMaxNormalization": MinMaxNormalization,
    "QuatTo6D":            QuatTo6D,
    "GaussianNoise":       GaussianNoise,
    "ik_loss":             ik_loss,
}

# ---------------------------------------------------------------------------
# Dataset di calibrazione per INT8 (legge N campioni casuali dal Parquet)
# ---------------------------------------------------------------------------

def build_calibration_dataset(parquet_path: str, n_samples: int = 1024):
    pf        = pq.ParquetFile(parquet_path)
    n_groups  = pf.metadata.num_row_groups
    rng       = np.random.default_rng(42)

    group_idx = int(rng.integers(0, n_groups))
    table     = pf.read_row_group(group_idx, columns=INPUT_COLS)
    data      = table.to_pandas().values.astype(np.float32)

    if len(data) > n_samples:
        idx  = rng.choice(len(data), n_samples, replace=False)
        data = data[idx]

    print(f"[CALIB] {len(data)} campioni da row group {group_idx}")

    # TFLite representative_dataset vuole un callable che yielda
    # una lista di array — uno per ogni input tensor del modello
    def representative_dataset():
        for row in data:
            yield [row.reshape(1, -1)]  # lista con un elemento: shape [1, 7]

    return representative_dataset

# ---------------------------------------------------------------------------
# Float16
# ---------------------------------------------------------------------------

def convert_float16(model: tf.keras.Model, out_path: str):
    print("\n[FLOAT16] Conversione in corso...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations          = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[FLOAT16] Salvato: {out_path}  ({size_mb:.2f} MB)")

# ---------------------------------------------------------------------------
# Float32
# ---------------------------------------------------------------------------

def convert_float32(model: tf.keras.Model, out_path: str):
    print("\n[FLOAT32] Conversione in corso...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations          = []
    # converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[FLOAT32] Salvato: {out_path}  ({size_mb:.2f} MB)")

# ---------------------------------------------------------------------------
# INT8 PTQ
# ---------------------------------------------------------------------------

def convert_int8(model: tf.keras.Model, parquet_path: str, out_path: str,
                 n_calib_samples: int = 1024):
    print("\n[INT8] Conversione con calibrazione in corso...")
    calib_ds = build_calibration_dataset(parquet_path, n_calib_samples)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations                      = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset             = calib_ds
    converter.target_spec.supported_ops          = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type               = tf.float32   # input resta float per comodità
    converter.inference_output_type              = tf.float32   # output resta float

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[INT8] Conversione INT8 fallita: {e}")
        print("[INT8] Provo fallback a INT8 con ops miste (alcune op restano float)...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[INT8] Salvato: {out_path}  ({size_mb:.2f} MB)")

# ---------------------------------------------------------------------------
# QAT (Quantization-Aware Training)
# ---------------------------------------------------------------------------

def run_qat(model: tf.keras.Model, parquet_path: str, out_dir: str,
            epochs: int = 10, batch_size: int = 512):
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        print("[QAT] tensorflow_model_optimization non installato.")
        print("      pip install tensorflow-model-optimization")
        sys.exit(1)

    print("\n[QAT] Annotazione layer per quantization-aware training...")

    # Annota solo i Dense standard — i layer custom non hanno pesi da quantizzare
    def annotate(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated = tf.keras.models.clone_model(model, clone_function=annotate)
    annotated.set_weights(model.get_weights())

    with tfmot.quantization.keras.quantize_scope(CUSTOM_OBJECTS):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated)

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    qat_model.summary()

    # Dataset di fine-tuning (riuso lo stesso loader minimale)
    pf       = pq.ParquetFile(parquet_path)
    n_groups = pf.metadata.num_row_groups
    output_cols_all = [n for n in pq.read_schema(parquet_path).names
                       if n.endswith("_sin") or n.endswith("_cos")]
    all_cols = INPUT_COLS + output_cols_all
    n_in  = len(INPUT_COLS)

    def _gen():
        for g in range(n_groups):
            table = pf.read_row_group(g, columns=all_cols)
            data  = table.to_pandas().values.astype(np.float32)
            np.random.shuffle(data)
            for i in range(0, len(data) - batch_size, batch_size):
                batch = data[i:i + batch_size]
                yield batch[:, :n_in], batch[:, n_in:]

    n_out = len(output_cols_all)
    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, n_in),  dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, n_out), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)

    print(f"[QAT] Fine-tuning per {epochs} epoche...")
    qat_model.fit(ds, epochs=epochs, verbose=1)

    # Salva come SavedModel
    os.makedirs(out_dir, exist_ok=True)
    qat_model.save(os.path.join(out_dir, "model_qat.keras"))
    print(f"[QAT] Modello QAT salvato in: {out_dir}")

    # Converti subito a TFLite
    tflite_out = os.path.join(out_dir, "model_qat.tflite")
    converter  = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_out, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(tflite_out) / 1024 / 1024
    print(f"[QAT] TFLite salvato: {tflite_out}  ({size_mb:.2f} MB)")

# ---------------------------------------------------------------------------
# Benchmark: confronta latenza Keras vs TFLite
# ---------------------------------------------------------------------------

def benchmark(keras_model: tf.keras.Model, tflite_path: str, n_runs: int = 500):
    import time

    dummy = np.random.randn(1, len(INPUT_COLS)).astype(np.float32)

    # Keras
    keras_model(dummy, training=False)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_runs):
        keras_model(dummy, training=False)
    keras_ms = (time.perf_counter() - t0) / n_runs * 1000

    # TFLite
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    interp.set_tensor(inp_idx, dummy)
    interp.invoke()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_runs):
        interp.set_tensor(inp_idx, dummy)
        interp.invoke()
    tflite_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\n[BENCHMARK] ({n_runs} run, batch=1)")
    print(f"  Keras   : {keras_ms:.3f} ms/inference")
    print(f"  TFLite  : {tflite_ms:.3f} ms/inference")
    print(f"  Speedup : {keras_ms / tflite_ms:.2f}x")

# ---------------------------------------------------------------------------
# Valutazione accuratezza: confronta angoli Keras vs TFLite su N campioni
# ---------------------------------------------------------------------------

def evaluate_accuracy(keras_model: tf.keras.Model, tflite_path: str,
                      parquet_path: str, n_samples: int = 2048):
    pf         = pq.ParquetFile(parquet_path)
    output_cols = [n for n in pq.read_schema(parquet_path).names
                   if n.endswith("_sin") or n.endswith("_cos")]
    all_cols   = INPUT_COLS + output_cols
    n_in       = len(INPUT_COLS)

    table = pf.read_row_group(0, columns=all_cols)
    data  = table.to_pandas().values.astype(np.float32)
    if len(data) > n_samples:
        data = data[:n_samples]

    x    = data[:, :n_in]
    y_gt = data[:, n_in:]

    # Keras
    pred_keras = keras_model(x, training=False).numpy()

    # TFLite
    interp  = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    pred_tflite = np.zeros_like(pred_keras)
    for i, row in enumerate(x):
        interp.set_tensor(inp_idx, row.reshape(1, -1))
        interp.invoke()
        pred_tflite[i] = interp.get_tensor(out_idx)[0]

    def to_deg(raw):
        return np.rad2deg(np.arctan2(raw[:, 0::2], raw[:, 1::2]))

    deg_gt     = to_deg(y_gt)
    deg_keras  = to_deg(pred_keras)
    deg_tflite = to_deg(pred_tflite)

    def angular_mae(a, b):
        diff = (a - b + 180.0) % 360.0 - 180.0
        return np.mean(np.abs(diff))

    mae_keras  = angular_mae(deg_gt,    deg_keras)
    mae_tflite = angular_mae(deg_gt,    deg_tflite)
    mae_delta  = angular_mae(deg_keras, deg_tflite)

    print(f"\n[ACCURACY] ({len(x)} campioni)")
    print(f"  MAE Keras   vs GT     : {mae_keras:.4f}°")
    print(f"  MAE TFLite  vs GT     : {mae_tflite:.4f}°")
    print(f"  MAE TFLite  vs Keras  : {mae_delta:.4f}°  ← degradazione da quantizzazione")

# ---------------------------------------------------------------------------
# TFLitePredictor – drop-in replacement per run_inference()
# ---------------------------------------------------------------------------

class TFLitePredictor:
    """
    Uso:
        predictor = TFLitePredictor("ik_model/quantized/model_float16.tflite")
        angles_rad = predictor.predict(x, y, z, qx, qy, qz, qw)
    """
    def __init__(self, tflite_path: str):
        self.interp = tf.lite.Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        self.inp_idx = self.interp.get_input_details()[0]["index"]
        self.out_idx = self.interp.get_output_details()[0]["index"]
        print(f"[TFLitePredictor] Caricato: {tflite_path}")

    def predict_raw(self, x: float, y: float, z: float,
                    qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Ritorna il vettore sin/cos raw (identico a run_inference)."""
        inp = np.array([[x, y, z, qx, qy, qz, qw]], dtype=np.float32)
        self.interp.set_tensor(self.inp_idx, inp)
        self.interp.invoke()
        return self.interp.get_tensor(self.out_idx)[0]

    def predict(self, x: float, y: float, z: float,
                qx: float, qy: float, qz: float, qw: float) -> List[float]:
        """Ritorna direttamente la lista di angoli in radianti."""
        raw     = self.predict_raw(x, y, z, qx, qy, qz, qw)
        sin_v   = raw[0::2]
        cos_v   = raw[1::2]
        return np.arctan2(sin_v, cos_v).tolist()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quantizzazione modello IK")
    p.add_argument("--model_dir",      default="ik_model")
    p.add_argument("--mode",           choices=["float16", "int8", "qat", "float32"], default="float16")
    p.add_argument("--parquet_file",   default=None,
                   help="Richiesto per int8 e qat (dataset di calibrazione/fine-tuning)")
    p.add_argument("--n_calib",        type=int, default=1024,
                   help="Campioni di calibrazione per INT8 (default: 1024)")
    p.add_argument("--qat_epochs",     type=int, default=10)
    p.add_argument("--qat_batch_size", type=int, default=512)
    p.add_argument("--benchmark",      action="store_true",
                   help="Confronta latenza Keras vs TFLite dopo conversione")
    p.add_argument("--eval_accuracy",  action="store_true",
                   help="Confronta MAE angolare Keras vs TFLite")
    p.add_argument("--n_eval",         type=int, default=2048,
                   help="Campioni per eval accuratezza (default: 2048)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode in ("int8", "qat") and not args.parquet_file:
        print(f"[ERRORE] --mode {args.mode} richiede --parquet_file")
        sys.exit(1)

    out_dir = os.path.join(args.model_dir, "quantized")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Caricamento modello da: {args.model_dir}")
    model, metadata = load_model_and_metadata(args.model_dir)
    model.summary()

    if args.mode == "float16":
        out_path = os.path.join(out_dir, "model_float16.tflite")
        convert_float16(model, out_path)
        if args.benchmark:
            benchmark(model, out_path)
        if args.eval_accuracy and args.parquet_file:
            evaluate_accuracy(model, out_path, args.parquet_file, args.n_eval)

    elif args.mode == "int8":
        out_path = os.path.join(out_dir, "model_int8.tflite")
        convert_int8(model, args.parquet_file, out_path, args.n_calib)
        if args.benchmark:
            benchmark(model, out_path)
        if args.eval_accuracy:
            evaluate_accuracy(model, out_path, args.parquet_file, args.n_eval)
    
    elif args.mode == "float32":
        out_path = os.path.join(out_dir, "model_float32.tflite")
        convert_float32(model, out_path)
        if args.benchmark:
            benchmark(model, out_path)
        if args.eval_accuracy:
            evaluate_accuracy(model, out_path, args.parquet_file, args.n_eval)

    elif args.mode == "qat":
        run_qat(model, args.parquet_file, out_dir,
                epochs=args.qat_epochs, batch_size=args.qat_batch_size)


if __name__ == "__main__":
    main()