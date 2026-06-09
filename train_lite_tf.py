#!/usr/bin/env python3
"""
IK trainer – feed-forward con orientamento mano come quaternione.

Input : target_x, target_y, target_z, hand_quat_qx/qy/qz/qw  (7 colonne)
Output: sin/cos degli angoli dei giunti  (colonne *_sin / *_cos già nel CSV)

Il CSV non viene mai caricato interamente: viene letto in chunk da un
generatore Python e alimentato a TF tramite tf.data.Dataset.from_generator.
In RAM vive al massimo un chunk alla volta.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------------------------
# Colonne fisse di input / output
# ---------------------------------------------------------------------------

INPUT_COLS = [
    "target_x", "target_y", "target_z",
    "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw",
]

# Colonne di output: tutto ciò che finisce con _sin o _cos
# (vengono rilevate leggendo solo l'header del CSV)

def detect_output_cols(csv_path: str) -> List[str]:
    """Legge solo l'header e restituisce le colonne *_sin / *_cos."""
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols = [c for c in header if c.endswith("_sin") or c.endswith("_cos")]
    if not cols:
        raise ValueError(
            f"Nessuna colonna *_sin/*_cos trovata nel CSV: {csv_path}\n"
            f"Colonne disponibili: {header}"
        )
    return cols


# ---------------------------------------------------------------------------
# Generatori per il dataset streaming
# ---------------------------------------------------------------------------

def _split_mask(row: tf.Tensor, lo: float, hi: float) -> tf.Tensor:
    weights = tf.constant([73856093, 19349663, 83492791, 46187201, 61339669, 91228751, 37312693], tf.int64)
    key = tf.reduce_sum(
        tf.cast(row[:7] * 1e4, tf.int64) * weights
    )
    h = tf.cast(
        tf.bitwise.bitwise_and(
            key * tf.constant(2654435761, tf.int64),
            tf.constant(0x7FFFFFFF, tf.int64)
        ),
        tf.float32
    ) / 2147483648.0
    return (h >= lo) & (h < hi)

def make_streaming_dataset(
    csv_path: str,
    output_cols: List[str],
    chunk_size: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    split_lo: float = 0.0,
    split_hi: float = 1.0
) -> tf.data.Dataset:
    """
    Crea un tf.data.Dataset che legge il CSV in streaming, senza caricare
    tutto in RAM.

    split_lo / split_hi permettono di suddividere il file in
    train / val / test in modo deterministico senza shuffle globale:
      - train : split_lo=0.0,  split_hi=0.80
      - val   : split_lo=0.80, split_hi=0.90
      - test  : split_lo=0.90, split_hi=1.0
    """
    n_in  = len(INPUT_COLS)
    n_out = len(output_cols)

    # Associa le colonne di output al generatore tramite attributo
    all_cols = INPUT_COLS + output_cols

    def _generator():
        for chunk in pd.read_csv(
            csv_path,
            usecols=all_cols,
            dtype=np.float32,
            chunksize=chunk_size,
            on_bad_lines="skip",
        ):
            chunk = chunk.dropna(subset=all_cols)
            if chunk.empty:
                continue
            yield from chunk[all_cols].to_numpy(dtype=np.float32)

    out_sig = tf.TensorSpec(shape=(n_in + n_out,), dtype=tf.float32)

    ds = tf.data.Dataset.from_generator(
        _generator,
        output_signature=out_sig,
    )

    # Suddivisione train/val/test per posizione nel file (hash delle prime 7 colonne)
    if not (split_lo == 0.0 and split_hi == 1.0):
        ds = ds.filter(lambda row: _split_mask(row, split_lo, split_hi))

    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    # Separa input / output
    ds = ds.map(
        lambda row: (row[:n_in], row[n_in:]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Stima rapida del numero di righe (per i log, non per caricamento)
# ---------------------------------------------------------------------------

def estimate_rows(csv_path: str) -> int:
    """Conta le righe tramite wc -l (veloce, stima)."""
    import subprocess
    try:
        out = subprocess.check_output(["wc", "-l", csv_path]).split()
        return int(out[0]) - 1  # sottrae header
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Normalizzazione: calcola media/std su un campione del CSV
# ---------------------------------------------------------------------------

class WelfordNormalization(tf.keras.layers.Layer):

    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features

    def build(self, input_shape):

        self.mean = self.add_weight(
            name="mean",
            shape=(self.n_features,),
            initializer="zeros",
            trainable=False,
        )

        self.std = self.add_weight(
            name="std",
            shape=(self.n_features,),
            initializer="ones",
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x, training=False):

        return (x - self.mean) / (self.std + 1e-6)

    def set_stats(self, mean, std):

        self.mean.assign(mean)

        self.std.assign(std)

def compute_streaming_stats(ds, n_features):
    """
    Calcola mean/std in streaming usando Welford batch-wise.
    Non carica mai l'intero dataset in RAM.
    """

    n = 0

    mean = np.zeros(n_features, dtype=np.float64)

    M2 = np.zeros(n_features, dtype=np.float64)

    for x_batch, _ in ds:

        batch = x_batch.numpy().astype(np.float64)

        batch_n = batch.shape[0]

        if batch_n == 0:
            continue

        batch_mean = np.mean(batch, axis=0)

        batch_M2 = np.sum(
            (batch - batch_mean) ** 2,
            axis=0,
        )

        if n == 0:

            mean = batch_mean

            M2 = batch_M2

            n = batch_n

            continue

        delta = batch_mean - mean

        new_n = n + batch_n

        mean = mean + delta * batch_n / new_n

        M2 = M2 + batch_M2 + (delta ** 2) * n * batch_n / new_n

        n = new_n

    std = np.sqrt(M2 / max(n - 1, 1))

    std[std < 1e-6] = 1.0

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )

# ---------------------------------------------------------------------------
# Architettura della rete
# ---------------------------------------------------------------------------

def build_model(
    input_dim: int,
    output_dim: int,
    hidden: List[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    loss_name: str,
    huber_delta: float,
) -> Tuple[tf.keras.Model, WelfordNormalization]:
    """
    Feed-forward con skip connection, LayerNorm, GELU.
    La normalizzazione degli input è integrata nel modello (Normalization layer).
    """
    inp  = tf.keras.Input(shape=(input_dim,), name="target_pos")
    norm = WelfordNormalization(n_features=input_dim, name="input_norm")
    x = norm(inp)

    for i, units in enumerate(hidden):
        residual = x
        x = tf.keras.layers.Dense(units, name=f"dense_{i+1}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"ln_{i+1}")(x)
        x = tf.keras.layers.Activation("gelu", name=f"act_{i+1}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"drop_{i+1}")(x)
        if residual.shape[-1] == units:
            x = tf.keras.layers.Add(name=f"res_{i+1}")([residual, x])

    out   = tf.keras.layers.Dense(output_dim, name="output")(x)
    model = tf.keras.Model(inp, out)

    if hasattr(tf.keras.optimizers, "AdamW") and weight_decay > 0:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.Huber(delta=huber_delta) if loss_name == "huber" else "mse"
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model, norm


# ---------------------------------------------------------------------------
# Callback: stop quando la loss scende sotto una soglia
# ---------------------------------------------------------------------------

class StopOnLossTarget(tf.keras.callbacks.Callback):
    def __init__(self, loss_target: float, monitor: str = "val_loss"):
        super().__init__()
        self.loss_target = loss_target
        self.monitor     = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        val = (logs or {}).get(self.monitor)
        if val is not None and val < self.loss_target:
            print(
                f"\nEpoch {epoch+1}: {self.monitor}={val:.6f} "
                f"< target {self.loss_target:.6f} → stop."
            )
            self.model.stop_training = True


# ---------------------------------------------------------------------------
# Valutazione finale su un set
# ---------------------------------------------------------------------------

def evaluate_set(
    name: str,
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    output_cols: List[str],
) -> Dict:
    """
    Valuta il modello su un Dataset e stampa MAE/RMSE per colonna.
    Decodifica sin/cos → angoli e usa differenza angolare wrapped.
    """
    all_true, all_pred = [], []
    for x_batch, y_batch in ds:
        pred = model(x_batch, training=False).numpy()
        all_true.append(y_batch.numpy())
        all_pred.append(pred)

    if not all_true:
        print(f"{name}: vuoto")
        return {}

    true = np.concatenate(all_true, axis=0)
    pred = np.concatenate(all_pred, axis=0)

    # Decodifica sin/cos → gradi, differenza wrapped in [-180, 180]
    n_joints = true.shape[1] // 2
    true_angles = np.rad2deg(np.arctan2(true[:, 0::2], true[:, 1::2]))
    pred_angles = np.rad2deg(np.arctan2(pred[:, 0::2], pred[:, 1::2]))
    diff = (pred_angles - true_angles + 180.0) % 360.0 - 180.0

    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    per_col_mae = np.mean(np.abs(diff), axis=0)

    # Nomi giunti (da colonne *_sin)
    joint_names = [c[:-4] for c in output_cols if c.endswith("_sin")]
    worst_idx   = np.argsort(per_col_mae)[-5:][::-1]

    print(f"\n{name}  MAE={mae:.4f}°  RMSE={rmse:.4f}°")
    print(f"{name} worst joints:")
    for i in worst_idx:
        jname = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        print(f"  {jname}: {float(per_col_mae[i]):.4f}°")

    return {
        "rows": int(true.shape[0]),
        "mae_deg": mae,
        "rmse_deg": rmse,
        "worst": [
            {"joint": joint_names[int(i)] if int(i) < len(joint_names) else f"joint_{i}",
             "mae_deg": float(per_col_mae[int(i)])}
            for i in worst_idx
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IK trainer – streaming dataset")
    p.add_argument("--csv",         required=True,      help="Percorso del file CSV")
    p.add_argument("--model_dir",   default="ik_model", help="Directory output")
    p.add_argument("--epochs",      type=int,   default=120)
    p.add_argument("--batch_size",  type=int,   default=4096,
                   help="Batch size (grande è ok perché i dati non stanno in RAM)")
    p.add_argument("--chunk_size",  type=int,   default=500_000,
                   help="Righe per chunk pandas (default 500k ~= pochi GB alla volta)")
    p.add_argument("--hidden",      type=int, nargs="+", default=[256, 256, 256, 128])
    p.add_argument("--dropout",     type=float, default=0.05)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--loss",        choices=["mse", "huber"], default="huber")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--val_split",   type=float, default=0.10,
                   help="Frazione del dataset per la validation (0.10 = 10%)")
    p.add_argument("--test_split",  type=float, default=0.10,
                   help="Frazione del dataset per il test (0.10 = 10%)")
    p.add_argument("--shuffle_buffer", type=int, default=50_000,
                   help="Buffer shuffle TF (0 = disabilita, riduce RAM)")
    p.add_argument("--early_stop_patience",   type=int,   default=20)
    p.add_argument("--early_stop_min_delta",  type=float, default=1e-5)
    p.add_argument("--reduce_lr_patience",    type=int,   default=6)
    p.add_argument("--reduce_lr_factor",      type=float, default=0.5)
    p.add_argument("--min_lr",                type=float, default=1e-6)
    p.add_argument("--loss_target", type=float, default=0.0,
                   help="Ferma il training se val_loss < soglia (0 = disabilita)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.val_split < 0 or args.test_split < 0 or (args.val_split + args.test_split) >= 1.0:
        raise ValueError("val_split + test_split deve essere < 1.0")

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    output_cols = detect_output_cols(csv_path)
    n_in  = len(INPUT_COLS)
    n_out = len(output_cols)
    train_fraction = 1.0 - args.val_split - args.test_split

    size_gb = os.path.getsize(csv_path) / 1024**3
    # est_rows = estimate_rows(csv_path)
    # print(f"CSV: {csv_path}  ({size_gb:.1f} GB, ~{est_rows:,} righe stimato)")
    print(f"Input  ({n_in}): {INPUT_COLS}")
    print(f"Output ({n_out}): {output_cols}")
    print(f"Split: train={train_fraction:.0%}  val={args.val_split:.0%}  test={args.test_split:.0%}")
    print(f"Chunk size pandas: {args.chunk_size:,} righe")

    # ------------------------------------------------------------------
    # Dataset streaming (nessun array globale)
    # ------------------------------------------------------------------
    common = dict(
        csv_path=csv_path,
        output_cols=output_cols,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        seed=42,
    )
    train_ds = make_streaming_dataset(**common, shuffle_buffer=args.shuffle_buffer,
                                    split_lo=0.0,                               split_hi=train_fraction)
    val_ds   = make_streaming_dataset(**common, shuffle_buffer=0,
                                    split_lo=train_fraction,                    split_hi=train_fraction + args.val_split)
    test_ds  = make_streaming_dataset(**common, shuffle_buffer=0,
                                    split_lo=train_fraction + args.val_split,   split_hi=1.0)
    
    print("\nCalcolo statistiche di normalizzazione...")

    norm_mean, norm_std = compute_streaming_stats(
        train_ds,
        n_in,
    )

    print("Statistiche calcolate.")

    # ------------------------------------------------------------------
    # Modello
    # ------------------------------------------------------------------
    model, norm_layer = build_model(
        input_dim=n_in,
        output_dim=n_out,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        huber_delta=args.huber_delta,
    )

    norm_layer.set_stats(
        norm_mean,
        norm_std,
    )
    model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    os.makedirs(args.model_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    monitor = "val_loss" if args.val_split > 0 else "loss"

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        # checkpoint per-epoca (per analisi post-training)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, f"epoch_{{epoch:04d}}.keras"),
            monitor=monitor,
            save_best_only=False,
            verbose=0,
        ),
        # checkpoint best (sovrascrive)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best_model.keras"),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            restore_best_weights=True,
        ),
    ]
    if args.loss_target > 0.0:
        callbacks.append(StopOnLossTarget(args.loss_target, monitor=monitor))
        print(f"Auto-stop quando {monitor} < {args.loss_target}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\nTraining → max {args.epochs} epoche, batch {args.batch_size}")
    history = model.fit(
        train_ds,
        validation_data=val_ds if args.val_split > 0 else None,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Salvataggio
    # ------------------------------------------------------------------
    model_path = os.path.join(args.model_dir, "model.keras")
    model.save(model_path)

    metadata = {
        "input_columns":  INPUT_COLS,
        "output_columns": output_cols,
        "hidden":         args.hidden,
        "dropout":        args.dropout,
        "loss":           args.loss,
        "huber_delta":    args.huber_delta if args.loss == "huber" else None,
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "weight_decay":   args.weight_decay,
        "norm_mean":      norm_layer.mean.numpy().tolist(),
        "norm_std":       norm_layer.std.numpy().tolist(),
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Valutazione finale
    # ------------------------------------------------------------------
    val_metrics  = evaluate_set("Validation", model, val_ds,  output_cols)
    test_metrics = evaluate_set("Test",        model, test_ds, output_cols)

    report = {
        "config":     vars(args),
        "csv_path":   csv_path,
        "history":    {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "validation": val_metrics,
        "test":       test_metrics,
    }
    with open(os.path.join(args.model_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModello salvato  → {model_path}")
    print(f"Metadata         → {os.path.join(args.model_dir, 'metadata.json')}")
    print(f"Report           → {os.path.join(args.model_dir, 'train_report.json')}")
    print(f"Checkpoints      → {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())