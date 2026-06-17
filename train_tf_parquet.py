#!/usr/bin/env python3
"""
train_ik.py  –  IK trainer con dataset Parquet, normalizzazione min-max fissa.

Flusso di lavoro:
  1. Converti il CSV una volta sola con convert_to_parquet.py
     → genera train.parquet, val.parquet, test.parquet
  2. Lancia questo script puntando alla directory Parquet

Miglioramenti rispetto alla versione CSV:
  - 0 iterazioni preliminari sul dataset (niente Welford, niente hash)
  - Lettura colonnare: PyArrow legge solo le colonne utili, skip delle altre
  - Split fisico: train/val/test sono file separati, nessun filtro per riga
  - Shuffle vero: mescolamento dell'ordine dei row group tra epoche (config.)
  - Compressione nativa Parquet: dataset su disco ~3-4x più piccolo
  - Normalizzazione min-max con range fissi noti a priori (hardcoded)

Input  : target_x, target_y, target_z, hand_quat_qx/qy/qz/qw  (7 col.)
Output : *_sin / *_cos per ogni giunto
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf

# ---------------------------------------------------------------------------
# Colonne
# ---------------------------------------------------------------------------

INPUT_COLS = [
    "target_x", "target_y", "target_z",
    "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw",
]

# Normalizzazione min-max fissa: scala per portare gli input in [-1, 1]
# x, y, z ∈ [-2, 2]  →  dividi per 2
# qx,qy,qz,qw ∈ [-1, 1]  →  già normalizzati (dividi per 1)
# Gli output (sin/cos) sono già in [-1,1] per definizione: nessuna normalizzazione.
INPUT_SCALE = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def detect_output_cols(parquet_path: str) -> List[str]:
    """Legge solo lo schema del Parquet (zero righe) e restituisce
    le colonne output in ordine [joint1_sin, joint1_cos, joint2_sin, ...]."""
    schema = pq.read_schema(parquet_path)
    sin_cols = [n for n in schema.names if n.endswith("_sin")]
    cos_cols = [n for n in schema.names if n.endswith("_cos")]

    if not sin_cols or not cos_cols:
        raise ValueError(
            f"Nessuna colonna *_sin/*_cos nel Parquet: {parquet_path}\n"
            f"Colonne disponibili: {schema.names}"
        )

    # Verifica che ogni _sin abbia il corrispondente _cos
    sin_joints = [c[:-4] for c in sin_cols]
    cos_joints = [c[:-4] for c in cos_cols]
    missing = set(sin_joints) ^ set(cos_joints)
    if missing:
        raise ValueError(f"Colonne sin/cos non appaiate per i giunti: {missing}")

    # Restituisce in ordine alternato: sin, cos, sin, cos, ...
    cols = []
    for joint in sin_joints:
        cols.append(f"{joint}_sin")
        cols.append(f"{joint}_cos")
    return cols


# ---------------------------------------------------------------------------
# Normalizzazione (layer senza pesi da apprendere o aggiornare)
# ---------------------------------------------------------------------------

class MinMaxNormalization(tf.keras.layers.Layer):
    """
    Normalizza gli input con scala fissa (range noti a priori).
    Non ha parametri trainable né variabili aggiornabili.
    x_norm = x / scale
    """

    def __init__(self, scale: np.ndarray, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self._scale = scale.astype(np.float32)

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


# ---------------------------------------------------------------------------
# Dataset Parquet → tf.data
# ---------------------------------------------------------------------------

def make_parquet_dataset(
    parquet_path: str,
    output_cols: List[str],
    batch_size: int,
    shuffle_row_groups: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Legge un file Parquet in streaming per row group.

    PyArrow legge un row group alla volta (default ~200k righe):
    - Mai più di un row group in RAM
    - Solo le colonne necessarie vengono deserializzate

    shuffle_row_groups=True: mescola l'ordine dei row group a ogni chiamata
    del generatore (cioè a ogni epoca). Non è uno shuffle globale, ma è
    molto più efficace dello shuffle CSV perché i row group sono stati
    scritti già con distribuzione spaziale diversa dallo split deterministico.
    """
    all_cols = INPUT_COLS + output_cols
    n_in  = len(INPUT_COLS)
    n_out = len(output_cols)

    n_row_groups = pq.ParquetFile(parquet_path).metadata.num_row_groups

    def _generator():
        # Apri il file: ogni worker ha il suo handle
        pf = pq.ParquetFile(parquet_path)
        rg_indices = list(range(n_row_groups))
        if shuffle_row_groups:
            random.shuffle(rg_indices)

        for rg_idx in rg_indices:
            table = pf.read_row_group(rg_idx, columns=all_cols)
            arr   = table.to_pydict()
            data  = np.column_stack(
                [np.asarray(arr[c], dtype=np.float32) for c in all_cols]
            )
            np.random.shuffle(data)
            yield from data

    out_sig = tf.TensorSpec(shape=(n_in + n_out,), dtype=tf.float32)
    ds = tf.data.Dataset.from_generator(_generator, output_signature=out_sig)
    ds = ds.map(lambda row: (row[:n_in], row[n_in:]),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Architettura
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
) -> tf.keras.Model:
    inp  = tf.keras.Input(shape=(input_dim,), name="target_pose")
    norm = MinMaxNormalization(scale=INPUT_SCALE, name="input_norm")
    x    = norm(inp)

    for i, units in enumerate(hidden):
        residual = x
        x = tf.keras.layers.Dense(units, name=f"dense_{i+1}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"ln_{i+1}")(x)
        x = tf.keras.layers.Activation("gelu", name=f"act_{i+1}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"drop_{i+1}")(x)
        if residual.shape[-1] != units:
            residual = tf.keras.layers.Dense(units, use_bias=False)(residual)
        x = tf.keras.layers.Add(name=f"res_{i+1}")([residual, x])

    out   = tf.keras.layers.Dense(output_dim, name="output")(x)
    model = tf.keras.Model(inp, out)

    if hasattr(tf.keras.optimizers, "AdamW") and weight_decay > 0:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.Huber(delta=huber_delta) if loss_name == "huber" else "mse"
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class StopOnLossTarget(tf.keras.callbacks.Callback):
    def __init__(self, loss_target: float, monitor: str = "val_loss"):
        super().__init__()
        self.loss_target = loss_target
        self.monitor     = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        val = (logs or {}).get(self.monitor)
        if val is not None and val < self.loss_target:
            print(f"\nEpoch {epoch+1}: {self.monitor}={val:.6f} < {self.loss_target:.6f} → stop.")
            self.model.stop_training = True

# class CheckpointEveryN(tf.keras.callbacks.ModelCheckpoint):
#     def __init__(self, every_n: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.every_n = every_n

#     def on_epoch_end(self, epoch, logs=None):
#         if (epoch + 1) % self.every_n == 0:
#             super().on_epoch_end(epoch, logs)

# ---------------------------------------------------------------------------
# Valutazione finale
# ---------------------------------------------------------------------------

def evaluate_set(
    name: str,
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    output_cols: List[str],
) -> Dict:
    n        = 0
    sum_abs  = None
    sum_sq   = None

    for x_batch, y_batch in ds:
        pred = model(x_batch, training=False).numpy()
        true = y_batch.numpy()

        true_angles = np.rad2deg(np.arctan2(true[:, 0::2], true[:, 1::2]))
        pred_angles = np.rad2deg(np.arctan2(pred[:, 0::2], pred[:, 1::2]))
        diff        = (pred_angles - true_angles + 180.0) % 360.0 - 180.0

        if sum_abs is None:
            sum_abs = np.zeros(diff.shape[1], dtype=np.float64)
            sum_sq  = np.zeros(diff.shape[1], dtype=np.float64)

        sum_abs += np.sum(np.abs(diff), axis=0)
        sum_sq  += np.sum(diff ** 2,    axis=0)
        n       += diff.shape[0]

    if n == 0:
        print(f"{name}: vuoto")
        return {}

    per_col_mae = (sum_abs / n).astype(np.float32)
    mae         = float(np.mean(per_col_mae))
    rmse        = float(np.sqrt(np.mean(sum_sq / n)))

    joint_names = [c[:-4] for c in output_cols if c.endswith("_sin")]
    worst_idx   = np.argsort(per_col_mae)[-5:][::-1]

    print(f"\n{name}  MAE={mae:.4f}°  RMSE={rmse:.4f}°")
    for i in worst_idx:
        jname = joint_names[i] if i < len(joint_names) else f"joint_{i}"
        print(f"  {jname}: {float(per_col_mae[i]):.4f}°")

    return {
        "rows":     n,
        "mae_deg":  mae,
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
    p = argparse.ArgumentParser(description="IK trainer – dataset Parquet")
    p.add_argument("--parquet_dir",  required=True,
                   help="Directory con train.parquet, val.parquet, test.parquet")
    p.add_argument("--model_dir",    default="ik_model")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=4096)
    p.add_argument("--hidden",       type=int,   nargs="+", default=[256, 256, 256, 128])
    p.add_argument("--dropout",      type=float, default=0.05)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--loss",         choices=["mse", "huber"], default="huber")
    p.add_argument("--huber_delta",  type=float, default=1.0)
    p.add_argument("--shuffle_row_groups", action="store_true",
                   help="Mescola l'ordine dei row group a ogni epoca (consigliato)")
    p.add_argument("--early_stop_patience",  type=int,   default=20)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    p.add_argument("--reduce_lr_patience",   type=int,   default=6)
    p.add_argument("--reduce_lr_factor",     type=float, default=0.5)
    p.add_argument("--min_lr",               type=float, default=1e-6)
    p.add_argument("--loss_target",          type=float, default=0.0)
    p.add_argument("--resume", action="store_true",
               help="Riprendi dal checkpoint migliore esistente in model_dir/checkpoints/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    print("IK Trainer – Parquet edition")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    parquet_dir  = Path(args.parquet_dir)
    train_path   = str(parquet_dir / "train.parquet")
    val_path     = str(parquet_dir / "val.parquet")
    test_path    = str(parquet_dir / "test.parquet")

    for p in [train_path, val_path, test_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File non trovato: {p}")

    output_cols = detect_output_cols(train_path)
    n_in  = len(INPUT_COLS)
    n_out = len(output_cols)
    print(f"Input  ({n_in}): {INPUT_COLS}")
    print(f"Output ({n_out}): {output_cols}")

    # Row group info
    pf = pq.ParquetFile(train_path)
    print(f"Train: {pf.metadata.num_row_groups} row group(s), "
          f"{pf.metadata.num_rows:,} righe")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    common = dict(output_cols=output_cols, batch_size=args.batch_size)

    train_ds = make_parquet_dataset(train_path, shuffle_row_groups=args.shuffle_row_groups, seed=42, **common).repeat()
    val_ds   = make_parquet_dataset(val_path,  shuffle_row_groups=False, seed=42, **common)
    test_ds  = make_parquet_dataset(test_path, shuffle_row_groups=False, seed=42, **common)

    # ------------------------------------------------------------------
    # Modello
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(args.model_dir, "checkpoints", "best_model.keras")

    if args.resume and os.path.isfile(ckpt_path):
        print(f"\nRipresa da checkpoint: {ckpt_path}")
        model = tf.keras.models.load_model(
            ckpt_path,
            custom_objects={"MinMaxNormalization": MinMaxNormalization},
        )
        metadata_path = os.path.join(args.model_dir, "metadata.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                saved_meta = json.load(f)
            resume_lr = saved_meta.get("lr", args.lr)
        else:
            resume_lr = args.lr
        print(f"  LR impostato a: {resume_lr}")
        tf.keras.backend.set_value(model.optimizer.learning_rate, resume_lr)
    else:
        print("\nCostruzione modello da zero...")
        model = build_model(
            input_dim=n_in,
            output_dim=n_out,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            huber_delta=args.huber_delta,
        )

    model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    os.makedirs(args.model_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    monitor = "val_loss"

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best_model.keras"),
            monitor=monitor, save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience, min_lr=args.min_lr, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta, restore_best_weights=True,
        ),
    ]
    if args.loss_target > 0.0:
        callbacks.append(StopOnLossTarget(args.loss_target, monitor=monitor))
        print(f"Auto-stop quando {monitor} < {args.loss_target}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\nTraining → max {args.epochs} epoche, batch {args.batch_size}")
    total_train_rows = pq.ParquetFile(train_path).metadata.num_rows
    steps_per_epoch  = total_train_rows // args.batch_size

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
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
        "input_scale":    INPUT_SCALE.tolist(),
        "hidden":         args.hidden,
        "dropout":        args.dropout,
        "loss":           args.loss,
        "huber_delta":    args.huber_delta if args.loss == "huber" else None,
        "lr": float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
        "batch_size":     args.batch_size,
        "weight_decay":   args.weight_decay,
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Valutazione
    # ------------------------------------------------------------------
    val_metrics  = evaluate_set("Validation", model, val_ds,  output_cols)
    test_metrics = evaluate_set("Test",        model, test_ds, output_cols)

    report = {
        "config":     vars(args),
        "history":    {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "validation": val_metrics,
        "test":       test_metrics,
    }
    with open(os.path.join(args.model_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModello salvato  → {model_path}")
    print(f"Metadata         → {os.path.join(args.model_dir, 'metadata.json')}")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)