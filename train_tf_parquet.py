#!/usr/bin/env python3
"""
Input  : target_x, target_y, target_z, hand_quat_qx/qy/qz/qw  (7 col.)
Output : *_sin / *_cos per ogni giunto

Virtual Split Edition: accetta un unico file .parquet e lo splitta a runtime usando i Row Groups.
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

INPUT_SCALE = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def detect_output_cols(parquet_path: str) -> List[str]:
    schema = pq.read_schema(parquet_path)
    sin_cols = [n for n in schema.names if n.endswith("_sin")]
    cos_cols = [n for n in schema.names if n.endswith("_cos")]

    if not sin_cols or not cos_cols:
        raise ValueError(
            f"Nessuna colonna *_sin/*_cos nel Parquet: {parquet_path}\n"
            f"Colonne disponibili: {schema.names}"
        )

    sin_joints = [c[:-4] for c in sin_cols]
    cos_joints = [c[:-4] for c in cos_cols]
    missing = set(sin_joints) ^ set(cos_joints)
    if missing:
        raise ValueError(f"Colonne sin/cos non appaiate per i giunti: {missing}")

    cols = []
    for joint in sin_joints:
        cols.append(f"{joint}_sin")
        cols.append(f"{joint}_cos")
    return cols


# ---------------------------------------------------------------------------
# Normalizzazione
# ---------------------------------------------------------------------------

class MinMaxNormalization(tf.keras.layers.Layer):
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
# Dataset Parquet → tf.data (Con supporto a Row Groups specifici)
# ---------------------------------------------------------------------------

def make_parquet_dataset(
    parquet_path: str,
    output_cols: List[str],
    batch_size: int,
    row_groups: List[int],
    shuffle_row_groups: bool = True,
    reader_batch_rows: int = 131072,
    shuffle_buffer_batches: int = 256,
    seed: int = 42,
) -> tf.data.Dataset:
    
    all_cols = INPUT_COLS + output_cols
    n_in = len(INPUT_COLS)
    n_out = len(output_cols)

    output_signature = (
        tf.TensorSpec(shape=(None, n_in),  dtype=tf.float32),
        tf.TensorSpec(shape=(None, n_out), dtype=tf.float32),
    )

    def _record_batch_to_numpy(record_batch) -> np.ndarray:
        schema = record_batch.schema
        cols = []
        for col_name in all_cols:
            col_idx = schema.get_field_index(col_name)
            arr = record_batch.column(col_idx).to_numpy(zero_copy_only=False)
            cols.append(np.asarray(arr, dtype=np.float32))
        return np.stack(cols, axis=1)

    def _generator():
        pf = pq.ParquetFile(parquet_path)
        rng = np.random.default_rng(seed)
        carry = None

        # Crea una copia locale per poter rimescolare l'ordine dei blocchi a ogni epoca
        active_groups = list(row_groups)
        if shuffle_row_groups:
            rng.shuffle(active_groups)

        for record_batch in pf.iter_batches(
            batch_size=reader_batch_rows,
            columns=all_cols,
            use_threads=True,
            row_groups=active_groups, # Chiediamo a PyArrow solo i nostri gruppi assegnati
        ):
            data = _record_batch_to_numpy(record_batch)

            # Shuffle intra-batch (dentro le righe caricate)
            if shuffle_row_groups:
                data = data[rng.permutation(len(data))]

            if carry is not None:
                data = np.concatenate([carry, data], axis=0)
                carry = None

            n_full = (len(data) // batch_size) * batch_size
            if n_full == 0:
                carry = data
                continue

            full = data[:n_full]
            carry = data[n_full:] if n_full < len(data) else None

            x = full[:, :n_in]
            y = full[:, n_in:]

            for i in range(0, n_full, batch_size):
                yield x[i:i + batch_size], y[i:i + batch_size]
                
        if carry is not None and len(carry) > 0:
            x = carry[:, :n_in]
            y = carry[:, n_in:]
            yield x, y

    ds = tf.data.Dataset.from_generator(
        _generator,
        output_signature=output_signature,
    )

    if shuffle_buffer_batches and shuffle_buffer_batches > 1:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer_batches,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    options = tf.data.Options()
    options.deterministic = False
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Architettura & Loss
# ---------------------------------------------------------------------------

class QuatTo6D(tf.keras.layers.Layer):
    def call(self, x, training=False):
        pos = x[..., :3]
        q   = x[..., 3:]
        q = tf.math.l2_normalize(q, axis=-1)
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
    sin_p_n, cos_p_n = sin_p / (norm + 1e-8), cos_p / (norm + 1e-8)
    sin_t, cos_t = y_true[:, 0::2], y_true[:, 1::2]
    angular = tf.reduce_mean(1.0 - (sin_t*sin_p_n + cos_t*cos_p_n))
    unit_pen = tf.reduce_mean((norm - 1.0)**2)
    return angular + 0.05 * unit_pen

def build_model(
    input_dim: int,
    output_dim: int,
    hidden: List[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    loss_name: str,
    huber_delta: float,
    first_decay_steps: int = 1000,
) -> tf.keras.Model:
    inp  = tf.keras.Input(shape=(input_dim,), name="target_pose")
    norm = MinMaxNormalization(scale=INPUT_SCALE, name="input_norm")
    x    = norm(inp)
    x    = QuatTo6D(trainable=False, name="quat_to_6d")(x)
    x    = GaussianNoise(stddev=0.01, trainable=False, name="input_noise")(x)

    for i, units in enumerate(hidden):
        residual = x
        x = tf.keras.layers.Dense(units, name=f"dense_{i+1}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"ln_{i+1}")(x)
        x = tf.keras.layers.Activation("gelu", name=f"act_{i+1}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"drop_{i+1}")(x)
        if residual.shape[-1] != units:
            residual = tf.keras.layers.Dense(units, name=f"dense_{i+1}_res", use_bias=False)(residual)
        x = tf.keras.layers.Add(name=f"res_{i+1}")([residual, x])

    out   = tf.keras.layers.Dense(output_dim, name="output")(x)
    model = tf.keras.Model(inp, out)

    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=first_decay_steps,
        t_mul=2.0, m_mul=0.9
    )

    if hasattr(tf.keras.optimizers, "AdamW") and weight_decay > 0:
        opt = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=weight_decay)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=schedule)

    loss = ik_loss if loss_name == "ik" else tf.keras.losses.Huber(delta=huber_delta) if loss_name == "huber" else "mse"
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

class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        if callable(opt.learning_rate):
            lr_val = float(opt.learning_rate(opt.iterations).numpy())
        else:
            lr_val = float(opt.learning_rate.numpy())
        print(f"\nEpoch {epoch+1}  –  lr: {lr_val:.6e}")
        if logs is not None:
            logs["lr"] = lr_val


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
    p = argparse.ArgumentParser(description="IK trainer – dataset Parquet unico con Virtual Split")
    p.add_argument("--parquet_file", required=True,
                   help="Path al file unico dataset.parquet")
    p.add_argument("--model_dir",    default="ik_model")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--hidden",       type=int,   nargs="+", default=[1024, 512, 512, 256])
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--lr",           type=float, default=1e-3, help="Learning rate iniziale")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss",         choices=["mse", "huber", "ik"], default="huber")
    p.add_argument("--huber_delta",  type=float, default=1.0)
    p.add_argument("--shuffle_row_groups", action="store_true",
                   help="Mescola l'ordine dei row group a ogni epoca (consigliato)")
    p.add_argument("--early_stop_patience",  type=int,   default=75)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--loss_target",          type=float, default=0.0)
    p.add_argument("--resume", action="store_true", help="Riprendi dal checkpoint migliore")
    p.add_argument("--reader_batch_rows", type=int, default=131072)
    p.add_argument("--train_shuffle_buffer_batches", type=int, default=256)
    p.add_argument("--eval_shuffle_buffer_batches", type=int, default=1)
    
    # Parametri aggiunti per gestire lo split virtuale
    p.add_argument("--train_prop", type=float, default=0.8, help="Proporzione di Row Groups per il Training")
    p.add_argument("--val_prop", type=float, default=0.1, help="Proporzione di Row Groups per la Validation")
    p.add_argument("--test_prop", type=float, default=0.1, help="Proporzione di Row Groups per il Test")
    p.add_argument("--split_seed", type=int, default=42, help="Seed deterministico per lo split iniziale dei gruppi")
    
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    print("IK Trainer – Virtual Split Edition")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    if not os.path.isfile(args.parquet_file):
        raise FileNotFoundError(f"File dataset unico non trovato: {args.parquet_file}")

    output_cols = detect_output_cols(args.parquet_file)
    n_in  = len(INPUT_COLS)
    n_out = len(output_cols)
    print(f"Input  ({n_in}): {INPUT_COLS}")
    print(f"Output ({n_out}): {output_cols}")

    # Lettura dei metadati dei Row Groups
    pf = pq.ParquetFile(args.parquet_file)
    total_groups = pf.metadata.num_row_groups
    print(f"Dataset unico rilevato: {total_groups} row group(s), {pf.metadata.num_rows:,} righe totali.")

    if total_groups < 3:
        raise ValueError(
            f"Il file parquet ha solo {total_groups} row groups. "
            "Impossibile effettuare uno split virtuale efficiente. Genera il file con un 'row_group_size' più piccolo."
        )

    # ------------------------------------------------------------------
    # Calcolo dello Split Virtuale basato sui Row Groups
    # ------------------------------------------------------------------
    all_group_indices = list(range(total_groups))
    # Usiamo un generatore ad-hoc isolato per non interferire con gli altri seed
    rng_splitter = random.Random(args.split_seed)
    rng_splitter.shuffle(all_group_indices)

    n_val = max(1, int(total_groups * args.val_prop))
    n_test = max(1, int(total_groups * args.test_prop))
    n_train = total_groups - n_val - n_test

    train_groups = all_group_indices[:n_train]
    val_groups   = all_group_indices[n_train : n_train + n_val]
    test_groups  = all_group_indices[n_train + n_val :]

    # Calcolo esatto del numero di righe per ogni split interrogando i metadati
    train_rows = sum(pf.metadata.row_group(g).num_rows for g in train_groups)
    val_rows   = sum(pf.metadata.row_group(g).num_rows for g in val_groups)
    test_rows  = sum(pf.metadata.row_group(g).num_rows for g in test_groups)

    train_steps = train_rows // args.batch_size
    val_steps   = val_rows // args.batch_size
    test_steps  = test_rows // args.batch_size

    print(f"\n--- Bilanciamento Split Virtuale ---")
    print(f"Train : {len(train_groups)} gruppi, {train_rows:,} righe, {train_steps:,} steps/epoch")
    print(f"Val   : {len(val_groups)} gruppi, {val_rows:,} righe, {val_steps:,} steps/epoch")
    print(f"Test  : {len(test_groups)} gruppi, {test_rows:,} righe, {test_steps:,} steps/epoch\n")

    if train_steps == 0 or val_steps == 0 or test_steps == 0:
        raise ValueError("Dataset o split troppo piccoli rispetto al batch_size impostato.")

    # ------------------------------------------------------------------
    # Dataset Inizializzazione
    # ------------------------------------------------------------------
    common = dict(
        parquet_path=args.parquet_file,
        output_cols=output_cols,
        batch_size=args.batch_size,
        reader_batch_rows=args.reader_batch_rows,
    )

    train_ds = make_parquet_dataset(
        row_groups=train_groups,
        shuffle_row_groups=args.shuffle_row_groups,
        shuffle_buffer_batches=args.train_shuffle_buffer_batches,
        seed=42,
        **common,
    ).repeat()

    val_ds = make_parquet_dataset(
        row_groups=val_groups,
        shuffle_row_groups=False,
        shuffle_buffer_batches=args.eval_shuffle_buffer_batches,
        seed=42,
        **common,
    )

    test_ds = make_parquet_dataset(
        row_groups=test_groups,
        shuffle_row_groups=False,
        shuffle_buffer_batches=args.eval_shuffle_buffer_batches,
        seed=42,
        **common,
    )

    # ------------------------------------------------------------------
    # Modello
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(args.model_dir, "checkpoints", "best_model.keras")

    if args.resume and os.path.isfile(ckpt_path):
        print(f"\nRipresa da checkpoint: {ckpt_path}")
        model = tf.keras.models.load_model(
            ckpt_path,
            custom_objects={
                "MinMaxNormalization": MinMaxNormalization,
                "QuatTo6D":            QuatTo6D,
                "GaussianNoise":       GaussianNoise,
                "ik_loss":             ik_loss,
            },
        )
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
            first_decay_steps=train_steps * 20,
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
        LRLogger(),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best_model.keras"),
            monitor=monitor, save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta, restore_best_weights=True,
        ),
    ]
    if args.loss_target > 0.0:
        callbacks.append(StopOnLossTarget(args.loss_target, monitor=monitor))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\nTraining → max {args.epochs} epoche, batch {args.batch_size}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Salvataggio & Reportistica
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
        "lr_initial":     args.lr,
        "batch_size":     args.batch_size,
        "weight_decay":   args.weight_decay,
        "epochs_trained": len(history.history["loss"]),
        "virtual_split": {
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
            "split_seed": args.split_seed
        }
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

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
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)