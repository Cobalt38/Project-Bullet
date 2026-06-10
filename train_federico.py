#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


BASE_REQUIRED_INPUTS = [
    "target_x",
    "target_y",
    "target_z",
    "hand_quat_qx",
    "hand_quat_qy",
    "hand_quat_qz",
    "hand_quat_qw",
]
BASE_FILTER_OUTPUTS = [
    "hand_mario_x",
    "hand_mario_y",
    "hand_mario_z",
    "hand_mario_distance_line",
    "distance_from_target",
    "orientation_error_deg",
    "use_rot",
    "hand_angle_x_sin",
    "hand_angle_x_cos",
    "hand_angle_y_sin",
    "hand_angle_y_cos",
    "hand_angle_z_sin",
    "hand_angle_z_cos",
    "hand_angle_y",
    "hand_angle_z",
    "hand_angle_x",
    "hand_quat_qx",
    "hand_quat_qy",
    "hand_quat_qz",
    "hand_quat_qw",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_angle_x",
    "ee_angle_y",
    "ee_angle_z",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "ee_qw",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = str(PROJECT_ROOT / "data_csv")
DEFAULT_MODEL_DIR = str(PROJECT_ROOT / "models" / "ik_model_research_quat")


def resolve_csv_path(path: str) -> str:
    if not path:
        raise FileNotFoundError("Empty CSV path")

    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        csv_like = [f for f in files if ".csv" in f.lower()]
        if not csv_like:
            raise FileNotFoundError(f"No CSV-like files found in directory: {path}")
        if len(csv_like) > 1:
            raise FileNotFoundError(
                f"Multiple CSV-like files found in directory: {path}\n"
                f"Specify one explicitly. Found: {', '.join(csv_like)}"
            )
        return os.path.join(path, csv_like[0])

    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".csv"):
        return path + ".csv"

    # Useful when files end up as "<name>.csv-XXXXXX"
    parent = os.path.dirname(path) or "."
    base = os.path.basename(path)
    if os.path.isdir(parent):
        prefixed = [
            os.path.join(parent, f)
            for f in os.listdir(parent)
            if f.startswith(base) and ".csv" in f.lower() and os.path.isfile(os.path.join(parent, f))
        ]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            prefixed.sort(key=os.path.getmtime, reverse=True)
            return prefixed[0]

    raise FileNotFoundError(f"CSV file not found: {path}")

def read_csv_header(path: str) -> List[str]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        return list(reader.fieldnames)

def infer_output_columns(header: List[str], required_inputs: List[str], filter_outputs: List[str]) -> List[str]:
    for req in required_inputs:
        if req not in header:
            raise ValueError(f"Missing required column: {req}")
    output_cols = [c for c in header if c not in required_inputs and c not in filter_outputs]
    if not output_cols:
        raise ValueError("No output columns found (all columns are inputs?)")
    return output_cols


def split_cli_columns(raw_cols: List[str]) -> List[str]:
    out: List[str] = []
    for item in raw_cols:
        for token in item.split(","):
            col = token.strip()
            if col:
                out.append(col)
    return out


def resolve_output_columns(
    header: List[str],
    required_inputs: List[str],
    filter_outputs: List[str],
    output_cols_arg: List[str],
) -> List[str]:
    manual = split_cli_columns(output_cols_arg)
    if not manual:
        return infer_output_columns(header, required_inputs, filter_outputs)

    unique_manual = list(dict.fromkeys(manual))
    missing = [c for c in unique_manual if c not in header]
    if missing:
        raise ValueError(f"--output_cols contains unknown columns: {', '.join(missing)}")

    overlap = [c for c in unique_manual if c in required_inputs]
    if overlap:
        raise ValueError(
            f"--output_cols includes input columns (not allowed): {', '.join(overlap)}"
        )

    return unique_manual

def infer_rotation_format(columns: List[str]) -> str:
    if all(c.endswith(("_qx", "_qy", "_qz", "_qw")) for c in columns):
        return "quat"
    if all(c.endswith(("_rx", "_ry", "_rz")) for c in columns):
        return "euler_deg"
    return "unknown"

def load_csv(
    path: str,
    required_inputs: List[str],
    output_cols: List[str],
    max_rows: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    skipped = 0
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            if max_rows > 0 and row_idx > max_rows:
                break
            try:
                x = [float(row[k]) for k in required_inputs]
                y = [float(row[k]) for k in output_cols]
            except (ValueError, TypeError, KeyError):
                skipped += 1
                continue
            xs.append(x)
            ys.append(y)
            if row_idx % 500_000 == 0:
                print(f"Load progress: {row_idx:,} rows", flush=True)
    if not xs:
        raise ValueError("No valid rows loaded from CSV")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), skipped


def count_valid_csv_rows(
    path: str,
    required_inputs: List[str],
    output_cols: List[str],
    max_rows: int,
) -> Tuple[int, int]:
    valid = 0
    skipped = 0
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            if max_rows > 0 and row_idx > max_rows:
                break
            try:
                [float(row[k]) for k in required_inputs]
                [float(row[k]) for k in output_cols]
            except (ValueError, TypeError, KeyError):
                skipped += 1
                continue
            valid += 1
            if row_idx % 500_000 == 0:
                print(f"Count progress: {row_idx:,} rows, valid={valid:,}", flush=True)
    return valid, skipped


def _cache_stem(path: str) -> str:
    stem = Path(path).stem
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
    return cleaned or "dataset"


def load_csv_memmap(
    path: str,
    required_inputs: List[str],
    output_cols: List[str],
    max_rows: int,
    cache_dir: str,
) -> Tuple[str, str, int, int, str]:
    valid_count, skipped = count_valid_csv_rows(path, required_inputs, output_cols, max_rows)
    if valid_count <= 0:
        raise ValueError("No valid rows loaded from CSV")

    run_cache_dir = os.path.join(cache_dir, f"{_cache_stem(path)}_quat_{os.getpid()}")
    os.makedirs(run_cache_dir, exist_ok=True)
    x_path = os.path.join(run_cache_dir, "x.npy")
    y_path = os.path.join(run_cache_dir, "y_angles.npy")

    x_mm = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(valid_count, len(required_inputs)),
    )
    y_mm = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.float32,
        shape=(valid_count, len(output_cols)),
    )

    write_idx = 0
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            if max_rows > 0 and row_idx > max_rows:
                break
            try:
                x_mm[write_idx, :] = [float(row[k]) for k in required_inputs]
                y_mm[write_idx, :] = [float(row[k]) for k in output_cols]
            except (ValueError, TypeError, KeyError):
                continue
            write_idx += 1
            if write_idx % 500_000 == 0:
                print(f"Memmap load progress: {write_idx:,}/{valid_count:,} valid rows", flush=True)

    if write_idx != valid_count:
        raise RuntimeError(f"Memmap row count mismatch: expected {valid_count}, wrote {write_idx}")

    x_mm.flush()
    y_mm.flush()
    return x_path, y_path, valid_count, skipped, run_cache_dir


def dedupe_by_target_quantization(
    x: np.ndarray, y: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    if eps <= 0.0:
        return x, y, 0
    cell = np.rint(x / eps).astype(np.int64)
    seen: Dict[Tuple[int, int, int], int] = {}
    keep = np.zeros((x.shape[0],), dtype=bool)
    for i in range(x.shape[0]):
        key = (int(cell[i, 0]), int(cell[i, 1]), int(cell[i, 2]))
        if key in seen:
            continue
        seen[key] = i
        keep[i] = True
    dropped = int(np.size(keep) - np.count_nonzero(keep))
    return x[keep], y[keep], dropped

def split_random(
    n: int, val_split: float, test_split: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Train split is empty; reduce val/test split")
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _spatial_hash_unit_values(pos: np.ndarray, seed: int, cell_size: float) -> np.ndarray:
    cell = np.floor(pos / cell_size).astype(np.int64)
    h = (
        (cell[:, 0] * 73856093)
        ^ (cell[:, 1] * 19349663)
        ^ (cell[:, 2] * 83492791)
        ^ np.int64(seed * 2654435761)
    )
    return (h.astype(np.uint64) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0


def split_spatial_hash(
    x: np.ndarray, val_split: float, test_split: float, seed: int, cell_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell_size <= 0.0:
        raise ValueError("cell_size must be > 0")
    u = _spatial_hash_unit_values(x[:, :3], seed, cell_size)
    test_mask = u < test_split
    val_mask = (u >= test_split) & (u < (test_split + val_split))
    train_mask = ~(test_mask | val_mask)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    if train_idx.size == 0:
        raise ValueError("Train split is empty; reduce val/test split or change cell size")
    return train_idx, val_idx, test_idx


def split_spatial_hash_memmap_indices(
    x_path: str,
    val_split: float,
    test_split: float,
    seed: int,
    cell_size: float,
    cache_dir: str,
    chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell_size <= 0.0:
        raise ValueError("cell_size must be > 0")
    chunk_rows = max(1, int(chunk_rows))
    x_mm = np.load(x_path, mmap_mode="r")
    n = int(x_mm.shape[0])
    counts = {"train": 0, "val": 0, "test": 0}

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        u = _spatial_hash_unit_values(np.asarray(x_mm[start:end, :3], dtype=np.float32), seed, cell_size)
        test_count = int(np.count_nonzero(u < test_split))
        val_count = int(np.count_nonzero((u >= test_split) & (u < (test_split + val_split))))
        counts["test"] += test_count
        counts["val"] += val_count
        counts["train"] += int((end - start) - test_count - val_count)
        if end % (chunk_rows * 10) == 0 or end == n:
            print(f"Split count progress: {end:,}/{n:,} rows", flush=True)

    if counts["train"] <= 0:
        raise ValueError("Train split is empty; reduce val/test split or change cell size")

    split_dir = os.path.join(cache_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    train_path = os.path.join(split_dir, "train_idx.npy")
    val_path = os.path.join(split_dir, "val_idx.npy")
    test_path = os.path.join(split_dir, "test_idx.npy")
    train_idx = np.lib.format.open_memmap(train_path, mode="w+", dtype=np.int64, shape=(counts["train"],))
    val_idx = np.lib.format.open_memmap(val_path, mode="w+", dtype=np.int64, shape=(counts["val"],))
    test_idx = np.lib.format.open_memmap(test_path, mode="w+", dtype=np.int64, shape=(counts["test"],))
    write_pos = {"train": 0, "val": 0, "test": 0}

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        row_idx = np.arange(start, end, dtype=np.int64)
        u = _spatial_hash_unit_values(np.asarray(x_mm[start:end, :3], dtype=np.float32), seed, cell_size)
        test_mask = u < test_split
        val_mask = (u >= test_split) & (u < (test_split + val_split))
        train_mask = ~(test_mask | val_mask)

        selected = row_idx[train_mask]
        pos = write_pos["train"]
        train_idx[pos : pos + selected.shape[0]] = selected
        write_pos["train"] += int(selected.shape[0])

        selected = row_idx[val_mask]
        pos = write_pos["val"]
        val_idx[pos : pos + selected.shape[0]] = selected
        write_pos["val"] += int(selected.shape[0])

        selected = row_idx[test_mask]
        pos = write_pos["test"]
        test_idx[pos : pos + selected.shape[0]] = selected
        write_pos["test"] += int(selected.shape[0])

        if end % (chunk_rows * 10) == 0 or end == n:
            print(f"Split write progress: {end:,}/{n:,} rows", flush=True)

    train_idx.flush()
    val_idx.flush()
    test_idx.flush()
    return (
        np.load(train_path, mmap_mode="r"),
        np.load(val_path, mmap_mode="r"),
        np.load(test_path, mmap_mode="r"),
    )


def encode_euler_deg(y: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(y)
    sin = np.sin(rad)
    cos = np.cos(rad)
    out = np.empty((y.shape[0], y.shape[1] * 2), dtype=np.float32)
    out[:, 0::2] = sin
    out[:, 1::2] = cos
    return out

def decode_euler_deg(y_enc: np.ndarray) -> np.ndarray:
    sin = y_enc[:, 0::2]
    cos = y_enc[:, 1::2]
    return np.rad2deg(np.arctan2(sin, cos))

def angular_diff_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return (pred - true + 180.0) % 360.0 - 180.0

def compute_inverse_std_weights(y: np.ndarray, min_w: float, max_w: float, eps: float = 1e-6) -> np.ndarray:
    std = np.std(y, axis=0)
    return compute_inverse_std_weights_from_std(std, min_w, max_w, eps)


def compute_inverse_std_weights_from_std(
    std: np.ndarray, min_w: float, max_w: float, eps: float = 1e-6
) -> np.ndarray:
    inv = 1.0 / np.maximum(std, eps)
    inv = inv / np.mean(inv)
    inv = np.clip(inv, min_w, max_w)
    return inv.astype(np.float32)


def iter_index_batches(
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_chunk_rows: int,
):
    n = int(indices.shape[0])
    if n == 0:
        return
    if not shuffle:
        for start in range(0, n, batch_size):
            yield indices[start : start + batch_size]
        return

    rng = np.random.default_rng(seed)
    if shuffle_chunk_rows <= 0 or shuffle_chunk_rows >= n:
        order = indices.copy()
        rng.shuffle(order)
        for start in range(0, n, batch_size):
            yield order[start : start + batch_size]
        return

    starts = np.arange(0, n, shuffle_chunk_rows, dtype=np.int64)
    rng.shuffle(starts)
    for chunk_start in starts:
        chunk = indices[chunk_start : chunk_start + shuffle_chunk_rows].copy()
        rng.shuffle(chunk)
        for start in range(0, int(chunk.shape[0]), batch_size):
            yield chunk[start : start + batch_size]


def make_memmap_dataset(
    x_path: str,
    y_path: str,
    indices: np.ndarray,
    batch_size: int,
    use_sincos: bool,
    shuffle: bool,
    seed: int,
    shuffle_chunk_rows: int,
) -> tf.data.Dataset:
    x_shape = np.load(x_path, mmap_mode="r").shape
    y_shape = np.load(y_path, mmap_mode="r").shape
    input_dim = int(x_shape[1])
    output_dim = int(y_shape[1] * 2) if use_sincos else int(y_shape[1])
    epoch = {"value": 0}

    def generator():
        x_mm = np.load(x_path, mmap_mode="r")
        y_mm = np.load(y_path, mmap_mode="r")
        local_seed = seed + epoch["value"]
        epoch["value"] += 1
        for batch_idx in iter_index_batches(indices, batch_size, shuffle, local_seed, shuffle_chunk_rows):
            x_batch = np.asarray(x_mm[batch_idx], dtype=np.float32)
            y_batch = np.asarray(y_mm[batch_idx], dtype=np.float32)
            if use_sincos:
                y_batch = encode_euler_deg(y_batch)
            yield x_batch, y_batch

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, output_dim), dtype=tf.float32),
        ),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def make_memmap_feature_dataset(
    x_path: str,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_chunk_rows: int,
) -> tf.data.Dataset:
    x_shape = np.load(x_path, mmap_mode="r").shape
    input_dim = int(x_shape[1])
    epoch = {"value": 0}

    def generator():
        x_mm = np.load(x_path, mmap_mode="r")
        local_seed = seed + epoch["value"]
        epoch["value"] += 1
        for batch_idx in iter_index_batches(indices, batch_size, shuffle, local_seed, shuffle_chunk_rows):
            yield np.asarray(x_mm[batch_idx], dtype=np.float32)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_inverse_std_weights_memmap(
    y_path: str,
    train_idx: np.ndarray,
    use_sincos: bool,
    batch_size: int,
    min_w: float,
    max_w: float,
) -> np.ndarray:
    y_mm = np.load(y_path, mmap_mode="r")
    sum_v: Optional[np.ndarray] = None
    sum_sq: Optional[np.ndarray] = None
    n = 0
    for batch_idx in iter_index_batches(train_idx, batch_size, False, 0, 0):
        y_batch = np.asarray(y_mm[batch_idx], dtype=np.float32)
        if use_sincos:
            y_batch = encode_euler_deg(y_batch)
        y64 = y_batch.astype(np.float64, copy=False)
        if sum_v is None:
            sum_v = np.zeros((y64.shape[1],), dtype=np.float64)
            sum_sq = np.zeros((y64.shape[1],), dtype=np.float64)
        sum_v += np.sum(y64, axis=0)
        sum_sq += np.sum(y64 * y64, axis=0)
        n += int(y64.shape[0])
    if n <= 0 or sum_v is None or sum_sq is None:
        raise ValueError("Train split is empty; cannot compute target weights")
    mean = sum_v / float(n)
    var = np.maximum((sum_sq / float(n)) - (mean * mean), 0.0)
    std = np.sqrt(var)
    return compute_inverse_std_weights_from_std(std, min_w=min_w, max_w=max_w)

def build_weighted_loss(loss_name: str, weights: np.ndarray, huber_delta: float) -> tf.keras.losses.Loss:
    w = tf.constant(weights.astype(np.float32), dtype=tf.float32)
    delta = tf.constant(float(huber_delta), dtype=tf.float32)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = y_pred - y_true
        if loss_name == "huber":
            abs_err = tf.abs(err)
            base = tf.where(
                abs_err <= delta,
                0.5 * tf.square(err),
                delta * (abs_err - 0.5 * delta),
            )
        else:
            base = tf.square(err)
        return tf.reduce_mean(base * w, axis=-1)

    return loss_fn

def build_model(
    input_dim: int,
    output_dim: int,
    hidden: List[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    loss_name: str,
    huber_delta: float,
    target_weights: Optional[np.ndarray],
) -> Tuple[tf.keras.Model, tf.keras.layers.Normalization]:
    inp = tf.keras.Input(shape=(input_dim,), name="target_pos")
    norm = tf.keras.layers.Normalization(name="input_norm")
    x = norm(inp)
    for i, units in enumerate(hidden):
        residual = x
        x = tf.keras.layers.Dense(units, name=f"dense_{i+1}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"ln_{i+1}")(x)
        x = tf.keras.layers.Activation("gelu", name=f"act_{i+1}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"drop_{i+1}")(x)
        if residual.shape[-1] == x.shape[-1]:
            x = tf.keras.layers.Add(name=f"res_{i+1}")([residual, x])
    out = tf.keras.layers.Dense(output_dim, name="rotations")(x)
    model = tf.keras.Model(inp, out)

    if hasattr(tf.keras.optimizers, "AdamW") and weight_decay > 0:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if target_weights is not None:
        loss_obj = build_weighted_loss(loss_name, target_weights, huber_delta)
    elif loss_name == "huber":
        loss_obj = tf.keras.losses.Huber(delta=huber_delta)
    else:
        loss_obj = "mse"
    model.compile(optimizer=optimizer, loss=loss_obj, metrics=["mae"])
    return model, norm


def evaluate_angles(
    name: str,
    model: tf.keras.Model,
    x_eval: np.ndarray,
    y_eval_angles: np.ndarray,
    use_sincos: bool,
    rotation_format: str,
    output_cols: List[str],
    batch_size: int,
) -> Dict:
    if x_eval.shape[0] == 0:
        print(f"{name}: empty")
        return {"rows": 0}
    preds = model.predict(x_eval, batch_size=batch_size, verbose=0)
    if rotation_format == "euler_deg":
        pred_angles = decode_euler_deg(preds) if use_sincos else preds
        diff = angular_diff_deg(pred_angles, y_eval_angles)
    else:
        diff = preds - y_eval_angles
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    per_col = np.mean(np.abs(diff), axis=0)
    worst = np.argsort(per_col)[-5:][::-1]
    print(f"{name} MAE: {mae:.6f}  RMSE: {rmse:.6f}")
    print(f"{name} worst 5 outputs (MAE):")
    for i in worst:
        print(f"  {output_cols[i]}: {float(per_col[i]):.6f}")
    return {
        "rows": int(x_eval.shape[0]),
        "mae": mae,
        "rmse": rmse,
        "worst_outputs": [{"name": output_cols[int(i)], "mae": float(per_col[int(i)])} for i in worst],
    }


def evaluate_angles_memmap(
    name: str,
    model: tf.keras.Model,
    x_path: str,
    y_path: str,
    indices: np.ndarray,
    use_sincos: bool,
    rotation_format: str,
    output_cols: List[str],
    batch_size: int,
) -> Dict:
    if indices.shape[0] == 0:
        print(f"{name}: empty")
        return {"rows": 0}

    x_mm = np.load(x_path, mmap_mode="r")
    y_mm = np.load(y_path, mmap_mode="r")
    sum_abs = np.zeros((len(output_cols),), dtype=np.float64)
    sum_sq = np.zeros((len(output_cols),), dtype=np.float64)
    rows = 0

    for batch_idx in iter_index_batches(indices, batch_size, False, 0, 0):
        x_batch = np.asarray(x_mm[batch_idx], dtype=np.float32)
        y_true = np.asarray(y_mm[batch_idx], dtype=np.float32)
        preds = model.predict(x_batch, batch_size=batch_size, verbose=0)
        if rotation_format == "euler_deg":
            pred_angles = decode_euler_deg(preds) if use_sincos else preds
            diff = angular_diff_deg(pred_angles, y_true)
        else:
            diff = preds - y_true
        diff64 = diff.astype(np.float64, copy=False)
        sum_abs += np.sum(np.abs(diff64), axis=0)
        sum_sq += np.sum(diff64 * diff64, axis=0)
        rows += int(diff64.shape[0])

    per_col = sum_abs / float(rows)
    mae = float(np.sum(sum_abs) / float(rows * len(output_cols)))
    rmse = float(np.sqrt(np.sum(sum_sq) / float(rows * len(output_cols))))
    worst = np.argsort(per_col)[-5:][::-1]
    print(f"{name} MAE: {mae:.6f}  RMSE: {rmse:.6f}")
    print(f"{name} worst 5 outputs (MAE):")
    for i in worst:
        print(f"  {output_cols[i]}: {float(per_col[i]):.6f}")
    return {
        "rows": int(rows),
        "mae": mae,
        "rmse": rmse,
        "worst_outputs": [{"name": output_cols[int(i)], "mae": float(per_col[int(i)])} for i in worst],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-oriented IK trainer (quaternion hand orientation input)"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="CSV file path or directory")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Output model directory")
    parser.add_argument("--max_rows", type=int, default=0, help="Limit loaded rows (0 = all)")
    parser.add_argument(
        "--data_backend",
        choices=["auto", "memory", "memmap"],
        default="auto",
        help="Dataset backend: memory keeps old eager loading; memmap stores arrays on disk; auto chooses by CSV size.",
    )
    parser.add_argument(
        "--memmap_threshold_mb",
        type=float,
        default=512.0,
        help="CSV size threshold for --data_backend auto.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        help="Directory for memmap cache files (default: <model_dir>/dataset_cache).",
    )
    parser.add_argument(
        "--shuffle_chunk_rows",
        type=int,
        default=262144,
        help="Rows shuffled at a time in memmap training; lower uses less RAM, 0 shuffles the whole index array.",
    )
    parser.add_argument(
        "--split_chunk_rows",
        type=int,
        default=1000000,
        help="Rows processed at a time while creating memmap split indices.",
    )
    parser.add_argument(
        "--split_mode",
        choices=["spatial_hash", "random"],
        default="spatial_hash",
        help="Split strategy for train/val/test",
    )
    parser.add_argument(
        "--spatial_cell_size",
        type=float,
        default=0.01,
        help="Cell size in target units for spatial-hash split (default 0.01 ~= 1 cm if units are meters)",
    )
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dedupe_target_eps",
        type=float,
        default=0.0,
        help="Drop duplicate targets by quantized position (0 disables)",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--target_weighting",
        choices=["none", "inverse_std"],
        default="inverse_std",
        help="Weight outputs in loss based on train-set std",
    )
    parser.add_argument("--weight_min", type=float, default=0.25)
    parser.add_argument("--weight_max", type=float, default=4.0)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    parser.add_argument("--reduce_lr_patience", type=int, default=6)
    parser.add_argument("--reduce_lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument(
        "--no_euler_sincos",
        action="store_true",
        help="Disable Euler sin/cos encoding even if outputs are *_rx/*_ry/*_rz",
    )
    parser.add_argument("--model_file", default="model.keras")
    parser.add_argument("--report_file", default="train_report.json")
    parser.add_argument(
        "--quat_input_prefix",
        default="hand_quat",
        help="Prefix for quaternion input columns: <prefix>_qx/_qy/_qz/_qw",
    )
    parser.add_argument(
        "--output_cols",
        nargs="+",
        default=[],
        help="Optional explicit output columns (space-separated or comma-separated).",
    )
    parser.add_argument(
        "--include_use_rot_input",
        action="store_true",
        help="Include use_rot as model input feature (expects 'use_rot' column in CSV)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.val_split < 0 or args.test_split < 0 or (args.val_split + args.test_split) >= 1.0:
        raise ValueError("Invalid val/test split")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    quat_cols = [
        f"{args.quat_input_prefix}_qx",
        f"{args.quat_input_prefix}_qy",
        f"{args.quat_input_prefix}_qz",
        f"{args.quat_input_prefix}_qw",
    ]

    required_inputs = [
        "target_x",
        "target_y",
        "target_z",
        *quat_cols,
    ]
    filter_outputs = list(BASE_FILTER_OUTPUTS)
    filter_outputs.extend(quat_cols)
    if args.include_use_rot_input:
        required_inputs.append("use_rot")
    else:
        filter_outputs.append("use_rot")

    csv_path = resolve_csv_path(args.csv)
    header = read_csv_header(csv_path)
    missing_required = [c for c in required_inputs if c not in header]
    if missing_required:
        raise ValueError(f"Missing required input columns: {', '.join(missing_required)}")
    output_cols = resolve_output_columns(
        header=header,
        required_inputs=required_inputs,
        filter_outputs=filter_outputs,
        output_cols_arg=args.output_cols,
    )
    rotation_format = infer_rotation_format(output_cols)
    use_sincos = rotation_format == "euler_deg" and not args.no_euler_sincos

    print("INPUTS :", required_inputs)
    print("OUTPUT: ", output_cols)


    csv_size_mb = os.path.getsize(csv_path) / (1024.0 * 1024.0)
    print(f"Loading CSV: {csv_path}")
    print(f"CSV size: {csv_size_mb:.1f} MB")
    print(f"Rotation format: {rotation_format}  Euler sin/cos: {use_sincos}")

    data_backend = args.data_backend
    if data_backend == "auto":
        data_backend = "memmap" if csv_size_mb >= args.memmap_threshold_mb else "memory"
        if args.dedupe_target_eps > 0:
            data_backend = "memory"
    if data_backend == "memmap" and args.dedupe_target_eps > 0:
        raise ValueError("--dedupe_target_eps is only supported with --data_backend memory")

    cache_dir_used: Optional[str] = None
    x_path: Optional[str] = None
    y_path: Optional[str] = None

    if data_backend == "memmap":
        cache_dir = args.cache_dir or os.path.join(args.model_dir, "dataset_cache")
        print(f"Dataset backend: memmap  cache={cache_dir}")
        x_path, y_path, total_rows, skipped, cache_dir_used = load_csv_memmap(
            csv_path, required_inputs, output_cols, args.max_rows, cache_dir
        )
        x_all = np.load(x_path, mmap_mode="r")
        y_all_angles = np.load(y_path, mmap_mode="r")
        print(f"Loaded samples to memmap: {total_rows:,}")
    else:
        print("Dataset backend: memory")
        x_all, y_all_angles, skipped = load_csv(csv_path, required_inputs, output_cols, args.max_rows)
        if args.dedupe_target_eps > 0:
            x_all, y_all_angles, dropped = dedupe_by_target_quantization(
                x_all, y_all_angles, args.dedupe_target_eps
            )
            print(
                f"Deduped by target eps={args.dedupe_target_eps}: "
                f"dropped {dropped:,}, remaining {x_all.shape[0]:,}"
            )
        total_rows = int(x_all.shape[0])
        print(f"Loaded samples: {total_rows:,}")

    if skipped > 0:
        print(f"Warning: skipped {skipped} invalid rows")

    if args.split_mode == "spatial_hash":
        if data_backend == "memmap":
            if x_path is None or cache_dir_used is None:
                raise RuntimeError("Internal error: memmap split paths are missing")
            train_idx, val_idx, test_idx = split_spatial_hash_memmap_indices(
                x_path,
                args.val_split,
                args.test_split,
                args.seed,
                args.spatial_cell_size,
                cache_dir_used,
                args.split_chunk_rows,
            )
        else:
            train_idx, val_idx, test_idx = split_spatial_hash(
                x_all, args.val_split, args.test_split, args.seed, args.spatial_cell_size
            )
    else:
        train_idx, val_idx, test_idx = split_random(x_all.shape[0], args.val_split, args.test_split, args.seed)

    train_rows = int(train_idx.shape[0])
    val_rows = int(val_idx.shape[0])
    test_rows = int(test_idx.shape[0])

    print(
        f"Split ({args.split_mode}) -> "
        f"train={train_rows:,} val={val_rows:,} test={test_rows:,}"
    )

    output_dim = len(output_cols) * 2 if use_sincos else len(output_cols)
    x_train = y_train_angles = x_val = y_val_angles = x_test = y_test_angles = None
    y_train = y_val = None

    target_weights: Optional[np.ndarray] = None
    if data_backend == "memory":
        x_train = x_all[train_idx]
        y_train_angles = y_all_angles[train_idx]
        x_val = x_all[val_idx]
        y_val_angles = y_all_angles[val_idx]
        x_test = x_all[test_idx]
        y_test_angles = y_all_angles[test_idx]

        if use_sincos:
            y_train = encode_euler_deg(y_train_angles)
            y_val = encode_euler_deg(y_val_angles) if val_rows > 0 else y_val_angles
        else:
            y_train = y_train_angles
            y_val = y_val_angles

        if args.target_weighting == "inverse_std":
            target_weights = compute_inverse_std_weights(
                y_train, min_w=args.weight_min, max_w=args.weight_max
            )
    else:
        if y_path is None:
            raise RuntimeError("Internal error: memmap y path is missing")
        if args.target_weighting == "inverse_std":
            target_weights = compute_inverse_std_weights_memmap(
                y_path,
                train_idx,
                use_sincos,
                args.batch_size,
                min_w=args.weight_min,
                max_w=args.weight_max,
            )

    if target_weights is not None:
        print(
            "Target weights ready "
            f"(min={float(np.min(target_weights)):.3f}, "
            f"max={float(np.max(target_weights)):.3f}, "
            f"mean={float(np.mean(target_weights)):.3f})"
        )

    model, norm = build_model(
        input_dim=int(x_all.shape[1]),
        output_dim=output_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        huber_delta=args.huber_delta,
        target_weights=target_weights,
    )
    if data_backend == "memory":
        norm.adapt(x_train)
    else:
        if x_path is None or y_path is None:
            raise RuntimeError("Internal error: memmap paths are missing")
        norm_ds = make_memmap_feature_dataset(
            x_path,
            train_idx,
            args.batch_size,
            False,
            args.seed,
            args.shuffle_chunk_rows,
        )
        norm.adapt(norm_ds)

    monitor_metric = "val_loss" if val_rows > 0 else "loss"
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            restore_best_weights=True,
        ),
    ]

    if data_backend == "memory":
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val) if val_rows > 0 else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        if x_path is None or y_path is None:
            raise RuntimeError("Internal error: memmap paths are missing")
        train_ds = make_memmap_dataset(
            x_path,
            y_path,
            train_idx,
            args.batch_size,
            use_sincos,
            True,
            args.seed,
            args.shuffle_chunk_rows,
        )
        val_ds = (
            make_memmap_dataset(
                x_path,
                y_path,
                val_idx,
                args.batch_size,
                use_sincos,
                False,
                args.seed,
                args.shuffle_chunk_rows,
            )
            if val_rows > 0
            else None
        )
        steps_per_epoch = max(1, (train_rows + args.batch_size - 1) // args.batch_size)
        validation_steps = max(1, (val_rows + args.batch_size - 1) // args.batch_size) if val_rows > 0 else None
        history = model.fit(
            train_ds.repeat(),
            validation_data=val_ds.repeat() if val_ds is not None else None,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, args.model_file)
    model.save(model_path)

    output_cols_encoded = []
    if use_sincos:
        for c in output_cols:
            output_cols_encoded.append(f"{c}_sin")
            output_cols_encoded.append(f"{c}_cos")
    else:
        output_cols_encoded = output_cols

    metadata = {
        "csv_path": csv_path,
        "input_columns": required_inputs,
        "output_columns": output_cols,
        "output_columns_encoded": output_cols_encoded,
        "rotation_format": rotation_format,
        "euler_sincos": use_sincos,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "split_mode": args.split_mode,
        "spatial_cell_size": args.spatial_cell_size if args.split_mode == "spatial_hash" else None,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "model_file": args.model_file,
        "loss": args.loss,
        "huber_delta": args.huber_delta if args.loss == "huber" else None,
        "target_weighting": args.target_weighting,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "quat_input_prefix": args.quat_input_prefix,
        "output_cols_manual": split_cli_columns(args.output_cols),
        "data_backend": data_backend,
        "memmap_cache_dir": cache_dir_used,
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if data_backend == "memory":
        val_metrics = evaluate_angles(
            "Validation", model, x_val, y_val_angles, use_sincos, rotation_format, output_cols, args.batch_size
        )
        test_metrics = evaluate_angles(
            "Test", model, x_test, y_test_angles, use_sincos, rotation_format, output_cols, args.batch_size
        )
    else:
        if x_path is None or y_path is None:
            raise RuntimeError("Internal error: memmap paths are missing")
        val_metrics = evaluate_angles_memmap(
            "Validation", model, x_path, y_path, val_idx, use_sincos, rotation_format, output_cols, args.batch_size
        )
        test_metrics = evaluate_angles_memmap(
            "Test", model, x_path, y_path, test_idx, use_sincos, rotation_format, output_cols, args.batch_size
        )

    report = {
        "config": vars(args),
        "csv_path_resolved": csv_path,
        "rotation_format": rotation_format,
        "euler_sincos": use_sincos,
        "data_backend": data_backend,
        "memmap_cache_dir": cache_dir_used,
        "rows": {
            "total": int(total_rows),
            "train": int(train_rows),
            "val": int(val_rows),
            "test": int(test_rows),
            "skipped_invalid": int(skipped),
        },
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(args.model_dir, args.report_file), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {os.path.join(args.model_dir, 'metadata.json')}")
    print(f"Saved report to: {os.path.join(args.model_dir, args.report_file)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
