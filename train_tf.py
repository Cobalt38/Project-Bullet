#!/usr/bin/env python3
"""
IK trainer orientato alla ricerca – input quaternione per l'orientamento della mano.

Pipeline principale:
  CSV  -->  pandas chunked loading  -->  split train/val/test
        -->  normalizzazione  -->  training con checkpoint  -->  salvataggio modello
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------------------------
# Costanti globali: colonne di input obbligatorie e colonne da escludere
# dall'output (colonne di diagnostica, feedback di errore, ecc.)
# ---------------------------------------------------------------------------

BASE_REQUIRED_INPUTS = [
    "target_x",
    "target_y",
    "target_z",
    "hand_quat_qx",
    "hand_quat_qy",
    "hand_quat_qz",
    "hand_quat_qw",
]

# Colonne presenti nel CSV che NON devono mai diventare target di output:
# includono le coordinate EE calcolate, gli errori di orientamento, ecc.
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


# ---------------------------------------------------------------------------
# Utility: gestione percorsi CSV
# ---------------------------------------------------------------------------

def resolve_csv_path(path: str) -> str:
    """
    Risolve il percorso del CSV.

    Accetta:
    - un file esatto (con o senza estensione .csv)
    - una directory contenente esattamente un file CSV
    - un prefisso di nome file (es. "data" trova "data.csv-ABC123")

    Solleva FileNotFoundError se il file non è trovato o ambiguo.
    """
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

    # Fallback: cerca file il cui nome inizia con il basename dato
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
            # In caso di ambiguità, sceglie il file più recente
            prefixed.sort(key=os.path.getmtime, reverse=True)
            return prefixed[0]

    raise FileNotFoundError(f"CSV file not found: {path}")


def read_csv_header(path: str) -> List[str]:
    """Legge solo la prima riga del CSV per ottenere i nomi delle colonne."""
    df = pd.read_csv(path, nrows=0)
    if df.columns.empty:
        raise ValueError("CSV has no header")
    return list(df.columns)


# ---------------------------------------------------------------------------
# Utility: risoluzione colonne di output
# ---------------------------------------------------------------------------

def infer_output_columns(
    header: List[str], required_inputs: List[str], filter_outputs: List[str]
) -> List[str]:
    """
    Deduce automaticamente le colonne di output come:
      header - required_inputs - filter_outputs

    Verifica che tutte le colonne di input obbligatorie siano presenti.
    """
    for req in required_inputs:
        if req not in header:
            raise ValueError(f"Missing required column: {req}")
    output_cols = [c for c in header if c not in required_inputs and c not in filter_outputs]
    if not output_cols:
        raise ValueError("No output columns found (all columns are inputs?)")
    return output_cols


def split_cli_columns(raw_cols: List[str]) -> List[str]:
    """
    Normalizza l'elenco di colonne passato da CLI:
    supporta sia spazi che virgole come separatori.
    Es: ["col1,col2", "col3"] --> ["col1", "col2", "col3"]
    """
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
    """
    Risolve le colonne di output da usare durante il training:
    - se --output_cols è vuoto, usa la logica di inferenza automatica
    - altrimenti valida e usa le colonne specificate manualmente
    """
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
    """
    Deduce il formato di rotazione usato nelle colonne di output:
    - 'quat'      se tutte finiscono con _qx/_qy/_qz/_qw
    - 'euler_deg' se tutte finiscono con _rx/_ry/_rz
    - 'unknown'   altrimenti (es. output già encodati sin/cos, o misti)
    """
    if all(c.endswith(("_qx", "_qy", "_qz", "_qw")) for c in columns):
        return "quat"
    if all(c.endswith(("_rx", "_ry", "_rz")) for c in columns):
        return "euler_deg"
    return "unknown"


# ---------------------------------------------------------------------------
# Caricamento CSV con pandas (chunked) e conversione in float32
# ---------------------------------------------------------------------------

def load_csv_chunked(
    path: str,
    required_inputs: List[str],
    output_cols: List[str],
    max_rows: int,
    chunk_size: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Carica il CSV in blocchi (chunk) con pandas, evitando di tenere l'intero
    file in memoria durante la lettura.

    Ogni chunk viene immediatamente convertito in float32 e accumulato in
    liste Python; al termine le liste vengono convertite in array NumPy.

    Parametri
    ----------
    path           : percorso del file CSV
    required_inputs: nomi delle colonne di input
    output_cols    : nomi delle colonne di output (angoli giunti)
    max_rows       : numero massimo di righe da caricare (0 = nessun limite)
    chunk_size     : righe per chunk (default 200 000)

    Ritorna
    -------
    x_all     : array float32 di shape (N, n_inputs)
    y_all     : array float32 di shape (N, n_outputs)
    skipped   : righe scartate per valori NaN / non numerici
    """
    all_cols = required_inputs + output_cols
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    skipped = 0
    loaded = 0

    reader = pd.read_csv(
        path,
        usecols=all_cols,      # legge solo le colonne necessarie
        dtype=np.float32,      # conversione diretta in float32
        chunksize=chunk_size,
        on_bad_lines="skip",   # salta righe malformate silenziosamente
    )

    for chunk_idx, chunk in enumerate(reader):
        # Rimuovi righe con NaN in qualsiasi colonna rilevante
        before = len(chunk)
        chunk = chunk.dropna(subset=all_cols)
        skipped += before - len(chunk)

        if max_rows > 0:
            remaining = max_rows - loaded
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        if len(chunk) == 0:
            continue

        xs.append(chunk[required_inputs].to_numpy(dtype=np.float32))
        ys.append(chunk[output_cols].to_numpy(dtype=np.float32))
        loaded += len(chunk)

        if (chunk_idx + 1) % 10 == 0 or (max_rows > 0 and loaded >= max_rows):
            print(f"  Load progress: {loaded:,} rows loaded so far...", flush=True)

        if max_rows > 0 and loaded >= max_rows:
            break

    if not xs:
        raise ValueError("No valid rows loaded from CSV")

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    print(f"  Loaded {loaded:,} rows total, skipped {skipped:,} invalid rows.")
    return x_all, y_all, skipped


# ---------------------------------------------------------------------------
# Deduplicazione per quantizzazione delle coordinate target
# ---------------------------------------------------------------------------

def dedupe_by_target_quantization(
    x: np.ndarray, y: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Rimuove righe duplicate in base alla posizione target quantizzata.

    La posizione (x, y, z) viene arrotondata alla griglia con passo `eps`;
    per ogni cella della griglia viene mantenuta solo la prima occorrenza.
    Utile per ridurre la densità di campionamento in zone molto popolate.

    Parametri
    ----------
    x   : array di input, prime 3 colonne = (target_x, target_y, target_z)
    y   : array di output corrispondente
    eps : dimensione della cella (es. 0.01 = 1 cm se le unità sono metri)
    """
    if eps <= 0.0:
        return x, y, 0
    cell = np.rint(x[:, :3] / eps).astype(np.int64)
    seen: Dict[Tuple[int, int, int], int] = {}
    keep = np.zeros((x.shape[0],), dtype=bool)
    for i in range(x.shape[0]):
        key = (int(cell[i, 0]), int(cell[i, 1]), int(cell[i, 2]))
        if key not in seen:
            seen[key] = i
            keep[i] = True
    dropped = int(np.size(keep) - np.count_nonzero(keep))
    return x[keep], y[keep], dropped


# ---------------------------------------------------------------------------
# Split train / val / test
# ---------------------------------------------------------------------------

def split_random(
    n: int, val_split: float, test_split: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split casuale del dataset in train / val / test.

    Mescola gli indici con il seed dato e li suddivide nelle tre partizioni.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Train split is empty; reduce val/test split")
    return idx[:n_train], idx[n_train: n_train + n_val], idx[n_train + n_val:]


def _spatial_hash_unit_values(pos: np.ndarray, seed: int, cell_size: float) -> np.ndarray:
    """
    Calcola un valore pseudo-casuale in [0, 1) per ogni punto 3D, in modo
    deterministico e stabile al seed.

    Tecnica: hash spaziale con costanti prime per minimizzare le collisioni.
    Punti nella stessa cella di griglia ricevono lo stesso hash (utile per
    garantire che campioni vicini nello spazio finiscano nella stessa split).
    """
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
    """
    Split basato su hash spaziale: separa i campioni in base alla loro
    posizione 3D anziché casualmente.

    Vantaggio rispetto allo split casuale: il set di test contiene pose
    MAI viste in training, evitando overfitting geografico.
    La suddivisione è deterministica dato il seed.
    """
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


# ---------------------------------------------------------------------------
# Encoding / decoding angoli Euler come sin+cos
# ---------------------------------------------------------------------------

def encode_euler_deg(y: np.ndarray) -> np.ndarray:
    """
    Converte angoli Euler in gradi nella rappresentazione sin/cos:
      [θ1, θ2, ...] --> [sin(θ1), cos(θ1), sin(θ2), cos(θ2), ...]

    Questo encoding risolve il problema della discontinuità degli angoli
    (es. 359° e 1° sono vicini ma numericamente distanti).
    L'output ha il doppio delle colonne rispetto all'input.
    """
    rad = np.deg2rad(y)
    sin_v = np.sin(rad)
    cos_v = np.cos(rad)
    out = np.empty((y.shape[0], y.shape[1] * 2), dtype=np.float32)
    out[:, 0::2] = sin_v  # posizioni pari   = sin
    out[:, 1::2] = cos_v  # posizioni dispari = cos
    return out


def decode_euler_deg(y_enc: np.ndarray) -> np.ndarray:
    """
    Inverso di encode_euler_deg: riconverte la rappresentazione sin/cos
    in angoli Euler in gradi usando arctan2.
    """
    sin_v = y_enc[:, 0::2]
    cos_v = y_enc[:, 1::2]
    return np.rad2deg(np.arctan2(sin_v, cos_v))


def angular_diff_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Calcola la differenza angolare "wrapped" in [-180, 180].
    Necessario per la metrica angolare: 1° e 359° hanno differenza 2°, non 358°.
    """
    return (pred - true + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Pesi inversi della deviazione standard per il loss pesato
# ---------------------------------------------------------------------------

def compute_inverse_std_weights(
    y: np.ndarray, min_w: float, max_w: float, eps: float = 1e-6
) -> np.ndarray:
    """
    Calcola pesi per il loss in base all'inverso della std per colonna.

    Colonne con alta varianza (es. angoli che coprono tutto il range) ricevono
    peso minore; colonne con bassa varianza (angoli stabili) ricevono peso
    maggiore, forzando la rete a imparare tutti i giunti in modo equilibrato.

    I pesi vengono normalizzati alla media=1 e poi clampati in [min_w, max_w].
    """
    std = np.std(y, axis=0)
    return _compute_inverse_std_weights_from_std(std, min_w, max_w, eps)


def _compute_inverse_std_weights_from_std(
    std: np.ndarray, min_w: float, max_w: float, eps: float = 1e-6
) -> np.ndarray:
    """Versione interna: riceve la std già calcolata e restituisce i pesi."""
    inv = 1.0 / np.maximum(std, eps)
    inv = inv / np.mean(inv)          # normalizza a media = 1
    inv = np.clip(inv, min_w, max_w)  # evita pesi estremi
    return inv.astype(np.float32)


# ---------------------------------------------------------------------------
# tf.data.Dataset da array NumPy in memoria
# ---------------------------------------------------------------------------

def make_memory_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Crea un tf.data.Dataset da array NumPy già in RAM.

    Con shuffle=True mescola i campioni a ogni epoca; il prefetch
    sovrappone CPU (preparazione batch) e GPU (training).
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 100_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Loss pesato per output con varianza diversa
# ---------------------------------------------------------------------------

def build_weighted_loss(
    loss_name: str, weights: np.ndarray, huber_delta: float
) -> tf.keras.losses.Loss:
    """
    Costruisce una funzione di loss (MSE o Huber) con pesi per colonna.

    I pesi moltiplicano l'errore per ogni output prima della media,
    permettendo di bilanciare l'importanza dei diversi giunti.
    """
    w = tf.constant(weights.astype(np.float32), dtype=tf.float32)
    delta = tf.constant(float(huber_delta), dtype=tf.float32)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = y_pred - y_true
        if loss_name == "huber":
            abs_err = tf.abs(err)
            base = tf.where(
                abs_err <= delta,
                0.5 * tf.square(err),          # zona quadratica (errori piccoli)
                delta * (abs_err - 0.5 * delta),  # zona lineare (errori grandi)
            )
        else:
            base = tf.square(err)  # MSE puro
        return tf.reduce_mean(base * w, axis=-1)

    return loss_fn


# ---------------------------------------------------------------------------
# Costruzione del modello
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
    target_weights: Optional[np.ndarray],
) -> Tuple[tf.keras.Model, tf.keras.layers.Normalization]:
    """
    Costruisce una rete feed-forward con connessioni residuali.

    Architettura:
      Input --> Normalization --> [Dense + LayerNorm + GELU + Dropout + Residual] x N --> Output

    La connessione residuale (skip connection) viene aggiunta solo quando
    il numero di unità del layer è uguale al layer precedente, migliorando
    il gradiente nelle reti profonde.

    Ritorna il modello compilato e il layer di normalizzazione (da adattare
    ai dati di training prima di iniziare il fit).
    """
    inp = tf.keras.Input(shape=(input_dim,), name="target_pos")
    norm = tf.keras.layers.Normalization(name="input_norm")
    x = norm(inp)

    for i, units in enumerate(hidden):
        residual = x
        x = tf.keras.layers.Dense(units, name=f"dense_{i + 1}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"ln_{i + 1}")(x)
        x = tf.keras.layers.Activation("gelu", name=f"act_{i + 1}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, name=f"drop_{i + 1}")(x)
        # Aggiungi skip connection solo se le dimensioni coincidono
        if residual.shape[-1] == x.shape[-1]:
            x = tf.keras.layers.Add(name=f"res_{i + 1}")([residual, x])

    out = tf.keras.layers.Dense(output_dim, name="rotations")(x)
    model = tf.keras.Model(inp, out)

    # Usa AdamW se disponibile (migliore regolarizzazione), fallback su Adam
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


# ---------------------------------------------------------------------------
# Valutazione finale (metriche angolari sul set val/test)
# ---------------------------------------------------------------------------

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
    """
    Valuta il modello sul set dato e stampa MAE / RMSE per colonna.

    Se l'output è in formato sin/cos (euler_deg), decodifica prima di
    calcolare le metriche angolari (usa differenza "wrapped" per angoli).

    Ritorna un dizionario con le metriche pronto per essere salvato nel report JSON.
    """
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
    rmse = float(np.sqrt(np.mean(diff ** 2)))
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
        "worst_outputs": [
            {"name": output_cols[int(i)], "mae": float(per_col[int(i)])} for i in worst
        ],
    }


# ---------------------------------------------------------------------------
# Parsing argomenti CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-oriented IK trainer (quaternion hand orientation input)"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="CSV file path or directory")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Output model directory")
    parser.add_argument("--max_rows", type=int, default=0, help="Limit loaded rows (0 = all)")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200_000,
        help="Rows per pandas chunk during CSV loading (default: 200000). "
             "Abbassare se la RAM è limitata.",
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
        help="Cell size in target units for spatial-hash split (default 0.01 ~= 1 cm if meters)",
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
    parser.add_argument(
        "--loss_target",
        type=float,
        default=0.0,
        help="Ferma il training se la loss di validazione (o di training se no val) "
             "scende sotto questa soglia. 0 = disabilitato.",
    )
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
        "--checkpoint_dir",
        default="",
        help="Directory per i checkpoint per-epoca (default: <model_dir>/checkpoints). "
             "I checkpoint permettono di riprendere il training o recuperare il miglior modello.",
    )
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


# ---------------------------------------------------------------------------
# Callback personalizzato: stop quando la loss è sotto una soglia
# ---------------------------------------------------------------------------

class StopOnLossTarget(tf.keras.callbacks.Callback):
    """
    Ferma il training non appena la loss monitored scende sotto `loss_target`.

    Utile per training interattivi: si avvia il training con molte epoche e si
    interrompe manualmente (ctrl+C) o automaticamente quando la qualità è
    soddisfacente, senza dover indovinare il numero di epoche a priori.
    """

    def __init__(self, loss_target: float, monitor: str = "val_loss"):
        super().__init__()
        self.loss_target = loss_target
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is not None and current < self.loss_target:
            print(
                f"\nEpoch {epoch + 1}: {self.monitor} = {current:.6f} < "
                f"target {self.loss_target:.6f} → training stopped."
            )
            self.model.stop_training = True


# ---------------------------------------------------------------------------
# Entry point principale
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Validazione split
    if args.val_split < 0 or args.test_split < 0 or (args.val_split + args.test_split) >= 1.0:
        raise ValueError("Invalid val/test split")

    # Seed globali per riproducibilità
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Costruisce i nomi delle colonne quaternione in base al prefisso scelto
    quat_cols = [
        f"{args.quat_input_prefix}_qx",
        f"{args.quat_input_prefix}_qy",
        f"{args.quat_input_prefix}_qz",
        f"{args.quat_input_prefix}_qw",
    ]

    required_inputs = ["target_x", "target_y", "target_z", *quat_cols]
    filter_outputs = list(BASE_FILTER_OUTPUTS)
    filter_outputs.extend(quat_cols)

    if args.include_use_rot_input:
        required_inputs.append("use_rot")
    else:
        filter_outputs.append("use_rot")

    # Risolve il percorso CSV e legge l'header
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
    print("OUTPUTS:", output_cols)
    csv_size_mb = os.path.getsize(csv_path) / (1024.0 * 1024.0)
    print(f"Loading CSV: {csv_path}  ({csv_size_mb:.1f} MB)")
    print(f"Rotation format: {rotation_format}  Euler sin/cos encoding: {use_sincos}")

    # -----------------------------------------------------------------------
    # Caricamento dati con pandas chunked (niente più memmap)
    # -----------------------------------------------------------------------
    print(f"Reading CSV in chunks of {args.chunk_size:,} rows ...")
    x_all, y_all_angles, skipped = load_csv_chunked(
        csv_path,
        required_inputs,
        output_cols,
        args.max_rows,
        args.chunk_size,
    )

    if skipped > 0:
        print(f"Warning: skipped {skipped:,} invalid rows")

    # Deduplicazione opzionale per densità uniforme nel workspace
    if args.dedupe_target_eps > 0:
        x_all, y_all_angles, dropped = dedupe_by_target_quantization(
            x_all, y_all_angles, args.dedupe_target_eps
        )
        print(
            f"Deduped by target eps={args.dedupe_target_eps}: "
            f"dropped {dropped:,}, remaining {x_all.shape[0]:,}"
        )

    total_rows = int(x_all.shape[0])
    print(f"Total samples after filtering: {total_rows:,}")

    # -----------------------------------------------------------------------
    # Split train / val / test
    # -----------------------------------------------------------------------
    if args.split_mode == "spatial_hash":
        train_idx, val_idx, test_idx = split_spatial_hash(
            x_all, args.val_split, args.test_split, args.seed, args.spatial_cell_size
        )
    else:
        train_idx, val_idx, test_idx = split_random(
            x_all.shape[0], args.val_split, args.test_split, args.seed
        )

    train_rows = int(train_idx.shape[0])
    val_rows = int(val_idx.shape[0])
    test_rows = int(test_idx.shape[0])
    print(
        f"Split ({args.split_mode}) → "
        f"train={train_rows:,}  val={val_rows:,}  test={test_rows:,}"
    )

    # Estrae i sottoset
    x_train = x_all[train_idx]
    y_train_angles = y_all_angles[train_idx]
    x_val = x_all[val_idx]
    y_val_angles = y_all_angles[val_idx]
    x_test = x_all[test_idx]
    y_test_angles = y_all_angles[test_idx]

    # Encoding sin/cos degli angoli Euler (solo per output euler_deg)
    if use_sincos:
        y_train = encode_euler_deg(y_train_angles)
        y_val = encode_euler_deg(y_val_angles) if val_rows > 0 else y_val_angles
    else:
        y_train = y_train_angles
        y_val = y_val_angles

    # Pesi per il loss pesato (opzionale)
    target_weights: Optional[np.ndarray] = None
    if args.target_weighting == "inverse_std":
        target_weights = compute_inverse_std_weights(
            y_train, min_w=args.weight_min, max_w=args.weight_max
        )

    if target_weights is not None:
        print(
            f"Target weights: "
            f"min={float(np.min(target_weights)):.3f}  "
            f"max={float(np.max(target_weights)):.3f}  "
            f"mean={float(np.mean(target_weights)):.3f}"
        )

    # -----------------------------------------------------------------------
    # Costruzione del modello
    # -----------------------------------------------------------------------
    output_dim = len(output_cols) * 2 if use_sincos else len(output_cols)
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

    # Adatta la normalizzazione sui dati di training
    norm.adapt(x_train)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    monitor_metric = "val_loss" if val_rows > 0 else "loss"
    os.makedirs(args.model_dir, exist_ok=True)

    # Directory checkpoint
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_{epoch:04d}_valloss_{val_loss:.6f}.keras")

    callbacks = [
        # Termina immediatamente se la loss diventa NaN (segnale di instabilità numerica)
        tf.keras.callbacks.TerminateOnNaN(),

        # Salva il modello a ogni epoca (solo quando migliorano le metriche)
        # save_best_only=False: salva OGNI epoca, utile per analisi post-training
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor_metric,
            save_best_only=False,   # salva tutti i checkpoint per poter analizzare l'andamento
            save_weights_only=False,
            verbose=0,
        ),

        # Salva un checkpoint separato "best" che sovrascrive il precedente
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor=monitor_metric,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),

        # Riduce il learning rate se la loss non migliora per patience epoche
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),

        # Ferma il training se la loss smette di migliorare
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            restore_best_weights=True,  # ripristina i pesi migliori trovati
        ),
    ]

    # Callback opzionale: ferma il training quando la loss scende sotto la soglia
    if args.loss_target > 0.0:
        callbacks.append(StopOnLossTarget(loss_target=args.loss_target, monitor=monitor_metric))
        print(f"Training will stop automatically when {monitor_metric} < {args.loss_target}")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    train_ds = make_memory_dataset(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = (
        make_memory_dataset(x_val, y_val, args.batch_size, shuffle=False, seed=args.seed)
        if val_rows > 0
        else None
    )

    print(f"\nStarting training: {train_rows:,} train samples, {val_rows:,} val samples")
    print(f"Checkpoints saved to: {checkpoint_dir}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # -----------------------------------------------------------------------
    # Salvataggio modello finale e metadata
    # -----------------------------------------------------------------------
    model_path = os.path.join(args.model_dir, args.model_file)
    model.save(model_path)

    # Nomi colonne output con encoding (per il file metadata)
    if use_sincos:
        output_cols_encoded = []
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
        "checkpoint_dir": checkpoint_dir,
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------------------------------------------------
    # Valutazione finale su val e test
    # -----------------------------------------------------------------------
    val_metrics = evaluate_angles(
        "Validation", model, x_val, y_val_angles, use_sincos, rotation_format, output_cols, args.batch_size
    )
    test_metrics = evaluate_angles(
        "Test", model, x_test, y_test_angles, use_sincos, rotation_format, output_cols, args.batch_size
    )

    report = {
        "config": vars(args),
        "csv_path_resolved": csv_path,
        "rotation_format": rotation_format,
        "euler_sincos": use_sincos,
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

    print(f"\nSaved model      → {model_path}")
    print(f"Saved metadata   → {os.path.join(args.model_dir, 'metadata.json')}")
    print(f"Saved report     → {os.path.join(args.model_dir, args.report_file)}")
    print(f"Checkpoints      → {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())