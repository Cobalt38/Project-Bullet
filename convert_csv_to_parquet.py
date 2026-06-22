#!/usr/bin/env python3
"""
convert_to_parquet.py
---------------------
Converte il CSV del dataset IK (~106 GB) in tre file Parquet distinti:
  - train.parquet  (80% righe)
  - val.parquet    (10% righe)
  - test.parquet   (10% righe)

Lo split è deterministico e basato sull'hash delle 7 colonne di input. Viene eseguito UNA SOLA VOLTA.

Requisiti:
  pip install pyarrow pandas tqdm

Uso:
  python convert_to_parquet.py --csv dataset.csv --out_dir parquet_data/
  python convert_to_parquet.py --csv dataset.csv --out_dir parquet_data/ --chunk_size 500000 --compression zstd --val_split 0.1 --test_split 0.1
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Colonne
# ---------------------------------------------------------------------------

INPUT_COLS = [
    "target_x", "target_y", "target_z",
    "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw",
]

HASH_WEIGHTS = np.array(
    [73856093, 19349663, 83492791, 46187201, 61339669, 91228751, 37312693],
    dtype=np.int64,
)

# ---------------------------------------------------------------------------
# Funzione di split (identica alla logica in TF del vecchio script)
# ---------------------------------------------------------------------------

def compute_split_bucket(chunk: pd.DataFrame, input_cols: list) -> np.ndarray:
    """
    Restituisce un array di float32 in [0,1) per ogni riga,
    usando la stessa formula hash del vecchio _split_mask TF.
    """
    vals = (chunk[input_cols].to_numpy(dtype=np.float32) * 1e4).astype(np.int64)
    keys = (vals * HASH_WEIGHTS).sum(axis=1)  # int64
    # Moltiplicazione di Knuth (modulo 2^31)
    h = ((keys * np.int64(2654435761)) & np.int64(0x7FFFFFFF)).astype(np.float32)
    return h / np.float32(2147483648.0)


def detect_output_cols(csv_path: str) -> list:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols = [c for c in header if c.endswith("_sin") or c.endswith("_cos")]
    if not cols:
        raise ValueError(f"Nessuna colonna *_sin/*_cos nel CSV: {csv_path}")
    return cols


# ---------------------------------------------------------------------------
# Conversione principale
# ---------------------------------------------------------------------------

def convert(
    csv_path: str,
    out_dir: str,
    chunk_size: int,
    compression: str,
    row_group_size: int,
    val_split: float,
    test_split: float,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_cols = detect_output_cols(csv_path)
    all_cols = INPUT_COLS + output_cols
    print(f"Colonne input : {len(INPUT_COLS)}")
    print(f"Colonne output: {len(output_cols)}")
    print(f"Totale colonne: {len(all_cols)}")
    print(f"Compressione  : {compression}")
    print(f"Chunk size    : {chunk_size:,} righe")
    print(f"Row group size: {row_group_size:,} righe")
    print()

    train_hi = 1.0 - val_split - test_split
    val_hi   = 1.0 - test_split

    # Schema PyArrow (float32 per tutto)
    schema = pa.schema([(c, pa.float32()) for c in all_cols])

    writers = {
        "train": pq.ParquetWriter(
            str(out / "train.parquet"), schema,
            compression=compression, write_batch_size=row_group_size,
        ),
        "val": pq.ParquetWriter(
            str(out / "val.parquet"), schema,
            compression=compression, write_batch_size=row_group_size,
        ),
        "test": pq.ParquetWriter(
            str(out / "test.parquet"), schema,
            compression=compression, write_batch_size=row_group_size,
        ),
    }

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    t0 = time.time()
    chunk_idx = 0

    try:
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

            buckets = compute_split_bucket(chunk, INPUT_COLS)

            mask_train = buckets < train_hi
            mask_val   = (buckets >= train_hi) & (buckets < val_hi)
            mask_test  = buckets >= val_hi

            for split_name, mask in [("train", mask_train), ("val", mask_val), ("test", mask_test)]:
                sub = chunk[mask][all_cols]
                if sub.empty:
                    continue
                table = pa.Table.from_pandas(sub, schema=schema, preserve_index=False)
                writers[split_name].write_table(table)
                counts[split_name] += len(sub)

            chunk_idx += 1
            elapsed = time.time() - t0
            total = counts["train"] + counts["val"] + counts["test"]
            print(
                f"\rChunk {chunk_idx:5d} | "
                f"train={counts['train']:>12,}  val={counts['val']:>10,}  test={counts['test']:>10,} | "
                f"elapsed={elapsed:.0f}s",
                end="",
            )

    finally:
        for w in writers.values():
            w.close()

    total = counts["train"] + counts["val"] + counts["test"]
    elapsed = time.time() - t0
    print(f"\n\nConversione completata in {elapsed:.1f}s")
    print(f"  train : {counts['train']:>12,} righe  → {out / 'train.parquet'}")
    print(f"  val   : {counts['val']:>12,} righe  → {out / 'val.parquet'}")
    print(f"  test  : {counts['test']:>12,} righe  → {out / 'test.parquet'}")
    print(f"  totale: {total:>12,} righe")

    # Verifica dimensioni file
    for name in ["train", "val", "test"]:
        p = out / f"{name}.parquet"
        size_gb = p.stat().st_size / 1e9
        print(f"  {name}.parquet: {size_gb:.2f} GB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CSV → Parquet split converter per dataset IK")
    p.add_argument("--csv",           required=True,         help="Percorso del CSV sorgente")
    p.add_argument("--out_dir",       default="parquet_data", help="Directory output (default: parquet_data/)")
    p.add_argument("--chunk_size",    type=int, default=500_000, help="Righe per chunk durante la lettura CSV")
    p.add_argument("--compression",   default="zstd",
                   choices=["snappy", "gzip", "zstd", "brotli", "none"],
                   help="Algoritmo di compressione Parquet (default: zstd)")
    p.add_argument("--row_group_size", type=int, default=200_000,
                   help="Righe per row group nel Parquet (default: 200k)")
    p.add_argument("--val_split",     type=float, default=0.10)
    p.add_argument("--test_split",    type=float, default=0.10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.val_split + args.test_split >= 1.0:
        raise ValueError("val_split + test_split deve essere < 1.0")
    convert(
        csv_path=args.csv,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        compression=args.compression,
        row_group_size=args.row_group_size,
        val_split=args.val_split,
        test_split=args.test_split,
    )