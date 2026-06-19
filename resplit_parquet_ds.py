#!/usr/bin/env python3
"""
reshuffle_parquet_split.py
--------------------------
Unisce logicamente train/val/test Parquet esistenti e li risplittal
in modo casuale a livello di ROW GROUP — senza mai caricare tutto in RAM.

Un row group alla volta viene letto e scritto nel nuovo file di destinazione.
Il picco di RAM è: dimensione_di_un_row_group * 1 (lettura) + buffer di scrittura.

Uso:
    python reshuffle_parquet_split.py \
        --input_dir parquet_data/ \
        --output_dir parquet_data_reshuffled/ \
        --seed 123 \
        --val_split 0.10 \
        --test_split 0.10

    # Per vedere quanti row group ci sono senza fare nulla:
    python reshuffle_parquet_split.py --input_dir parquet_data/ --dry_run
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_row_groups(input_dir: Path, splits: list[str]) -> list[tuple[str, int]]:
    """
    Restituisce una lista di (filepath, row_group_index) per tutti i file.
    Non legge nessun dato, solo i metadati.
    """
    entries = []
    for split in splits:
        path = input_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        pf = pq.ParquetFile(str(path))
        n_rg = pf.metadata.num_row_groups
        n_rows = pf.metadata.num_rows
        print(f"  {split}.parquet  →  {n_rg:,} row groups, {n_rows:,} righe")
        for rg_idx in range(n_rg):
            entries.append((str(path), rg_idx))
    return entries


def get_schema(input_dir: Path) -> pa.Schema:
    """Legge lo schema dal primo file disponibile."""
    for name in ["train", "val", "test"]:
        p = input_dir / f"{name}.parquet"
        if p.exists():
            return pq.read_schema(str(p))
    raise FileNotFoundError("Nessun file train/val/test trovato in input_dir")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reshuffle(
    input_dir: Path,
    output_dir: Path,
    val_split: float,
    test_split: float,
    seed: int,
    compression: str,
    dry_run: bool,
) -> None:
    print(f"\nScansione row group in: {input_dir}")
    all_rg = collect_row_groups(input_dir, ["train", "val", "test"])
    total_rg = len(all_rg)
    print(f"\nTotale row group: {total_rg:,}")

    if dry_run:
        print("\n[dry_run] Nessuna scrittura eseguita.")
        return

    # Permutazione casuale degli indici dei row group
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_rg)

    # Calcolo soglie
    train_hi = 1.0 - val_split - test_split
    val_hi   = 1.0 - test_split

    n_train = int(total_rg * train_hi)
    n_val   = int(total_rg * val_split)
    n_test  = total_rg - n_train - n_val

    train_indices = set(perm[:n_train].tolist())
    val_indices   = set(perm[n_train:n_train + n_val].tolist())
    test_indices  = set(perm[n_train + n_val:].tolist())

    print(f"\nRow group per split:")
    print(f"  train : {len(train_indices):,}  (~{len(train_indices)/total_rg*100:.1f}%)")
    print(f"  val   : {len(val_indices):,}  (~{len(val_indices)/total_rg*100:.1f}%)")
    print(f"  test  : {len(test_indices):,}  (~{len(test_indices)/total_rg*100:.1f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = get_schema(input_dir)

    writers = {
        "train": pq.ParquetWriter(str(output_dir / "train.parquet"), schema, compression=compression),
        "val":   pq.ParquetWriter(str(output_dir / "val.parquet"),   schema, compression=compression),
        "test":  pq.ParquetWriter(str(output_dir / "test.parquet"),  schema, compression=compression),
    }

    index_to_split = {}
    for idx in train_indices: index_to_split[idx] = "train"
    for idx in val_indices:   index_to_split[idx] = "val"
    for idx in test_indices:  index_to_split[idx] = "test"

    counts = {"train": 0, "val": 0, "test": 0}
    t0 = time.time()

    # Cache dei ParquetFile aperti per evitare di riaprire lo stesso file ad ogni row group
    open_files: dict[str, pq.ParquetFile] = {}

    try:
        for global_idx, (filepath, rg_idx) in enumerate(all_rg):
            split_name = index_to_split[global_idx]

            # Apri il file solo se non è già in cache
            if filepath not in open_files:
                open_files[filepath] = pq.ParquetFile(filepath)

            pf = open_files[filepath]
            table = pf.read_row_group(rg_idx)
            writers[split_name].write_table(table)
            counts[split_name] += table.num_rows

            # Progress ogni 100 row group
            if (global_idx + 1) % 100 == 0 or global_idx == total_rg - 1:
                elapsed = time.time() - t0
                pct = (global_idx + 1) / total_rg * 100
                rg_per_sec = (global_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_rg - global_idx - 1) / rg_per_sec if rg_per_sec > 0 else 0
                print(
                    f"\r  [{global_idx+1:,}/{total_rg:,}] {pct:.1f}%  "
                    f"| train={counts['train']:,}  val={counts['val']:,}  test={counts['test']:,}  "
                    f"| {rg_per_sec:.1f} rg/s  ETA={eta:.0f}s   ",
                    end="",
                )

    finally:
        for w in writers.values():
            w.close()
        for pf in open_files.values():
            pass  # ParquetFile non ha close() esplicito, GC se ne occupa

    elapsed = time.time() - t0
    print(f"\n\nCompletato in {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    print("\nDimensioni file output:")
    for name in ["train", "val", "test"]:
        p = output_dir / f"{name}.parquet"
        size_gb = p.stat().st_size / 1e9
        pf = pq.ParquetFile(str(p))
        print(f"  {name}.parquet  →  {pf.metadata.num_rows:,} righe  |  {size_gb:.2f} GB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Reshuffle split Parquet per row group")
    p.add_argument("--input_dir",   required=True,  help="Directory con train/val/test.parquet originali")
    p.add_argument("--output_dir",  required=True,  help="Directory output per i nuovi file")
    p.add_argument("--val_split",   type=float, default=0.10)
    p.add_argument("--test_split",  type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42,   help="Seed per la permutazione casuale")
    p.add_argument("--compression", default="zstd",
                   choices=["snappy", "gzip", "zstd", "brotli", "none"])
    p.add_argument("--dry_run", action="store_true",
                   help="Mostra solo quanti row group ci sono, senza scrivere nulla")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.val_split + args.test_split >= 1.0:
        raise ValueError("val_split + test_split deve essere < 1.0")
    reshuffle(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        compression=args.compression,
        dry_run=args.dry_run,
    )