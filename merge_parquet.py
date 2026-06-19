#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(description="Unisce train, val e test Parquet in un unico file con Row Groups controllati.")
    parser.add_argument("--input_dir", required=True, help="Directory contenente train.parquet, val.parquet, test.parquet")
    parser.add_argument("--output_file", default="dataset.parquet", help="Nome del file di output unico")
    parser.add_argument("--row_group_size", type=int, default=100000, help="Numero di righe per ogni Row Group")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = ["train.parquet", "val.parquet", "test.parquet"]
    tables = []

    print("--- Recupero e unione dei file Parquet ---")
    for f in files:
        file_path = input_dir / f
        if file_path.is_file():
            print(f"Lettura in corso: {file_path}")
            tables.append(pq.read_table(file_path))
        else:
            print(f"Nota: {file_path} non trovato, salto.")

    if not tables:
        print("Errore: Nessun file trovato da unire!")
        return 1

    # Concatenazione dei passati split in un'unica tabella in memoria
    merged_table = pa.concat_tables(tables)
    total_rows = merged_table.num_rows
    print(f"\nTotale righe accumulate: {total_rows:,}")

    # Scrittura con dimensione dei Row Group esplicita
    print(f"Scrittura del file unico '{args.output_file}'...")
    pq.write_table(merged_table, args.output_file, row_group_size=args.row_group_size)
    
    # Verifica finale dello schema appena creato
    final_pf = pq.ParquetFile(args.output_file)
    print("\n--- Verifica Output ---")
    print(f"File creato con successo: {args.output_file}")
    print(f"Numero totale di Row Groups generati: {final_pf.metadata.num_row_groups}")
    print("Pronto per il training virtuale!")
    return 0

if __name__ == "__main__":
    main()