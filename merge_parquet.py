#!/usr/bin/env python3
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(
        description="Unisce train, val e test Parquet in un unico file in modalità STREAMING (RAM-safe)."
    )
    parser.add_argument("--input_dir", required=True, help="Directory contenente train.parquet, val.parquet, test.parquet")
    parser.add_argument("--output_file", default="dataset.parquet", help="Nome del file di output unico")
    parser.add_argument("--row_group_size", type=int, default=100000, help="Numero di righe per ogni Row Group finale")
    parser.add_argument("--buffer_size", type=int, default=500000, help="Quante righe caricare in RAM alla volta prima di scrivere")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = ["train.parquet", "val.parquet", "test.parquet"]
    
    # 1. Identifica i file validi ed estrai lo schema iniziale senza caricare i dati
    schema = None
    valid_files = []
    for f in files:
        file_path = input_dir / f
        if file_path.is_file():
            valid_files.append(file_path)
            if schema is None:
                # Legge solo i metadati dello schema strutturale
                schema = pq.read_schema(file_path)
        else:
            print(f"Nota: {file_path} non trovato, salto.")

    if not valid_files:
        print("Errore: Nessun file trovato da unire!")
        return 1

    print(f"Schema individuato correttamente. Inizio fusione in streaming su: {args.output_file}")
    total_rows_written = 0

    # 2. Apri il file di output usando ParquetWriter (scrive direttamente su disco in chunk)
    with pq.ParquetWriter(args.output_file, schema=schema) as writer:
        for file_path in valid_files:
            print(f"Leggendo in streaming: {file_path}...")
            pf = pq.ParquetFile(file_path)
            
            # Leggiamo il file originale a blocchi controllati (es. 500k righe alla volta)
            for record_batch in pf.iter_batches(batch_size=args.buffer_size):
                # Convertiamo il singolo blocco in una tabella temporanea Arrow
                table_chunk = pa.Table.from_batches([record_batch])
                
                # Scriviamo la tabella intermedia nel file finale.
                # 'row_group_size' dice a PyArrow di spezzare questo chunk in piccoli 
                # Row Groups (es. da 100k righe ciascuno) prima di fare il flush su disco.
                writer.write_table(table_chunk, row_group_size=args.row_group_size)
                
                total_rows_written += table_chunk.num_rows

    # 3. Verifica finale senza caricare i dati (legge solo i metadati di coda del file)
    final_pf = pq.ParquetFile(args.output_file)
    print("\n--- Unione Completata (RAM Safe) ---")
    print(f"File generato con successo: {args.output_file}")
    print(f"Totale righe elaborate e scritte: {total_rows_written:,}")
    print(f"Numero totale di Row Groups creati sul file unico: {final_pf.metadata.num_row_groups}")
    print("Il dataset unico è pronto per il Virtual Splitting nel codice di training.")
    return 0

if __name__ == "__main__":
    main()