import os
import math
import numpy as np
import pandas as pd
import tqdm
import argparse

def get_joint_angles_columns_name(df):
    column_names = df.columns
    joint_angles_columns = [col for col in column_names if 'joint' in col]
    return joint_angles_columns

def estimate_total_chunks(df_path, chunk_size):
    file_size = os.path.getsize(df_path)
    # Leggi un campione di 1000 righe per stimare la dimensione media per riga
    sample = pd.read_csv(df_path, nrows=1000)
    sample_bytes = sum(len(str(row)) + 1 for _, row in sample.iterrows())  # +1 per \n
    bytes_per_row = sample_bytes / 1000
    estimated_rows = file_size / bytes_per_row
    print(f"File size: {file_size} bytes, Estimated rows: {estimated_rows}, Bytes per row: {bytes_per_row}")
    return math.ceil(estimated_rows / chunk_size)

def process_csv_in_chunks(df_path, chunk_size, output_path):
    df_head = pd.read_csv(df_path, nrows=1)
    cols = get_joint_angles_columns_name(df_head)

    first_chunk = True
    for chunk in tqdm.tqdm(pd.read_csv(df_path, chunksize=chunk_size),
                           total=estimate_total_chunks(df_path, chunk_size),
                           desc="Elaborazione",
                           unit="chunk"):
        for col in cols:
            chunk[f'{col}_sin'] = np.sin(chunk[col].values)
            chunk[f'{col}_cos'] = np.cos(chunk[col].values)

        chunk.drop(columns=cols, inplace=True)
        chunk.to_csv(output_path, mode='a', header=first_chunk, index=False)
        first_chunk = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte angoli in sin/cos in un CSV.")
    parser.add_argument("input", help="Percorso al CSV di input")
    parser.add_argument("output", help="Percorso al CSV di output")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Il file di input '{args.input}' non esiste.")
        if os.path.exists(args.output):
            raise FileExistsError(f"Il file '{args.output}' esiste già. Rimuovilo prima di eseguire lo script.")
        process_csv_in_chunks(df_path=args.input, chunk_size=500000, output_path=args.output)
    except (FileNotFoundError, FileExistsError) as e:
        print(e)