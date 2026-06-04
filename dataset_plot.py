"""
Analisi statistica del dataset IK.

Uso:
    python analyze_dataset.py dataset.csv
    python analyze_dataset.py dataset.csv --save-plots
    python analyze_dataset.py dataset.csv --save-plots --out-dir risultati/
    python analyze_dataset.py dataset.csv --no-plots
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CHUNK_SIZE = 500_000

# ─────────────────────────────────────────────
#  CARICAMENTO
# ─────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    print(f"Caricamento {path} ...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} righe  |  colonne: {list(df.columns)}")
    return df

def load_chunks(path: str):
    print(f"Caricamento {path} ...")
    return pd.read_csv(path, chunksize=CHUNK_SIZE)


# ─────────────────────────────────────────────
#  RILEVAMENTO COLONNE  (adattivo)
# ─────────────────────────────────────────────

def detect_columns_from_file(path):
    sample = pd.read_csv(path, nrows=1000)
    return detect_cols(sample)

def detect_cols(df: pd.DataFrame):
    """
    Rileva automaticamente le colonne di posizione, quaternione e giunti.
    Compatibile con i nomi prodotti da main_parallelizzato.py:
        target_x / target_y / target_z
        hand_quat_qx / hand_quat_qy / hand_quat_qz / hand_quat_qw
        joint_0 / joint_1 / ...
    e con schemi legacy (qx/qy/qz/qw, _qx/_qy/_qz/_qw, ecc.).
    """
    cols = df.columns.tolist()

    # ── Posizione ──
    pos_exact = ["target_x", "target_y", "target_z"]
    if all(c in cols for c in pos_exact):
        pos_cols = pos_exact
    else:
        pos_cols = [c for c in cols if c in ("target_x", "target_y", "target_z")]
        if len(pos_cols) < 3:
            pos_cols = [c for c in cols
                        if "target" in c and any(c.endswith(s) for s in ("_x", "_y", "_z"))]
        if len(pos_cols) < 3:
            pos_cols = [c for c in cols if c.lower() in ("x", "y", "z")]

    # ── Quaternioni ──
    for suffix_set in [
        ("hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"),
    ]:
        if all(c in cols for c in suffix_set):
            quat_cols = list(suffix_set)
            break
    else:
        quat_cols = [c for c in cols if c in ("qx", "qy", "qz", "qw")]
        if not quat_cols:
            quat_cols = [c for c in cols if any(c.endswith(s) for s in ("_qx", "_qy", "_qz", "_qw"))]
        if not quat_cols:
            quat_cols = [c for c in cols if any(s in c for s in ("qx", "qy", "qz", "qw"))]

    # ── Giunti ──
    used = set(pos_cols + quat_cols)
    joint_cols = [c for c in cols
                  if c not in used
                  and not c.endswith("_sin") and not c.endswith("_cos")]

    return pos_cols, quat_cols, joint_cols


def _qw_col(quat_cols):
    """Restituisce la colonna qw tra quat_cols, o None."""
    return next(
        (c for c in quat_cols if c.endswith("qw") or c == "qw"),
        quat_cols[-1] if quat_cols else None
    )


# ─────────────────────────────────────────────
#  REPORT TESTUALE
# ─────────────────────────────────────────────

def text_report(df: pd.DataFrame):
    pos_cols, quat_cols, joint_cols = detect_cols(df)
    sep = "─" * 60

    print(f"\n{sep}")
    print("REPORT DATASET IK")
    print(sep)
    print(f"  [colonne rilevate automaticamente]")
    print(f"    posizione  : {pos_cols}")
    print(f"    quaternioni: {quat_cols}")
    print(f"    giunti     : {joint_cols}")

    if len(pos_cols) < 3:
        print(f"\n  ⚠️  Impossibile trovare 3 colonne di posizione. Trovate: {pos_cols}")
        return pos_cols, quat_cols, joint_cols

    cx, cy, cz = pos_cols[0], pos_cols[1], pos_cols[2]

    # ── Posizioni ──
    print("\n[POSIZIONI TARGET]")
    unique_pos = df[pos_cols].drop_duplicates()
    n_unique = len(unique_pos)
    ratio = len(df) / n_unique if n_unique > 0 else 0
    print(f"  posizioni uniche  : {n_unique:,}")
    print(f"  orient/posizione  : {ratio:.1f} (media)")
    for c in pos_cols:
        print(f"  {c:28s}: min={df[c].min():.4f}  max={df[c].max():.4f}  "
              f"mean={df[c].mean():.4f}  std={df[c].std():.4f}")

    # ── Quadranti XY — solo testo, basta per il controllo ──
    print("\n[QUADRANTI XY]")
    q1 = ((df[cx] >= 0) & (df[cy] >= 0)).sum()
    q2 = ((df[cx] <  0) & (df[cy] >= 0)).sum()
    q3 = ((df[cx] <  0) & (df[cy] <  0)).sum()
    q4 = ((df[cx] >= 0) & (df[cy] <  0)).sum()
    for label, cnt in [("Q1 (+x+y)", q1), ("Q2 (-x+y)", q2),
                        ("Q3 (-x-y)", q3), ("Q4 (+x-y)", q4)]:
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        warn = " ← VUOTO" if cnt == 0 else ""
        print(f"  {label}: {cnt:10,}  ({pct:5.1f}%)  {bar}{warn}")
    mn, mx = min(q1, q2, q3, q4), max(q1, q2, q3, q4)
    if mn == 0:
        xmin, xmax = df[cx].min(), df[cx].max()
        ymin, ymax = df[cy].min(), df[cy].max()
        print(f"  ⚠️  PROBLEMA: un quadrante è vuoto.")
        print(f"     Range X: [{xmin:.4f}, {xmax:.4f}]")
        print(f"     Range Y: [{ymin:.4f}, {ymax:.4f}]")
        if xmin >= 0:
            print(f"     → {cx} è sempre positivo: esplori solo x>0")
        elif xmax <= 0:
            print(f"     → {cx} è sempre negativo (MAPOFFSET sbagliato?)")
        if ymin >= 0:
            print(f"     → {cy} è sempre positivo")
        elif ymax <= 0:
            print(f"     → {cy} è sempre negativo")
    elif mn > 0 and mx / mn > 3:
        print(f"  ⚠️  SQUILIBRIO: rapporto max/min quadrante = {mx / mn:.1f}x")
    else:
        print("  ✓  Copertura quadranti bilanciata")

    # ── Quaternioni ──
    if quat_cols:
        print("\n[QUATERNIONI]")
        qw = _qw_col(quat_cols)
        norms = np.linalg.norm(df[quat_cols].values, axis=1)
        neg_qw = (df[qw] < 0).sum()
        print(f"  colonne rilevate  : {quat_cols}")
        print(f"  colonna qw        : {qw}")
        print(f"  qw < 0            : {neg_qw:,}  ({neg_qw / len(df) * 100:.1f}%)")
        print(f"  norma ||q|| min   : {norms.min():.6f}")
        print(f"  norma ||q|| max   : {norms.max():.6f}")
        print(f"  norma ||q|| mean  : {norms.mean():.6f}")
        if norms.max() > 1.01 or norms.min() < 0.99:
            print("  ⚠️  Alcune norme si discostano da 1.0")
        else:
            print("  ✓  Tutte le norme sono ≈ 1.0")
        if neg_qw > 0:
            print(f"\n  ⚠️  ATTENZIONE: {neg_qw:,} righe con qw<0.")
            print(f"     Se hai applicato il flip 'if qw<0: q=-q', rimuovilo e")
            print(f"     usa una loss invariante al segno nel training:")
            print(f"       loss = min(||q_pred - q_true||², ||q_pred + q_true||²)")
    else:
        print("\n[QUATERNIONI] — nessuna colonna quaternione rilevata")

    # ── Giunti ──
    if joint_cols:
        print(f"\n[GIUNTI]  ({len(joint_cols)} trovati: {joint_cols})")
        for jc in joint_cols:
            v = df[jc]
            flag = "  ⚠️  std≈0 (bloccato al seed?)" if v.std() < 0.01 else ""
            print(f"  {jc:28s}: min={v.min():7.3f}  max={v.max():7.3f}  "
                  f"mean={v.mean():7.3f}  std={v.std():.3f}{flag}")

    # ── Duplicati ──
    print("\n[DUPLICATI]")
    dup = df.duplicated().sum()
    print(f"  righe duplicate   : {dup:,}")
    if dup > 0:
        print("  ⚠️  Considera df.drop_duplicates() prima del training")

    # ── Densità: soluzioni per punto spaziale ──
    print("\n[DENSITÀ SOLUZIONI]")
    sols_per_pos = df.groupby(pos_cols).size()
    print(f"  min soluzioni/punto : {sols_per_pos.min():,}")
    print(f"  max soluzioni/punto : {sols_per_pos.max():,}")
    print(f"  media               : {sols_per_pos.mean():.1f}")
    print(f"  mediana             : {sols_per_pos.median():.1f}")
    low_coverage = (sols_per_pos < 10).sum()
    if low_coverage > 0:
        pct = low_coverage / len(sols_per_pos) * 100
        print(f"  ⚠️  {low_coverage:,} punti ({pct:.1f}%) con meno di 10 soluzioni")
    else:
        print("  ✓  Tutti i punti hanno ≥10 soluzioni")

    print(f"\n{sep}\n")
    return pos_cols, quat_cols, joint_cols


# ─────────────────────────────────────────────
#  GRAFICI
# ─────────────────────────────────────────────

def plots(df: pd.DataFrame, pos_cols, quat_cols, joint_cols,
          save: bool, out_path: str):

    if len(pos_cols) < 3:
        print("⚠️  Grafici non generati: meno di 3 colonne di posizione.")
        return

    cx, cy, cz = pos_cols[0], pos_cols[1], pos_cols[2]
    qw = _qw_col(quat_cols)

    fig = plt.figure(figsize=(16, 12))
    dataset_label = os.path.splitext(os.path.basename(out_path))[0] if save else "dataset"
    fig.suptitle(f"Analisi dataset IK — {dataset_label}",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Campione per scatter (evita rendering lento su dataset enormi)
    sample = df.sample(min(8000, len(df)), random_state=42)

    # 1 — Scatter XY (copertura spaziale piano orizzontale)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(sample[cx], sample[cy], s=2, alpha=0.25, c="#378ADD", rasterized=True)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(cx); ax.set_ylabel(cy)
    ax.set_title("Copertura XY (piano orizzontale)")
    ax.set_aspect("equal")

    # 2 — Scatter XZ (copertura spaziale piano verticale)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(sample[cx], sample[cz], s=2, alpha=0.25, c="#1D9E75", rasterized=True)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(cx); ax.set_ylabel(cz)
    ax.set_title("Copertura XZ (piano verticale)")

    # 3 — Densità soluzioni per slice Z
    #     → rivela se le zone alte/basse del workspace hanno meno soluzioni
    ax = fig.add_subplot(gs[0, 2])
    z_bins = pd.cut(df[cz], bins=25)
    z_count = df.groupby(z_bins, observed=True).size()
    z_mid = [iv.mid for iv in z_count.index]
    height = z_count.index[0].length * 0.8 if len(z_count) else 0.01
    bars = ax.barh(z_mid, z_count.values, height=height, color="#7F77DD", edgecolor="white", lw=0.3)
    ax.set_xlabel("n° soluzioni"); ax.set_ylabel(cz)
    ax.set_title(f"Soluzioni per slice {cz}")
    # Evidenzia slice con densità < 50% della mediana
    median_count = np.median(z_count.values)
    for bar, cnt in zip(bars, z_count.values):
        if cnt < median_count * 0.5:
            bar.set_color("#D85A30")
    ax.axvline(median_count, color="black", lw=0.8, ls="--", label="mediana")
    ax.legend(fontsize=8)

    # 4 — Istogramma qw  (check flip quaternioni)
    if qw:
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(df[qw], bins=60, color="#D85A30", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", lw=1, ls="--", label="qw=0")
        neg_count = (df[qw] < 0).sum()
        ax.set_xlabel(qw); ax.set_ylabel("conteggio")
        ax.set_title(f"Distribuzione {qw}  (neg: {neg_count:,})")
        ax.legend(fontsize=8)

    # 5 — Norma quaternioni  (check corruzione dati)
    if quat_cols:
        ax = fig.add_subplot(gs[1, 1])
        norms = np.linalg.norm(df[quat_cols].values, axis=1)
        norm_range = norms.max() - norms.min()
        if norm_range > 1e-9:
            ax.hist(norms, bins=60, color="#7F77DD", edgecolor="white", linewidth=0.3)
        else:
            ax.text(0.5, 0.5, f"Tutte le norme = {norms.mean():.6f}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axvline(1.0, color="black", lw=1, ls="--", label="||q||=1")
        ax.set_xlabel("||q||"); ax.set_ylabel("conteggio")
        ax.set_title("Norma quaternioni  (deve essere ≈1)")
        ax.legend(fontsize=8)

    # 6 — Boxplot giunti  (check joint bloccati / fuori range)
    if joint_cols:
        ax = fig.add_subplot(gs[1, 2])
        data_j = [df[jc].values for jc in joint_cols]
        bp = ax.boxplot(data_j, labels=joint_cols, patch_artist=True,
                        medianprops=dict(color="white", lw=2))
        palette = ["#185FA5", "#1D9E75", "#D85A30", "#7F77DD",
                   "#BA7517", "#993556", "#3B6D11"]
        for patch, color in zip(bp["boxes"], palette * (len(joint_cols) // len(palette) + 1)):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xlabel("giunto"); ax.set_ylabel("rad")
        ax.set_title(f"Distribuzione angoli giunti  ({len(joint_cols)} giunti)")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()

    if save:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Grafici salvati in: {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analisi statistica del dataset IK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Esempi:
            python dataset_plot.py dataset.csv
            python dataset_plot.py dataset.csv --save-plots
            python dataset_plot.py dataset.csv --save-plots --out-dir risultati/
            python dataset_plot.py dataset.csv --no-plots
        """
    )
    parser.add_argument("csv",
                        help="Percorso al dataset CSV da analizzare")
    parser.add_argument("--save-plots", action="store_true", default=True,
                        help="Salva i grafici su file invece di mostrarli")
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Non generare grafici (solo report testuale)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory di output per i grafici "
                             "(default: stessa directory del CSV)")
    parser.add_argument("--out-name", default=None,
                        help="Nome base del file PNG di output, senza estensione "
                             "(default: <nome_csv>_analysis)")
    cli = parser.parse_args()

    csv_path = os.path.abspath(cli.csv)
    if not os.path.exists(csv_path):
        print(f"Errore: file '{csv_path}' non trovato.")
        sys.exit(1)

    csv_dir  = os.path.dirname(csv_path)
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]

    out_dir  = os.path.abspath(cli.out_dir) if cli.out_dir else csv_dir
    out_name = cli.out_name if cli.out_name else f"{csv_stem}_analysis"
    plot_path = os.path.join(out_dir, f"{out_name}.png")

    if cli.save_plots and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Directory di output creata: {out_dir}")

    df = load(csv_path)
    pos_cols, quat_cols, joint_cols = text_report(df)

    if not cli.no_plots:
        plots(df, pos_cols, quat_cols, joint_cols,
              save=cli.save_plots, out_path=plot_path)
    else:
        print("(grafici disabilitati con --no-plots)")


if __name__ == "__main__":
    main()