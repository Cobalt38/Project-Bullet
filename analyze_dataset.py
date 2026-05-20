"""
Analisi statistica del dataset IK.

Uso:
    python analyze_dataset.py dataset.csv
    python analyze_dataset.py dataset.csv --save-plots
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


def load(path):
    print(f"Caricamento {path} ...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} righe  |  colonne: {list(df.columns)}")
    return df


def detect_cols(df):
    cols = df.columns.tolist()

    # posizione
    pos_cols = [c for c in cols if c in ("target_x", "target_y", "target_z")]
    if not pos_cols:
        pos_cols = [c for c in cols if "target" in c and any(c.endswith(s) for s in ("_x","_y","_z"))]

    # quaternioni — cerca suffissi _qx/_qy/_qz/_qw oppure qx/qy/qz/qw esatti
    quat_cols = [c for c in cols if c in ("qx","qy","qz","qw")]
    if not quat_cols:
        quat_cols = [c for c in cols if any(c.endswith(s) for s in ("_qx","_qy","_qz","_qw"))]
    if not quat_cols:
        quat_cols = [c for c in cols if any(s in c for s in ("qx","qy","qz","qw"))]

    # giunti: tutto il resto, esclusi _sin/_cos
    joint_cols = [c for c in cols
                  if c not in pos_cols + quat_cols
                  and not c.endswith("_sin") and not c.endswith("_cos")]

    return pos_cols, quat_cols, joint_cols


def text_report(df):
    pos_cols, quat_cols, joint_cols = detect_cols(df)
    sep = "─" * 60
    print(f"\n{sep}")
    print("REPORT DATASET IK")
    print(sep)
    print(f"  [colonne rilevate automaticamente]")
    print(f"    posizione  : {pos_cols}")
    print(f"    quaternioni: {quat_cols}")
    print(f"    giunti     : {joint_cols}")

    cx, cy, cz = pos_cols[0], pos_cols[1], pos_cols[2]

    # ── posizioni ──
    print("\n[POSIZIONI TARGET]")
    unique_pos = df[pos_cols].drop_duplicates()
    print(f"  posizioni uniche  : {len(unique_pos):,}")
    print(f"  orient/posizione  : {len(df)/len(unique_pos):.1f} (media)")
    for c in pos_cols:
        print(f"  {c:20s}: min={df[c].min():.4f}  max={df[c].max():.4f}  "
              f"mean={df[c].mean():.4f}  std={df[c].std():.4f}")

    # ── quadranti XY ──
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
    mn, mx = min(q1,q2,q3,q4), max(q1,q2,q3,q4)
    if mn == 0:
        xmin, xmax = df[cx].min(), df[cx].max()
        ymin, ymax = df[cy].min(), df[cy].max()
        print(f"  ⚠️  PROBLEMA: un quadrante è vuoto.")
        print(f"     Range X: [{xmin:.4f}, {xmax:.4f}]")
        print(f"     Range Y: [{ymin:.4f}, {ymax:.4f}]")
        if xmin >= 0:
            print(f"     → target_x è sempre positivo: esplori solo x>0")
        elif xmax <= 0:
            print(f"     → target_x è sempre negativo: esplori solo x<0  (MAPOFFSET sbagliato?)")
        if ymin >= 0:
            print(f"     → target_y è sempre positivo")
        elif ymax <= 0:
            print(f"     → target_y è sempre negativo")
    elif mx / max(mn, 1) > 3:
        print(f"  ⚠️  SQUILIBRIO: rapporto max/min quadrante = {mx/mn:.1f}x")
    else:
        print("  ✓  Copertura quadranti bilanciata")

    # ── quaternioni ──
    if quat_cols:
        print("\n[QUATERNIONI]")
        # identifica qw: cerca _qw o qw
        qw_col = next((c for c in quat_cols if c.endswith("qw") or c == "qw"), quat_cols[-1])
        norms = np.linalg.norm(df[quat_cols].values, axis=1)
        neg_qw = (df[qw_col] < 0).sum()
        print(f"  colonna qw        : {qw_col}")
        print(f"  qw < 0            : {neg_qw:,}  ({neg_qw/len(df)*100:.1f}%)")
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

    # ── giunti ──
    if joint_cols:
        print("\n[GIUNTI]")
        for jc in joint_cols:
            v = df[jc]
            flag = "  ⚠️  std≈0 (bloccato al seed?)" if v.std() < 0.01 else ""
            print(f"  {jc:25s}: min={v.min():7.3f}  max={v.max():7.3f}  "
                  f"mean={v.mean():7.3f}  std={v.std():.3f}{flag}")

    # ── duplicati ──
    print("\n[DUPLICATI]")
    dup = df.duplicated().sum()
    print(f"  righe duplicate   : {dup:,}")
    if dup > 0:
        print("  ⚠️  Considera df.drop_duplicates() prima del training")

    print(f"\n{sep}\n")
    return pos_cols, quat_cols, joint_cols


def plots(df, pos_cols, quat_cols, joint_cols, save):
    cx, cy, cz = pos_cols[0], pos_cols[1], pos_cols[2]
    qw_col = next((c for c in quat_cols if c.endswith("qw") or c == "qw"), None) if quat_cols else None

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Analisi dataset IK", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    sample = df.sample(min(5000, len(df)), random_state=42)

    # 1. Scatter XY
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(sample[cx], sample[cy], s=3, alpha=0.3, c="#378ADD", rasterized=True)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(cx); ax.set_ylabel(cy)
    ax.set_title("Distribuzione XY")
    ax.set_aspect("equal")

    # 2. Scatter XZ
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(sample[cx], sample[cz], s=3, alpha=0.3, c="#1D9E75", rasterized=True)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(cx); ax.set_ylabel(cz)
    ax.set_title("Distribuzione XZ")

    # 3. Quadranti
    ax = fig.add_subplot(gs[0, 2])
    q1 = ((df[cx] >= 0) & (df[cy] >= 0)).sum()
    q2 = ((df[cx] <  0) & (df[cy] >= 0)).sum()
    q3 = ((df[cx] <  0) & (df[cy] <  0)).sum()
    q4 = ((df[cx] >= 0) & (df[cy] <  0)).sum()
    counts = [q1, q2, q3, q4]
    colors_q = ["#5DCAA5","#378ADD","#FAC775","#F0997B"]
    bars = ax.bar(["Q1\n+x+y","Q2\n-x+y","Q3\n-x-y","Q4\n+x-y"],
                  counts, color=colors_q, edgecolor="white", linewidth=0.5)
    ax.set_title("Righe per quadrante XY"); ax.set_ylabel("righe")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(counts)*0.01,
                f"{cnt:,}", ha="center", va="bottom", fontsize=8)

    # 4. Istogramma qw
    if qw_col:
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(df[qw_col], bins=60, color="#D85A30", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", lw=1, ls="--", label="qw=0")
        ax.set_xlabel(qw_col); ax.set_ylabel("conteggio")
        ax.set_title(f"Distribuzione qw  (neg: {(df[qw_col]<0).sum():,})")
        ax.legend(fontsize=8)

    # 5. Norma quaternioni
    if quat_cols:
        ax = fig.add_subplot(gs[1, 1])
        norms = np.linalg.norm(df[quat_cols].values, axis=1)
        if norms.max() - norms.min() > 1e-9:
            ax.hist(norms, bins=60, color="#7F77DD", edgecolor="white", linewidth=0.3)
            ax.axvline(1.0, ...)
        else:
            ax.text(0.5, 0.5, f"Tutte le norme = {norms.mean():.6f}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("Norma quaternioni — costante")
        ax.axvline(1.0, color="black", lw=1, ls="--", label="||q||=1")
        ax.set_xlabel("||q||"); ax.set_ylabel("conteggio")
        ax.set_title("Norma quaternioni")
        ax.legend(fontsize=8)

    # 6. qx vs qz colorato per qw
    if len(quat_cols) >= 4 and qw_col:
        qx_col = next((c for c in quat_cols if c.endswith("qx") or c=="qx"), quat_cols[0])
        qz_col = next((c for c in quat_cols if c.endswith("qz") or c=="qz"), quat_cols[2])
        ax = fig.add_subplot(gs[1, 2])
        sc = ax.scatter(sample[qx_col], sample[qz_col], s=2, alpha=0.2,
                        c=sample[qw_col], cmap="coolwarm", rasterized=True)
        plt.colorbar(sc, ax=ax, label="qw")
        ax.set_xlabel(qx_col); ax.set_ylabel(qz_col)
        ax.set_title("qx vs qz (colore = qw)")

    # 7. Boxplot giunti
    if joint_cols:
        ax = fig.add_subplot(gs[2, :2])
        data_j = [df[jc].values for jc in joint_cols]
        bp = ax.boxplot(data_j, labels=joint_cols, patch_artist=True,
                        medianprops=dict(color="white", lw=2))
        palette = ["#185FA5","#1D9E75","#D85A30","#7F77DD","#BA7517","#993556","#3B6D11"]
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xlabel("giunto"); ax.set_ylabel("rad")
        ax.set_title("Distribuzione angoli giunti")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    # 8. qw medio per slice Z
    if qw_col:
        ax = fig.add_subplot(gs[2, 2])
        z_bins = pd.cut(df[cz], bins=20)
        z_mean = df.groupby(z_bins, observed=True)[qw_col].mean()
        z_mid  = [iv.mid for iv in z_mean.index]
        ax.barh(z_mid, z_mean.values, height=z_mean.index[0].length * 0.8,
                color=["#D85A30" if v < 0 else "#1D9E75" for v in z_mean.values])
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("qw medio"); ax.set_ylabel(cz)
        ax.set_title("qw medio per slice Z")
        legend_elems = [Patch(color="#1D9E75", label="qw>0"),
                        Patch(color="#D85A30", label="qw<0")]
        ax.legend(handles=legend_elems, fontsize=8)

    plt.tight_layout()

    if save:
        out = "dataset_analysis.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Grafici salvati in: {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analisi dataset IK")
    parser.add_argument("csv", help="Percorso al dataset.csv")
    parser.add_argument("--save-plots", action="store_true")
    cli = parser.parse_args()

    if not os.path.exists(cli.csv):
        print(f"Errore: file '{cli.csv}' non trovato.")
        sys.exit(1)

    df = load(cli.csv)
    pos_cols, quat_cols, joint_cols = text_report(df)
    plots(df, pos_cols, quat_cols, joint_cols, cli.save_plots)


if __name__ == "__main__":
    main()
