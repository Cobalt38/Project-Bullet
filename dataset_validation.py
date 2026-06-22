"""
validate_dataset.py
====================
Script di validazione per il dataset di cinematica inversa.
Legge il CSV in chunk per gestire file di grandi dimensioni (>100 GB).

Sezioni:
  1. Integrità strutturale   — NaN/Inf, tipi, conteggio righe
  2. Vincoli sin/cos         — sin²+cos²=1 per ogni giunto
  3. Distribuzione spaziale  — scatter 3D interattivo + istogrammi marginali x/y/z
  4. Boxplot giunti          — distribuzione degli angoli ricostruiti (atan2)
  6. Verifica FK             — ricalcolo cinematica diretta su campione casuale via PyBullet

Uso:
  python3 validate_dataset.py --csv dataset.csv [opzioni]

  --csv        path al file CSV (obbligatorio)
  --robot      nome del robot da usare (default: openarmRight)
               scelte: xarm | panda | biarm | openarmRight | openarmv2right
  --chunksize  righe per chunk durante la lettura (default: 500_000)
  --fk_samples numero di righe da campionare per la verifica FK (default: 5000)
  --fk_tol     tolleranza errore posizione FK in metri (default: 0.012)
  --out        cartella di output per i plot (default: validation_output/)
  --seed       seed random per il campionamento FK (default: 42)
  --skip_fk    salta la verifica FK
"""

import argparse
import os
import sys
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly non trovato: lo scatter 3D sarà salvato come PNG statico.")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURAZIONI ROBOT  (specchio di main_parallelizzato.py)
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_CONFIGS = {
    "xarm": {
        "rest_pose":     [0, 0.5, -0.5, 0.5, 0],
        "arm_joints":    [0, 1, 2, 3, 4],
        "ee_link_index": 5,
        "isLocalpath":   True,
        "urdf_path":     "xarm_model/urdf/xarm_fixed.urdf",
    },
    "panda": {
        "rest_pose":     [0, -0.215, 0, -2.57, 0, 2.356, 2.356],
        "arm_joints":    [0, 1, 2, 3, 4, 5, 6],
        "ee_link_index": 11,
        "isLocalpath":   False,
        "urdf_path":     "franka_panda/panda.urdf",
    },
    "biarm": {
        "rest_pose":     [0, 0, 0, 1.2, 0, 0, 0],
        "arm_joints":    [0, 1, 2, 3, 4, 5, 6],
        "ee_link_index": 8,
        "isLocalpath":   True,
        "urdf_path":     "biarm_model/openarm.urdf",
    },
    "openarmRight": {
        "rest_pose":     [0, 0, 0, 1.2, 0, 0, 0],
        "arm_joints":    list(range(2, 9)),   # joint id 2..8
        "ee_link_index": 10,                  # TCP
        "isLocalpath":   True,
        "urdf_path":     "biarm_model/openarm_right.urdf",
        "max_reach":     0.7,
    },
    "openarmv2right": {
        "rest_pose":     [0, 0, 0, 0, 0, 0, 0],
        "arm_joints":    list(range(2, 9)),
        "ee_link_index": 8,
        "isLocalpath":   True,
        "urdf_path":     "biarm_model/openarm_v2_right.urdf",
    },
}

DEFAULT_ROBOT = "openarmRight"

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_csv_cols(n_joints: int):
    """Restituisce la lista ordinata di colonne CSV per un robot con n_joints giunti."""
    return (
        ["target_x", "target_y", "target_z",
         "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"]
        + [f"joint_{i}_{sc}" for i in range(n_joints) for sc in ("sin", "cos")]
    )

def make_out(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

def save_fig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → salvato: {path}")

def fmt_num(n):
    return f"{n:,}".replace(",", "_")

# ─────────────────────────────────────────────────────────────────────────────
#  SETUP PYBULLET: carica il robot e costruisce la mappa joint_id → indice IK
#  Speculare esatto di main_parallelizzato.py (vedi commento "MAPPATURA GIUNTI")
# ─────────────────────────────────────────────────────────────────────────────
def load_robot_pybullet(urdf_full: str, robot_cfg: dict):
    """
    Carica il robot in PyBullet (modalità DIRECT) e restituisce:
      client, robo_id, arm_joints, ee_link_index, joint_limits
      (joint_limits = lista di (lower, upper) per ogni arm_joint)

    La mappa ik_index_of NON serve per la FK: resetJointState usa direttamente
    i joint_id, non gli indici dell'array IK. La costruiamo comunque e la
    stampiamo come sanity-check per debug.
    """
    import pybullet as p
    import pybullet_data

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    robo = p.loadURDF(urdf_full, basePosition=[0, 0, 0],
                      useFixedBase=True, physicsClientId=client)

    arm_joints    = robot_cfg["arm_joints"]
    ee_link_index = robot_cfg["ee_link_index"]

    # Stessa logica di main_parallelizzato.py: tutti i joint non-FIXED in ordine
    all_movable = [
        i for i in range(p.getNumJoints(robo, physicsClientId=client))
        if p.getJointInfo(robo, i, physicsClientId=client)[2] != 4  # non FIXED
    ]
    ik_index_of = {jid: idx for idx, jid in enumerate(all_movable)}

    # Verifica che arm_joints siano tutti presenti nella mappa
    missing = [jid for jid in arm_joints if jid not in ik_index_of]
    if missing:
        print(f"  [WARN] I seguenti arm_joints non sono joint mobili nel URDF: {missing}")
        print(f"  Joint mobili trovati: {all_movable}")

    # Limiti dai joint info
    joint_limits = []
    for jid in arm_joints:
        info = p.getJointInfo(robo, jid, physicsClientId=client)
        joint_limits.append((float(info[8]), float(info[9])))

    print(f"  Robot caricato: {p.getNumJoints(robo, physicsClientId=client)} joint totali, "
          f"{len(all_movable)} mobili")
    print(f"  arm_joints    : {arm_joints}  (ik_indices: {[ik_index_of[j] for j in arm_joints]})")
    print(f"  ee_link_index : {ee_link_index}")

    return client, robo, arm_joints, ee_link_index, joint_limits

# ─────────────────────────────────────────────────────────────────────────────
#  SEZIONE 1 — INTEGRITÀ STRUTTURALE
# ─────────────────────────────────────────────────────────────────────────────
def check_integrity(csv_path: str, chunksize: int, out_dir: str, csv_cols: list):
    print("\n" + "="*60)
    print("SEZIONE 1 — Integrità strutturale")
    print("="*60)

    total_rows  = 0
    nan_rows    = 0
    inf_rows    = 0
    wrong_cols  = False
    first_chunk = True

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if header != csv_cols:
        wrong_cols = True
        print(f"  [WARN] Header inatteso!")
        print(f"  Atteso : {csv_cols}")
        print(f"  Trovato: {header}")
    else:
        print(f"  [OK] Header corretto ({len(header)} colonne)")

    print(f"  Scansione chunk (chunksize={fmt_num(chunksize)})...")
    t0 = time.time()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=np.float32,
                              names=csv_cols, header=0):
        total_rows += len(chunk)
        nan_rows   += int(chunk.isnull().any(axis=1).sum())
        inf_rows   += int(np.isinf(chunk.select_dtypes(include=[np.number]).values).any(axis=1).sum())

        if first_chunk:
            non_numeric = [c for c in chunk.columns
                           if not pd.api.types.is_numeric_dtype(chunk[c])]
            if non_numeric:
                print(f"  [WARN] Colonne non numeriche: {non_numeric}")
            else:
                print("  [OK] Tutti i tipi sono numerici")
            first_chunk = False

    elapsed = time.time() - t0
    report_lines = [
        f"  Righe totali   : {fmt_num(total_rows)}",
        f"  Righe con NaN  : {fmt_num(nan_rows)}  {'[OK]' if nan_rows == 0 else '[WARN]'}",
        f"  Righe con Inf  : {fmt_num(inf_rows)}  {'[OK]' if inf_rows == 0 else '[WARN]'}",
        f"  Header corretto: {'sì' if not wrong_cols else 'NO — vedi sopra'}",
        f"  Tempo scansione: {elapsed:.1f}s",
    ]
    for line in report_lines:
        print(line)

    report_path = os.path.join(out_dir, "1_integrity_report.txt")
    with open(report_path, "w") as f:
        f.write("SEZIONE 1 — Integrità strutturale\n")
        f.write("\n".join(report_lines) + "\n")
    print(f"  → salvato: {report_path}")
    return total_rows

# ─────────────────────────────────────────────────────────────────────────────
#  SEZIONE 2 — VINCOLI SIN/COS
# ─────────────────────────────────────────────────────────────────────────────
def check_sincos(csv_path: str, chunksize: int, out_dir: str,
                 csv_cols: list, n_joints: int):
    print("\n" + "="*60)
    print("SEZIONE 2 — Vincoli sin²+cos²=1")
    print("="*60)

    TOLS       = [1e-4, 1e-3, 1e-2]
    violations = {t: np.zeros(n_joints, dtype=np.int64) for t in TOLS}
    max_err    = np.zeros(n_joints)
    mean_err   = np.zeros(n_joints)
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=np.float64,
                              names=csv_cols, header=0):
        total_rows += len(chunk)
        for j in range(n_joints):
            s   = chunk[f"joint_{j}_sin"].values
            c   = chunk[f"joint_{j}_cos"].values
            err = np.abs(s**2 + c**2 - 1.0)
            for t in TOLS:
                violations[t][j] += int((err > t).sum())
            max_err[j]  = max(max_err[j], float(err.max()))
            mean_err[j] += float(err.mean()) * len(chunk)

    mean_err /= total_rows

    header_line = f"  {'Giunto':<10} {'max_err':>12} {'mean_err':>12}"
    for t in TOLS:
        header_line += f"  {'viol@'+str(t):>14}"
    print(header_line)
    print("  " + "-" * 80)
    for j in range(n_joints):
        line = f"  joint_{j:<4}  {max_err[j]:12.2e} {mean_err[j]:12.2e}"
        for t in TOLS:
            flag = " !" if violations[t][j] > 0 else "  "
            line += f"  {violations[t][j]:>13,}{flag}"
        print(line)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Sezione 2 — Errore sin²+cos²−1 per giunto", fontsize=13)
    xticks = range(n_joints)
    xlabels = [f"j{j}" for j in range(n_joints)]

    axes[0].bar(xticks, max_err, color="steelblue")
    axes[0].set_title("Errore massimo")
    axes[0].set_xlabel("Giunto"); axes[0].set_ylabel("|sin²+cos²−1|")
    axes[0].set_xticks(xticks); axes[0].set_xticklabels(xlabels)
    axes[0].axhline(1e-3, color="red", ls="--", lw=1, label="soglia 1e-3")
    axes[0].legend()

    axes[1].bar(xticks, mean_err, color="darkorange")
    axes[1].set_title("Errore medio")
    axes[1].set_xlabel("Giunto"); axes[1].set_ylabel("|sin²+cos²−1| (media)")
    axes[1].set_xticks(xticks); axes[1].set_xticklabels(xlabels)
    axes[1].axhline(1e-4, color="red", ls="--", lw=1, label="soglia 1e-4")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "2_sincos_errors.png"))

    report_path = os.path.join(out_dir, "2_sincos_report.txt")
    with open(report_path, "w") as f:
        f.write("SEZIONE 2 — Vincoli sin²+cos²=1\n")
        for j in range(n_joints):
            f.write(f"joint_{j}: max={max_err[j]:.2e}  mean={mean_err[j]:.2e}  "
                    + "  ".join(f"viol@{t}={violations[t][j]}" for t in TOLS) + "\n")
    print(f"  → salvato: {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  SEZIONE 3 — DISTRIBUZIONE SPAZIALE
# ─────────────────────────────────────────────────────────────────────────────
def check_spatial(csv_path: str, chunksize: int, out_dir: str, csv_cols: list,
                  sample_frac: float = 0.02):
    print("\n" + "="*60)
    print("SEZIONE 3 — Distribuzione spaziale")
    print("="*60)

    POS_COLS = ["target_x", "target_y", "target_z"]
    every_n  = max(1, int(1.0 / sample_frac))
    print(f"  Campionamento: 1 riga ogni {every_n} (≈{sample_frac*100:.1f}%)")

    xyz_chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize,
                              usecols=POS_COLS, dtype=np.float32,
                              names=csv_cols, header=0):
        mask = np.arange(len(chunk)) % every_n == 0
        xyz_chunks.append(chunk.loc[mask, POS_COLS].values)

    xyz = np.concatenate(xyz_chunks, axis=0)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    print(f"  Punti campionati: {fmt_num(len(x))}")
    print(f"  Range X: [{x.min():.3f}, {x.max():.3f}]  "
          f"Y: [{y.min():.3f}, {y.max():.3f}]  "
          f"Z: [{z.min():.3f}, {z.max():.3f}]")

    # Scatter 3D
    if HAS_PLOTLY:
        max_plot = 200_000
        idx = np.random.choice(len(x), min(max_plot, len(x)), replace=False)
        fig3d = go.Figure(data=[go.Scatter3d(
            x=x[idx], y=y[idx], z=z[idx], mode="markers",
            marker=dict(size=1.5, color=z[idx], colorscale="Viridis",
                        opacity=0.5, colorbar=dict(title="Z (m)")),
        )])
        fig3d.update_layout(
            title="Sezione 3 — Distribuzione spaziale target (campione)",
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
            width=900, height=700,
        )
        html_path = os.path.join(out_dir, "3a_scatter3d.html")
        fig3d.write_html(html_path)
        print(f"  → salvato: {html_path}")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Sezione 3 — Proiezioni spaziali (campione)", fontsize=13)
        for ax, (lx, ly, vx, vy) in zip(axes,
                [("X","Y",x,y), ("X","Z",x,z), ("Y","Z",y,z)]):
            ax.hexbin(vx, vy, gridsize=60, cmap="viridis", mincnt=1)
            ax.set_xlabel(f"{lx} (m)"); ax.set_ylabel(f"{ly} (m)")
            ax.set_title(f"{lx} vs {ly}")
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, "3a_projections.png"))

    # Istogrammi marginali
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Sezione 3 — Istogrammi marginali target_x/y/z", fontsize=13)
    for ax, lbl, val, col in zip(axes,
            ["target_x","target_y","target_z"], [x,y,z],
            ["steelblue","darkorange","forestgreen"]):
        ax.hist(val, bins=100, color=col, alpha=0.85, edgecolor="none")
        ax.set_xlabel(f"{lbl} (m)"); ax.set_ylabel("Conteggio"); ax.set_title(lbl)
        ax.axvline(val.mean(), color="red", ls="--", lw=1.2,
                   label=f"μ={val.mean():.3f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "3b_histograms_xyz.png"))

# ─────────────────────────────────────────────────────────────────────────────
#  SEZIONE 4 — BOXPLOT GIUNTI
# ─────────────────────────────────────────────────────────────────────────────
def check_joints_boxplot(csv_path: str, chunksize: int, out_dir: str,
                          csv_cols: list, n_joints: int,
                          joint_limits: list = None,
                          sample_frac: float = 0.02):
    print("\n" + "="*60)
    print("SEZIONE 4 — Distribuzione angoli giunti (boxplot)")
    print("="*60)

    every_n = max(1, int(1.0 / sample_frac))
    print(f"  Campionamento: 1 riga ogni {every_n}")

    usecols = [f"joint_{i}_{sc}" for i in range(n_joints) for sc in ("sin", "cos")]
    angle_chunks = [[] for _ in range(n_joints)]

    for chunk in pd.read_csv(csv_path, chunksize=chunksize,
                              usecols=usecols, dtype=np.float32,
                              names=csv_cols, header=0):
        mask = np.arange(len(chunk)) % every_n == 0
        sub  = chunk.loc[mask]
        for j in range(n_joints):
            s = sub[f"joint_{j}_sin"].values
            c = sub[f"joint_{j}_cos"].values
            angle_chunks[j].append(np.arctan2(s, c))   # rad, [-π, π]

    angles_all = [np.concatenate(ch) for ch in angle_chunks]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(angles_all, patch_artist=True, notch=False,
                    medianprops=dict(color="red", lw=2))
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)

    # Sovrapponi limiti di giunto da URDF se disponibili
    if joint_limits:
        for ji, (lo, hi) in enumerate(joint_limits):
            x_pos = ji + 1
            ax.plot([x_pos - 0.4, x_pos + 0.4], [lo, lo],
                    color="navy", ls="--", lw=1.2)
            ax.plot([x_pos - 0.4, x_pos + 0.4], [hi, hi],
                    color="navy", ls="--", lw=1.2)
        # Aggiunge entry alla legenda
        ax.plot([], [], color="navy", ls="--", lw=1.2, label="limiti URDF")
        ax.legend(fontsize=8)

    ax.set_xticks(range(1, n_joints + 1))
    ax.set_xticklabels([f"joint_{j}" for j in range(n_joints)], rotation=30)
    ax.set_ylabel("Angolo atan2(sin,cos) [rad]")
    ax.set_title("Sezione 4 — Distribuzione angoli giunti (campione)")
    ax.axhline(0,          color="grey", ls=":",  lw=1)
    ax.axhline( math.pi,   color="grey", ls="--", lw=1, label="±π")
    ax.axhline(-math.pi,   color="grey", ls="--", lw=1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "4_joint_boxplot.png"))

    print(f"  {'Giunto':<10} {'min':>8} {'q25':>8} {'median':>8} {'q75':>8} {'max':>8}  [rad]")
    for j, ang in enumerate(angles_all):
        q = np.quantile(ang, [0, 0.25, 0.5, 0.75, 1.0])
        lim_str = ""
        if joint_limits:
            lo, hi = joint_limits[j]
            out_of_range = ((ang < lo) | (ang > hi)).sum()
            lim_str = f"  limiti URDF=[{lo:.3f},{hi:.3f}]"
            if out_of_range > 0:
                lim_str += f"  [WARN] {out_of_range} angoli fuori range"
        print(f"  joint_{j:<4}  {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f} {q[3]:8.3f} {q[4]:8.3f}{lim_str}")

# ─────────────────────────────────────────────────────────────────────────────
#  SEZIONE 6 — VERIFICA FK
# ─────────────────────────────────────────────────────────────────────────────
def check_fk(csv_path: str, chunksize: int, out_dir: str,
             robot_cfg: dict, urdf_full: str,
             fk_samples: int, fk_tol: float, seed: int, n_joints: int,
             csv_cols: list):
    print("\n" + "="*60)
    print("SEZIONE 6 — Verifica cinematica diretta (FK)")
    print("="*60)

    try:
        import pybullet as p
    except ImportError:
        print("  [SKIP] PyBullet non disponibile. Installa con: pip install pybullet")
        return

    if not os.path.exists(urdf_full):
        print(f"  [WARN] URDF non trovato: {urdf_full}")
        print("  Imposta --urdf con il path corretto. Verifica FK saltata.")
        return

    # ── Campionamento reservoir (1 passata, O(fk_samples) RAM) ────────────────
    print(f"  Campionamento reservoir: {fmt_num(fk_samples)} righe...")
    rng        = np.random.default_rng(seed)
    reservoir  = []
    total_seen = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=np.float64,
                              names=csv_cols, header=0):
        for row in chunk.itertuples(index=False):
            total_seen += 1
            if len(reservoir) < fk_samples:
                reservoir.append(row)
            else:
                j = int(rng.integers(0, total_seen))
                if j < fk_samples:
                    reservoir[j] = row

    sample_df = pd.DataFrame(reservoir)
    print(f"  Campione ottenuto: {len(sample_df)} righe su {fmt_num(total_seen)} totali")

    # ── Carica robot ──────────────────────────────────────────────────────────
    client, robo, arm_joints, ee_link_index, _ = load_robot_pybullet(urdf_full, robot_cfg)

    import pybullet as p

    pos_errors  = []
    quat_errors = []

    for row in sample_df.itertuples(index=False):
        # Ricostruisci angoli da sin/cos
        # NOTA: j qui è l'indice logico nel dataset (0..n_joints-1),
        #       arm_joints[j] è il joint_id reale nel URDF.
        #       resetJointState vuole il joint_id, NON l'indice IK — nessuna
        #       mappa necessaria per la FK diretta.
        angles = [math.atan2(getattr(row, f"joint_{j}_sin"),
                             getattr(row, f"joint_{j}_cos"))
                  for j in range(n_joints)]

        for j, jid in enumerate(arm_joints):
            p.resetJointState(robo, jid, angles[j], physicsClientId=client)
        p.stepSimulation(physicsClientId=client)

        ls      = p.getLinkState(robo, ee_link_index, physicsClientId=client)
        fk_pos  = np.array(ls[4])   # world position
        fk_quat = np.array(ls[5])   # quaternione (x,y,z,w)

        target_pos = np.array([row.target_x, row.target_y, row.target_z])
        target_q   = np.array([row.hand_quat_qx, row.hand_quat_qy,
                                row.hand_quat_qz, row.hand_quat_qw])

        pos_err  = float(np.linalg.norm(fk_pos - target_pos))
        dot      = min(abs(float(np.dot(fk_quat, target_q))), 1.0)
        quat_err = float(2 * math.acos(dot))   # radianti

        pos_errors.append(pos_err)
        quat_errors.append(quat_err)

    p.disconnect(client)

    pos_errors  = np.array(pos_errors)
    quat_errors = np.array(quat_errors)
    n_fail_pos  = int((pos_errors  > fk_tol).sum())
    n_fail_quat = int((quat_errors > 0.1).sum())

    print(f"\n  Errore posizione FK (m):")
    print(f"    media={pos_errors.mean():.5f}  mediana={np.median(pos_errors):.5f}  "
          f"max={pos_errors.max():.5f}  p99={np.percentile(pos_errors,99):.5f}")
    print(f"    Righe > {fk_tol}m : {n_fail_pos} / {len(pos_errors)}  "
          f"({'[OK]' if n_fail_pos == 0 else '[WARN]'})")

    print(f"\n  Errore orientazione FK (rad):")
    print(f"    media={quat_errors.mean():.5f}  mediana={np.median(quat_errors):.5f}  "
          f"max={quat_errors.max():.5f}  p99={np.percentile(quat_errors,99):.5f}")
    print(f"    Righe > 0.1 rad : {n_fail_quat} / {len(quat_errors)}  "
          f"({'[OK]' if n_fail_quat == 0 else '[WARN]'})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sezione 6 — Errori FK (cinematica diretta)", fontsize=13)

    axes[0].hist(pos_errors, bins=100, color="steelblue", edgecolor="none")
    axes[0].axvline(fk_tol, color="red", ls="--", lw=1.5, label=f"soglia {fk_tol}m")
    axes[0].set_xlabel("Errore posizione (m)"); axes[0].set_ylabel("Conteggio")
    axes[0].set_title("Distribuzione errore posizione")
    axes[0].set_yscale("log"); axes[0].legend()

    axes[1].hist(np.degrees(quat_errors), bins=100, color="darkorange", edgecolor="none")
    axes[1].axvline(math.degrees(0.1), color="red", ls="--", lw=1.5, label="soglia 5.7°")
    axes[1].set_xlabel("Errore orientazione (°)"); axes[1].set_ylabel("Conteggio")
    axes[1].set_title("Distribuzione errore orientazione")
    axes[1].set_yscale("log"); axes[1].legend()

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "6_fk_errors.png"))

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(pos_errors, np.degrees(quat_errors), s=3, alpha=0.3, c="steelblue")
    ax2.axvline(fk_tol,           color="red",    ls="--", lw=1, label=f"pos soglia {fk_tol}m")
    ax2.axhline(math.degrees(0.1),color="orange", ls="--", lw=1, label="quat soglia 5.7°")
    ax2.set_xlabel("Errore posizione (m)"); ax2.set_ylabel("Errore orientazione (°)")
    ax2.set_title("Sezione 6 — Errore pos vs quat"); ax2.legend()
    fig2.tight_layout()
    save_fig(fig2, os.path.join(out_dir, "6_fk_scatter.png"))

    report_path = os.path.join(out_dir, "6_fk_report.txt")
    with open(report_path, "w") as f:
        f.write("SEZIONE 6 — Verifica FK\n")
        f.write(f"Robot: {robot_cfg.get('urdf_path','?')}\n")
        f.write(f"arm_joints: {arm_joints}  ee_link_index: {ee_link_index}\n")
        f.write(f"Campione: {len(pos_errors)} righe su {total_seen} totali\n\n")
        f.write(f"Errore posizione (m):\n"
                f"  mean={pos_errors.mean():.5f}  median={np.median(pos_errors):.5f}  "
                f"max={pos_errors.max():.5f}  p99={np.percentile(pos_errors,99):.5f}\n"
                f"  Righe > {fk_tol}m: {n_fail_pos}\n\n")
        f.write(f"Errore orientazione (rad):\n"
                f"  mean={quat_errors.mean():.5f}  median={np.median(quat_errors):.5f}  "
                f"max={quat_errors.max():.5f}  p99={np.percentile(quat_errors,99):.5f}\n"
                f"  Righe > 0.1 rad: {n_fail_quat}\n")
    print(f"  → salvato: {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validazione dataset cinematica inversa"
    )
    parser.add_argument("--csv",        required=True,
                        help="Path al file CSV del dataset")
    parser.add_argument("--robot",      default=DEFAULT_ROBOT,
                        choices=list(ROBOT_CONFIGS.keys()),
                        help=f"Robot da usare (default: {DEFAULT_ROBOT})")
    parser.add_argument("--urdf",       default=None,
                        help="Override path URDF (default: usa quello della config robot)")
    parser.add_argument("--chunksize",  type=int, default=500_000)
    parser.add_argument("--fk_samples", type=int, default=5_000)
    parser.add_argument("--fk_tol",     type=float, default=0.012)
    parser.add_argument("--out",        default="validation_output")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--skip_fk",    action="store_true",
                        help="Salta la verifica FK")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERRORE] File non trovato: {args.csv}")
        sys.exit(1)

    robot_cfg = ROBOT_CONFIGS[args.robot]
    n_joints  = len(robot_cfg["arm_joints"])
    csv_cols  = get_csv_cols(n_joints)

    # Risolvi path URDF
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_rel   = args.urdf if args.urdf else robot_cfg["urdf_path"]
    urdf_full  = urdf_rel if os.path.isabs(urdf_rel) \
                 else os.path.join(script_dir, urdf_rel)

    # Carica limiti giunto da URDF se disponibile (per boxplot sezione 4)
    joint_limits = None
    if not args.skip_fk or True:   # li carichiamo sempre se l'URDF esiste
        try:
            import pybullet as p
            if os.path.exists(urdf_full):
                _client, _robo, _aj, _ee, joint_limits = \
                    load_robot_pybullet(urdf_full, robot_cfg)
                p.disconnect(_client)
        except Exception as e:
            print(f"  [WARN] Impossibile caricare URDF per i limiti: {e}")

    from datetime import datetime
    run_dir = os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.out = run_dir
    make_out(args.out)
    np.random.seed(args.seed)

    t_start = time.time()
    print(f"\nDataset    : {args.csv}")
    print(f"Robot      : {args.robot}  ({n_joints} giunti)")
    print(f"URDF       : {urdf_full}")
    print(f"Output dir : {args.out}")
    print(f"Chunksize  : {fmt_num(args.chunksize)}")

    check_integrity(args.csv, args.chunksize, args.out, csv_cols)
    check_sincos   (args.csv, args.chunksize, args.out, csv_cols, n_joints)
    check_spatial  (args.csv, args.chunksize, args.out, csv_cols)
    check_joints_boxplot(args.csv, args.chunksize, args.out, csv_cols,
                         n_joints, joint_limits)

    if not args.skip_fk:
        check_fk(args.csv, args.chunksize, args.out,
                 robot_cfg, urdf_full,
                 args.fk_samples, args.fk_tol, args.seed,
                 n_joints, csv_cols)
    else:
        print("\n[SKIP] Verifica FK saltata (--skip_fk)")

    elapsed = time.time() - t_start
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"\n{'='*60}")
    print(f"Validazione completata in {h:02d}:{m:02d}:{s:02d}")
    print(f"Output: {args.out}/")
    print("="*60)


if __name__ == "__main__":
    main()