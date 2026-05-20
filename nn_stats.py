# eval_ik.py
import json
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from tqdm import tqdm

IK_HOST   = "192.168.102.188"
IK_PORT   = 8001
CSV_PATH  = "csv_belli/dataset_fixed.csv"
N_SAMPLES = 500000
SEED      = 42

JOINT_COLS = ["Root.001_ry", "Root.002_rx", "Root.003_rx", "Root.004_rx", "Head_ry"]
QUAT_COLS  = ["hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"]
POS_COLS   = ["target_x", "target_y", "target_z"]
BONE_MAP   = [("Root.001","ry"),("Root.002","rx"),("Root.003","rx"),("Root.004","rx"),("Head","ry")]
OUTLIER_THRESHOLD_DEG = 5.0

def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IK_HOST, IK_PORT))
    s.settimeout(5.0)
    return s

def predict(s, target_pos, target_ori):
    payload = {"target": list(target_pos) + list(target_ori)}
    s.sendall((json.dumps(payload) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        buf += s.recv(4096)
    resp = json.loads(buf.split(b"\n")[0])
    if not resp.get("ok"):
        return None
    r = resp["result"]
    return np.array([r[bone][axis] for bone, axis in BONE_MAP])

def plot_overhead(rec: pd.DataFrame, threshold: float):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0e0e0e")
    for ax in axes:
        ax.set_facecolor("#0e0e0e")

    # --- plot 1: scatter colorato per max_err ---
    ax = axes[0]
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=threshold, vmax=rec["max_err"].quantile(0.999))
    sc = ax.scatter(
        rec["x"], rec["y"],
        c=rec["max_err"],
        cmap="RdYlGn_r",
        norm=norm,
        s=6, alpha=0.6, linewidths=0,
    )
    # evidenzia outlier
    out = rec[rec["max_err"] > threshold]
    ax.scatter(out["x"], out["y"], c="red", s=40, zorder=5,
               linewidths=0.8, edgecolors="white", label=f"outlier (>{threshold}°)")

    # simbolo robot al centro (0,0)
    ax.plot(0, 0, marker=(5, 1), color="cyan", markersize=14, zorder=10)
    ax.annotate("robot", (0, 0), color="cyan", fontsize=8,
                textcoords="offset points", xytext=(6, 6))

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("max joint error (°)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("target_x (m)", color="white")
    ax.set_ylabel("target_y (m)", color="white")
    ax.set_title("Errore massimo — vista dall'alto", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)

    # --- plot 2: heatmap 2D binned dell'errore medio ---
    ax = axes[1]
    bins = 40
    xedges = np.linspace(rec["x"].min(), rec["x"].max(), bins + 1)
    yedges = np.linspace(rec["y"].min(), rec["y"].max(), bins + 1)

    # calcola errore medio per bin
    heatmap_sum, _, _ = np.histogram2d(rec["x"], rec["y"], bins=[xedges, yedges],
                                        weights=rec["max_err"])
    heatmap_cnt, _, _ = np.histogram2d(rec["x"], rec["y"], bins=[xedges, yedges])
    with np.errstate(invalid="ignore"):
        heatmap_mean = np.where(heatmap_cnt > 0, heatmap_sum / heatmap_cnt, np.nan)

    im = ax.imshow(
        heatmap_mean.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="inferno",
        aspect="equal",
        interpolation="bilinear",
    )
    ax.plot(0, 0, marker=(5, 1), color="cyan", markersize=14, zorder=10)
    ax.annotate("robot", (0, 0), color="cyan", fontsize=8,
                textcoords="offset points", xytext=(6, 6))

    cb2 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("errore medio per cella (°)", color="white", fontsize=9)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("target_x (m)", color="white")
    ax.set_ylabel("target_y (m)", color="white")
    ax.set_title("Heatmap errore medio — vista dall'alto", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#2a2a2a", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("ik_error_overhead.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\nPlot salvato: ik_error_overhead.png")
    plt.show()


def main():
    df = pd.read_csv(CSV_PATH)
    if N_SAMPLES:
        df = df.sample(n=min(N_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)

    print(f"Valutazione su {len(df)} campioni...")
    s = connect()

    errors  = []
    records = []   # x, y, max_err per ogni campione
    failed  = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pos = row[POS_COLS].values.astype(float)
        ori = row[QUAT_COLS].values.astype(float)
        gt  = row[JOINT_COLS].values.astype(float)

        pred = predict(s, pos, ori)
        if pred is None:
            continue

        err     = pred - gt
        max_err = np.max(np.abs(err))
        errors.append(err)
        records.append({"x": pos[0], "y": pos[1], "max_err": max_err})

        if max_err > OUTLIER_THRESHOLD_DEG:
            failed.append({"pos": pos, "ori": ori, "gt": gt,
                           "pred": pred, "err": err, "max_err": max_err})

    s.close()
    errors = np.array(errors)
    rec    = pd.DataFrame(records)

    # --- statistiche ---
    print("\n=== STATISTICHE PER JOINT ===")
    print(f"{'joint':<14} {'MAE':>7} {'RMSE':>7} {'max|err|':>9} {'std':>7}")
    print("-" * 44)
    for i, col in enumerate(JOINT_COLS):
        e = errors[:, i]
        print(f"{col:<14} {np.mean(np.abs(e)):>7.3f} {np.sqrt(np.mean(e**2)):>7.3f} "
              f"{np.max(np.abs(e)):>9.3f} {np.std(e):>7.3f}")

    abs_max = rec["max_err"].values
    print(f"\n=== ERRORE MASSIMO PER CAMPIONE ===")
    print(f"  mediana:    {np.median(abs_max):.3f}°")
    print(f"  95° pct:    {np.percentile(abs_max, 95):.3f}°")
    print(f"  99° pct:    {np.percentile(abs_max, 99):.3f}°")
    print(f"  max:        {np.max(abs_max):.3f}°")
    print(f"  outlier (>{OUTLIER_THRESHOLD_DEG}°): {len(failed)} / {len(errors)}  "
          f"({100*len(failed)/len(errors):.2f}%)")

    if failed:
        print(f"\n=== TOP 5 OUTLIER ===")
        for f in sorted(failed, key=lambda x: -x["max_err"])[:5]:
            print(f"\n  pos={np.round(f['pos'],3)}  max_err={f['max_err']:.2f}°")
            for i, col in enumerate(JOINT_COLS):
                print(f"    {col:<14} gt={f['gt'][i]:>8.2f}  pred={f['pred'][i]:>8.2f}  "
                      f"Δ={f['err'][i]:>+8.2f}")

    plot_overhead(rec, OUTLIER_THRESHOLD_DEG)

if __name__ == "__main__":
    main()