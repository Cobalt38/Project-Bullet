"""
test_ik_roundtrip.py
--------------------
Test di round-trip IK per OpenArm (destra e sinistra).

Pipeline per ogni test case:
  1. Imposta gli angoli hardcoded sul robot PyBullet
  2. FK: legge posizione e orientamento dell'EE dal simulatore
  3. (opzionale mirroring se braccio sinistro)
  4. IK: chiede al modello gli angoli corrispondenti
  5. (opzionale joint-fix se braccio sinistro)
  6. FK di verifica: applica i giunti predetti e rilegge l'EE
  7. Stampa errori di posizione e orientamento

Uso:
  python test_ik_roundtrip.py --model_path path/al/modello [--model_dir ik_model]
                              [--side right|left|both] [--gui]
"""

import argparse
import math
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data

from ik_inference import load_model_and_metadata, run_inference, decode_output, build_input

# ---------------------------------------------------------------------------
# Angoli di test hardcoded (radianti) — modifica liberamente
# ---------------------------------------------------------------------------
# Ogni entry è (nome_descrittivo, angoli_7_giunti)
# Ordine giunti: indici [2,3,4,5,6,7,8] dell'URDF

TEST_CASES = [
    ("rest_pose",       [0.0,   0.0,   0.0,   1.2,   0.0,   0.0,   0.0]),
    ("elbow_up",        [0.0,  -0.5,   0.0,   0.8,   0.0,  -0.5,   0.0]),
    ("side_reach",      [0.5,   0.3,  -0.3,   1.0,   0.2,  -0.4,   0.1]),
    ("wrist_roll",      [0.0,   0.0,   0.0,   1.2,   0.0,   0.0,   1.57]),
    ("full_extension",  [0.0,  -0.8,   0.0,   0.3,   0.0,  -0.8,   0.0]),
    ("complex_pose",    [0.4,  -0.6,   0.2,   0.9,  -0.3,  -0.5,   0.8]),
]

# ---------------------------------------------------------------------------
# Costanti (stessa configurazione di run_mp.py)
# ---------------------------------------------------------------------------

ARM_JOINTS    = [2, 3, 4, 5, 6, 7, 8]
EE_LINK_INDEX = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

URDF_PATHS = {
    "right": "biarm_model/openarm_right.urdf",
    "left":  "biarm_model/openarm_left.urdf",
}

# ---------------------------------------------------------------------------
# Mirroring XZ (identico a run_mp.py — piano di simmetria XZ, Y invertita)
# ---------------------------------------------------------------------------

def left_to_right_mirroring(pos: np.ndarray, quat: np.ndarray):
    """Piano di simmetria XZ (Y invertita). quat in xyzw."""
    pos_m  = pos  * np.array([1.0, -1.0,  1.0])
    quat_m = quat * np.array([1.0, -1.0,  1.0, -1.0])  # nega qy
    return pos_m, quat_m


def openarm_left_joint_fix(joints: np.ndarray) -> np.ndarray:
    """Inversione giunti per il braccio sinistro (identico a run_mp.py)."""
    j = joints.copy()
    j[3] *= -1
    return j * np.array([-1.0] * len(j))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUAT_EPS = 1e-8

def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Errore angolare (rad) tra due quaternioni xyzw.
    Gestisce il doppio-segno: q e -q rappresentano la stessa rotazione.
    """
    q1 = q1 / (np.linalg.norm(q1) + QUAT_EPS)
    q2 = q2 / (np.linalg.norm(q2) + QUAT_EPS)
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * math.acos(dot)


def set_joints(robot_id: int, angles):
    for joint_idx, val in zip(ARM_JOINTS, angles):
        p.resetJointState(robot_id, joint_idx, float(val))


def read_ee(robot_id: int) -> tuple:
    """Restituisce (pos_np, quat_np) dell'EE in coordinate mondo. quat = xyzw."""
    state = p.getLinkState(robot_id, EE_LINK_INDEX,
                           computeForwardKinematics=True)
    pos  = np.array(state[4])   # worldLinkFramePosition
    quat = np.array(state[5])   # worldLinkFrameOrientation (xyzw)
    return pos, quat

# ---------------------------------------------------------------------------
# Singolo test case
# ---------------------------------------------------------------------------

def run_test_case(name: str, gt_joints: list, robot_id: int,
                  model, meta, side: str, verbose: bool = True) -> dict:
    """
    Esegue un singolo round-trip e restituisce un dict con i risultati.
    """
    gt_joints = np.array(gt_joints, dtype=np.float64)

    # ── STEP 1: FK con angoli ground-truth ──────────────────────────────────
    set_joints(robot_id, gt_joints)
    p.stepSimulation()
    pos_gt, quat_gt = read_ee(robot_id)

    # ── STEP 2: prepara input per il modello (mirroring se sinistra) ────────
    pos_in  = pos_gt.copy()
    quat_in = quat_gt.copy()

    if side == "left":
        pos_in, quat_in = left_to_right_mirroring(pos_in, quat_in)

    # ── STEP 3: inferenza IK ────────────────────────────────────────────────
    raw    = run_inference(model, build_input(*pos_in, *quat_in))
    result = decode_output(raw, meta.get("output_columns", []))
    pred_joints = result.get("angles_rad")

    if pred_joints is None:
        return {"name": name, "error": "Il modello ha restituito None"}

    pred_joints = np.array(pred_joints, dtype=np.float64)

    # ── STEP 4: joint-fix per il braccio sinistro ───────────────────────────
    if side == "left":
        pred_joints = openarm_left_joint_fix(pred_joints)

    # ── STEP 5: FK di verifica ───────────────────────────────────────────────
    set_joints(robot_id, pred_joints)
    p.stepSimulation()
    pos_pred, quat_pred = read_ee(robot_id)

    # ── STEP 6: calcolo errori ───────────────────────────────────────────────
    pos_err_m   = np.linalg.norm(pos_pred - pos_gt)
    ori_err_rad = quat_angle_error(quat_pred, quat_gt)
    ori_err_deg = math.degrees(ori_err_rad)
    joint_err   = np.linalg.norm(pred_joints - gt_joints)

    res = {
        "name":         name,
        "gt_joints":    gt_joints,
        "pred_joints":  pred_joints,
        "pos_gt":       pos_gt,
        "pos_pred":     pos_pred,
        "quat_gt":      quat_gt,
        "quat_pred":    quat_pred,
        "pos_err_m":    pos_err_m,
        "ori_err_deg":  ori_err_deg,
        "joint_err_l2": joint_err,
    }

    if verbose:
        _print_result(res, side)

    return res


def _print_result(r: dict, side: str):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Test: {r['name']}   (braccio: {side.upper()})")
    print(sep)

    if "error" in r:
        print(f"  ❌  ERRORE: {r['error']}")
        return

    gt  = r["gt_joints"]
    pr  = r["pred_joints"]
    dj  = pr - gt

    print("  Giunti (rad):")
    print(f"    {'idx':>4}  {'GT':>8}  {'Pred':>8}  {'Δ':>8}")
    for i, (g, p_val, d) in enumerate(zip(gt, pr, dj)):
        flag = "  ←" if abs(d) > 0.1 else ""
        print(f"    {ARM_JOINTS[i]:>4}  {g:>8.4f}  {p_val:>8.4f}  {d:>+8.4f}{flag}")

    print(f"\n  FK ground-truth :  pos={_fmt_v(r['pos_gt'])}  quat={_fmt_v(r['quat_gt'])}")
    print(f"  FK predetta     :  pos={_fmt_v(r['pos_pred'])}  quat={_fmt_v(r['quat_pred'])}")

    ok_pos = r["pos_err_m"]   < 0.01   # 1 cm
    ok_ori = r["ori_err_deg"] < 2.0    # 2°

    status_pos = "✅" if ok_pos else "⚠️ "
    status_ori = "✅" if ok_ori else "⚠️ "

    print(f"\n  {status_pos}  Errore posizione  : {r['pos_err_m']*1000:.2f} mm")
    print(f"  {status_ori}  Errore orientamento: {r['ori_err_deg']:.3f}°")
    print(f"       Errore joint L2 : {r['joint_err_l2']:.4f} rad")


def _fmt_v(v):
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"

# ---------------------------------------------------------------------------
# Runner per un singolo braccio
# ---------------------------------------------------------------------------

def run_arm(side: str, model_path: str, model_dir: str, use_gui: bool):
    urdf_path = os.path.join(SCRIPT_DIR, URDF_PATHS[side])
    if not os.path.isfile(urdf_path):
        print(f"[{side.upper()}] URDF non trovato: {urdf_path}")
        return

    print(f"\n{'═'*60}")
    print(f"  BRACCIO: {side.upper()}")
    print(f"{'═'*60}")

    # ── PyBullet ────────────────────────────────────────────────────────────
    mode = p.GUI if use_gui else p.DIRECT
    client = p.connect(mode)
    p.resetSimulation(physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, 0, physicsClientId=client)

    # Nota: le chiamate p.xxx usano il client di default (l'ultimo connesso).
    # Con DIRECT non c'è ambiguità; con GUI si usa un singolo client per braccio.

    p.loadURDF("plane.urdf")
    robot = p.loadURDF(urdf_path, basePosition=[0, 0, 0],
                       useFixedBase=True, globalScaling=1.0)

    # disabilita self-collision (non influisce sulla FK ma è coerente con run_mp)
    n = p.getNumJoints(robot)
    links = [-1] + list(range(n))
    for i in links:
        for j in links:
            if i != j:
                p.setCollisionFilterPair(robot, robot, i, j, enableCollision=0)

    # ── Modello ─────────────────────────────────────────────────────────────
    print(f"[{side.upper()}] Caricamento modello...")
    model, meta = load_model_and_metadata(model_path=model_path, model_dir=model_dir)
    print(f"[{side.upper()}] Modello caricato. Output joints: {meta.get('output_columns', [])}")

    # ── Esecuzione test ──────────────────────────────────────────────────────
    results = []
    for name, joints in TEST_CASES:
        r = run_test_case(name, joints, robot, model, meta, side=side)
        results.append(r)

    # ── Sommario ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SOMMARIO — {side.upper()}")
    print(f"{'═'*60}")
    print(f"  {'Test':<20}  {'Pos err (mm)':>14}  {'Ori err (°)':>12}  {'Status'}")
    print(f"  {'─'*20}  {'─'*14}  {'─'*12}  {'─'*6}")

    n_ok = 0
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<20}  {'N/A':>14}  {'N/A':>12}  ❌ {r['error']}")
            continue
        ok = r["pos_err_m"] < 0.01 and r["ori_err_deg"] < 2.0
        n_ok += ok
        sym = "✅" if ok else "⚠️ "
        print(f"  {r['name']:<20}  {r['pos_err_m']*1000:>13.2f}  "
              f"{r['ori_err_deg']:>11.3f}°  {sym}")

    valid = [r for r in results if "error" not in r]
    if valid:
        avg_pos = np.mean([r["pos_err_m"] for r in valid]) * 1000
        avg_ori = np.mean([r["ori_err_deg"] for r in valid])
        print(f"\n  Media errore pos: {avg_pos:.2f} mm   "
              f"Media errore ori: {avg_ori:.3f}°   "
              f"Passed: {n_ok}/{len(valid)}")

    try:
        p.disconnect()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Round-trip IK test: FK → IK → FK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model_path", required=True,
                    help="Percorso al modello TF")
    ap.add_argument("--model_dir", default="ik_model",
                    help="Directory opzionale del modello (default: ik_model)")
    ap.add_argument("--side", default="both", choices=["right", "left", "both"],
                    help="Quale braccio testare (default: both)")
    ap.add_argument("--gui", action="store_true",
                    help="Apri finestra PyBullet GUI (una per braccio, sequenziale)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    sides = ["right", "left"] if args.side == "both" else [args.side]

    for side in sides:
        run_arm(side, args.model_path, args.model_dir, args.gui)

    print("\n[DONE] Test completati.")