"""
udp_joint_viewer.py
───────────────────
Carica un URDF in PyBullet (GUI) e aggiorna i giunti + il visual target
in real-time ricevendo i dati via UDP.

Protocollo UDP
──────────────
Ogni pacchetto è un JSON su singola riga (terminato da \\n oppure no):

{
    "joints":     [j1, j2, j3, j4, j5, j6, j7],   // gradi, len == ARM_JOINTS
    "target_pos": [x, y, z],                        // metri, world frame
    "target_quat": [qx, qy, qz, qw]                // quaternione xyzw
}

I campi "target_pos" e "target_quat" sono opzionali: se assenti il visual
rimane nell'ultima posizione ricevuta.

Esempio Python mittente
───────────────────────
    import socket, json
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = {
        "joints":      [0, 45, -30, 10, 0, 0, 90],
        "target_pos":  [0.4, 0.0, 0.5],
        "target_quat": [0.0, 0.0, 0.0, 1.0],
    }
    sock.sendto(json.dumps(payload).encode(), ("127.0.0.1", 5005))

Uso
───
    python udp_joint_viewer.py [--urdf PATH] [--host HOST] [--port PORT]

Default:
    --urdf  biarm_model/openarm_right.urdf
    --host  0.0.0.0
    --port  5005
"""

import argparse
import json
import math
import os
import socket
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data

# ─── Configurazione di default ───────────────────────────────────────────────

DEFAULT_URDF  = "biarm_model/openarm_right.urdf"
DEFAULT_HOST  = "0.0.0.0"
DEFAULT_PORT  = 5005

ARM_JOINTS    = list(range(2, 9))   # openarm_joint2 … openarm_joint8
EE_LINK_INDEX = 10

ROBOT_BASE_POS = [0.0, 0.0, 0.0]
ROBOT_BASE_ORI = [0.0, 0.0, 0.0, 1.0]

# Geometria visual target (speculare a run_mp.py)
CYL_COLOR    = [0.0, 0.5, 1.0, 0.35]   # blu semitrasparente
EAR_OFFSET   = 0.015                    # distanza orecchie dal centro lungo X

# ─── Stato condiviso tra thread ───────────────────────────────────────────────

_lock        = threading.Lock()
_latest: dict | None = None   # {"joints": [...], "target_pos": [...], "target_quat": [...]}
_running     = True

# ─── Thread UDP ───────────────────────────────────────────────────────────────

def udp_receiver(host: str, port: int, n_joints: int) -> None:
    global _latest, _running

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(0.5)

    print(f"[UDP] In ascolto su {host}:{port}  —  attendo dict JSON con {n_joints} giunti")

    while _running:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            pkt = json.loads(data.decode().strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[UDP] Pacchetto malformato da {addr}: {e}")
            continue

        if "joints" not in pkt:
            print(f"[UDP] Campo 'joints' mancante — pacchetto ignorato")
            continue

        joints = pkt["joints"]
        if len(joints) != n_joints:
            print(f"[UDP] Attesi {n_joints} valori in 'joints', ricevuti {len(joints)} — ignorato")
            continue
        
        #print(f"RICEVUTO: {joints}")

        with _lock:
            _latest = pkt

    sock.close()
    print("[UDP] Thread ricevitore chiuso.")


# ─── Debug visuals ────────────────────────────────────────────────────────────

def create_debug_visuals() -> tuple:
    """
    Crea il set di visual bodies speculare a ArmProcess._create_debug_visuals_set:
      cyl    – cilindro semitrasparente orientabile
      tip    – sfera viola al centro
      ear_l  – piccolo cilindro arancione (lato +X)
      ear_r  – piccolo cilindro verde    (lato -X)
    Restituisce (cyl, tip, ear_l, ear_r).
    """
    cyl = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.015, length=0.10,
            rgbaColor=CYL_COLOR,
            visualFramePosition=[0, 0, -0.05],
            visualFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        ),
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
    )
    tip = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.015,
            rgbaColor=[0.6, 0.1, 0.6, 0.6],
        ),
        basePosition=[0, 0, 0],
    )
    ear_l = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.006, length=0.01,
            rgbaColor=[1, 0.3, 0.0, 0.5],
            visualFramePosition=[0, 0, 0],
            visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]),
        ),
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
    )
    ear_r = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.006, length=0.01,
            rgbaColor=[0.2, 0.85, 0.1, 0.5],
            visualFramePosition=[0, 0, 0],
            visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi / 2, 0]),
        ),
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
    )
    return cyl, tip, ear_l, ear_r


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def update_debug_visuals(cyl, tip, ear_l, ear_r,
                         target_pos: list, target_quat: list) -> None:
    """Aggiorna posizione/orientamento di tutti e quattro i bodies."""
    p.resetBasePositionAndOrientation(cyl,   target_pos, target_quat)
    p.resetBasePositionAndOrientation(tip,   target_pos, [0, 0, 0, 1])

    R      = _quat_to_matrix(target_quat)
    x_axis = R[:, 0]
    el     = np.array(target_pos) + x_axis * EAR_OFFSET
    er     = np.array(target_pos) - x_axis * EAR_OFFSET
    p.resetBasePositionAndOrientation(ear_l, el.tolist(), target_quat)
    p.resetBasePositionAndOrientation(ear_r, er.tolist(), target_quat)


# ─── Setup PyBullet ───────────────────────────────────────────────────────────

def setup_simulation(urdf_path: str):
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.3],
    )

    p.loadURDF("plane.urdf")

    urdf_abs = os.path.abspath(urdf_path)
    if not os.path.exists(urdf_abs):
        raise FileNotFoundError(f"URDF non trovato: {urdf_abs}")

    robot = p.loadURDF(
        urdf_abs,
        basePosition=ROBOT_BASE_POS,
        baseOrientation=ROBOT_BASE_ORI,
        useFixedBase=True,
        globalScaling=1.0,
    )

    for i in range(-1, p.getNumJoints(robot)):
        p.setCollisionFilterGroupMask(robot, i, collisionFilterGroup=0, collisionFilterMask=0)

    print(f"\n=== Giunti attivi ({len(ARM_JOINTS)}) ===")
    for idx in ARM_JOINTS:
        info = p.getJointInfo(robot, idx)
        lo, hi = math.degrees(info[8]), math.degrees(info[9])
        print(f"  [{idx:2d}] {info[1].decode():<40}  range [{lo:.1f}°, {hi:.1f}°]")
    print("===\n")

    return robot


# ─── Applica angoli ───────────────────────────────────────────────────────────

def apply_joints(robot: int, angles_deg: list[float]) -> None:
    for joint_idx, angle_deg in zip(ARM_JOINTS, angles_deg):
        p.resetJointState(robot, joint_idx, math.radians(float(angle_deg)))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _running

    parser = argparse.ArgumentParser(description="PyBullet UDP joint + target viewer")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Percorso URDF")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Indirizzo bind UDP")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="Porta UDP")
    args = parser.parse_args()

    robot = setup_simulation(args.urdf)
    cyl, tip, ear_l, ear_r = create_debug_visuals()

    rx_thread = threading.Thread(
        target=udp_receiver,
        args=(args.host, args.port, len(ARM_JOINTS)),
        daemon=True,
    )
    rx_thread.start()

    print("Simulazione avviata. Premi Ctrl+C per uscire.\n")

    apply_joints(robot, [0.0] * len(ARM_JOINTS))

    try:
        while True:
            with _lock:
                pkt = dict(_latest) if _latest is not None else None

            if pkt is not None:
                apply_joints(robot, pkt["joints"])

                if "target_pos" in pkt and "target_quat" in pkt:
                    update_debug_visuals(
                        cyl, tip, ear_l, ear_r,
                        pkt["target_pos"],
                        pkt["target_quat"],
                    )

            p.stepSimulation()
            time.sleep(1 / 240)

    except KeyboardInterrupt:
        print("\nInterruzione richiesta.")
    finally:
        _running = False
        rx_thread.join(timeout=1.0)
        p.disconnect()
        print("Disconnesso da PyBullet.")


if __name__ == "__main__":
    main()