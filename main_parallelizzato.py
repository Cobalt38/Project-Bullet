import signal

import pybullet as p
import pybullet_data
import math
import numpy as np
import os
import csv
import argparse
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURAZIONE ROBOT
# ─────────────────────────────────────────────
xarmConfig = {
    "rest_pose": [0, 0.5, -0.5, 0.5, 0],
    "arm_joints": [0, 1, 2, 3, 4],
    "ee_link_index": 5,
    "isLocalpath": True,
    "urdf_path": "xarm_model/urdf/xarm_fixed.urdf"
}

pandaConfig = {
    "rest_pose": [0, -0.215, 0, -2.57, 0, 2.356, 2.356],
    "arm_joints": [0, 1, 2, 3, 4, 5, 6],
    "ee_link_index": 11,
    "isLocalpath": False,
    "urdf_path": "franka_panda/panda.urdf"
}

biarmConfig = {
    "rest_pose": [0, 0, 0, 1.2, 0, 0, 0],
    "arm_joints": [0, 1, 2, 3, 4, 5, 6],  # lascia vuoto per dedurlo automaticamente dal robot
    "ee_link_index": 8,
    "isLocalpath": True,
    "urdf_path": "biarm_model/openarm.urdf"
}

openarmRightConfig = {
    "rest_pose": [0, 0, 0, 1.2, 0, 0, 0],
    "arm_joints": [i for i in range(2, 9)],  # 2..8, lascia vuoto per dedurlo automaticamente dal robot
    "ee_link_index": 10,  # (TCP)
    "isLocalpath": True,
    "urdf_path": "biarm_model/openarm_right.urdf",
    "max_reach": 0.8  # stima della lunghezza massima del braccio, usata per filtrare i target troppo lontani
}

openarmv2rightConfig = {
    "rest_pose": [0, 0, 0, 0, 0, 0, 0],
    "arm_joints": [i for i in range(2, 9)],  # 2..8, lascia vuoto per dedurlo automaticamente dal robot
    "ee_link_index": 8,
    "isLocalpath": True,
    "urdf_path": "biarm_model/openarm_v2_right.urdf"
}

ROBOT_CONFIG = openarmRightConfig

# ─────────────────────────────────────────────
#  PARAMETRI CAMPIONAMENTO
# ─────────────────────────────────────────────
roboPos         = [0, 0, 0]
THRESHOLD       = 0.01  # max errore IK accettabile (in metri) — se troppo basso, rischia di non trovare soluzioni per target lontani; se troppo alto, rischia di accettare soluzioni con errori visibili
APPROACH_OFFSET = [0, 0, 0]

MAPSIZE = [0.6, 0.5, 1.0]      # dimensione del cubo di test (in metri)
MAPSTEPS = [15, 15, 25]           # quanti step di offset testare lungo ogni asse 
MAPOFFSET = [0, -0.3, 0.6]     # offset del centro del cubo rispetto alla base globale (in metri) 

XRANGE, XSAMPLES = math.pi,   45
YRANGE, YSAMPLES = math.pi,   45
ZRANGE, ZSAMPLES = math.pi*2, 36

# ─────────────────────────────────────────────
#  FUNZIONI MATEMATICHE  (usate in ogni worker)
# ─────────────────────────────────────────────
def quaternion_multiply(q1, q2):
    """q1 * q2 — composizione di quaternioni (x, y, z, w)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )

def quaternion_to_matrix(q):
    """Converte quaternione (x,y,z,w) in matrice di rotazione 3x3"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def axis_angle_to_quaternion(axis, angle):
    """Quaternione da asse (vettore 3D normalizzato) + angolo in radianti"""
    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle / 2)
    return (ax[0]*s, ax[1]*s, ax[2]*s, math.cos(angle / 2))

def local_axes_to_quaternion(q_ref, rx, ry, rz):
    """
    Offset combinato sui tre assi locali, indipendenti tra loro.
    Porta gli assi locali nel frame world, costruisce lì i quaternioni
    di rotazione, poi ricompone.
    """
    R = quaternion_to_matrix(q_ref)
    x_local = R[:, 0]
    y_local = R[:, 1]
    z_local = R[:, 2]
    q_rx = axis_angle_to_quaternion(x_local, rx)
    q_ry = axis_angle_to_quaternion(y_local, ry)
    q_rz = axis_angle_to_quaternion(z_local, rz)
    return quaternion_multiply(q_rx,
           quaternion_multiply(q_ry,
           quaternion_multiply(q_rz, q_ref)))

def make_samples(n, half_range):
    if n == 1:
        return [0.0]
    return list(np.linspace(-half_range, half_range, n))

def get_model_joints(robot_id, client):
    """Ritorna i joint revolute del robot"""
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)
    print(f'#Joints: {num_joints}')
    controllable_joints = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i, physicsClientId=client)
        if joint_info[2] == 0:  # revolute
            controllable_joints.append(i)
    return controllable_joints

def estimate_max_reach(robot_id, ee_link_index, arm_joints):
    """Stima la lunghezza massima del braccio usando le combinazioni estreme dei limiti di giunto."""
    base_pos = p.getLinkState(robot_id, arm_joints[0])[4]  # posizione del link base (assumendo che sia il primo)
    for j in arm_joints:
        p.resetJointState(robot_id, j, 0) # max estensione possibile stimata
    ee_pos = p.getLinkState(robot_id, ee_link_index)[4]
    return math.dist(base_pos, ee_pos) * 1.5  # moltiplica per 1.5 per avere un margine di sicurezza (da tarare in base al robot)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
def init_log(log_path, n_workers, n_chunks, total_points, arm_joints, ee_link_index, rest_pose):
    """Apre (o crea) il file di log e scrive l'header della run corrente."""
    file_exists = os.path.exists(log_path)
    log = open(log_path, "a", buffering=1)  # line-buffered: ogni riga va subito su disco
    if file_exists:
        log.write("\n")  # riga vuota di separazione dalla run precedente
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.write(f"{'='*60}\n")
    log.write(f"  RUN  {ts}\n")
    log.write(f"{'='*60}\n")
    log.write(f"  Workers     : {n_workers}\n")
    log.write(f"  Chunks      : {n_chunks}\n")
    log.write(f"  Punti totali: {total_points}\n")
    log.write(f"  ARM_JOINTS  : {arm_joints}\n")
    log.write(f"  EE_LINK_IDX : {ee_link_index}\n")
    log.write(f"  REST_POSE   : {rest_pose}\n")
    log.write(f"  MAPSIZE     : {MAPSIZE}  MAPSTEPS: {MAPSTEPS}  MAPOFFSET: {MAPOFFSET}\n")
    log.write(f"  Rotazioni   : X={XSAMPLES} Y={YSAMPLES} Z={ZSAMPLES}\n")
    log.write(f"  THRESHOLD   : {THRESHOLD}\n")
    log.write(f"{'-'*60}\n")
    return log

def log_worker(log, chunk_id, processed_points, n_results):
    ts = datetime.now().strftime("%H:%M:%S")
    log.write(f"  [{ts}] Worker chunk {chunk_id:>4} | punti: {processed_points:>5} | soluzioni: {n_results:>7}\n")

def log_summary(log, total_saved, total_points, elapsed_sec):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h, rem = divmod(int(elapsed_sec), 3600)
    m, s   = divmod(rem, 60)
    log.write(f"{'-'*60}\n")
    log.write(f"  Fine run     : {ts}\n")
    log.write(f"  Durata       : {h:02d}:{m:02d}:{s:02d}\n")
    log.write(f"  Righe salvate: {total_saved} / {total_points} punti processati\n")
    log.write(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  WORKER  — gira in un processo separato
# ─────────────────────────────────────────────
def worker_task(args):
    chunk_id, point_chunk, config = args

    import pybullet as p
    import pybullet_data
    import os, math
    import numpy as np

    (THRESHOLD, APPROACH_OFFSET, REST_POSE, arm_joints,
     EE_LINK_INDEX, xSamples, ySamples, zSamples,
     roboPos, script_dir, robot_config) = config

    client = p.connect(p.DIRECT)
    # for(i_map, j_map, k_map) in point_chunk:
    #     pos = (i_map, j_map, k_map)
    #     p.addUserDebugPoints(
    #                     pointPositions=[pos],
    #                     pointColorsRGB=[[1, 0.5, 0]],   # arancione
    #                     pointSize=3,
    #                     lifeTime=0
    #                 )

    p.setGravity(0, 0, 0, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

    if robot_config["isLocalpath"]:
        urdf_path = os.path.join(script_dir, robot_config["urdf_path"])
    else:
        urdf_path = robot_config["urdf_path"]

    robo = p.loadURDF(urdf_path, basePosition=roboPos,
                      useFixedBase=True, physicsClientId=client)

    # FIX CONFIGURAZIONE ROBOT (se necessario)
    actual_arm_joints = arm_joints
    actual_rest_pose  = REST_POSE
    actual_ee_index   = EE_LINK_INDEX

    if actual_arm_joints is None or actual_arm_joints == []:
        num_joints = p.getNumJoints(robo, physicsClientId=client)
        actual_arm_joints = [
            i for i in range(num_joints)
            if p.getJointInfo(robo, i, physicsClientId=client)[2] == 0
        ]
    if actual_rest_pose is None or len(actual_rest_pose) != len(actual_arm_joints):
        actual_rest_pose = [0] * len(actual_arm_joints)
    if actual_ee_index is None:
        actual_ee_index = actual_arm_joints[-1]

    # Disabilita collisioni robot (evita incastri durante il campionamento IK)
    for i in range(-1, p.getNumJoints(robo, physicsClientId=client)):
        p.setCollisionFilterGroupMask(robo, i,
                                      collisionFilterGroup=0,
                                      collisionFilterMask=0,
                                      physicsClientId=client)

    # Limiti IK estratti dal robot
    lower_limits, upper_limits, joint_ranges = [], [], []
    for j in actual_arm_joints:
        info = p.getJointInfo(robo, j, physicsClientId=client)
        lower_limits.append(info[8])
        upper_limits.append(info[9])
        joint_ranges.append(info[9] - info[8])

    results = []
    processed_points = 0

    max_reach = robot_config.get("max_reach", estimate_max_reach(robo, actual_ee_index, actual_arm_joints))

    for (i_map, j_map, k_map) in point_chunk:

        target_position = [i_map, j_map, k_map]
        desired_pos = list(np.array(target_position) + np.array(APPROACH_OFFSET))

        if math.dist(p.getLinkState(robo, actual_arm_joints[0], physicsClientId=client)[4], desired_pos) > max_reach*1.05:
            continue

        # Reset alla rest pose — seed consistente per l'IK
        for ji, jid in enumerate(actual_arm_joints):
            p.resetJointState(robo, jid, actual_rest_pose[ji],
                              physicsClientId=client)

        # IK senza orientamento — trova la posa di riferimento naturale
        ik_angles = p.calculateInverseKinematics(
            robo, actual_ee_index, desired_pos,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=actual_rest_pose,
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=client
        )

        for ji, jid in enumerate(actual_arm_joints):
            p.resetJointState(robo, jid, ik_angles[ji], physicsClientId=client)

        p.stepSimulation(physicsClientId=client)

        ee_ref_orientation = p.getLinkState(
            robo, actual_ee_index, physicsClientId=client)[5]

        # Campionamento rotazioni
        for sampleX in xSamples:
            for sampleY in ySamples:
                for sampleZ in zSamples:

                    q_target = local_axes_to_quaternion(
                        ee_ref_orientation, sampleX, sampleY, sampleZ)

                    # Reset alla rest pose prima di ogni IK con orientamento
                    # for ji, jid in enumerate(actual_arm_joints):
                    #     p.resetJointState(robo, jid, actual_rest_pose[ji],
                    #                       physicsClientId=client)

                    ik_angles = p.calculateInverseKinematics(
                        robo, actual_ee_index, desired_pos,
                        targetOrientation=q_target,
                        lowerLimits=lower_limits,
                        upperLimits=upper_limits,
                        jointRanges=joint_ranges,
                        restPoses=actual_rest_pose,
                        maxNumIterations=200,
                        residualThreshold=1e-5,
                        physicsClientId=client
                    )

                    for ji, jid in enumerate(actual_arm_joints):
                        p.resetJointState(robo, jid, ik_angles[ji],
                                          physicsClientId=client)

                    p.stepSimulation(physicsClientId=client)

                    # Verifica IK
                    ee_pos = p.getLinkState(robo, actual_ee_index,
                                            physicsClientId=client)[4]

                    ik_error = math.sqrt(
                        sum((ee_pos[i] - desired_pos[i])**2 for i in range(3)))

                    if ik_error < THRESHOLD:
                        results.append([
                            *desired_pos,                               # x, y, z target
                            *q_target,                                  # qx, qy, qz, qw
                            *[ik_angles[ji] for ji in range(len(actual_arm_joints))]  # j0..jN
                        ])

        processed_points += 1

    p.disconnect(client)
    return chunk_id, processed_points, results


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="Numero di processi (default: cpu_count - 1)")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Risolvi ARM_JOINTS e REST_POSE: se vuoti, carica temporaneamente il robot
    # nel processo principale per dedurli, poi passa i valori risolti ai worker
    arm_joints    = ROBOT_CONFIG["arm_joints"]
    rest_pose     = ROBOT_CONFIG["rest_pose"]
    ee_link_index = ROBOT_CONFIG["ee_link_index"] or None
    

    if arm_joints == [] or rest_pose == [] or \
       len(rest_pose) != len(arm_joints) or ee_link_index is None:

        print("Deduzione ARM_JOINTS / REST_POSE dal robot...")
        client_tmp = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=client_tmp)
        if ROBOT_CONFIG["isLocalpath"]:
            urdf_tmp = os.path.join(script_dir, ROBOT_CONFIG["urdf_path"])
        else:
            urdf_tmp = ROBOT_CONFIG["urdf_path"]

        robo_tmp = p.loadURDF(urdf_tmp, useFixedBase=True,
                              physicsClientId=client_tmp)

        if arm_joints == []:
            arm_joints = get_model_joints(robo_tmp, client_tmp)
            print(f"  Arm joints dedotti: {arm_joints}")

        if rest_pose == [] or len(rest_pose) != len(arm_joints):
            rest_pose = [0] * len(arm_joints)
            print(f"  REST_POSE impostata a zero per {len(arm_joints)} giunti.")

        if ee_link_index is None:
            ee_link_index = arm_joints[-1]
            print(f"  EE_LINK_INDEX impostato a {ee_link_index}")

        print("=== ALL JOINTS ===")
        for i in range(p.getNumJoints(robo_tmp, physicsClientId=client_tmp)):
            info = p.getJointInfo(robo_tmp, i, physicsClientId=client_tmp)
            jname = info[1].decode()
            jtype = info[2]
            lname = info[12].decode()
            type_str = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(jtype, str(jtype))
            print(f"  [{i}] {jname} ({type_str}) → link: {lname}")

        p.disconnect(client_tmp)

    # Campionamento
    xSamples = make_samples(XSAMPLES, XRANGE/2)
    ySamples = make_samples(YSAMPLES, YRANGE/2)
    zSamples = make_samples(ZSAMPLES, ZRANGE/2)

    iSteps = make_samples(MAPSTEPS[0], MAPSIZE[0]/2)
    jSteps = make_samples(MAPSTEPS[1], MAPSIZE[1]/2)
    kSteps = make_samples(MAPSTEPS[2], MAPSIZE[2]/2)

    # MAPOFFSET applicato qui — i worker ricevono già le coordinate assolute
    point_combinations = [
        (i + MAPOFFSET[0], j + MAPOFFSET[1], k + MAPOFFSET[2])
        for i in iSteps for j in jSteps for k in kSteps
    ]
    total_points = len(point_combinations)

    config = (
        THRESHOLD, APPROACH_OFFSET, rest_pose, arm_joints,
        ee_link_index, xSamples, ySamples, zSamples,
        roboPos, script_dir, ROBOT_CONFIG
    )

    chunk_size = 100
    chunks = [
        point_combinations[i:i+chunk_size]
        for i in range(0, total_points, chunk_size)
    ]

    n_workers = args.workers if args.workers is not None \
                else max(1, mp.cpu_count() - 1)

    print(f"{n_workers} worker | {len(chunks)} chunk | {total_points} punti")

    csv_path = os.path.join(script_dir, "dataset.csv")
    log_path = os.path.join(script_dir, "capture.log")

    header_row = [
        "target_x", "target_y", "target_z",
        "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"
    ] + [f"joint_{i}" for i in range(len(arm_joints))]

    total_saved = 0
    start_time  = datetime.now()

    args_iter = [(i, chunk, config) for i, chunk in enumerate(chunks)]

    log = init_log(log_path, n_workers, len(chunks), total_points,
                   arm_joints, ee_link_index, rest_pose)

    def setup_signal_handlers(pool):
        pgid = os.getpgid(os.getpid())  # process group id del padre

        def handle_termination(signum, frame):
            print(f"\nSegnale {signum} ricevuto, termino tutti i processi...")
            pool.terminate()
            pool.join()
            # scrivi sul log prima di uscire
            log_summary(log, total_saved, total_points,
                        (datetime.now() - start_time).total_seconds())
            log.write("  *** Terminato da segnale esterno ***\n")
            log.close()
            os.killpg(pgid, signal.SIGKILL)

        signal.signal(signal.SIGTERM, handle_termination)
        signal.signal(signal.SIGINT,  handle_termination)

    try:
        with mp.Pool(n_workers) as pool, \
             open(csv_path, "w", newline="") as csv_file, \
             tqdm(total=total_points, desc="Punti", unit="pt",
                  bar_format="{l_bar}{bar:25}{r_bar}{postfix}") as pbar:
            
            setup_signal_handlers(pool)

            writer = csv.writer(csv_file)
            writer.writerow(header_row)
            print(len(args_iter))
            for chunk_id, processed_points, results in pool.imap_unordered(worker_task, args_iter):
                pbar.update(processed_points)

                for row in results:
                    writer.writerow(row)

                total_saved += len(results)
                pbar.set_postfix({"saved": total_saved}, refresh=False)

                log_worker(log, chunk_id, processed_points, len(results))

                # Flush CSV periodico ogni ~1000 righe salvate
                if total_saved % 1000 < len(results):
                    csv_file.flush()

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
        log.write("  *** Interrotto dall'utente ***\n")
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()
        log.write(f"  *** ERRORE: {e} ***\n")
    finally:
        elapsed = (datetime.now() - start_time).total_seconds()
        log_summary(log, total_saved, total_points, elapsed)
        log.close()
        print(f"\nDataset salvato: {csv_path} ({total_saved} righe)")
        print(f"Log salvato    : {log_path}")


if __name__ == "__main__":
    main()