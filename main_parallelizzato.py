import pybullet as p
import pybullet_data
import math
import numpy as np
import os
import csv
import argparse
import multiprocessing as mp
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIGURAZIONE
# ─────────────────────────────────────────────
roboPos = [0, 0, 0]
THRESHOLD      = 0.005
APPROACH_OFFSET = [0, 0, 0]

MAPSIZE   = [0.4, 0.4, 0.2]
MAPSTEPS  = [200, 200, 100]
MAPOFFSET = [0, 0, 0.05]

XRANGE, XSAMPLES = math.pi/2, 45
YRANGE, YSAMPLES = math.pi/2, 1
ZRANGE, ZSAMPLES = math.pi*2, 36

NUM_WORKERS = max(1, mp.cpu_count() - 1)  # lascia un core libero al sistema

# ─────────────────────────────────────────────
#  FUNZIONI MATEMATICHE  (usate in ogni worker)
# ─────────────────────────────────────────────
def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )

def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),  2*(x*z+w*y)],
        [  2*(x*y+w*z),1-2*(x*x+z*z),  2*(y*z-w*x)],
        [  2*(x*z-w*y),  2*(y*z+w*x),1-2*(x*x+y*y)],
    ])

def axis_angle_to_quaternion(axis, angle):
    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle / 2)
    return (ax[0]*s, ax[1]*s, ax[2]*s, math.cos(angle / 2))

def local_axes_to_quaternion(q_ref, rx, ry, rz):
    R      = quaternion_to_matrix(q_ref)
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

# ─────────────────────────────────────────────
#  WORKER  — gira in un processo separato
# ─────────────────────────────────────────────
def worker_task(args):
    chunk_id, point_chunk, config = args

    import pybullet as p
    import pybullet_data
    import os, csv, math, numpy as np

    (THRESHOLD, APPROACH_OFFSET, REST_POSE, arm_joints,
     EE_LINK_INDEX, xSamples, ySamples, zSamples,
     roboPos, script_dir) = config

    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

    urdf_path = os.path.join(script_dir, "xarm_model/urdf/xarm_fixed.urdf")
    robo = p.loadURDF(urdf_path, basePosition=roboPos,
                      useFixedBase=True, physicsClientId=client)

    lower_limits, upper_limits, joint_ranges = [], [], []
    for j in arm_joints:
        info = p.getJointInfo(robo, j, physicsClientId=client)
        lower_limits.append(info[8])
        upper_limits.append(info[9])
        joint_ranges.append(info[9] - info[8])

    results = []
    processed_points = 0

    for (i_map, j_map, k_map) in point_chunk:

        target_position = list(np.array([i_map, j_map, k_map]))
        desired_pos = list(np.array(target_position) + np.array(APPROACH_OFFSET))

        # IK base
        for ji, jid in enumerate(arm_joints):
            p.resetJointState(robo, jid, REST_POSE[ji], physicsClientId=client)

        ik_ref = p.calculateInverseKinematics(
            robo, EE_LINK_INDEX, desired_pos,
            lowerLimits=lower_limits, upperLimits=upper_limits,
            jointRanges=joint_ranges, restPoses=REST_POSE,
            maxNumIterations=100,
            physicsClientId=client
        )

        for jid in arm_joints:
            p.resetJointState(robo, jid, ik_ref[jid], physicsClientId=client)

        ee_ref_orientation = p.getLinkState(
            robo, EE_LINK_INDEX, physicsClientId=client)[5]

        # rotazioni
        for sampleX in xSamples:
            for sampleY in ySamples:
                for sampleZ in zSamples:

                    q_target = local_axes_to_quaternion(
                        ee_ref_orientation, sampleX, sampleY, sampleZ)

                    for ji, jid in enumerate(arm_joints):
                        p.resetJointState(robo, jid, REST_POSE[ji],
                                          physicsClientId=client)

                    ik_angles = p.calculateInverseKinematics(
                        robo, EE_LINK_INDEX, desired_pos,
                        targetOrientation=q_target,
                        lowerLimits=lower_limits, upperLimits=upper_limits,
                        jointRanges=joint_ranges, restPoses=REST_POSE,
                        maxNumIterations=100,
                        physicsClientId=client
                    )

                    for jid in arm_joints:
                        p.resetJointState(robo, jid, ik_angles[jid],
                                          physicsClientId=client)

                    ee_pos = p.getLinkState(
                        robo, EE_LINK_INDEX, physicsClientId=client)[4]

                    ik_error = math.sqrt(
                        sum((ee_pos[i] - desired_pos[i])**2 for i in range(3)))

                    if ik_error < THRESHOLD:
                        results.append([
                            *desired_pos,
                            *q_target,
                            *[ik_angles[j] for j in arm_joints]
                        ])

        processed_points += 1

    p.disconnect(client)

    return processed_points, results


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    import multiprocessing as mp
    from tqdm import tqdm
    import os, csv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="Numero di processi (default: cpu_count - 1)")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    # --- setup identico al tuo ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    xSamples = make_samples(XSAMPLES, XRANGE/2)
    ySamples = make_samples(YSAMPLES, YRANGE/2)
    zSamples = make_samples(ZSAMPLES, ZRANGE/2)

    iSteps = make_samples(MAPSTEPS[0], MAPSIZE[0]/2)
    jSteps = make_samples(MAPSTEPS[1], MAPSIZE[1]/2)
    kSteps = make_samples(MAPSTEPS[2], MAPSIZE[2]/2)

    point_combinations = [
        (i + MAPOFFSET[0], j + MAPOFFSET[1], k + MAPOFFSET[2])
        for i in iSteps for j in jSteps for k in kSteps
    ]

    total_points = len(point_combinations)

    REST_POSE     = [0, 0.5, -0.5, 0.5, 0]
    arm_joints    = [0, 1, 2, 3, 4]
    EE_LINK_INDEX = 5

    config = (
        THRESHOLD, APPROACH_OFFSET, REST_POSE, arm_joints,
        EE_LINK_INDEX, xSamples, ySamples, zSamples,
        roboPos, script_dir
    )

    # --- chunk piccoli (CRUCIALE) ---
    chunk_size = 50   # 🔥 puoi sperimentare (20–200)
    chunks = [
        point_combinations[i:i+chunk_size]
        for i in range(0, total_points, chunk_size)
    ]

    if args.workers is not None:
        n_workers = args.workers
    else:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"{n_workers} worker | {len(chunks)} chunk | {total_points} punti")

    csv_path = os.path.join(script_dir, "dataset.csv")

    header = (
        ["target_x", "target_y", "target_z",
         "qx","qy","qz","qw"]
        + [f"joint_{i}" for i in range(len(arm_joints))]
    )

    total_saved = 0

    with mp.Pool(n_workers) as pool, \
         open(csv_path, "w", newline="") as f, \
         tqdm(total=total_points, desc="Punti", unit="pt") as pbar:

        writer = csv.writer(f)
        writer.writerow(header)

        args_iter = [
            (i, chunk, config)
            for i, chunk in enumerate(chunks)
        ]

        for processed_points, results in pool.imap_unordered(worker_task, args_iter):

            # aggiorna progress
            pbar.update(processed_points)

            # scrivi risultati
            for row in results:
                writer.writerow(row)

            total_saved += len(results)
            pbar.set_postfix({"saved": total_saved}, refresh=False)

    print(f"\nDataset salvato: {csv_path} ({total_saved} righe)")

# ─────────────────────────────────────────────
#  SINGLE-PROCESS (modalità GUI)
# ─────────────────────────────────────────────
def _run_single(config, point_combinations, script_dir, args):
    """Modalità GUI: singolo processo con debug visivi, identica al main originale."""
    (THRESHOLD, APPROACH_OFFSET, REST_POSE, arm_joints,
     EE_LINK_INDEX, xSamples, ySamples, zSamples,
     roboPos, script_dir) = config

    p.connect(p.GUI)
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    target = p.loadURDF("cube_small.urdf", basePosition=[1,0.75,0.1],
                        useFixedBase=True, globalScaling=0.1)
    p.changeVisualShape(target, -1, rgbaColor=[1,0,0,1])

    urdf_path = os.path.join(script_dir, "xarm_model/urdf/xarm_fixed.urdf")
    robo = p.loadURDF(urdf_path, basePosition=roboPos,
                      useFixedBase=True, globalScaling=1.0)

    lower_limits, upper_limits, joint_ranges = [], [], []
    for j in arm_joints:
        info = p.getJointInfo(robo, j)
        lower_limits.append(info[8])
        upper_limits.append(info[9])
        joint_ranges.append(info[9] - info[8])

    # Debug visivi
    Q_FIX = p.getQuaternionFromEuler([0, 0, 0])
    ear_offset = 0.015

    debug_cylinder = p.createMultiBody(0,
        p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.10,
            rgbaColor=[0,0.5,1,0.35],
            visualFramePosition=[0,0,-0.05],
            visualFrameOrientation=Q_FIX))
    debug_tip = p.createMultiBody(0,
        p.createVisualShape(p.GEOM_SPHERE, radius=0.015,
            rgbaColor=[0.6,0.1,0.6,0.6]))
    debug_ear_left = p.createMultiBody(0,
        p.createVisualShape(p.GEOM_CYLINDER, radius=0.006, length=0.01,
            rgbaColor=[1,0.3,0,0.5],
            visualFrameOrientation=p.getQuaternionFromEuler([0,math.pi/2,0])))
    debug_ear_right = p.createMultiBody(0,
        p.createVisualShape(p.GEOM_CYLINDER, radius=0.006, length=0.01,
            rgbaColor=[0.2,0.85,0.1,0.5],
            visualFrameOrientation=p.getQuaternionFromEuler([0,math.pi/2,0])))

    # Volume di campionamento
    half   = np.array(MAPSIZE) / 2
    center = np.array(MAPOFFSET)
    p.createMultiBody(0, p.createVisualShape(p.GEOM_BOX,
        halfExtents=half, rgbaColor=[1,0.5,0,0.08]),
        basePosition=center.tolist())
    corners = [center + np.array([sx,sy,sz])*half
               for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]]
    for a,b in [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]:
        p.addUserDebugLine(corners[a].tolist(), corners[b].tolist(),
                           [1,0.5,0], lineWidth=1.5, lifeTime=0)

    csv_path = os.path.join(script_dir, "dataset.csv")
    points_added = 0
    debug_line_id = None
    debug_text_id = None

    try:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["target_x","target_y","target_z",
                 "hand_quat_qx","hand_quat_qy","hand_quat_qz","hand_quat_qw"]
                + [f"joint_{i}" for i in range(len(arm_joints))]
            )

            for i_map, j_map, k_map in (pbar := tqdm(
                    point_combinations, desc="Punti", unit="pt",
                    bar_format="{l_bar}{bar:25}{r_bar}{postfix}")):

                target_position = list(np.array([i_map, j_map, k_map]))
                p.resetBasePositionAndOrientation(target, target_position, [0,0,0,1])
                desired_pos = list(np.array(target_position) + np.array(APPROACH_OFFSET))

                for ji, jid in enumerate(arm_joints):
                    p.resetJointState(robo, jid, REST_POSE[ji])
                ik_ref = p.calculateInverseKinematics(
                    robo, EE_LINK_INDEX, desired_pos,
                    lowerLimits=lower_limits, upperLimits=upper_limits,
                    jointRanges=joint_ranges, restPoses=REST_POSE,
                    maxNumIterations=200, residualThreshold=1e-5)
                for jid in arm_joints:
                    p.resetJointState(robo, jid, ik_ref[jid])
                p.stepSimulation()

                ee_ref_orientation = p.getLinkState(robo, EE_LINK_INDEX)[5]

                for sampleX in xSamples:
                    for sampleY in ySamples:
                        for sampleZ in zSamples:
                            q_target = local_axes_to_quaternion(
                                ee_ref_orientation, sampleX, sampleY, sampleZ)

                            # Debug visivi
                            p.resetBasePositionAndOrientation(
                                debug_cylinder, desired_pos, q_target)
                            p.resetBasePositionAndOrientation(
                                debug_tip, desired_pos, [0,0,0,1])
                            R      = quaternion_to_matrix(q_target)
                            x_axis = R[:, 0]
                            mid    = np.array(desired_pos)
                            p.resetBasePositionAndOrientation(
                                debug_ear_left,
                                (mid + x_axis*ear_offset).tolist(), q_target)
                            p.resetBasePositionAndOrientation(
                                debug_ear_right,
                                (mid - x_axis*ear_offset).tolist(), q_target)

                            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                            for ji, jid in enumerate(arm_joints):
                                p.resetJointState(robo, jid, REST_POSE[ji])
                            ik_angles = p.calculateInverseKinematics(
                                robo, EE_LINK_INDEX, desired_pos,
                                targetOrientation=q_target,
                                lowerLimits=lower_limits, upperLimits=upper_limits,
                                jointRanges=joint_ranges, restPoses=REST_POSE,
                                maxNumIterations=200, residualThreshold=1e-5)
                            for jid in arm_joints:
                                p.resetJointState(robo, jid, ik_angles[jid])
                            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                            p.stepSimulation()

                            ee_pos = p.getLinkState(robo, EE_LINK_INDEX)[4]
                            ik_error = math.sqrt(
                                sum((ee_pos[i]-desired_pos[i])**2 for i in range(3)))
                            dist_to_cube = math.sqrt(
                                sum((ee_pos[i]-target_position[i])**2 for i in range(3)))

                            color = [0,1,0] if ik_error < THRESHOLD else [1,0,0]
                            mid_pos = [(ee_pos[i]+target_position[i])/2 for i in range(3)]
                            mid_pos[2] += 0.05
                            label = f"{dist_to_cube*100:.1f} cm"

                            if all(math.isfinite(v) for v in list(ee_pos)+target_position+mid_pos):
                                if debug_line_id is None:
                                    debug_line_id = p.addUserDebugLine(
                                        ee_pos, target_position, color, lineWidth=2, lifeTime=0)
                                    debug_text_id = p.addUserDebugText(
                                        label, mid_pos, color, textSize=1.2, lifeTime=0)
                                else:
                                    debug_line_id = p.addUserDebugLine(
                                        ee_pos, target_position, color, lineWidth=2, lifeTime=0,
                                        replaceItemUniqueId=debug_line_id)
                                    debug_text_id = p.addUserDebugText(
                                        label, mid_pos, color, textSize=1.2, lifeTime=0,
                                        replaceItemUniqueId=debug_text_id)

                            if ik_error < THRESHOLD:
                                writer.writerow([
                                    *desired_pos, *q_target,
                                    *[ik_angles[j] for j in arm_joints]
                                ])
                                points_added += 1
                                pbar.set_postfix({"saved": points_added}, refresh=False)

    except KeyboardInterrupt:
        print("Interrotto.")
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        print(f"Dataset salvato: {csv_path}  ({points_added} righe)")
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
