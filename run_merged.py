import pybullet as p
import pybullet_data
import math
import numpy as np
import os
from ik_inference import load_model_and_metadata, run_inference, decode_output, build_input

openarmRightConfig = {
    "rest_pose": [0, 0, 0, 1.2, 0, 0, 0],
    "arm_joints": [2, 3, 4, 5, 6, 7, 8],  # openarm_right_joint1..7
    "ee_link_index": 10,                   # openarm_right_hand_tcp
    "urdf_path": "biarm_model/openarm_right.urdf",
    "robopos": [0, 0, 0],
}

openarmLeftConfig = {
    "rest_pose": [0, 0, 0, 1.2, 0, 0, 0],
    "arm_joints": [2, 3, 4, 5, 6, 7, 8],  # openarm_left_joint1..7
    "ee_link_index": 10,                   # openarm_left_hand_tcp
    "urdf_path": "biarm_model/openarm_left.urdf",
    "robopos": [0, 0, 0],
}

USE_LEFT = True  # False per solo braccio destro

CONFIG_R = openarmRightConfig
CONFIG_L = openarmLeftConfig if USE_LEFT else None

MAPSIZE     = [2.0, 2.0, 2.0] # dimensione del "cubo" di test (in metri)
MAPOFFSET   = [0, -0.9, 1]    # centro workspace braccio destro
MAPOFFSET_L = [0,  0.9, 1]    # centro workspace braccio sinistro (Y specchiata)

ARM_JOINTS_R    = CONFIG_R.get("arm_joints")
EE_LINK_INDEX_R = CONFIG_R.get("ee_link_index")
REST_POSE_R     = CONFIG_R.get("rest_pose")
URDF_PATH_R     = CONFIG_R.get("urdf_path")
ROBOPOS_R       = CONFIG_R.get("robopos")

ARM_JOINTS_L    = CONFIG_L.get("arm_joints")    if CONFIG_L else None
EE_LINK_INDEX_L = CONFIG_L.get("ee_link_index") if CONFIG_L else None
REST_POSE_L     = CONFIG_L.get("rest_pose")     if CONFIG_L else None
URDF_PATH_L     = CONFIG_L.get("urdf_path")     if CONFIG_L else None
ROBOPOS_L       = CONFIG_L.get("robopos")       if CONFIG_L else None

# ---------------------------------------------------------------------------
# Definizione parametri GUI: (nome_display, min, max, default)
# ---------------------------------------------------------------------------
PARAM_DEFS = [
    ("⚡ Predict (all)",   1,                            0,                           1              ),
    ("Euler X - dx",      -math.pi,                     math.pi,                     0              ),
    ("Euler Y - dx",      -math.pi,                     math.pi,                     0              ),
    ("Euler Z - dx",      -math.pi,                     math.pi,                     0              ),
    ("Position X - dx",   -MAPSIZE[0]/2 + MAPOFFSET[0], MAPSIZE[0]/2 + MAPOFFSET[0], MAPOFFSET[0]  ),
    ("Position Y - dx",   -MAPSIZE[1]/2 + MAPOFFSET[1], MAPSIZE[1]/2 + MAPOFFSET[1], MAPOFFSET[1]  ),
    ("Position Z - dx",   -MAPSIZE[2]/2 + MAPOFFSET[2], MAPSIZE[2]/2 + MAPOFFSET[2], MAPOFFSET[2]  ),
]
PARAM_KEYS = ["predict_btn", "eulerX_r", "eulerY_r", "eulerZ_r", "posX_r", "posY_r", "posZ_r"]

if CONFIG_L:
    PARAM_DEFS.extend([
        ("Euler X - sx",    -math.pi,                      math.pi,                      0               ),
        ("Euler Y - sx",    -math.pi,                      math.pi,                      0               ),
        ("Euler Z - sx",    -math.pi,                      math.pi,                      0               ),
        ("Position X - sx", -MAPSIZE[0]/2 + MAPOFFSET_L[0], MAPSIZE[0]/2 + MAPOFFSET_L[0], MAPOFFSET_L[0] ),
        ("Position Y - sx", -MAPSIZE[1]/2 + MAPOFFSET_L[1], MAPSIZE[1]/2 + MAPOFFSET_L[1], MAPOFFSET_L[1] ),
        ("Position Z - sx", -MAPSIZE[2]/2 + MAPOFFSET_L[2], MAPSIZE[2]/2 + MAPOFFSET_L[2], MAPOFFSET_L[2] ),
    ])
    PARAM_KEYS.extend(["eulerX_l", "eulerY_l", "eulerZ_l", "posX_l", "posY_l", "posZ_l"])
# ---------------------------------------------------------------------------
# Stato globale
# ---------------------------------------------------------------------------
_model = None
_meta  = None

_target_r = None
_target_l = None

_robot_r = None
_robot_l = None

_debug_cylinder_r  = None
_debug_tip_r       = None
_debug_ear_left_r  = None
_debug_ear_right_r = None

_debug_cylinder_l  = None
_debug_tip_l       = None
_debug_ear_left_l  = None
_debug_ear_right_l = None

_ear_offset = 0.015

handles   = None
last_vals = None

# ---------------------------------------------------------------------------
# Inferenza
# ---------------------------------------------------------------------------

def load_model(model_path, model_dir):
    global _model, _meta
    _model, _meta = load_model_and_metadata(model_path=model_path, model_dir=model_dir)
    print(f"[MODEL] Caricato. Output joints: {_meta.get('output_columns', [])}")

def inference(x, y, z, qx, qy, qz, qw):
    """Ritorna la lista di angoli in radianti (uno per giunto)."""
    global _model, _meta
    raw    = run_inference(_model, build_input(x, y, z, qx, qy, qz, qw))
    result = decode_output(raw, _meta.get("output_columns", []))
    print(result)
    return result["angles_rad"]

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def left_to_right_mirroring(pos_left, quat_left):
    """
        Converte input del braccio sinistro in input equivalenti per il modello
        addestrato sul braccio destro. Piano di simmetria: XZ (Y invertita).
    """

    pos_right = pos_left * np.array([1.0, -1.0, 1.0])
    quat_right = quat_left * np.array([1.0, -1.0, 1.0, 1.0])  # nega qy (xyzw)
    print(f"POS_LEFT: {pos_left}, POS_RIGHT: {pos_right}")

    return pos_right, quat_right

def diagnose_robot():
    for label, robot in [("RIGHT", _robot_r), ("LEFT", _robot_l) if CONFIG_L else (None, None)]:
        if robot is None:
            continue
        print(f"\n=== JOINT MAP {label} ({p.getNumJoints(robot)} joints totali) ===")
        for i in range(p.getNumJoints(robot)):
            info  = p.getJointInfo(robot, i)
            jtype = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(info[2], str(info[2]))
            print(f"  [{i:2d}] {jtype:<10} name={info[1].decode():<40} parent={info[12].decode()}")
        print("===\n")

def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def setup_debug_params() -> dict:
    handles = {}
    for key, (label, lo, hi, default) in zip(PARAM_KEYS, PARAM_DEFS):
        handles[key] = p.addUserDebugParameter(label, lo, hi, default)
    return handles

def read_debug_params(handles: dict) -> dict:
    return {key: p.readUserDebugParameter(h) for key, h in handles.items()}

def disable_self_collisions(robot_id: int):
    """Disabilita tutte le collisioni interne al robot."""
    joint_count = p.getNumJoints(robot_id)
    link_indices = [-1] + list(range(joint_count))
    for i in link_indices:
        for j in link_indices:
            if i == j:
                continue
            p.setCollisionFilterPair(robot_id, robot_id, i, j, enableCollision=0)

def disable_collision_between(robot_a: int, robot_b: int):
    """Disabilita tutte le collisioni tra due robot."""
    joint_count_a = p.getNumJoints(robot_a)
    joint_count_b = p.getNumJoints(robot_b)
    link_indices_a = [-1] + list(range(joint_count_a))
    link_indices_b = [-1] + list(range(joint_count_b))
    for i in link_indices_a:
        for j in link_indices_b:
            p.setCollisionFilterPair(robot_a, robot_b, i, j, enableCollision=0)

# ---------------------------------------------------------------------------
# Debug visivo
# ---------------------------------------------------------------------------

def _create_debug_visuals(cylinder_color):
    """Crea e ritorna (cylinder, tip, ear_l, ear_r) per un braccio."""
    cyl = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.015, length=0.10,
            rgbaColor=cylinder_color,
            visualFramePosition=[0, 0, -0.05],
            visualFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
        ),
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1]
    )
    tip = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.015,
            rgbaColor=[0.6, 0.1, 0.6, 0.6]
        ),
        basePosition=[0, 0, 0]
    )
    ear_l = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.006, length=0.01,
            rgbaColor=[1, 0.3, 0.0, 0.5],
            visualFramePosition=[0, 0, 0],
            visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])
        ),
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1]
    )
    ear_r = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.006, length=0.01,
            rgbaColor=[0.2, 0.85, 0.1, 0.5],
            visualFramePosition=[0, 0, 0],
            visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])
        ),
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1]
    )
    return cyl, tip, ear_l, ear_r


def createTargetDebug():
    global _debug_cylinder_r, _debug_tip_r, _debug_ear_left_r, _debug_ear_right_r
    global _debug_cylinder_l, _debug_tip_l, _debug_ear_left_l, _debug_ear_right_l

    # Braccio destro: cilindro blu
    _debug_cylinder_r, _debug_tip_r, _debug_ear_left_r, _debug_ear_right_r = _create_debug_visuals([0, 0.5, 1, 0.35])
    # Braccio sinistro: cilindro arancione (solo se bimanuale)
    if CONFIG_L:
        _debug_cylinder_l, _debug_tip_l, _debug_ear_left_l, _debug_ear_right_l = _create_debug_visuals([1, 0.5, 0, 0.35])
    p.stepSimulation()

def _update_debug_visuals(cyl, tip, ear_l, ear_r, target_pos, target_ori):
    p.resetBasePositionAndOrientation(cyl, target_pos, target_ori)
    p.resetBasePositionAndOrientation(tip, target_pos, [0, 0, 0, 1])
    R      = quaternion_to_matrix(target_ori)
    x_axis = R[:, 0]
    el     = np.array(target_pos) + x_axis * _ear_offset
    er     = np.array(target_pos) - x_axis * _ear_offset
    p.resetBasePositionAndOrientation(ear_l, el.tolist(), target_ori)
    p.resetBasePositionAndOrientation(ear_r, er.tolist(), target_ori)


def updateTargetDebug_r(target_pos, target_ori):
    global _debug_cylinder_r, _debug_tip_r, _debug_ear_left_r, _debug_ear_right_r
    _update_debug_visuals(
        _debug_cylinder_r, _debug_tip_r, _debug_ear_left_r, _debug_ear_right_r,
        target_pos, target_ori
    )

def updateTargetDebug_l(target_pos, target_ori):
    _update_debug_visuals(
        _debug_cylinder_l, _debug_tip_l, _debug_ear_left_l, _debug_ear_right_l,
        target_pos, target_ori
    )

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup(model_path, model_dir):
    print("Setting up simulation...")
    global handles, last_vals, _target_r, _target_l, _robot_r, _robot_l
    script_dir = os.path.dirname(os.path.abspath(__file__))

    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # p.setAdditionalSearchPath(script_dir)
    p.setGravity(0, 0, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.3]
    )

    handles   = setup_debug_params()
    last_vals = read_debug_params(handles)
    createTargetDebug()

    p.loadURDF("plane.urdf")

    # Target visivi: rosso per destro, blu per sinistro
    _target_r = p.loadURDF("cube_small.urdf",
                            basePosition=[1, -0.75, 0.1],
                            useFixedBase=True, globalScaling=0.3)
    p.changeVisualShape(objectUniqueId=_target_r, linkIndex=-1, rgbaColor=[1, 0, 0, 1])

    if CONFIG_L:
        _target_l = p.loadURDF("cube_small.urdf",
                                basePosition=[1, 0.75, 0.1],
                                useFixedBase=True, globalScaling=0.3)
        p.changeVisualShape(objectUniqueId=_target_l, linkIndex=-1, rgbaColor=[0, 0.4, 1, 1])

    urdf_path_r = os.path.join(script_dir, URDF_PATH_R)
    print(f"Loading RIGHT robot from URDF: {urdf_path_r}")
    _robot_r = p.loadURDF(urdf_path_r, basePosition=ROBOPOS_R, useFixedBase=True, globalScaling=1.0)
    disable_self_collisions(_robot_r)

    if CONFIG_L:
        urdf_path_l = os.path.join(script_dir, URDF_PATH_L)
        print(f"Loading LEFT robot from URDF: {urdf_path_l}")
        _robot_l = p.loadURDF(urdf_path_l, basePosition=ROBOPOS_L, useFixedBase=True, globalScaling=1.0)
        disable_self_collisions(_robot_l)
        disable_collision_between(_robot_r, _robot_l)

    load_model(model_path, model_dir)

# ---------------------------------------------------------------------------
# IK test + visuals
# ---------------------------------------------------------------------------

def test_ik_right(pos_t, quat_t):
    """Inferenza diretta per il braccio destro."""
    print(f"\n[PREDICT RIGHT] pos={[f'{v:.3f}' for v in pos_t]}  "
          f"quat={[f'{q:.3f}' for q in quat_t]}")

    joints_prediction = inference(*pos_t, *quat_t)
    if joints_prediction is None:
        print("[WARN] Inferenza RIGHT ha restituito None, skip.")
        return
    update_visuals_right(joints_prediction, pos_t, quat_t)


def test_ik_left(pos_t, quat_t):
    """Inferenza per il braccio sinistro: rispecchia input, chiama il modello destro."""
    print(f"\n[PREDICT LEFT] pos={[f'{v:.3f}' for v in pos_t]}  "
          f"quat={[f'{q:.3f}' for q in quat_t]}")

    pos_arr  = np.array(pos_t)
    quat_arr = np.array(quat_t)   # xyzw
    pos_mirrored, quat_mirrored = left_to_right_mirroring(pos_arr, quat_arr)

    print(f"           mirrored pos={[f'{v:.3f}' for v in pos_mirrored]}  "
          f"quat={[f'{q:.3f}' for q in quat_mirrored]}")

    joints_prediction = inference(*pos_mirrored, *quat_mirrored)
    if joints_prediction is None:
        print("[WARN] Inferenza LEFT ha restituito None, skip.")
        return
    update_visuals_left(joints_prediction, pos_t, quat_t)


def update_visuals_right(joints, pos, ori):
    for joint_idx, val in zip(ARM_JOINTS_R, joints):
        p.resetJointState(_robot_r, joint_idx, float(val))
    p.resetBasePositionAndOrientation(_target_r, pos, ori)
    updateTargetDebug_r(pos, ori)
    p.stepSimulation()


def update_visuals_left(joints, pos, ori):
    for joint_idx, val in zip(ARM_JOINTS_L, joints):
        p.resetJointState(_robot_l, joint_idx, float(val))
    p.resetBasePositionAndOrientation(_target_l, pos, ori)
    updateTargetDebug_l(pos, ori)
    p.stepSimulation()

import argparse
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IK Inference with pybullet interface – OpenArm Right",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model_path", required=True,
        help="Path al modello",
    )
    p.add_argument(
        "--model_dir", default="ik_model",
        help="Directory opzionale del modello (default: ik_model)",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        args = parse_args()
        setup(args.model_path, args.model_dir)
        diagnose_robot()

        curr_vals = read_debug_params(handles)
        pos_r   = [curr_vals["posX_r"], curr_vals["posY_r"], curr_vals["posZ_r"]]
        euler_r = [curr_vals["eulerX_r"], curr_vals["eulerY_r"], curr_vals["eulerZ_r"]]
        updateTargetDebug_r(pos_r, p.getQuaternionFromEuler(euler_r))

        if CONFIG_L:
            pos_l   = [curr_vals["posX_l"], curr_vals["posY_l"], curr_vals["posZ_l"]]
            euler_l = [curr_vals["eulerX_l"], curr_vals["eulerY_l"], curr_vals["eulerZ_l"]]
            updateTargetDebug_l(pos_l, p.getQuaternionFromEuler(euler_l))

        p.stepSimulation()
        last_vals = curr_vals

        slider_keys_r = ["eulerX_r", "eulerY_r", "eulerZ_r", "posX_r", "posY_r", "posZ_r"]
        slider_keys_l = ["eulerX_l", "eulerY_l", "eulerZ_l", "posX_l", "posY_l", "posZ_l"] if CONFIG_L else []
        slider_keys   = slider_keys_r + slider_keys_l

        while True:
            curr_vals = read_debug_params(handles)

            predict_triggered   = (last_vals["predict_btn"] != curr_vals["predict_btn"])
            sliders_changed_r   = any(curr_vals[k] != last_vals[k] for k in slider_keys_r)
            sliders_changed_l   = any(curr_vals[k] != last_vals[k] for k in slider_keys_l)

            # Aggiorna debug visivo dei target al movimento degli slider (senza inferenza)
            if sliders_changed_r:
                pos_r   = [curr_vals["posX_r"],   curr_vals["posY_r"],   curr_vals["posZ_r"]]
                euler_r = [curr_vals["eulerX_r"],  curr_vals["eulerY_r"], curr_vals["eulerZ_r"]]
                quat_r  = p.getQuaternionFromEuler(euler_r)
                updateTargetDebug_r(pos_r, quat_r)

            if sliders_changed_l:
                pos_l   = [curr_vals["posX_l"],   curr_vals["posY_l"],   curr_vals["posZ_l"]]
                euler_l = [curr_vals["eulerX_l"],  curr_vals["eulerY_l"], curr_vals["eulerZ_l"]]
                quat_l  = p.getQuaternionFromEuler(euler_l)
                updateTargetDebug_l(pos_l, quat_l)

            # Al click del pulsante: inferenza per entrambi i bracci
            if predict_triggered:
                pos_r   = [curr_vals["posX_r"],   curr_vals["posY_r"],   curr_vals["posZ_r"]]
                euler_r = [curr_vals["eulerX_r"],  curr_vals["eulerY_r"], curr_vals["eulerZ_r"]]
                quat_r  = p.getQuaternionFromEuler(euler_r)
                test_ik_right(pos_r, quat_r)

                if CONFIG_L:
                    pos_l   = [curr_vals["posX_l"],   curr_vals["posY_l"],   curr_vals["posZ_l"]]
                    euler_l = [curr_vals["eulerX_l"],  curr_vals["eulerY_l"], curr_vals["eulerZ_l"]]
                    quat_l  = p.getQuaternionFromEuler(euler_l)
                    test_ik_left(pos_l, quat_l)

            last_vals = curr_vals
            p.stepSimulation()

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        p.disconnect()