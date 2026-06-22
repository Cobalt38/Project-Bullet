import pybullet as p
import pybullet_data
import math
import numpy as np
import os
from ik_inference import load_model_and_metadata, run_inference, decode_output, build_input


openarmRightConfig = {
    "rest_pose": [0, 0, 0, 1.2, 0, 0, 0],
    "arm_joints": [i for i in range(2, 9)],  # 2..8
    "ee_link_index": 10,  # TCP
    "isLocalpath": True,
    "urdf_path": "biarm_model/openarm_right.urdf",
    "max_reach": 0.7
}

CONFIG = openarmRightConfig

ROBOPOS    = [0, 0, 0]
MAPSIZE    = [2.0, 2.0, 2.0]
MAPOFFSET  = [0, -0.9, 1]

ARM_JOINTS    = CONFIG.get("arm_joints")
EE_LINK_INDEX = CONFIG.get("ee_link_index")
REST_POSE     = CONFIG.get("rest_pose")
URDF_PATH     = CONFIG.get("urdf_path")

# ---------------------------------------------------------------------------
# Definizione parametri GUI: (nome_display, min, max, default)
# ---------------------------------------------------------------------------
PARAM_DEFS = [
    ("⚡ Predict (slider)", 1,              0,             1  ),
    ("Euler X",            -math.pi,        math.pi,       0  ),
    ("Euler Y",            -math.pi,        math.pi,       0  ),
    ("Euler Z",            -math.pi,        math.pi,       0  ),
    ("Position X",         -MAPSIZE[0]/2,   MAPSIZE[0]/2,  0  ),
    ("Position Y",         -MAPSIZE[1]/2,   MAPSIZE[1]/2,  0  ),
    ("Position Z",          0,              MAPSIZE[2],    0.1),
]

PARAM_KEYS = ["predict_btn", "eulerX", "eulerY", "eulerZ", "posX", "posY", "posZ"]

# ---------------------------------------------------------------------------
# Stato globale
# ---------------------------------------------------------------------------
_model = None
_meta  = None

_target = None
_robot  = None

_debug_cylinder  = None
_debug_tip       = None
_debug_ear_left  = None
_debug_ear_right = None
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

def diagnose_robot():
    """Stampa tutti i joint del robot per trovare gli indici corretti."""
    print(f"\n=== JOINT MAP ({p.getNumJoints(_robot)} joints totali) ===")
    for i in range(p.getNumJoints(_robot)):
        info  = p.getJointInfo(_robot, i)
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

# ---------------------------------------------------------------------------
# Debug visivo
# ---------------------------------------------------------------------------

def createTargetDebug():
    global _debug_cylinder, _debug_tip, _debug_ear_left, _debug_ear_right

    debug_cylinder_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER, radius=0.015, length=0.10,
        rgbaColor=[0, 0.5, 1, 0.35],
        visualFramePosition=[0, 0, -0.05],
        visualFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    _debug_cylinder = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=debug_cylinder_visual,
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1]
    )

    debug_tip_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, radius=0.015,
        rgbaColor=[0.6, 0.1, 0.6, 0.6]
    )
    _debug_tip = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=debug_tip_visual, basePosition=[0, 0, 0]
    )

    debug_ear_left_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER, radius=0.006, length=0.01,
        rgbaColor=[1, 0.3, 0.0, 0.5],
        visualFramePosition=[0, 0, 0],
        visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])
    )
    _debug_ear_left = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=debug_ear_left_visual,
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1]
    )

    debug_ear_right_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER, radius=0.006, length=0.01,
        rgbaColor=[0.2, 0.85, 0.1, 0.5],
        visualFramePosition=[0, 0, 0],
        visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])
    )
    _debug_ear_right = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=debug_ear_right_visual,
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1]
    )


def updateTargetDebug(target_pos, target_ori):
    p.resetBasePositionAndOrientation(_debug_cylinder, target_pos, target_ori)
    p.resetBasePositionAndOrientation(_debug_tip,      target_pos, [0, 0, 0, 1])
    R      = quaternion_to_matrix(target_ori)
    x_axis = R[:, 0]
    ear_l  = np.array(target_pos) + x_axis * _ear_offset
    ear_r  = np.array(target_pos) - x_axis * _ear_offset
    p.resetBasePositionAndOrientation(_debug_ear_left,  ear_l.tolist(), target_ori)
    p.resetBasePositionAndOrientation(_debug_ear_right, ear_r.tolist(), target_ori)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup(model_path, model_dir):
    print("Setting up simulation...")
    global handles, last_vals, _target, _robot

    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.3]
    )

    handles   = setup_debug_params()
    last_vals = read_debug_params(handles)

    p.loadURDF("plane.urdf")

    _target = p.loadURDF("cube_small.urdf",
                         basePosition=[1, 0.75, 0.1],
                         useFixedBase=True, globalScaling=0.3)
    p.changeVisualShape(objectUniqueId=_target, linkIndex=-1, rgbaColor=[1, 0, 0, 1])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path  = os.path.join(script_dir, URDF_PATH) if CONFIG["isLocalpath"] else URDF_PATH

    print(f"Loading robot from URDF: {urdf_path}")
    _robot = p.loadURDF(urdf_path, basePosition=ROBOPOS, useFixedBase=True, globalScaling=1.0)

    createTargetDebug()

    load_model(model_path, model_dir)

# ---------------------------------------------------------------------------
# IK test + visuals
# ---------------------------------------------------------------------------

def test_ik(pos_t, quat_t):
    print(f"\n[PREDICT] pos={[f'{v:.3f}' for v in pos_t]}  "
          f"quat={[f'{q:.3f}' for q in quat_t]}")

    joints_prediction = inference(*pos_t, *quat_t)

    if joints_prediction is None:
        print("[WARN] Inferenza ha restituito None, skip.")
        return

    update_visuals(joints_prediction, pos_t, quat_t)


def update_visuals(joints, pos, ori):
    # Applica giunti predetti
    for joint_idx, val in zip(ARM_JOINTS, joints):
        p.resetJointState(_robot, joint_idx, float(val))

    # BUG FIX: il target si orientava sempre con [0,0,0,1]
    #          ora usa ori (il quaternione reale dal slider)
    p.resetBasePositionAndOrientation(_target, pos, ori)
    updateTargetDebug(pos, ori)
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

        slider_keys = [k for k in PARAM_KEYS if k != "predict_btn"]

        while True:
            curr_vals = read_debug_params(handles)

            predict_triggered = (last_vals["predict_btn"] != curr_vals["predict_btn"])
            sliders_changed   = any(curr_vals[k] != last_vals[k] for k in slider_keys)

            if sliders_changed:
                pos   = [curr_vals["posX"],   curr_vals["posY"],   curr_vals["posZ"]]
                euler = [curr_vals["eulerX"],  curr_vals["eulerY"], curr_vals["eulerZ"]]
                quat  = p.getQuaternionFromEuler(euler)
                updateTargetDebug(pos, quat)

            if predict_triggered:
                pos   = [curr_vals["posX"],   curr_vals["posY"],   curr_vals["posZ"]]
                euler = [curr_vals["eulerX"],  curr_vals["eulerY"], curr_vals["eulerZ"]]
                quat  = p.getQuaternionFromEuler(euler)
                test_ik(pos, quat)

            last_vals = curr_vals
            p.stepSimulation()

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        p.disconnect()