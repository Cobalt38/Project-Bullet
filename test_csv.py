import pybullet as p
import pybullet_data
import pandas as pd
import os
import random
import math
import numpy as np
import socket
import json

# --- configurazione connessione IK server ---
IK_HOST = "192.168.102.188" # "127.0.0.1"
IK_PORT = 8001  # default: HTTP port+1

ROBOPOS = [0,0,0]
ARM_JOINTS = [0, 1, 2, 3, 4]
EE_LINK_INDEX = 5 
MAPSIZE = [0.4, 0.4, 0.2] 
MAPOFFSET = [0,0,0.05]

_target = None
_robot = None
_df = None
_debug_cylinder = None
_debug_tip = None
_debug_ear_left = None
_debug_ear_right = None
_ear_offset = 0.015 # distanza laterale delle "orecchie" di debug dal centro del cilindro
_btn_next = None
_predict_btn = None
_ik_socket = None

_eulerX, _eulerY, _eulerZ = None, None, None
_posX, _posY, _posZ = None, None, None

def ik_connect():
    print("Connecting to IK server...")
    global _ik_socket
    _ik_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _ik_socket.connect((IK_HOST, IK_PORT))
    _ik_socket.settimeout(2.0)
    print(f"[IK] connesso a {IK_HOST}:{IK_PORT}")

def ik_predict(target_pos, target_ori):
    """
    target_pos: [x, y, z]
    target_ori: [qx, qy, qz, qw]
    Restituisce dict tipo {"Root.001": {"ry": ...}, ...} oppure None se errore
    """
    payload = {
        "target": list(target_pos) + list(target_ori)
    }
    line = json.dumps(payload) + "\n"
    _ik_socket.sendall(line.encode("utf-8"))

    # leggi risposta (una riga)
    buf = b""
    while b"\n" not in buf:
        chunk = _ik_socket.recv(4096)
        if not chunk:
            raise ConnectionError("Server ha chiuso la connessione")
        buf += chunk

    response = json.loads(buf.split(b"\n")[0].decode("utf-8"))
    if not response.get("ok"):
        print(f"[IK] errore server: {response.get('error')}")
        return None
    return response["result"]

def ik_disconnect():
    global _ik_socket
    if _ik_socket:
        _ik_socket.close()
        _ik_socket = None

def quaternion_to_matrix(q):
    """Converte quaternione (x,y,z,w) in matrice di rotazione 3x3"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def setup():
    print("Setting up simulation...")
    global _df, _btn_next, _eulerX, _eulerY, _eulerZ, _posX, _posY, _posZ, _target, _robot, _predict_btn
    print("Loading CSV dataset...")
    _df = pd.read_csv("csv_belli/dataset_fixed.csv")
    print(f"Csv loaded, columns: {_df.columns}")
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    _btn_next = p.addUserDebugParameter("▶ Prossima riga", 1, 0, 1) 
    _predict_btn = p.addUserDebugParameter("Predict", 1, 0, 1)
    _eulerX = p.addUserDebugParameter("Euler X", -math.pi, math.pi, 0)
    _eulerY = p.addUserDebugParameter("Euler Y", -math.pi, math.pi, 0)
    _eulerZ = p.addUserDebugParameter("Euler Z", -math.pi, math.pi, 0)
    _posX = p.addUserDebugParameter("Position X", -MAPSIZE[0]/2, MAPSIZE[0]/2, 0)
    _posY = p.addUserDebugParameter("Position Y", -MAPSIZE[1]/2, MAPSIZE[1]/2, 0)
    _posZ = p.addUserDebugParameter("Position Z", -MAPSIZE[2]/2, MAPSIZE[2]/2, 0)

    global _target
    global _robot
    _target = p.loadURDF("cube_small.urdf", basePosition=[1,0.75,0.1], useFixedBase=True, globalScaling=0.3)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "xarm_model/urdf/xarm_fixed.urdf")
    _robot = p.loadURDF(urdf_path, basePosition=ROBOPOS, useFixedBase=True, globalScaling=1.0)
    createTargetDebug()

def test_csv(user_input=False, use_ik=False):
    print("Esecuzione test_csv(", user_input, use_ik, ")")
    global _df, _target, _robot, _btn_next, _eulerX, _eulerY, _eulerZ, _posX, _posY, _posZ
    row = None
    if user_input:
        target_ori = p.getQuaternionFromEuler([p.readUserDebugParameter(_eulerX), p.readUserDebugParameter(_eulerY), p.readUserDebugParameter(_eulerZ)])
        target_pos = [p.readUserDebugParameter(_posX), p.readUserDebugParameter(_posY), p.readUserDebugParameter(_posZ)]
        print(f"User input - Position: {target_pos}, Orientation (Euler): ({p.readUserDebugParameter(_eulerX):.1f}, {p.readUserDebugParameter(_eulerY):.1f}, {p.readUserDebugParameter(_eulerZ):.1f})")
    else:
        row = _df.iloc[random.randint(0, len(_df)-1)]
        print(f"Row {row.name}: {row.values}")
        target_pos = row[["target_x", "target_y", "target_z"]].values
        target_ori = row[["hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"]].values
        target_ori = target_ori + np.array([random.uniform(-0.001, 0.001), random.uniform(-0.001, 0.001), random.uniform(-0.001, 0.001) , random.uniform(-0.001, 0.001)])  # assicurati che sia un quaternione valido (w=1 se mancante)
    if use_ik:
        # chiedi angoli alla rete
        result = ik_predict(target_pos, target_ori)
        if result:
            if not user_input:
                csv_angles  = row[["Root.001_ry", "Root.002_rx", "Root.003_rx", "Root.004_rx", "Head_ry"]].values
                csv_angles_deg = [math.degrees(a) for a in csv_angles]
                # appiattisci result nello stesso ordine
                net_angles = [
                    result["Root.001"]["ry"],
                    result["Root.002"]["rx"],
                    result["Root.003"]["rx"],
                    result["Root.004"]["rx"],
                    result["Head"]["ry"]
                ]
                print(f"{'joint':<12} {'CSV':>8} {'rete':>8} {'diff':>8}")
                print("-" * 40)
                labels = ["Root.001_ry", "Root.002_rx", "Root.003_rx", "Root.004_rx", "Head_ry"]
                for label, csv_val, net_val in zip(labels, csv_angles_deg, net_angles):
                    diff = net_val - csv_val
                    print(f"{label:<12} {csv_val:>8.2f} {net_val:>8.2f} {diff:>+8.2f}")

            # mappa il risultato sui joint — esempio, adatta ai tuoi nomi:
            joint_map = {
                "Root.001": (0, "ry"),
                "Root.002": (1, "rx"),
                "Root.003": (2, "rx"),
                "Root.004": (3, "rx"),
                "Head":     (4, "ry"),
            }
            for bone, (joint_idx, axis) in joint_map.items():
                if bone in result and axis in result[bone]:
                    angle_deg = result[bone][axis]
                    p.resetJointState(_robot, joint_idx, math.radians(angle_deg))

        p.resetBasePositionAndOrientation(_target, target_pos, [0, 0, 0, 1])
        updateTargetDebug(target_pos, target_ori)
        p.stepSimulation()
    else:
        #probabilmente crasha
        joint_angles = row[
            ["Root.001_ry", 
            "Root.002_rx", 
            "Root.003_rx", 
            "Root.004_rx", 
            "Head_ry"]
            ].values  
        p.resetBasePositionAndOrientation(_target, target_pos, [0, 0, 0, 1])
        updateTargetDebug(target_pos, target_ori)
        for i, joint_index in enumerate(ARM_JOINTS):
            p.resetJointState(_robot, joint_index, joint_angles[i])
        p.stepSimulation()

def createTargetDebug():
    global _debug_cylinder,_debug_tip, _debug_ear_left, _debug_ear_right, _ear_offset

    q_cylinder_fix = p.getQuaternionFromEuler([0,0,0])
    debug_cylinder_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=0.015,
        length=0.10,
        rgbaColor=[0, 0.5, 1, 0.35],
        visualFramePosition=[0, 0, -0.05],        # cilindro esteso -10cm dal frame
        visualFrameOrientation=q_cylinder_fix     # fix asse Y→Z 
    )
    _debug_cylinder = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=debug_cylinder_visual,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1]
    )
    debug_tip_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[0.6, 0.1, 0.6, 0.6] 
    )
    _debug_tip = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=debug_tip_visual,
        basePosition=[0, 0, 0]
    )

    debug_ear_left_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=0.006,
        length=0.01,
        rgbaColor=[1, 0.3, 0.0, 0.5],  # arancione
        visualFramePosition=[0, 0, 0],
        visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])  # perpendicolare
    )
    _debug_ear_left = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=debug_ear_left_visual,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1]
    )

    debug_ear_right_visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=0.006,
        length=0.01,
        rgbaColor=[0.2, 0.85, 0.1, 0.5],  # verde
        visualFramePosition=[0, 0, 0],
        visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])
    )
    _debug_ear_right = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=debug_ear_right_visual,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1]
    )

def updateTargetDebug(target_pos, target_ori):
    global _debug_cylinder
    global _debug_tip
    global _debug_ear_left
    global _debug_ear_right
    global _ear_offset
    p.resetBasePositionAndOrientation(_debug_cylinder, target_pos, target_ori)
    p.resetBasePositionAndOrientation(_debug_tip, target_pos, [0, 0, 0, 1])
    R = quaternion_to_matrix(target_ori)
    x_axis = R[:, 0]  # asse X locale del cilindro nel world

    # Le orecchie stanno all'inizio del cilindro 
    z_axis = R[:, 2]
    ears_start_pos = np.array(target_pos) #- z_axis * 0.01

    ear_left_pos  = ears_start_pos + x_axis * _ear_offset
    ear_right_pos = ears_start_pos - x_axis * _ear_offset
    p.resetBasePositionAndOrientation(_debug_ear_left,  ear_left_pos.tolist(),  target_ori)
    p.resetBasePositionAndOrientation(_debug_ear_right, ear_right_pos.tolist(), target_ori)

if __name__ == "__main__":   
    try: 
        setup()
        ik_connect()
        test_csv()
        last_btn_val = p.readUserDebugParameter(_btn_next)
        _predict_btn_val = p.readUserDebugParameter(_predict_btn)
        _eulerX_val = p.readUserDebugParameter(_eulerX)
        _eulerY_val = p.readUserDebugParameter(_eulerY)
        _eulerZ_val = p.readUserDebugParameter(_eulerZ)
        _posX_val = p.readUserDebugParameter(_posX)
        _posY_val = p.readUserDebugParameter(_posY)
        _posZ_val = p.readUserDebugParameter(_posZ)
        _eulerX_val_last = _eulerX_val
        _eulerY_val_last = _eulerY_val
        _eulerZ_val_last = _eulerZ_val
        _posX_val_last = _posX_val
        _posY_val_last = _posY_val
        _posZ_val_last = _posZ_val

        while True:
            current_val = p.readUserDebugParameter(_btn_next)
            _eulerX_val = p.readUserDebugParameter(_eulerX)
            _eulerY_val = p.readUserDebugParameter(_eulerY)
            _eulerZ_val = p.readUserDebugParameter(_eulerZ)
            _posX_val = p.readUserDebugParameter(_posX)
            _posY_val = p.readUserDebugParameter(_posY)
            _posZ_val = p.readUserDebugParameter(_posZ)
            if _eulerX_val != _eulerX_val_last or _eulerY_val != _eulerY_val_last or _eulerZ_val != _eulerZ_val_last or _posX_val != _posX_val_last or _posY_val != _posY_val_last or _posZ_val != _posZ_val_last:
                _eulerX_val_last = _eulerX_val
                _eulerY_val_last = _eulerY_val
                _eulerZ_val_last = _eulerZ_val
                _posX_val_last = _posX_val
                _posY_val_last = _posY_val
                _posZ_val_last = _posZ_val
                updateTargetDebug([_posX_val, _posY_val, _posZ_val], p.getQuaternionFromEuler([_eulerX_val, _eulerY_val, _eulerZ_val]))
            elif current_val != last_btn_val:   # click rilevato
                last_btn_val = current_val
                test_csv(use_ik=True)
            if p.readUserDebugParameter(_predict_btn) != _predict_btn_val:
                _predict_btn_val = p.readUserDebugParameter(_predict_btn)
                test_csv(user_input=True, use_ik=True)
            p.stepSimulation()

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:  
        ik_disconnect()      
        p.disconnect()  
