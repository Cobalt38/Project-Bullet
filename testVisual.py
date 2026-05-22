import pybullet as p
import pybullet_data
import pandas as pd
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import joblib

# ──────────────────────────────────────────
#  CONFIGURAZIONE  — adatta questi valori al tuo setup
# ──────────────────────────────────────────

CKPT_PATH    = "ik_training/best_model.ckpt"
SCALER_Y_PATH = "ik_training/scaler_y.pkl"
CSV_PATH     = "dataset.csv"
URDF_PATH    = "biarm_model/openarm.urdf"

ROBOPOS      = [0, 0, 0]
ARM_JOINTS   = [0, 1, 2, 3, 4, 5, 6]   # 7 giunti
EE_LINK_INDEX = 6                        # ultimo link = end effector
MAPSIZE      = [0.8, 0.8, 0.6]
MAPOFFSET    = [0, 0, 0.3]

OUTPUT_COLS  = [f"joint_{i}" for i in range(7)]
INPUT_COLS   = ["target_x", "target_y", "target_z",
                "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"]

REST_POSE    = [0, 0, 0, 1.2, 0, 0, 0]

# ──────────────────────────────────────────
#  ARCHITETTURA  (identica a train.py)
# ──────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class IKNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, dropout):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class LitIKRegressor(pl.LightningModule):
    def __init__(self, input_dim=7, output_dim=7,
                 hidden_dim=512, n_blocks=4, dropout=0.1,
                 lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = IKNet(input_dim, output_dim, hidden_dim, n_blocks, dropout)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────
#  INFERENZA LOCALE 
# ──────────────────────────────────────────

_model    = None
_scaler_y = None
_scaler_x = None

def ik_connect():
    """Carica modello e scaler in memoria."""
    global _model, _scaler_y, _scaler_x
    print(f"Caricamento modello da {CKPT_PATH} ...")
    _model = LitIKRegressor.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    _model.eval()
    _scaler_y = joblib.load(SCALER_Y_PATH)
    _scaler_x = joblib.load(os.path.join(os.path.dirname(SCALER_Y_PATH), "scaler_x.pkl"))
    hp = _model.hparams
    params = sum(p_.numel() for p_ in _model.parameters())
    print(f"  hidden={hp.hidden_dim}, blocks={hp.n_blocks}, params={params:,}")
    print("Modello pronto.")

def ik_predict(target_pos, target_ori):
    """
    target_pos : [x, y, z]
    target_ori : [qx, qy, qz, qw]
    Restituisce array numpy (7,) con i valori dei giunti in radianti.
    """
    pos_norm = _scaler_x.transform([target_pos])[0].tolist()
    pose = pos_norm + list(target_ori)   # xyz normalizzati + quaternioni raw
    x = torch.tensor([pose], dtype=torch.float32)
    with torch.no_grad():
        pred_norm = _model(x).numpy()
    return _scaler_y.inverse_transform(pred_norm)[0]

# ──────────────────────────────────────────
#  STATO GLOBALE
# ──────────────────────────────────────────

_target = None
_robot  = None
_ik_robot = None
_df     = None
_debug_cylinder   = None
_debug_tip        = None
_debug_ear_left   = None
_debug_ear_right  = None
_ear_offset = 0.015
_btn_next    = None
_predict_btn = None

_eulerX = _eulerY = _eulerZ = None
_posX   = _posY   = _posZ   = None

lower_limits = []
upper_limits = []
joint_ranges = []


# ──────────────────────────────────────────
#  UTILS
# ──────────────────────────────────────────

def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def quat_angle_diff(q1, q2):
    dot = abs(np.dot(np.array(q1), np.array(q2)))
    return 2 * np.arccos(np.clip(dot, -1.0, 1.0))

def get_ee_pose():
    state = p.getLinkState(_robot, EE_LINK_INDEX, computeForwardKinematics=True)
    return np.array(state[4]), np.array(state[5])


# ──────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────

def setup():
    print("Setting up simulation...")
    global _df, _btn_next, _predict_btn
    global _eulerX, _eulerY, _eulerZ, _posX, _posY, _posZ
    global _target, _robot, _ik_robot

    print(f"Caricamento CSV: {CSV_PATH}")
    _df = pd.read_csv(CSV_PATH, usecols=INPUT_COLS + OUTPUT_COLS)
    print(f"  {len(_df):,} righe  |  colonne: {list(_df.columns)}")

    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.3]
    )

    _btn_next    = p.addUserDebugParameter("▶ Riga casuale CSV", 1, 0, 1)
    _predict_btn = p.addUserDebugParameter("⚡ Predict (slider)", 1, 0, 1)
    _eulerX = p.addUserDebugParameter("Euler X", -math.pi, math.pi, 0)
    _eulerY = p.addUserDebugParameter("Euler Y", -math.pi, math.pi, 0)
    _eulerZ = p.addUserDebugParameter("Euler Z", -math.pi, math.pi, 0)
    _posX   = p.addUserDebugParameter("Position X", -MAPSIZE[0]/2, MAPSIZE[0]/2, 0)
    _posY   = p.addUserDebugParameter("Position Y", -MAPSIZE[1]/2, MAPSIZE[1]/2, 0)
    _posZ   = p.addUserDebugParameter("Position Z",  0, MAPSIZE[2], 0.1)

    p.loadURDF("plane.urdf")

    _target = p.loadURDF("cube_small.urdf",
                         basePosition=[1, 0.75, 0.1],
                         useFixedBase=True, globalScaling=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_abs   = os.path.join(script_dir, URDF_PATH)
    _robot = p.loadURDF(urdf_abs, basePosition=ROBOPOS,
                        useFixedBase=True, globalScaling=1.0)

    # _ik_robot = p.loadURDF(urdf_abs, basePosition=ROBOPOS,
    #                        useFixedBase=True, globalScaling=1.0)
    
    # p.changeVisualShape(objectUniqueId=_ik_robot, linkIndex=-1, rgbaColor=[0.3, 1.0, 0.5, 0.3])
    # # Disabilita collisioni tra TUTTI i link dei due robot
    # for i in range(-1, p.getNumJoints(_robot)):
    #     for j in range(-1, p.getNumJoints(_ik_robot)):
    #         p.setCollisionFilterPair(
    #             _robot,
    #             _ik_robot,
    #             i,
    #             j,
    #             enableCollision=0
    #         )

    # p.changeDynamics(_ik_robot, -1, mass=0)
    # for j in range(p.getNumJoints(_ik_robot)):
    #     p.setJointMotorControl2(
    #         _ik_robot,
    #         j,
    #         p.VELOCITY_CONTROL,
    #         force=0
    #     )

    # for j in ARM_JOINTS:
    #     info = p.getJointInfo(_ik_robot, j)
    #     lower_limits.append(info[8])
    #     upper_limits.append(info[9])
    #     joint_ranges.append(info[9] - info[8])

    createTargetDebug()


# ──────────────────────────────────────────
#  TEST PRINCIPALE
# ──────────────────────────────────────────

def test_csv(user_input=False):
    """
    user_input=False → pesca una riga casuale dal CSV e confronta rete vs ground truth
    user_input=True  → usa i valori degli slider
    """
    global _df, _target, _robot, _ik_robot

    # ── Scegli target ──
    if user_input:
        euler = [p.readUserDebugParameter(_eulerX),
                 p.readUserDebugParameter(_eulerY),
                 p.readUserDebugParameter(_eulerZ)]
        target_pos = [p.readUserDebugParameter(_posX),
                      p.readUserDebugParameter(_posY),
                      p.readUserDebugParameter(_posZ)]
        target_ori = list(p.getQuaternionFromEuler(euler))
        print(f"\n[SLIDER] pos={[f'{v:.3f}' for v in target_pos]}  "
              f"euler={[f'{math.degrees(e):.1f}°' for e in euler]}")
        row = None
    else:
        row = _df.iloc[random.randint(0, len(_df) - 1)]
        target_pos = row[["target_x", "target_y", "target_z"]].values.tolist()
        target_ori = row[["hand_quat_qx", "hand_quat_qy",
                          "hand_quat_qz", "hand_quat_qw"]].values.tolist()
        print(f"\n[CSV row {row.name}] pos={[f'{v:.3f}' for v in target_pos]}")

    # ── Inferenza ──
    joints_pred = ik_predict(target_pos, target_ori)

    # ── Applica giunti predetti ──
    for joint_idx, val in zip(ARM_JOINTS, joints_pred):
        p.resetJointState(_robot, joint_idx, float(val))

    # ik_calculated = p.calculateInverseKinematics(
    #                     bodyUniqueId=_ik_robot,
    #                     endEffectorLinkIndex=EE_LINK_INDEX,
    #                     targetPosition=target_pos,
    #                     targetOrientation=target_ori,
    #                     lowerLimits=lower_limits,
    #                     upperLimits=upper_limits,
    #                     jointRanges=joint_ranges,
    #                     restPoses=REST_POSE,
    #                     maxNumIterations=200,
    #                     residualThreshold=1e-5
    #                 )
    # for ji, joint_id in enumerate(ARM_JOINTS):
    #         p.resetJointState(_ik_robot, joint_id, targetValue=ik_calculated[ji])

    # ── Aggiorna target visivo ──
    p.resetBasePositionAndOrientation(_target, target_pos, [0, 0, 0, 1])
    updateTargetDebug(target_pos, target_ori)
    p.stepSimulation()

    # ── Calcola errore FK ──
    ee_pos, ee_ori = get_ee_pose()
    pos_err_cm = np.linalg.norm(np.array(target_pos) - ee_pos) * 100
    orn_err_deg = math.degrees(quat_angle_diff(target_ori, ee_ori))

    # ── Stampa tabella confronto ──
    print(f"\n  {'giunto':<12} {'predetto':>10}  {'predetto°':>10}", end="")
    if row is not None:
        print(f"  {'GT':>10}  {'diff°':>8}", end="")
    print()
    print("  " + "─" * (48 if row is None else 68))

    for i, (name, pred_val) in enumerate(zip(OUTPUT_COLS, joints_pred)):
        line = f"  {name:<12} {pred_val:>+10.4f}  {math.degrees(pred_val):>+9.1f}°"
        if row is not None:
            gt_val  = row[name]
            diff    = math.degrees(pred_val - gt_val)
            line   += f"  {math.degrees(gt_val):>+9.1f}°  {diff:>+7.1f}°"
        print(line)

    print(f"\n  FK end effector → pos={np.round(ee_pos, 3).tolist()}")
    print(f"  Errore posizione : {pos_err_cm:.2f} cm", end="")
    if pos_err_cm < 2.0:
        print("  ✓")
    elif pos_err_cm < 5.0:
        print("  ~")
    else:
        print("  ✗")
    print(f"  Errore rotazione : {orn_err_deg:.2f}°")


# ──────────────────────────────────────────
#  DEBUG VISIVO  (identico al tuo originale)
# ──────────────────────────────────────────

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
    p.resetBasePositionAndOrientation(_debug_tip, target_pos, [0, 0, 0, 1])
    R       = quaternion_to_matrix(target_ori)
    x_axis  = R[:, 0]
    ear_l   = np.array(target_pos) + x_axis * _ear_offset
    ear_r   = np.array(target_pos) - x_axis * _ear_offset
    p.resetBasePositionAndOrientation(_debug_ear_left,  ear_l.tolist(), target_ori)
    p.resetBasePositionAndOrientation(_debug_ear_right, ear_r.tolist(), target_ori)


# ──────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────

if __name__ == "__main__":
    try:
        setup()
        ik_connect()
        test_csv()   # prima riga casuale all'avvio

        last_btn_val     = p.readUserDebugParameter(_btn_next)
        predict_btn_val  = p.readUserDebugParameter(_predict_btn)
        ex_val = p.readUserDebugParameter(_eulerX)
        ey_val = p.readUserDebugParameter(_eulerY)
        ez_val = p.readUserDebugParameter(_eulerZ)
        px_val = p.readUserDebugParameter(_posX)
        py_val = p.readUserDebugParameter(_posY)
        pz_val = p.readUserDebugParameter(_posZ)
        ex_last = ex_val; ey_last = ey_val; ez_last = ez_val
        px_last = px_val; py_last = py_val; pz_last = pz_val

        while True:
            # Leggi slider
            ex_val = p.readUserDebugParameter(_eulerX)
            ey_val = p.readUserDebugParameter(_eulerY)
            ez_val = p.readUserDebugParameter(_eulerZ)
            px_val = p.readUserDebugParameter(_posX)
            py_val = p.readUserDebugParameter(_posY)
            pz_val = p.readUserDebugParameter(_posZ)

            # Slider mossi → aggiorna solo il debug visivo (non inferenza)
            sliders_changed = (
                ex_val != ex_last or ey_val != ey_last or ez_val != ez_last or
                px_val != px_last or py_val != py_last or pz_val != pz_last
            )
            if sliders_changed:
                ex_last = ex_val; ey_last = ey_val; ez_last = ez_val
                px_last = px_val; py_last = py_val; pz_last = pz_val
                updateTargetDebug(
                    [px_val, py_val, pz_val],
                    p.getQuaternionFromEuler([ex_val, ey_val, ez_val])
                )

            # Bottone "Riga casuale CSV"
            current_btn = p.readUserDebugParameter(_btn_next)
            if current_btn != last_btn_val:
                last_btn_val = current_btn
                test_csv(user_input=False)

            # Bottone "Predict (slider)"
            current_pred = p.readUserDebugParameter(_predict_btn)
            if current_pred != predict_btn_val:
                predict_btn_val = current_pred
                test_csv(user_input=True)

            p.stepSimulation()

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        p.disconnect()