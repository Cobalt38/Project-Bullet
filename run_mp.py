"""
use_model_both_arms.py
----------------------
Lancia un processo PyBullet separato per ogni braccio robotico.
Ogni processo gestisce la propria GUI, i propri slider e l'inferenza IK.

Configurazione:
  - "side": "right" | "left"  → determina se applicare il mirroring degli input
  - "urdf_path": se None o non specificato, il braccio non viene avviato
  - "model_path" / "model_dir": percorso al modello TF da caricare

Uso:
  python use_model_both_arms.py --model_path path/al/modello [--model_dir ik_model]
"""

import argparse
import math
import multiprocessing
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data

from ik_inference import load_model_and_metadata, run_inference, decode_output, build_input

# ---------------------------------------------------------------------------
# Configurazione bracci
# ---------------------------------------------------------------------------

CONFIGS = {
    "right": {
        "side":          "right",
        "rest_pose":     [0, 0, 0, 1.2, 0, 0, 0],
        "arm_joints":    [2, 3, 4, 5, 6, 7, 8],
        "ee_link_index": 10,
        "urdf_path":     "biarm_model/openarm_right.urdf",
        "robopos":       [0, 0, 0],
        "mapsize":       [2.0, 2.0, 2.0],
        "mapoffset":     [0, -0.9, 1],
        "target_color":  [1, 0, 0, 1],           # rosso
        "cylinder_color":[0, 0.5, 1, 0.35],      # blu
        "target_start":  [1, -0.75, 0.1],
    },
    "left": {
        "side":          "left",
        "rest_pose":     [0, 0, 0, 1.2, 0, 0, 0],
        "arm_joints":    [2, 3, 4, 5, 6, 7, 8],
        "ee_link_index": 10,
        "urdf_path":     "biarm_model/openarm_left.urdf",  # impostare None per disabilitare
        "robopos":       [0, 0, 0],
        "mapsize":       [2.0, 2.0, 2.0],
        "mapoffset":     [0,  0.9, 1],
        "target_color":  [0, 0.4, 1, 1],          # blu
        "cylinder_color":[1, 0.5, 0, 0.35],       # arancione
        "target_start":  [1,  0.75, 0.1],
    },
}

# ---------------------------------------------------------------------------
# Mirroring XZ-plane: input sinistro → spazio destro
# ---------------------------------------------------------------------------

def left_to_right_mirroring(pos: np.ndarray, quat: np.ndarray):
    """Piano di simmetria XZ (Y invertita). quat in xyzw."""
    pos_m  = pos  * np.array([1.0, -1.0,  1.0])
    quat_m = quat * np.array([1.0, -1.0,  1.0, -1.0])  # nega qy
    return pos_m, quat_m

# ---------------------------------------------------------------------------
# Processo per singolo braccio
# ---------------------------------------------------------------------------

class ArmProcess:
    """Tutto ciò che gira in-process per un singolo braccio."""

    # ---- geometria debug ------------------------------------------------

    _ear_offset = 0.015

    def __init__(self, cfg: dict, model_path: str, model_dir: str):
        self.cfg        = cfg
        self.model_path = model_path
        self.model_dir  = model_dir
        self.side       = cfg["side"]
        self.label      = self.side.upper()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # stato pybullet
        self._robot        = None
        self._target       = None
        self._dbg_cyl      = None
        self._dbg_tip      = None
        self._dbg_ear_l    = None
        self._dbg_ear_r    = None
        self._dbg_cyl_m    = None   # mirrored (solo braccio sinistro)
        self._dbg_tip_m    = None
        self._dbg_ear_l_m  = None
        self._dbg_ear_r_m  = None
        self._handles      = {}
        self._last_vals    = {}

        # modello
        self._model = None
        self._meta  = None

    # ---- helpers --------------------------------------------------------

    def _load_model(self):
        self._model, self._meta = load_model_and_metadata(
            model_path=self.model_path, model_dir=self.model_dir
        )
        print(f"[{self.label}][MODEL] Caricato. Output joints: "
              f"{self._meta.get('output_columns', [])}")

    def _inference(self, x, y, z, qx, qy, qz, qw):
        raw    = run_inference(self._model, build_input(x, y, z, qx, qy, qz, qw))
        result = decode_output(raw, self._meta.get("output_columns", []))
        print(f"[{self.label}] {result}")
        return result["angles_rad"]

    @staticmethod
    def _quat_to_matrix(q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    @staticmethod
    def _openarm_left_joint_fix(joints):
        print(joints)
        joints[3] *= -1
        ret = np.array(joints) * np.array([-1] * len(joints))
        # print("---\nFROM ", joints, " TO ", ret, "\n---")
        return ret

    # ---- setup pybullet -------------------------------------------------

    def _connect(self):
        p.connect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3]
        )

    def _setup_params(self):
        cfg = self.cfg
        mo  = cfg["mapoffset"]
        ms  = cfg["mapsize"]
        defs = [
            ("⚡ Predict", 1, 0, 1),
            ("Euler X",   -math.pi, math.pi, 0),
            ("Euler Y",   -math.pi, math.pi, 0),
            ("Euler Z",   -math.pi, math.pi, 0),
            ("Position X", -ms[0]/2 + mo[0], ms[0]/2 + mo[0], mo[0]),
            ("Position Y", -ms[1]/2 + mo[1], ms[1]/2 + mo[1], mo[1]),
            ("Position Z", -ms[2]/2 + mo[2], ms[2]/2 + mo[2], mo[2]),
        ]
        keys = ["predict_btn", "eulerX", "eulerY", "eulerZ", "posX", "posY", "posZ"]
        self._handles = {
            k: p.addUserDebugParameter(label, lo, hi, default)
            for k, (label, lo, hi, default) in zip(keys, defs)
        }
        self._last_vals = self._read_params()

    def _read_params(self) -> dict:
        return {k: p.readUserDebugParameter(h) for k, h in self._handles.items()}

    def _load_robot(self):
        urdf = os.path.join(self.script_dir, self.cfg["urdf_path"])
        print(f"[{self.label}] Loading URDF: {urdf}")
        self._robot = p.loadURDF(urdf, basePosition=self.cfg["robopos"],
                                 useFixedBase=True, globalScaling=1.0)
        self._disable_self_collisions()

    @staticmethod
    def _disable_self_collisions(robot_id=None):
        # chiamato anche come metodo di istanza
        rid = robot_id
        n   = p.getNumJoints(rid)
        links = [-1] + list(range(n))
        for i in links:
            for j in links:
                if i != j:
                    p.setCollisionFilterPair(rid, rid, i, j, enableCollision=0)

    # chiama con self per usare self._robot
    def _disable_self_collisions(self):  # noqa: F811
        rid   = self._robot
        n     = p.getNumJoints(rid)
        links = [-1] + list(range(n))
        for i in links:
            for j in links:
                if i != j:
                    p.setCollisionFilterPair(rid, rid, i, j, enableCollision=0)

    def _create_debug_visuals_set(self, color) -> tuple:
        """Crea un set di visuals (cyl, tip, ear_l, ear_r) con il colore dato e li ritorna."""
        cyl = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_CYLINDER, radius=0.015, length=0.10,
                rgbaColor=color,
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

    def _create_debug_visuals(self):
        self._dbg_cyl, self._dbg_tip, self._dbg_ear_l, self._dbg_ear_r = \
            self._create_debug_visuals_set(self.cfg["cylinder_color"])
        # braccio sinistro: secondo set (verde) che mostra l'input post-mirroring
        if self.side == "left":
            self._dbg_cyl_m, self._dbg_tip_m, self._dbg_ear_l_m, self._dbg_ear_r_m = \
                self._create_debug_visuals_set([0.1, 0.9, 0.1, 0.35])
        else:
            self._dbg_cyl_m = self._dbg_tip_m = self._dbg_ear_l_m = self._dbg_ear_r_m = None

    def _update_debug_visuals_set(self, cyl, tip, ear_l, ear_r, target_pos, target_ori):
        p.resetBasePositionAndOrientation(cyl, target_pos, target_ori)
        p.resetBasePositionAndOrientation(tip, target_pos, [0, 0, 0, 1])
        R      = self._quat_to_matrix(target_ori)
        x_axis = R[:, 0]
        el     = np.array(target_pos) + x_axis * self._ear_offset
        er     = np.array(target_pos) - x_axis * self._ear_offset
        p.resetBasePositionAndOrientation(ear_l, el.tolist(), target_ori)
        p.resetBasePositionAndOrientation(ear_r, er.tolist(), target_ori)

    def _update_debug_visuals(self, target_pos, target_ori):
        self._update_debug_visuals_set(
            self._dbg_cyl, self._dbg_tip, self._dbg_ear_l, self._dbg_ear_r,
            target_pos, target_ori
        )
        if self.side == "left":
            pos_m, quat_m = left_to_right_mirroring(
                np.array(target_pos), np.array(target_ori)
            )
            self._update_debug_visuals_set(
                self._dbg_cyl_m, self._dbg_tip_m, self._dbg_ear_l_m, self._dbg_ear_r_m,
                pos_m.tolist(), quat_m.tolist()
            )

    # ---- IK + visuals ---------------------------------------------------

    def _run_ik(self, pos_t, quat_t):
        pos_arr  = np.array(pos_t,  dtype=np.float64)
        quat_arr = np.array(quat_t, dtype=np.float64)

        if self.side == "left":
            print(f"[{self.label}] pos_input={[f'{v:.3f}' for v in pos_arr]}  "
                  f"quat_input={[f'{q:.3f}' for q in quat_arr]}")
            pos_arr, quat_arr = left_to_right_mirroring(pos_arr, quat_arr)
            print(f"[{self.label}] pos_mirrored={[f'{v:.3f}' for v in pos_arr]}  "
                  f"quat_mirrored={[f'{q:.3f}' for q in quat_arr]}")
        else:
            print(f"[{self.label}] pos={[f'{v:.3f}' for v in pos_arr]}  "
                  f"quat={[f'{q:.3f}' for q in quat_arr]}")

        joints = self._inference(*pos_arr, *quat_arr)
        if joints is None:
            print(f"[{self.label}][WARN] Inferenza ha restituito None, skip.")
            return
        
        if self.side == "left":
            joints = self._openarm_left_joint_fix(np.array(joints))

        for joint_idx, val in zip(self.cfg["arm_joints"], joints):
            p.resetJointState(self._robot, joint_idx, float(val))
        p.resetBasePositionAndOrientation(self._target, pos_t, quat_t)
        self._update_debug_visuals(pos_t, quat_t)
        p.stepSimulation()

    # ---- diagnosi -------------------------------------------------------

    def _diagnose(self):
        print(f"\n=== JOINT MAP {self.label} ({p.getNumJoints(self._robot)} joints totali) ===")
        for i in range(p.getNumJoints(self._robot)):
            info  = p.getJointInfo(self._robot, i)
            jtype = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(info[2], str(info[2]))
            print(f"  [{i:2d}] {jtype:<10} name={info[1].decode():<40} parent={info[12].decode()}")
        print("===\n")

    # ---- entry point (lanciato in processo separato) --------------------

    def run(self):
        try:
            self._connect()
            self._setup_params()
            self._create_debug_visuals()

            p.loadURDF("plane.urdf")

            self._target = p.loadURDF(
                "cube_small.urdf",
                basePosition=self.cfg["target_start"],
                useFixedBase=True, globalScaling=0.3
            )
            p.changeVisualShape(objectUniqueId=self._target, linkIndex=-1,
                                rgbaColor=self.cfg["target_color"])

            self._load_robot()
            self._load_model()
            self._diagnose()

            # inizializza i visual target con i valori di default degli slider
            vals  = self._read_params()
            pos   = [vals["posX"], vals["posY"], vals["posZ"]]
            euler = [vals["eulerX"], vals["eulerY"], vals["eulerZ"]]
            quat  = p.getQuaternionFromEuler(euler)
            self._update_debug_visuals(pos, quat)
            p.stepSimulation()
            self._last_vals = vals

            slider_keys = ["eulerX", "eulerY", "eulerZ", "posX", "posY", "posZ"]

            while True:
                curr = self._read_params()

                predict_triggered = (self._last_vals["predict_btn"] != curr["predict_btn"])
                sliders_changed   = any(curr[k] != self._last_vals[k] for k in slider_keys)

                if sliders_changed:
                    pos   = [curr["posX"],   curr["posY"],   curr["posZ"]]
                    euler = [curr["eulerX"],  curr["eulerY"], curr["eulerZ"]]
                    quat  = p.getQuaternionFromEuler(euler)
                    self._update_debug_visuals(pos, quat)

                if predict_triggered:
                    pos   = [curr["posX"],   curr["posY"],   curr["posZ"]]
                    euler = [curr["eulerX"],  curr["eulerY"], curr["eulerZ"]]
                    quat  = p.getQuaternionFromEuler(euler)
                    self._run_ik(pos, quat)

                self._last_vals = curr
                p.stepSimulation()

        except KeyboardInterrupt:
            print(f"\n[{self.label}] Interrotto dall'utente.")
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            try:
                p.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Funzione target per multiprocessing.Process
# ---------------------------------------------------------------------------

def _arm_process_target(cfg: dict, model_path: str, model_dir: str):
    arm = ArmProcess(cfg, model_path, model_dir)
    arm.run()


# ---------------------------------------------------------------------------
# Argomenti CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="IK Inference dual-arm – processi separati per braccio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model_path", required=True,
                    help="Percorso al modello TF (condiviso da entrambi i processi)")
    ap.add_argument("--model_dir", default="ik_model",
                    help="Directory opzionale del modello (default: ik_model)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    processes = []

    for name, cfg in CONFIGS.items():
        urdf = cfg.get("urdf_path")
        if not urdf:
            print(f"[MAIN] Braccio '{name}': urdf_path non specificato, processo non avviato.")
            continue

        proc = multiprocessing.Process(
            target=_arm_process_target,
            args=(cfg, args.model_path, args.model_dir),
            name=f"arm_{name}",
            daemon=False,       # daemon=False: il main aspetta che finiscano
        )
        proc.start()
        print(f"[MAIN] Avviato processo per braccio '{name}' (PID {proc.pid})")
        processes.append(proc)

    if not processes:
        print("[MAIN] Nessun processo avviato. Verifica le configurazioni in CONFIGS.")
        sys.exit(0)

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt: termino i processi...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()

    print("[MAIN] Tutti i processi terminati.")