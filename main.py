import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import os
import csv
from tqdm import tqdm

roboPos = [0,0,0]
THRESHOLD = 0.02  # max errore IK accettabile (in metri)
APPROACH_OFFSET = [0,0,0]  # offset rispetto al target

MAPSIZE = [0.4, 0.4, 0.2]      # dimensione del cubo di test (in metri)
MAPSTEPS = [6, 6, 5]         # quanti step di offset testare lungo ogni asse 
MAPOFFSET = [0, 0, 0.05]       # offset del centro del cubo rispetto alla base globale (in metri)

#ROTATION SAMPLES
XRANGE, XSAMPLES = math.pi/2, 100
YRANGE, YSAMPLES = math.pi/2, 1
ZRANGE, ZSAMPLES = math.pi*2, 100

## setup
p.connect(p.GUI)
p.resetSimulation()
p.setGravity(gravX=0, gravY=0, gravZ=0)
p.setAdditionalSearchPath(path=pybullet_data.getDataPath())

def render(sec=1):
    for _ in range(int(240*sec)):
        p.stepSimulation()
        time.sleep(1/240)

def quaternion_multiply(q1, q2):
    """q1 * q2 — composizione di quaternioni (x, y, z, w)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    )

def make_samples(n, half_range):
        if n == 1:
            return [0.0]
        return list(np.linspace(-half_range, half_range, n))

def local_axes_to_quaternion(q_ref, rx, ry, rz):
    """
    Offset combinato sui tre assi locali, indipendenti tra loro.
    
    La chiave: porta gli assi locali nel frame world, costruisci lì
    i quaternioni di rotazione, poi ricomponi.
    """
    # Matrici di rotazione da q_ref → ricava gli assi locali nel world frame
    R = quaternion_to_matrix(q_ref)
    x_local = R[:, 0]  # asse X dell'EE nel world
    y_local = R[:, 1]  # asse Y dell'EE nel world
    z_local = R[:, 2]  # asse Z dell'EE nel world

    # Costruisci i tre quaternioni di rotazione attorno agli assi world-space
    # (che corrispondono agli assi locali dell'EE)
    q_rx = axis_angle_to_quaternion(x_local, rx)
    q_ry = axis_angle_to_quaternion(y_local, ry)
    q_rz = axis_angle_to_quaternion(z_local, rz)

    # Applica le tre rotazioni a q_ref (ordine Z→Y→X)
    q_result = quaternion_multiply(q_rx,
               quaternion_multiply(q_ry,
               quaternion_multiply(q_rz, q_ref)))

    return q_result


def quaternion_to_matrix(q):
    """Converte quaternione (x,y,z,w) in matrice di rotazione 3x3"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def axis_angle_to_quaternion(axis, angle):
    """Quaternione da asse (vettore 3D normalizzato) + angolo in radianti"""
    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle / 2)
    return (ax[0]*s, ax[1]*s, ax[2]*s, math.cos(angle / 2))

def is_valid_pos(pos):
    return all(math.isfinite(v) for v in pos)

# --- CARICAMENTO SCENA ---
target = p.loadURDF("cube_small.urdf", basePosition=[1,0.75,0.1], useFixedBase=True, globalScaling=0.1)
p.changeVisualShape(objectUniqueId=target, linkIndex=-1, rgbaColor=[1,0,0,1]) 
#robo2 = p.loadURDF("franka_panda/panda.urdf", basePosition=roboPos, useFixedBase=True, globalScaling=1.0)
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "xarm_model/urdf/xarm_fixed.urdf")
print(f"Loading URDF from: {urdf_path}")
robo = p.loadURDF(urdf_path, basePosition=roboPos, useFixedBase=True, globalScaling=1.0)

# --- CONFIGURAZIONE ROBOT ---
# Rest pose "neutra" - usata come seed per l'IK ad ogni step
REST_POSE = [0, 0.5, -0.5, 0.5, 0] #[0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
arm_joints = [0, 1, 2, 3, 4] #[0, 1, 2, 3, 4, 5, 6]
EE_LINK_INDEX = 5 #(panda:11, xarm:5)

# Verifica i limiti di ogni giunto per configurare correttamente l'IK
# for j in arm_joints:
#     info = p.getJointInfo(robo, j)
#     name = info[1].decode()
#     lo = math.degrees(info[8])
#     hi = math.degrees(info[9])
#     print(f"  {name}: [{lo:.1f}°, {hi:.1f}°]")

# Limiti per l'IK: estratti direttamente dal robot per evitare errori di configurazione
lower_limits = []
upper_limits = []
joint_ranges = []

for j in arm_joints:
    info = p.getJointInfo(robo, j)
    lower_limits.append(info[8])
    upper_limits.append(info[9])
    joint_ranges.append(info[9] - info[8])

# Verifica che la rest pose sia entro i limiti di ogni giunto, altrimenti l'IK potrebbe fallire
# for i, j in enumerate(arm_joints):
#     info = p.getJointInfo(robo, j)
#     lo, hi = info[8], info[9]
#     r = REST_POSE[i]
#     ok = lo <= r <= hi
#     print(f"  j{i}: REST={math.degrees(r):.1f}° range=[{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°] {'OK' if ok else '*** FUORI RANGE ***'}")

# --- CSV SETUP ---
csv_path = os.path.join(script_dir, "dataset.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
header_row = [
    "target_x", "target_y", "target_z",
    "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"
]
for i in range(len(arm_joints)):
    header_row += [f"joint_{i}"]
csv_writer.writerow(header_row)

print("=== ALL JOINTS ===")
for i in range(p.getNumJoints(robo)):
    info = p.getJointInfo(robo, i)
    joint_name = info[1].decode()
    joint_type = info[2]  # 0=revolute, 1=prismatic, 2=spherical, 3=planar, 4=fixed
    link_name  = info[12].decode()
    type_str   = {0:"REVOLUTE", 1:"PRISMATIC", 4:"FIXED"}.get(joint_type, str(joint_type))
    print(f"  [{i}] {joint_name} ({type_str}) → link: {link_name}")

# Offset fisso di rotazione per allineare l'asse "lungo" del cilindro con Z locale
Q_CYLINDER_FIX = p.getQuaternionFromEuler([0,0,0])#[math.pi/2, 0, 0])

debug_cylinder_visual = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER,
    radius=0.015,
    length=0.10,
    rgbaColor=[0, 0.5, 1, 0.35],
    visualFramePosition=[0, 0, -0.05],        # cilindro esteso -10cm dal frame
    visualFrameOrientation=Q_CYLINDER_FIX     # fix asse Y→Z 
)
debug_cylinder = p.createMultiBody(
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
debug_tip = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=debug_tip_visual,
    basePosition=[0, 0, 0]
)
ear_offset = 0.015 # distanza laterale delle "orecchie" dal centro del cilindro

debug_ear_left_visual = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER,
    radius=0.006,
    length=0.01,
    rgbaColor=[1, 0.3, 0.0, 0.5],  # arancione
    visualFramePosition=[0, 0, 0],
    visualFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0])  # perpendicolare
)
debug_ear_left = p.createMultiBody(
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
debug_ear_right = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=debug_ear_right_visual,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1]
)

# --- Distance line debug ---
debug_line_id = None
debug_text_id = None

# --- PREPARAZIONE CAMPIONAMENTO ---
xSamples = make_samples(XSAMPLES, XRANGE/2)
ySamples = make_samples(YSAMPLES, YRANGE/2)
zSamples = make_samples(ZSAMPLES, ZRANGE/2)

iSteps = make_samples(MAPSTEPS[0], MAPSIZE[0]/2)
jSteps = make_samples(MAPSTEPS[1], MAPSIZE[1]/2)
kSteps = make_samples(MAPSTEPS[2], MAPSIZE[2]/2)

# --- VISUALIZZA PUNTI DI CAMPIONAMENTO ---
for i_map in iSteps:
    for j_map in jSteps:
        for k_map in kSteps:
            pos = list(np.array([i_map, j_map, k_map]) + np.array(MAPOFFSET))
            p.addUserDebugPoints(
                pointPositions=[pos],
                pointColorsRGB=[[1, 0.5, 0]],   # arancione
                pointSize=3,
                lifeTime=0
            )

# --- VISUALIZZA VOLUME DI CAMPIONAMENTO ---
half = np.array(MAPSIZE) / 2
center = np.array(MAPOFFSET)

volume_visual = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=half,
    rgbaColor=[1, 0.5, 0, 0.08]
)
volume_body = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=volume_visual,
    basePosition=center.tolist(),
    baseOrientation=[0, 0, 0, 1]
)

# Box wireframe
corners = [
    center + np.array([sx, sy, sz]) * half
    for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]
]
edges = [
    (0,1),(2,3),(4,5),(6,7),  # Z edges
    (0,2),(1,3),(4,6),(5,7),  # Y edges
    (0,4),(1,5),(2,6),(3,7)   # X edges
]
for a, b in edges:
    p.addUserDebugLine(corners[a].tolist(), corners[b].tolist(),
                       [1, 0.5, 0], lineWidth=1.5, lifeTime=0)
    
point_combinations = [(i,j,k) for i in iSteps for j in jSteps for k in kSteps]
    
#CAMPIONAMENTO
try:
    for i_map, j_map, k_map in tqdm(point_combinations, desc="Punti", unit="pt", bar_format="{l_bar}{bar:25}{r_bar}"):
        
        target_position = list(np.array([i_map, j_map, k_map]) + np.array(MAPOFFSET))  ##[i_map + MAPOFFSET[0], j_map + MAPOFFSET[1], k_map + MAPOFFSET[2]] 
        #print(f"\n--- Target position: ({i_map:.2f}, {j_map:.2f}, {k_map:.2f}) ---")
        p.resetBasePositionAndOrientation(target, target_position, [0, 0, 0, 1])
        desired_pos = list(np.array(target_position) + np.array(APPROACH_OFFSET))
        
        # ee_pos = p.getLinkState(robo2, 4)[4]
        # p.addUserDebugPoints([ee_pos], [[1,0,0]], pointSize=10, lifeTime=2)
        
        # Reset alla rest pose PRIMA di calcolare l'IK
        # così il seed è sempre consistente e l'IK non diverge
        for i, joint_id in enumerate(arm_joints):
            p.resetJointState(robo, joint_id, REST_POSE[i])

        ik_angles = p.calculateInverseKinematics(
            bodyUniqueId=robo,
            endEffectorLinkIndex=EE_LINK_INDEX,
            targetPosition=desired_pos,
            #non specifico targetOrientation così da ottenere la soluzione più naturale e usala come riferimento per i campionamenti di rotazione locali
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=REST_POSE,
            maxNumIterations=200,
            residualThreshold=1e-5
        )

        for joint_id in arm_joints:
            p.resetJointState(
                bodyUniqueId=robo,
                jointIndex=joint_id,
                targetValue=ik_angles[joint_id]
            )

        p.stepSimulation()

        ee_ref_orientation = p.getLinkState(robo, EE_LINK_INDEX)[5]
        #print(f"EE orientation (quaternion): {ee_ref_orientation}")

        for sampleX in xSamples: 
            for sampleY in ySamples:
                for sampleZ in zSamples:
                    #print(f"Trying offset = ({sampleX:.2f}, {sampleY:.2f}, {sampleZ:.2f})")

                    # --- Calcola orientamento target combinando gli offset di rotazione locali ---
                    q_target = local_axes_to_quaternion(ee_ref_orientation, sampleX, sampleY, sampleZ)

                    # --- DEBUG: visualizza orientamento target con cilindro e sfera ---
                    p.resetBasePositionAndOrientation(debug_cylinder, desired_pos, q_target)
                    tip_offset_local = np.array([0, 0, 0])
                    R = quaternion_to_matrix(q_target)
                    tip_offset_world = R @ tip_offset_local
                    tip_pos = [desired_pos[i] + tip_offset_world[i] for i in range(3)]
                    p.resetBasePositionAndOrientation(debug_tip, tip_pos, [0,0,0,1])

                    p.resetBasePositionAndOrientation(debug_cylinder, desired_pos, q_target)

                    # Calcola posizione orecchie nel world frame
                    R = quaternion_to_matrix(q_target)
                    x_axis = R[:, 0]  # asse X locale del cilindro nel world

                    # Le orecchie stanno all'inizio del cilindro 
                    z_axis = R[:, 2]
                    mid_cyl = np.array(desired_pos) #- z_axis * 0.01

                    ear_left_pos  = mid_cyl + x_axis * ear_offset
                    ear_right_pos = mid_cyl - x_axis * ear_offset

                    p.resetBasePositionAndOrientation(debug_ear_left,  ear_left_pos.tolist(),  q_target)
                    p.resetBasePositionAndOrientation(debug_ear_right, ear_right_pos.tolist(), q_target)

                    # --- fine debug orientamento target ---

                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # spegni renderer
                    
                    # Reset alla rest pose PRIMA di calcolare l'IK
                    # così il seed è sempre consistente e l'IK non diverge
                    for i, joint_id in enumerate(arm_joints):
                        p.resetJointState(robo, joint_id, REST_POSE[i])

                    ik_angles = p.calculateInverseKinematics(
                        bodyUniqueId=robo,
                        endEffectorLinkIndex=EE_LINK_INDEX,
                        targetPosition=desired_pos,
                        targetOrientation=q_target,
                        lowerLimits=lower_limits,
                        upperLimits=upper_limits,
                        jointRanges=joint_ranges,
                        restPoses=REST_POSE,
                        maxNumIterations=200,
                        residualThreshold=1e-5
                    )

                    for joint_id in arm_joints:
                        p.resetJointState(
                            bodyUniqueId=robo,
                            jointIndex=joint_id,
                            targetValue=ik_angles[joint_id]
                        )

                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # riaccendi renderer
                    p.stepSimulation()

                    # --- Verifica IK ---
                    ee_state = p.getLinkState(robo, EE_LINK_INDEX)
                    ee_pos = ee_state[4]  # punta dell'EEF

                    ik_error = math.sqrt(sum((ee_pos[i] - desired_pos[i])**2 for i in range(3)))
                    dist_to_cube = math.sqrt(sum((ee_pos[i] - target_position[i])**2 for i in range(3)))

                    status = "OK" if ik_error < THRESHOLD else "FAIL"

                    # Punto medio tra EEF e target — dove mettere il testo
                    mid_pos = [(ee_pos[i] + target_position[i]) / 2 for i in range(3)]
                    mid_pos[2] += 0.05  # solleva un po' il testo per non sovrapporsi alla linea

                    label = f"{dist_to_cube*100:.1f} cm [{status}]"

                    # Colore: verde se OK, rosso se FAIL
                    color = [0, 1, 0] if ik_error < THRESHOLD else [1, 0, 0]

                    # # Aggiorna linea (replaceItemUniqueId evita di accumulare debug line ad ogni iterazione)
                    # if is_valid_pos(ee_pos) and is_valid_pos(target_position) and is_valid_pos(mid_pos):
                    #     if debug_text_id is None:
                    #         #debug_line_id = p.addUserDebugLine(ee_pos, target_position, color, lineWidth=2, lifeTime=0.01)
                    #         debug_text_id = p.addUserDebugText(label, mid_pos, color, textSize=1.2, lifeTime=0)
                    #     else:
                    #         #debug_line_id = p.addUserDebugLine(ee_pos, target_position, color, lineWidth=2, lifeTime=0.01, replaceItemUniqueId=debug_line_id)
                    #         debug_text_id = p.addUserDebugText(label, mid_pos, color, textSize=1.2, lifeTime=0, replaceItemUniqueId=debug_text_id)
                    # else:
                    #     print(f"Skipping debug draw — invalid pos: ee={ee_pos} target={target_position}")
                    
                    # --- Salva su CSV solo se IK OK ---
                    if ik_error < THRESHOLD:
                        csv_writer.writerow([
                            *desired_pos,                        # x, y, z target
                            *q_target,                           # qx, qy, qz, qw orientamento
                            *[ik_angles[j] for j in arm_joints]  # j0..j4
                        ])
                # time.sleep(0.01)  # rallenta un po' per vedere meglio i debug (opzionale)        
except KeyboardInterrupt:
    print("Interrotto dall'utente.")
except Exception as e:
    print(f"Errore: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Chiusura...")
    csv_file.close()
    if p.isConnected():
        p.disconnect()
    print("Fatto.")