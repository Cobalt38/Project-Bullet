import pybullet as p
import pybullet_data
import pandas as pd
import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

URDF_PATH    = "biarm_model/openarm_right.urdf"
ARM_JOINTS    = [i for i in range(2, 9)]   # openarm_joint2 … openarm_joint8
EE_LINK_INDEX = 10
ROBOPOS      = [0, 0, 0]
DEFAULT_JOINTS = [32.0, 30.0, 0.0, 60.0, 20.0, -16.0, 52.0]

_robot = None

_joint1 = _joint2 = _joint3 = _joint4 = _joint5 = _joint6 = _joint7 = None

palla1 = palla2 = palla3 = None

calc_dist = None
get_pose = None
apply_default = None

lower_limits = []
upper_limits = []
joint_ranges = []

def calculate_distance():
    global palla1, palla2
    ee_pos = p.getLinkState(_robot, EE_LINK_INDEX)[4]
    first_joint_pos = p.getLinkState(_robot, ARM_JOINTS[1])[4]
    dist = math.dist(ee_pos, first_joint_pos)
    print(f"Distance from EE to first joint: {dist:.3f} m")
    palla1 = p.addUserDebugPoints(pointPositions=[ee_pos], pointColorsRGB=[[1, 0, 0]], pointSize=50, lifeTime=1)
    palla2 = p.addUserDebugPoints(pointPositions=[first_joint_pos], pointColorsRGB=[[0, 1, 0]], pointSize=50, lifeTime=1)

def get_ee_pose(robot_id, ee_link_index):
    state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
    print(f"[CIAO]: {state[0] + state[1]}")
    pos  = state[4]   # (x, y, z)
    quat = state[5]   # (x, y, z, w)
    euler = p.getEulerFromQuaternion(quat)   # (roll, pitch, yaw) in radianti
    quat2 = p.getQuaternionFromEuler(euler)
    print(f"PYB quat1: {quat}\nPYB quat2: {quat2}")
    # eulerScipy = R.from_quat(quat).as_euler("XYZ")
    # print(f"EULERI SCIPY: {eulerScipy}\nEULERI PYBULLET: {euler}")
    quatScipy = R.from_euler("xyz", euler).as_quat() #uguale a p.getQuaternionFromEuler(euler)
    print(f"\nQUAT SCIPY: {quatScipy}\nQUAT PYBULLET: {quat}")
    print(f"[EE POS]   x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
    print(f"[EE QUAT]  x={quat[0]:.4f}  y={quat[1]:.4f}  z={quat[2]:.4f}  w={quat[3]:.4f}")
    print(f"[EE EULER DEG] r={math.degrees(euler[0]):.2f}°  p={math.degrees(euler[1]):.2f}°  y={math.degrees(euler[2]):.2f}°")
    print(f"[EE EULER RAD] r={euler[0]:.2f}-r  p={euler[1]:.2f}-r  y={euler[2]:.2f}-r")
    return pos, quat, euler

def diagnose_robot():
    """Stampa tutti i joint del robot per trovare gli indici corretti."""
    print(f"\n=== JOINT MAP ({p.getNumJoints(_robot)} joints totali) ===")
    for i in range(p.getNumJoints(_robot)):
        info = p.getJointInfo(_robot, i)
        jtype = {0:"REVOLUTE", 1:"PRISMATIC", 4:"FIXED"}.get(info[2], str(info[2]))
        print(f"  [{i:2d}] {jtype:<10} name={info[1].decode():<40} parent={info[12].decode()}")
    print("===\n")

def setup():
    print("Setting up simulation...")
    global _robot, _joint1, _joint2, _joint3, _joint4, _joint5, _joint6, _joint7, lower_limits, upper_limits, joint_ranges, calc_dist, get_pose, apply_default, palla1, palla2, palla3
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)


    p.loadURDF("plane.urdf")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_abs   = os.path.join(script_dir, URDF_PATH)
    _robot = p.loadURDF(urdf_abs, basePosition=ROBOPOS,
                        useFixedBase=True, globalScaling=1.0)
    for i in range(-1, p.getNumJoints(_robot)):
        p.setCollisionFilterGroupMask(_robot, i, collisionFilterGroup=0, collisionFilterMask=0)
    for j in ARM_JOINTS:
        info = p.getJointInfo(_robot, j)
        lower_limits.append(info[8])
        upper_limits.append(info[9])
        joint_ranges.append(info[9] - info[8])
    calc_dist = p.addUserDebugParameter("Calculate distance", 1, 0, 0)
    get_pose = p.addUserDebugParameter("Get pose", 1, 0, 0)
    apply_default = p.addUserDebugParameter("Default", 1, 0, 0)
    _joint1 = p.addUserDebugParameter(f"Joint 1", math.degrees(lower_limits[0]), math.degrees(upper_limits[0]), 0)
    _joint2 = p.addUserDebugParameter(f"Joint 2", math.degrees(lower_limits[1]), math.degrees(upper_limits[1]), 90)
    _joint3 = p.addUserDebugParameter(f"Joint 3", math.degrees(lower_limits[2]), math.degrees(upper_limits[2]), 0)
    _joint4 = p.addUserDebugParameter(f"Joint 4", math.degrees(lower_limits[3]), math.degrees(upper_limits[3]), 0)
    _joint5 = p.addUserDebugParameter(f"Joint 5", math.degrees(lower_limits[4]), math.degrees(upper_limits[4]), 0)
    _joint6 = p.addUserDebugParameter(f"Joint 6", math.degrees(lower_limits[5]), math.degrees(upper_limits[5]), 0)
    _joint7 = p.addUserDebugParameter(f"Joint 7", math.degrees(lower_limits[6]), math.degrees(upper_limits[6]), 0)

    palla1 = p.addUserDebugPoints(pointPositions=[(0, 0, 0)], pointColorsRGB=[[1, 0, 0]], pointSize=10, lifeTime=0)
    palla2 = p.addUserDebugPoints(pointPositions=[(0, 0, 0)], pointColorsRGB=[[0, 1, 0]], pointSize=10, lifeTime=0)
    palla3 = p.addUserDebugPoints(pointPositions=[(0, 0, 0)], pointColorsRGB=[[0, 0, 1]], pointSize=10, lifeTime=0)

if __name__ == "__main__":
    try:
        setup()
        diagnose_robot()
        print(p.getLinkState(_robot, EE_LINK_INDEX)[5])
        last_calc = p.readUserDebugParameter(calc_dist)
        last_pose = p.readUserDebugParameter(get_pose)
        last_apply = p.readUserDebugParameter(apply_default)
        while True:
            new_calc = p.readUserDebugParameter(calc_dist)
            new_pose = p.readUserDebugParameter(get_pose)
            new_apply = p.readUserDebugParameter(apply_default)
            if new_calc != last_calc:
                last_calc = new_calc
                calculate_distance()
            targetJoints=[
                p.readUserDebugParameter(_joint1),
                p.readUserDebugParameter(_joint2),
                p.readUserDebugParameter(_joint3),
                p.readUserDebugParameter(_joint4),
                p.readUserDebugParameter(_joint5),
                p.readUserDebugParameter(_joint6),
                p.readUserDebugParameter(_joint7)
            ]
            if new_apply != last_apply:
                targetJoints=DEFAULT_JOINTS
            if new_pose != last_pose:
                last_pose = new_pose
                print(f"[CURRENT JOINTS]: {targetJoints}")
                get_ee_pose(_robot, EE_LINK_INDEX)
            for joint_idx, val in zip(ARM_JOINTS, targetJoints):
                p.resetJointState(_robot, joint_idx, math.radians(float(val)))
            p.stepSimulation()
    except KeyboardInterrupt:
        print("Exiting simulation...")
    finally:
        p.disconnect()