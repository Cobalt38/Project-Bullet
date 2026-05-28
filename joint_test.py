import pybullet as p
import pybullet_data
import pandas as pd
import os
import math
import numpy as np

URDF_PATH    = "biarm_model/openarm_right.urdf"
ARM_JOINTS    = [i for i in range(2, 9)]   # openarm_joint2 … openarm_joint8
EE_LINK_INDEX = 9
# ARM_JOINTS            = [12, 13, 14, 15, 16, 17, 18] 
# EE_LINK_INDEX         = 18 
ROBOPOS      = [0, 0, 0]
MAPSIZE      = [0.8, 0.8, 0.6]
MAPOFFSET    = [0, 0, 0.3]
RIGHT_BASE_TRANSLATION = np.array([0.0, -0.031, 0.698])  # xyz del giunto di montaggio
RIGHT_BASE_RPY         = [1.5708, 0.0, 0.0]              # rpy del giunto di montaggio

_robot = None

_joint1 = _joint2 = _joint3 = _joint4 = _joint5 = _joint6 = _joint7 = None

calc_dist = None

lower_limits = []
upper_limits = []
joint_ranges = []

def calculate_distance():
    ee_pos = p.getLinkState(_robot, EE_LINK_INDEX)[4]
    first_joint_pos = p.getLinkState(_robot, ARM_JOINTS[0])[4]
    dist = math.dist(ee_pos, first_joint_pos)
    print(f"Distance from EE to first joint: {dist:.3f} m")


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
    global _robot, _joint1, _joint2, _joint3, _joint4, _joint5, _joint6, _joint7, lower_limits, upper_limits, joint_ranges, calc_dist
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
    _joint1 = p.addUserDebugParameter(f"Joint 1 - {lower_limits[0]} to {upper_limits[0]}", lower_limits[0], upper_limits[0], 0)
    _joint2 = p.addUserDebugParameter(f"Joint 2 - {lower_limits[1]} to {upper_limits[1]}", lower_limits[1], upper_limits[1], 0)
    _joint3 = p.addUserDebugParameter(f"Joint 3 - {lower_limits[2]} to {upper_limits[2]}", lower_limits[2], upper_limits[2], 0)
    _joint4 = p.addUserDebugParameter(f"Joint 4 - {lower_limits[3]} to {upper_limits[3]}", lower_limits[3], upper_limits[3], 0)
    _joint5 = p.addUserDebugParameter(f"Joint 5 - {lower_limits[4]} to {upper_limits[4]}", lower_limits[4], upper_limits[4], 0)
    _joint6 = p.addUserDebugParameter(f"Joint 6 - {lower_limits[5]} to {upper_limits[5]}", lower_limits[5], upper_limits[5], 0)
    _joint7 = p.addUserDebugParameter(f"Joint 7 - {lower_limits[6]} to {upper_limits[6]}", lower_limits[6], upper_limits[6], 0)

if __name__ == "__main__":
    try:
        setup()
        diagnose_robot()
        print(p.getLinkState(_robot, EE_LINK_INDEX)[5])
        last_calc = p.readUserDebugParameter(calc_dist)
        while True:
            new_calc = p.readUserDebugParameter(calc_dist)
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
            for joint_idx, val in zip(ARM_JOINTS, targetJoints):
                p.resetJointState(_robot, joint_idx, float(val))
            p.stepSimulation()
    except KeyboardInterrupt:
        print("Exiting simulation...")
    finally:
        p.disconnect()