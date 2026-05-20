import pybullet as p
import pybullet_data
import pandas as pd
import os
import random
import math
import numpy as np
import socket
import json

from test_csv import createTargetDebug, updateTargetDebug

ROBOPOS = [0,0,0]
MAPSIZE = [2, 2, 2] 
MAPOFFSET = [0,0,0.05]
#Robot XARM
ARM_JOINTS = [0, 1, 2, 3, 4]
EE_LINK_INDEX = 5 
REST_POSE = [0, 0.5, -0.5, 0.5, 0]

#Robot panda
# ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
# EE_LINK_INDEX = 11  # link dell'end effector del Panda
# REST_POSE = [0, -0.215, 0, -2.57, 0, 2.356, 2.356]  # posa a riposo standard Panda

_target = None
_robot = None
_debug_cylinder = None
_debug_tip = None
_debug_ear_left = None
_debug_ear_right = None
_ear_offset = 0.015 # distanza laterale delle "orecchie" di debug dal centro del cilindro
_posX, _posY, _posZ = None, None, None
def setup():
    print("Setting up simulation...")

    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    global _posX, _posY, _posZ
    _posX = p.addUserDebugParameter("Position X", -MAPSIZE[0]/2, MAPSIZE[0]/2, 0)
    _posY = p.addUserDebugParameter("Position Y", -MAPSIZE[1]/2, MAPSIZE[1]/2, 0)
    _posZ = p.addUserDebugParameter("Position Z", -MAPSIZE[2]/2, MAPSIZE[2]/2, 0)

    global _target
    global _robot
    _target = p.loadURDF("cube_small.urdf", basePosition=[1,0.75,0.1], useFixedBase=True, globalScaling=1.0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "xarm_model/urdf/xarm_fixed.urdf")
    _robot = p.loadURDF(urdf_path, basePosition=ROBOPOS, useFixedBase=True, globalScaling=1.0)
    #_robot = p.loadURDF("franka_panda/panda.urdf", basePosition=ROBOPOS, useFixedBase=True, globalScaling=1.0)
    #createTargetDebug()



if __name__ == "__main__":
    try:
        setup()
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        for j in ARM_JOINTS:
            info = p.getJointInfo(_robot, j)
            lower_limits.append(info[8])
            upper_limits.append(info[9])
            joint_ranges.append(info[9] - info[8])
        while True:
            _posX_val = p.readUserDebugParameter(_posX)
            _posY_val = p.readUserDebugParameter(_posY)
            _posZ_val = p.readUserDebugParameter(_posZ)
            p.resetBasePositionAndOrientation(_target, [_posX_val, _posY_val, _posZ_val], [0,0,0,1])
            ik_angles = p.calculateInverseKinematics(
                        bodyUniqueId=_robot,
                        endEffectorLinkIndex=EE_LINK_INDEX,
                        targetPosition=p.getBasePositionAndOrientation(_target)[0],
                        #targetOrientation=p.getBasePositionAndOrientation(_target)[1],
                        lowerLimits=lower_limits,
                        upperLimits=upper_limits,
                        jointRanges=joint_ranges,
                        restPoses=REST_POSE,
                        maxNumIterations=200,
                        residualThreshold=1e-5
                    )
            for joint_id in ARM_JOINTS:
                p.resetJointState(
                    bodyUniqueId=_robot,
                    jointIndex=joint_id,
                    targetValue=ik_angles[joint_id]
                )
            p.stepSimulation() 
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:       
        p.disconnect()  