# Adds a several planes to the environment

import numpy as np
import pybullet as pb
import time
import pybullet_data
import os

physicsClient = pb.connect(pb.GUI)#or pb.DIRECT for non-graphical version
pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
pb.setGravity(0,0,-10)
planeId = pb.loadURDF("plane.urdf")
startPos = [0,0,1]
wall_orientation = pb.getQuaternionFromEuler([np.pi/2, 0, 0])
wall_id = pb.loadURDF("plane.urdf", basePosition=[0, 20, 0], baseOrientation=wall_orientation)
startOrientation = pb.getQuaternionFromEuler([0,0,0])
sphere_id = pb.loadURDF("sphere2.urdf",
    [0, 2, .5],
    pb.getQuaternionFromEuler([0,0,0]),
    globalScaling=0.0625,
    physicsClientId=physicsClient
)
pb.changeDynamics(sphere_id, -1, mass=0.1)
print("sphere id:", sphere_id)
first_time = True
drone_urdf = "cf2x.urdf"
pb.loadURDF(
    os.path.dirname(os.path.abspath(__file__))+"/../gym_pybullet_drones/assets/" + drone_urdf,
    [0, 0, 0],
    pb.getQuaternionFromEuler([0, 0, 0]),
    physicsClientId=physicsClient
)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    if first_time:
        pb.applyExternalForce(
            sphere_id,
            -1,
            forceObj=[0, 0, 50],
            posObj=[0, 0, 0],
            flags=pb.LINK_FRAME,
            physicsClientId=physicsClient
        )
        print("pushed the ball up?")
        first_time = False
    pb.stepSimulation()
    time.sleep(1./2000.)
# cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
pb.disconnect()
