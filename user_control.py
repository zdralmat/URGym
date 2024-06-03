import os

import numpy as np

from tqdm import tqdm
from urgym.envs.env_box_v0 import BoxManipulation
import time
import math
import pybullet as p

def test_demo():
    env = BoxManipulation()
    # Show right and left menus
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    while True:
        x, y, z, roll, pitch, yaw, gripper_opening_length = env.read_debug_parameter()
        qx,qy,qz,qw = p.getQuaternionFromEuler((roll, pitch, yaw))
        env.test()
        print(env.robot.get_ee_pose()[3:])

def user_control_demo():
    env = BoxManipulation()
    # Show right and left menus
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    while True:
        x, y, z, roll, pitch, yaw, gripper_opening_length = env.read_debug_parameter()
        qx,qy,qz,qw = p.getQuaternionFromEuler((roll, pitch, yaw))
        action = np.array([x, y, z, qx, qy, qz, qw, gripper_opening_length])
        obs, reward, terminated, truncated, info = env.step(action)
        print(qx, qy, qz, qw)



if __name__ == '__main__':
    # user_control_demo()
    test_demo()
