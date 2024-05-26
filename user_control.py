import os

import numpy as np

from tqdm import tqdm
from ur5.envs.env_box import BoxManipulation
import time
import math
import pybullet as p


def user_control_demo():
    env = BoxManipulation()
    # Show right and left menus
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.read_debug_parameter(), 'end')



if __name__ == '__main__':
    user_control_demo()
