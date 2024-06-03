import time
import math
import numpy as np
from typing import Optional
import os

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.envs.registration import register

import pybullet as p
import pybullet_data
from urgym.utilities import YCBModels, Models, Camera
from urgym.robot import Panda, UR5Robotiq85, UR5Robotiq140

class BoxManipulation(Env):

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, button_touch_mode:str='any', camera=None, render_mode='human') -> None:
            """
            Initialize the environment.

            Parameters:
            - button_touch_mode (str): The mode, one of 'any', 'robot' or 'fingers', to determine the valid element that may press the button.
            - camera (optional): The camera object.
            - render_mode (str): The rendering mode of the environment, either 'human' or None.

            Returns:
            None
            """
            if render_mode == 'human':
                self.vis = True
            else:
                self.vis = False

            self.button_touch_mode = button_touch_mode

            # Set observation and action spaces
            # Observations: end-effector pose (x,y,z, qx, qy, qz, qw)
            self.observation_space = Box(low=np.array([-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi, -math.pi]), high=np.array([1.0, 1.0, 1.0, math.pi, math.pi, math.pi, math.pi]), dtype=np.float64)
            # Actions: (x, y, z, qx, qy, qz, qw, gripper_opening_length [0, 0.085]) for End Effector Position Control
            self.action_space = Box(low=np.array([-0.05]*3 + [-math.pi/10]*4 + [0.0]), high=np.array([+0.05]*3 + [+math.pi/10]*4 + [0.085]), dtype=np.float32)

            current_dir = os.path.dirname(__file__)
            ycb_models = YCBModels(
                os.path.join(current_dir, './data/ycb', '**', 'textured-decmp.obj'),
            )
            """camera = Camera((1, 1, 1),
                            (0, 0, 0),
                            (0, 0, 1),
                            0.1, 5, (320, 320), 40)"""
            self.camera = camera
            # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
            self.robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))

            # define environment        
            self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -10)
            # Hide right and left menus
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            # Reorient the debug camera
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[-0.5,+0.5,0])
            self.planeID = p.loadURDF("plane.urdf")

            self.robot.load()
            self.robot.step_simulation = self.step_simulation
            self.control_method = 'end'

            # custom sliders to tune parameters (name of the parameter,range,initial value)
            self.xin = p.addUserDebugParameter("x", -1, 1, 0)
            self.yin = p.addUserDebugParameter("y", -1, 1, 0)
            self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
            self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
            self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
            self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
            self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

            self.boxID = p.loadURDF(os.path.join(current_dir, "../urdf/skew-box-button.urdf"),
                                    [0.0, 0.0, 0.0],
                                    # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                    p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

            # For calculating the reward
            self.box_opened = False
            self.button_pressed = False
            self.box_closed = False


    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)


    def test(self):
        self.robot.move_ee(self.read_debug_parameter()[:6], "joint")
        self.step_simulation()

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action) -> tuple[list, float, bool, bool, dict]:
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        new_pose = self.robot.get_ee_pose() + action[:-1]
        self.robot.move_ee(new_pose, self.control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()
        terminated = self.box_closed
        self.episode_steps += 1
        truncated = False # Managed by the environment automatically 
        #info = dict(box_opened=self.box_opened, button_pressed=self.button_pressed, box_closed=self.box_closed)
        info = dict(is_success=self.box_closed)

        reward -= 0.1 # Step penalty
        return self.get_observation(), reward, terminated, truncated, info

    def update_reward(self):
        reward = 0
        if not self.box_opened:
            if p.getJointState(self.boxID, 1)[0] > 1.9:
                self.box_opened = True
                print('Box opened!')
                reward = 1
        elif not self.button_pressed:
            if self.valid_button_press():
                self.button_pressed = True
                print('Button pressed!')
                reward = 1
        else:
            # If it was opened previously and now closed
            if self.box_opened and p.getJointState(self.boxID, 1)[0] < 0.1:
                print('Box closed!')
                self.box_closed = True
                reward = 1
        return reward

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return np.array(obs["ee_pos"])

    def reset_box(self):
        p.resetJointState(self.boxID, 0, targetValue=0)
        p.resetJointState(self.boxID, 1, targetValue=0)
        p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=1)
        p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)
        self.box_opened = self.box_closed = self.button_pressed = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.reset()
        ee_pose = self.robot.get_ee_pose()
        self.robot.move_ee(ee_pose, self.control_method)
        self.reset_box()
        self.episode_steps = 0
        return self.get_observation(), {}

    def close(self):
        p.disconnect(self.physicsClient)

    def valid_button_press(self):
        # First check if the button is down
        if p.getJointState(self.boxID, 0)[0] < -0.02:
            if self.button_touch_mode == 'any':
                return True  # Button down and it doesn't matter which element pressed it
            else:
                button_index = 0
                contact_points = p.getContactPoints()
                if self.button_touch_mode == 'robot':
                    # Filter contact points to find those involving the button of box and any part of the robot
                    contact_with_links = any(
                        point[1] == self.boxID and point[3] == button_index and point[2] == self.robot.id or
                        point[1] == self.robot.id and point[2] == self.boxID and point[4] == button_index
                        for point in contact_points
                    )
                elif self.button_touch_mode == 'fingers':
                    # Filter contact points to find those involving the button of the box and the set of fingers links of the robot
                    fingers_indices = list(range(9, 19))
                    contact_with_links = any(
                        point[1] == self.boxID and point[3] == button_index and point[2] == self.robot.id and point[4] in fingers_indices or
                        point[2] == self.boxID and point[4] == button_index and point[1] == self.robot.id and point[3] in fingers_indices
                        for point in contact_points
                    )
                if contact_with_links:
                    return True
        return False
