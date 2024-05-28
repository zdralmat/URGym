from sre_compile import dis
import time
import math
from turtle import position
import numpy as np
from typing import Optional
import os

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.envs.registration import register

import pybullet as p
import pybullet_data
from ur5.utilities import YCBModels, Models, Camera, rotate_quaternion, geometric_distance_reward
from ur5.robot import Panda, UR5Robotiq85, UR5Robotiq140

class CubesManipulation(Env):

    SIMULATION_STEP_DELAY = 1 / 960.
    CUBE_RAISE_HEIGHT = 0.2

    def __init__(self, camera=None, render_mode='human') -> None:
        if render_mode == 'human':
            self.vis = True
        else:
            self.vis = False

        # Set observation and action spaces
        # Observations: the end-effector position and quaternion (x, y, z, qx, qy, qz, qw)
        # And one of the cubes position and quaternion (x, y, z, qx, qy, qz, qw)
        self.observation_space = Box(low=np.array([-1.0]*3 + [-math.pi]*4 + [-1.0]*3 + [-math.pi]*4), high=np.array([1.0]*3 + [math.pi]*4 + [1.0]*3 + [math.pi]*4), dtype=np.float64)
        # Actions: prob1,prob2 (x, y, z, qx, qy, qz, qw, mode [0,1]). Mode 0 means sequence open_gripper, move, close_gripper. Model 1 means sequence close_gripper, move, open_gripper
        self.action_space = Box(low=np.array([0]*2 + [-0.05]*3 + [-math.pi/10]*4 + [0]), high=np.array([1]*2 + [+0.05]*3 + [+math.pi/10]*4 + [1]), dtype=np.float32)

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
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        self.cubes = []


    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)

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

    def wait_simulation_steps(self, sim_steps):
        for _ in range(sim_steps):
            self.step_simulation()

    def step(self, action):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """

        if action[0] >= action[1]:
            # Move the end effector
            new_pose = self.robot.get_ee_pose() + action[2:-1]
            self.robot.move_ee(new_pose, self.control_method)
        else:
            # Move the gripper
            if action[-1] < 0.5:
                self.robot.open_gripper()
            else:
                self.robot.close_gripper()
 
        self.wait_simulation_steps(120)
            
        reward = self.update_reward()
        
        info = {"is_success": False}

        if self.on_top(self.cubes[1], self.cubes[0]):
            reward += 5
            terminated = True
            info["is_success"] = True
        else:
            terminated = False

        self.episode_steps += 1

        truncated = False # Managed by the environment automatically

        reward -= 0.1 # Step penalty
        return self.get_observation(), reward, terminated, truncated, info

    def update_reward(self):
        reward = 0

        # For getting close to the cube
        ee_pos = self.robot.get_ee_pose()[:3]
        cube_pos = self.get_cube_pose(self.cubes[0])[:3]

        distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        distance_reward = geometric_distance_reward(distance, 0.2, 0.5)

        reward += distance_reward

        for id in self.cubes:
            pose = p.getBasePositionAndOrientation(id)
            position = pose[0]
            if position[2] > self.CUBE_RAISE_HEIGHT: # If a cube is raised
                reward += 1
        return reward
    
    def on_top(self, lower_cube_id, upper_cube_id):
        lower_cube_pos, _ = p.getBasePositionAndOrientation(lower_cube_id)
        upper_cube_pos, _ = p.getBasePositionAndOrientation(upper_cube_id)
        contact_points = p.getContactPoints(lower_cube_id, upper_cube_id)
        # Check if the upper cube is placed on top of the lower cube
        if upper_cube_pos[2] > lower_cube_pos[2] and contact_points:
            print(f"Cube {upper_cube_id} is placed on top of cube {lower_cube_id} and in contact")
            return True
        return False

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None

        obs.update(self.robot.get_joint_obs())

        obs = np.array(obs["ee_pos"])
        obs = np.append(obs, self.get_cube_pose(self.cubes[0]))
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.reset()
        self.create_cubes()

        # Position the end effector above the cube
        new_pose = list(self.get_cube_pose(self.cubes[0]))
        new_pose[2] += 0.3 # A little bit higher
        new_pose[3:] = rotate_quaternion(new_pose[3:], math.pi/2, [1, 0, 0]) # Reorient the end effector
        new_pose[3:] = rotate_quaternion(new_pose[3:], math.pi/2, [0, 1, 0]) # Reorient the end effector
        self.robot.move_ee(new_pose, self.control_method)
        self.robot.open_gripper()
        self.wait_simulation_steps(120)

        self.episode_steps = 0
        return self.get_observation(), {}

    def close(self):
        p.disconnect(self.physicsClient)

    def create_cube(self, x: float, y: float, z: float, color:list=None):
        id = p.loadURDF("cube.urdf", [x, y, z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling = 0.04)
        if color != None:
            p.changeVisualShape(id, -1, rgbaColor=color)
        self.cubes.append(id)

    def create_cubes(self):
        for id in self.cubes:
            p.removeBody(id)
        self.cubes = []
        self.create_cube(0.0, -0.1, 0.1)
        self.create_cube(0.0, -0.2, 0.1, [1,0,0,1])
        #self.create_cube(0.1, -0.1, 0.1, [0,0,1,1])
        #self.create_cube(0.1, -0.2, 0.1, [0,1,0,1])

    def get_cube_pose(self, cube_id):
        position, orientation = p.getBasePositionAndOrientation(cube_id)
        pose = position + orientation
        return pose
    
# Register the environment
register(
    id='ur5/cubes_test',
    entry_point='ur5.envs.env_cubes_test:CubesManipulation',
    max_episode_steps=50,
)