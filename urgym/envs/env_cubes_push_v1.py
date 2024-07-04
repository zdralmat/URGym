from copy import deepcopy
from sre_compile import dis
import time
import math
from turtle import position
import numpy as np
from typing import Optional
import os
import random

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.envs.registration import register

import pybullet as p
import pybullet_data
from urgym.utilities import YCBModels, Models, Camera, rotate_quaternion, geometric_distance_reward, print_link_names_and_indices
from urgym.robot import Panda, UR5Robotiq85, UR5Robotiq140

class CubesPush(Env):

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, camera=None, reward_type='dense',  render_mode='human') -> None:

        self.reward_type = reward_type

        if render_mode == 'human':
            self.visualize = True
        else:
            self.visualize = False

        # Set observation and action spaces
        # Observations: the end-effector position
        # And position (x, y, z) of the two cubes
        self.observation_space = Box(low=np.array([-1.0]*3 + [-1.0]*6), high=np.array([1.0]*3 + [1.0]*6), dtype=np.float64)
        # Actions: joints 1 to 6
        #self.action_space = Box(low=np.array([-math.pi]*6), high=np.array([+math.pi]*6), dtype=np.float32)
        self.action_space = Box(low=np.array([-0.1]*3), high=np.array([+0.1]*3), dtype=np.float32)

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
        self.physicsClient = p.connect(p.GUI if self.visualize else p.DIRECT)
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
        if self.visualize:
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

    def wait_until_stable(self, sim_steps=480):
        pos = self.robot.get_joint_obs()['positions']
        for _ in range(sim_steps):
            self.step_simulation()
            new_pos = self.robot.get_joint_obs()['positions']
            if np.sum(np.abs(np.array(pos)-np.array(new_pos))) < 5e-3: # Threshold based on experience
                return True
            pos = new_pos
        print("Warning: The robot configuration did not stabilize")
        return False

    def step(self, action):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """

        reward = 0
        
        # Differential version
        ee_pose = list(self.robot.get_ee_pose())
        ee_pose[:3] += action
        self.robot.move_ee(ee_pose, self.control_method)

        self.wait_until_stable()
                
        if self.is_floor_collision():
            reward -= 1
            success = False
            terminated = False
        elif self.is_robot_touching_cube(self.cubes[0]) or self.is_robot_touching_cube(self.cubes[1]):
            reward, success = self.update_reward()
            reward += 0.1 # Reward for touching the cube
            terminated = success
        else:
            success = False
            terminated = False

        info = {"is_success": success}

        if success:
            print("SUCCESS!")

        truncated = False # Managed by the environment automatically

        reward -= 0.1 # Step penalty

        #print(f"Reward: {reward:.2f}")

        return self.get_observation(), reward, terminated, truncated, info

    def update_reward(self) -> tuple[float, bool]:
        reward = 0

        if self.reward_type == 'dense':
            # Dense reward: the distance between the cubes
            cube1_pos = self.get_cube_pose(self.cubes[0])[:3]
            cube2_pos = self.get_cube_pose(self.cubes[1])[:3]
            distance = np.linalg.norm(np.array(cube1_pos) - np.array(cube2_pos))

            reward += geometric_distance_reward(distance, 0.1, 0.5) / 10

        # Sparse reward: if the distance between the cubes is less than 0.05
        if self.are_cubes_close(self.cubes[0], self.cubes[1], 0.05):
            reward += 10
            sucess = True
        else:
            sucess = False

        return reward, sucess
    
    def is_floor_collision(self):
        contact_points = p.getContactPoints(bodyA=self.robot.id, bodyB=self.planeID)
        if contact_points:
            for point in contact_points:
                if point[3] != 0: # 0 is the base link, is always touching the floor
                    print("Collision with the floor!")
                    return True
        return False

    def is_robot_touching_cube(self, cube_id):
        contact_points = p.getContactPoints(bodyA=self.robot.id, bodyB=cube_id)
        if contact_points:
            return True
        return False

    def are_cubes_close(self, cube1_id, cube2_id, threshold=0.05):
        cube1_pos = self.get_cube_pose(cube1_id)[:3]
        cube2_pos = self.get_cube_pose(cube2_id)[:3]
        distance = np.linalg.norm(np.array(cube1_pos) - np.array(cube2_pos))
        return distance < threshold
    
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
        cube1_position = self.get_cube_pose(self.cubes[0])[:3]
        cube2_position = self.get_cube_pose(self.cubes[1])[:3]
        obs = np.append(obs, [cube1_position, cube2_position])
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.reset()
        self.create_cubes()

        # Position the end effector above the cube
        new_pose = list(self.get_cube_pose(self.cubes[0]))
        new_pose[2] += 0.2 # A little bit higher
        #new_pose[3:] = rotate_quaternion(new_pose[3:], math.pi/2, [1, 0, 0]) # Reorient the end effector
        new_pose[3:] = rotate_quaternion(new_pose[3:], math.pi/2, [0, 1, 0]) # Reorient the end effector
        self.robot.move_ee(new_pose, 'end')
        self.robot.close_gripper()
        self.wait_until_stable()

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
        """coors = [random.uniform(-0.3, -0.1) + (0.4 if random.random() > 0.5 else 0) for _ in range(4)]

        self.create_cube(coors[0], coors[1], 0.2)
        self.create_cube(coors[2], coors[3], 0.2, [1,0,0,1]) """
        self.create_cube(0.1, -0.1, 0.1)
        self.create_cube(0.1, -0.2, 0.1, [1,0,0,1])

    def get_cube_pose(self, cube_id):
        position, orientation = p.getBasePositionAndOrientation(cube_id)
        pose = position + orientation
        return pose
    
