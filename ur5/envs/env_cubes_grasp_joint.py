import time
import math
import numpy as np
from typing import Optional
import os

from gymnasium import Env
from gymnasium.spaces import Box

import pybullet as p
import pybullet_data
from ur5.utilities import YCBModels, Camera, rotate_quaternion, geometric_distance_reward
from ur5.robot import UR5Robotiq85
import random

class CubesGrasp(Env):

    SIMULATION_STEP_DELAY = 1 / 240.
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
        # Actions: prob1,prob2, joints 1 to 6, gripper action (open/close)
        self.action_space = Box(low=np.array([0]*2 + [-math.pi/10]*6 + [0]), high=np.array([1]*2 + [+math.pi/10]*6 + [1]), dtype=np.float32)

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
        self.control_method = 'joint'

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
        action1_prob = action[0]
        action2_prob = action[1]

        action1_actions = action[2:-1]
        action2_actions = action[-1]

        action_selected = random.choices([0, 1], weights=[action1_prob, action2_prob], k=1)[0]

        if action_selected == 0:
            # Move the end effector
            joint_states = self.robot.get_joint_states()
            self.robot.move_ee(joint_states + action[2:-1], self.control_method)
        else:
            # Move the gripper
            if action[-1] < 0.5:
                self.robot.open_gripper()
            else:
                self.robot.close_gripper()
 
        self.wait_simulation_steps(120)
            
        reward = self.update_reward()
        
        info = {"is_success": False}

        # Stacked version
        """if self.on_top(self.cubes[1], self.cubes[0]):
            reward += 10
            terminated = True
            info["is_success"] = True
        else:
            terminated = False"""

        # Grasp version
        if self.cube_grasped(self.cubes[0]):
            reward += 10
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

        """# For getting close to the cube
        ee_pos = self.robot.get_ee_pose()[:3]
        cube_pos = self.get_cube_pose(self.cubes[0])[:3]

        distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        distance_reward = geometric_distance_reward(distance, 0.2, 0.5)

        reward += distance_reward"""

        if self.touched_with_fingers(self.cubes[0]):
            print("Touched with fingers!")
            reward += 0.1

        return reward
    
    def any_cube_grasped(self):
        for id in self.cubes:
            contact_points = p.getContactPoints(self.robot.id, id)
            position,_ = p.getBasePositionAndOrientation(id)
            if contact_points and position[2] > self.CUBE_RAISE_HEIGHT: # If a cube is raised
                return True
        return False
    
    def cube_grasped(self, cube_id):
        contact_points = p.getContactPoints(self.robot.id, cube_id)
        position,_ = p.getBasePositionAndOrientation(cube_id)
        if contact_points and position[2] > self.CUBE_RAISE_HEIGHT: # If a cube is raised
            return True
        return False
    
    def on_top(self, lower_cube_id, upper_cube_id):
        lower_cube_pos, _ = p.getBasePositionAndOrientation(lower_cube_id)
        upper_cube_pos, _ = p.getBasePositionAndOrientation(upper_cube_id)
        contact_points = p.getContactPoints(lower_cube_id, upper_cube_id)
        # Check if the upper cube is placed on top of the lower cube
        if upper_cube_pos[2] > lower_cube_pos[2] and contact_points:
            print(f"Cube {upper_cube_id} is placed on top of cube {lower_cube_id} and in contact")
            return True
        return False

    def touched_with_fingers(self, cube_id):
        contact_points = p.getContactPoints(cube_id, self.robot.id)
        fingers_indices = list(range(9, 19))
        contact_with_links = any(
            point[4] in fingers_indices or  point[3] in fingers_indices
            for point in contact_points
        )
        if contact_with_links:
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
        self.robot.move_ee(new_pose, 'end')
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
    