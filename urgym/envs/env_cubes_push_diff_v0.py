import time
import math
import numpy as np
from typing import NoReturn, Optional
import random
import os

from gymnasium import Env
from gymnasium.spaces import Box

import pybullet as p
import pybullet_data
from urgym.base.utilities import rotate_quaternion, geometric_distance_reward
from urgym.base.robot import UR5Robotiq85

class CubesPush(Env):
    """
    Reinforcement learning environment for pushing two cubes with a robot arm.

    Observation Space:
    The observation space has 9 items consisting of the end-effector position (x, y, z) and the position differences (dx, dy, dz) between the end-effector and the two cubes.
    All the coordinates (x, y, z) are within bounds [-1.0, +1.0]

    Action Space:
    The action space consists of the desired end-effector displacement (dx, dy, dz) within bounds [-0.1, +0.1]
    The end-effector orientation is always vertical pointing down.

    Rewards:
    The environment provides different rewards.
    - Touch reward: If the robot touches the cube with its fingers, a reward of +0.1 is given.
    - Distance reward: Based on the distance between the two cubes, as follows:
        -0.1 if the distance is greater than 0.5, cubes are far away.
        [-0.1, 0] if the distance is between 0.5 and 0.1.
        [0, 0.1] if the distance is between 0.1 and 0.05.
    - Success reward: If the distance between the two cubes is less than 0.05 (close contact), a reward of 10 is given.
    - Collision penalty: A penalty of -0.01 is given if the robot collides with the table.
    - Time penalty: A penalty of -0.1 is given at each step.
    """

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, camera=None, reward_type='dense',  render_mode='human') -> None:
        """
        Initialize the CubesPush environment.

        Parameters:
        - reward_type (str): The type of reward to use. Options are 'dense' and 'sparse'. Default is 'dense'.
        - render_mode (str): The rendering mode. Options are 'human' and 'non-human'. Default is 'human'.
        """

        self.reward_type = reward_type

        if render_mode == 'human':
            self.visualize = True
        else:
            self.visualize = False

        # Set observation and action spaces
        # Observations: the end-effector position and position (x, y, z) of the two cubes
        self.observation_space = Box(low=np.array([-1.0]*3 + [-1.0]*6), high=np.array([1.0]*3 + [1.0]*6), dtype=np.float64)
        # Actions: (dx, dy, dz) for end-effector displacement 
        self.action_space = Box(low=np.array([-0.1]*3), high=np.array([+0.1]*3), dtype=np.float32)

        # Define environment        
        self.physicsClient = p.connect(p.GUI if self.visualize else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Reorient the debug camera
        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[-0.5,+0.2,0])

        # Hide right and left menus
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        # Robot will be centered at the origin (0,0,0), the plane and the table will be below z=0
        table_dimensions = [1.0, 2.0, 0.5]
        self.planeID = p.loadURDF("plane.urdf", [0, 0, -table_dimensions[2]])
        self.table_id = self.create_table_block(table_dimensions, [0, 0, -table_dimensions[2]/2], color=[0.5, 0.5, 0.5, 1])

        self.robot = UR5Robotiq85((0, 0, 0), (0, 0, 0))

        self.robot.load()
        self.robot.step_simulation = self.step_simulation # type: ignore
        self.control_method = 'end'

        self.cubes = []
        self.positive_reward_radius = 0.1

        # Draw a green circle around the target cube
        self.draw_circle_area(radius=self.positive_reward_radius)

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.visualize:
            time.sleep(self.SIMULATION_STEP_DELAY)

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
        action: (dx, dy, dz) for End-Effector Displacement Control
        """

        reward = 0
        
        # Differential version
        ee_pose = list(self.robot.get_ee_pose())
        ee_pose[:3] += action
        ee_pose[3:] = self.vertical_quaternion # to keep the end effector vertical at all times
        self.robot.move_ee(ee_pose, self.control_method)

        self.wait_until_stable()
                
        if self.is_table_collision():
            reward -= 0.01
            success = False
            terminated = False
        elif self.touched_with_fingers(self.cubes[1]):
            reward, success = self.update_reward()
            reward += 0.1 # Reward for touching the cube
            print("Touched!")
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
            distance = float(np.linalg.norm(np.array(cube1_pos) - np.array(cube2_pos)))

            reward += geometric_distance_reward(distance, self.positive_reward_radius, 0.5) / 10

        # Sparse reward: if the distance between the cubes is less than 0.05
        if self.are_cubes_close(self.cubes[0], self.cubes[1], 0.05):
            reward += 10
            sucess = True
        else:
            sucess = False

        return reward, sucess
    
    def is_table_collision(self):
        contact_points = p.getContactPoints(bodyA=self.robot.id, bodyB=self.table_id)
        if contact_points:
            for point in contact_points:
                if point[3] != 0: # 0 is the base link, is always touching the table
                    print("Collision with the table!")
                    return True
        return False

    def touched_with_fingers(self, object_id):
        contact_points = p.getContactPoints(object_id, self.robot.id)
        fingers_indices = list(range(9, 19))
        contact_with_links = any(
            point[4] in fingers_indices or  point[3] in fingers_indices
            for point in contact_points
        )
        if contact_with_links:
            return True
        return False

    def are_cubes_close(self, cube1_id, cube2_id, threshold=0.05):
        cube1_pos = self.get_cube_pose(cube1_id)[:3]
        cube2_pos = self.get_cube_pose(cube2_id)[:3]
        distance = np.linalg.norm(np.array(cube1_pos) - np.array(cube2_pos))
        return distance < threshold
    
    def get_observation(self):
        obs = dict()
        obs.update(self.robot.get_joint_obs())

        obs = np.array(obs["ee_pos"])[:3]
        cube1_position = self.get_cube_pose(self.cubes[0])[:3]
        cube2_position = self.get_cube_pose(self.cubes[1])[:3]
        cube1_diff = np.array(cube1_position) - np.array(obs)
        cube2_diff = np.array(cube2_position) - np.array(obs)
        obs = np.append(obs, [cube1_diff, cube2_diff])
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.reset()
        self.create_cubes()

        # Position the end effector above the cube
        new_pose = list(self.get_cube_pose(self.cubes[1]))
        new_pose[2] += 0.2 # A little bit higher
        new_pose[3:] = rotate_quaternion(new_pose[3:], math.pi/2, [0, 1, 0]) # Reorient the end effector
        self.vertical_quaternion = new_pose[3:]

        self.robot.move_ee(new_pose, 'end')
        self.robot.close_gripper()
        self.wait_until_stable()

        return self.get_observation(), {}

    def close(self):
        p.disconnect(self.physicsClient)

    def create_cube(self, x: float, y: float, z: float, color:list=[1,1,1,1]):
        id = p.loadURDF("cube.urdf", [x, y, z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling = 0.04)
        p.changeVisualShape(id, -1, rgbaColor=color)
        self.cubes.append(id)

    def create_cubes(self):
        for id in self.cubes:
            p.removeBody(id)
        self.cubes = []

        # Create target cube
        self.create_cube(0.0, -0.5, 0.1)
        # Create second cube
        x = random.uniform(-0.1, +0.1)
        self.create_cube(x, -0.7, 0.1, [1,0,0,1])

    def get_cube_pose(self, cube_id):
        position, orientation = p.getBasePositionAndOrientation(cube_id)
        pose = position + orientation
        return pose
    
    def draw_circle_area(self, radius=0.1, center_x=0.0, center_y=-0.5, segments=100, color=[0, 1.0, 0, 0.4]):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=0.001,  # Very thin to act as a visual aid
            rgbaColor=color
        )
        p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[center_x, center_y, 0.0001]  # Slightly above the ground to avoid z-fighting
        )

    def create_table_block(self, size, position, color=[1, 0, 0, 1]):
        """
        Create a block in the PyBullet simulation.

        Args:
        - size: A list of three floats representing the half extents of the block [x, y, z].
        - position: A list of three floats representing the position of the block [x, y, z].
        - color: A list of four floats representing the RGBA color of the block.

        Returns:
        - block_body_id: The ID of the created block.
        """
        block_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                    rgbaColor=color,
                                                    halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        block_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                        halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        block_body_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=block_collision_shape_id,
                                        baseVisualShapeIndex=block_visual_shape_id,
                                        basePosition=position)  
        
        # Load the texture
        script_dir = os.path.dirname(__file__)
        texture_path = os.path.join(script_dir, "../meshes/textures/steel.jpg")
 
        texture_id = p.loadTexture(texture_path)

        # Apply the texture to the block
        p.changeVisualShape(block_body_id, -1, textureUniqueId=texture_id)

        return block_body_id
