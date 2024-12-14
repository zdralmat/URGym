import time
import math
import numpy as np
from typing import NoReturn, Optional
import random
import os
from copy import deepcopy

from gymnasium import Env
from gymnasium.spaces import Box

import pybullet as p
import pybullet_data
from urgym.base.utilities import rotate_quaternion, geometric_distance_reward, print_link_names_and_indices
from urgym.base.robot import UR5Robotiq85

class TwoBallsBalance(Env):
    """
    Reinforcement learning environment for balancing two balls with a robot arm.

    Observation Space:
    The observation space has 9 items consisting of the end-effector orientation (roll, pitch, yaw) and the position (x, y, z) of the two balls with respect to the paddle.

    Action Space:
    The action space consists of the desired end-effector displacement (droll, dpitch, dyaw) within bounds [-0.1, +0.1]

    Rewards:
    The environment provides a single reward.
    - Time reward: A reward of +1 is given at each step that the two balls are on the paddle.
    """

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, camera=None, reward_type='dense',  render_mode='human') -> None:
        """
        Initialize the environment.

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
        # Observations: the paddle roll, pitch, yaw and the two balls position relative the center of the paddle
        self.observation_space = Box(low=np.array([-math.pi]*3 + [-1.0]*6), high=np.array([math.pi]*3 + [1.0]*6), dtype=np.float64)
        # Actions: (dx, dy, dz) for end-effector euler displacement 
        self.action_space = Box(low=np.array([-0.1]*3), high=np.array([+0.1]*3), dtype=np.float32)

        # Define environment        
        self.physicsClient = p.connect(p.GUI if self.visualize else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Reorient the debug camera
        #p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[-0.5,+0.2,0])
        p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=27, cameraPitch=-33, cameraTargetPosition=[-0.2,-0.2,0.2])

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

        self.paddle_id = None
        self.ball1_id = None
        self.ball2_id = None
        self.paddle_size = None
        self.ellipse_ids = None

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        #self.update_ellipses()
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
        action: (dx, dy, dz) for End-Effector Queternion Displacement Control
        """
        reward = 0
        for i in action:
            reward += self.punishment(i)

        
        
        # Differential version
        ee_pose = list(self.robot.get_ee_pose())
        euler = p.getEulerFromQuaternion(ee_pose[3:])
        euler += action
        quaternion = p.getQuaternionFromEuler(euler)
        ee_pose[3:] = quaternion
        self.robot.move_ee(ee_pose, self.control_method)

        self.wait_until_stable()
                
        truncated = False # Managed by the environment automatically

        reward += 1 #Balance reward

        ball1_position = self.get_ball_position(self.ball1_id)
        ball2_position = self.get_ball_position(self.ball2_id)
        paddle_position = self.get_paddle_pose()[:3]
        
        

        '''reward += self.reward_function(abs(ball1_position[0]))
        reward += self.reward_function(abs(ball1_position[1]))
        reward += self.reward_function(abs(ball2_position[0]))
        reward += self.reward_function(abs(ball2_position[1]))'''

        reward += self.reward_function()

        if ball1_position[2] < paddle_position[2] or ball2_position[2] < paddle_position[2]:
            terminated = True
        else:
            terminated = False

        return self.get_observation(), reward, terminated, truncated, {}
    '''
    def reward_function(self, x):
        if x > 1:
            return 0
        return 1 - x**2
    '''
    def reward_function(self):
        reward = 0
        results1 = self.check_ball_in_ellipses(self.ball1_id)
        results2 = self.check_ball_in_ellipses(self.ball2_id)
        for i in range(len(results1)):
            if results1[i] and results2[i]:
                reward += 1
                
        return reward
    
    def punishment(self, x):
        x = abs(x)
        return -1 * x
    
    
    def get_observation(self):
        quaternion = np.array(self.get_paddle_pose()[3:])
        euler = p.getEulerFromQuaternion(quaternion) # To roll, pitch, yaw
        relative_ball1_position = np.array(self.get_paddle_pose()[:3]) - np.array(self.get_ball_position(self.ball1_id))
        relative_ball2_position = np.array(self.get_paddle_pose()[:3]) - np.array(self.get_ball_position(self.ball2_id))
        obs = np.append(euler, [relative_ball1_position, relative_ball2_position])
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.reset()

        if self.paddle_id is not None:
            p.removeBody(self.paddle_id)
        if self.ball1_id is not None:
            p.removeBody(self.ball1_id)
        if self.ball2_id is not None:
            p.removeBody(self.ball2_id)
        if self.ellipse_ids is not None:
            for ellipse_id in self.ellipse_ids:
                p.removeBody(ellipse_id)

        new_pose = [0.0, -0.60, 0.60, 0.50, -0.50, -0.50, 0.50]

        self.robot.move_ee(new_pose, 'end')
        self.robot.open_gripper(20)
        self.wait_until_stable()

        pose = np.array(self.robot.get_joint_obs()['ee_pos'][:3])
        gripper_center = self.get_gripper_geometrical_center()
        pose[1] -= 0.2
        pose[2] = gripper_center[2]
        self.create_balance_paddle(pose)
        self.ball1_id, self.ball2_id = self.create_balls()
        #self.ellipse_ids = self.create_ellipses(n=5)
        #print(self.ellipse_ids)

        self.robot.close_gripper()
        self.wait_until_stable()
        
       

        return self.get_observation(), {}

    def close(self):
        p.disconnect(self.physicsClient)

    def create_balance_paddle(self, base_position):
        self.paddle_size = [0.13, 0.1, 0.005]  # [0.05, 0.1, 0.005]

        # Create a rectangular collision shape
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=self.paddle_size)
        
        # Create a visual shape (optional, for better visualization)
        visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=self.paddle_size, rgbaColor=[1, 0.5, 0, 1])
        
        # Create the multi-body using the collision shape and visual shape
        base_mass = 0.1  # mass of the object
        self.paddle_id = p.createMultiBody(base_mass, collision_shape, visual_shape, base_position, [0, 0, 0, 1])

    def create_balls(self, radius=0.015):
        # Get the position and orientation of the paddle
        paddle_position, paddle_orientation = p.getBasePositionAndOrientation(self.paddle_id)
        
        # Calculate the ball position (centered on top of the paddle)
        ball1_position = [paddle_position[0], paddle_position[1], paddle_position[2] + 0.2 + radius]
        ball2_position = [paddle_position[0], paddle_position[1], paddle_position[2] + 0.2 - radius]
        # Randomize the balls position a little bit
        ball1_position[0] += random.uniform(-0.03, 0.03)
        ball1_position[1] += random.uniform(-0.08, -0.045) # Front part of the paddle
        # Randomize the balls position a little bit
        ball2_position[0] += random.uniform(-0.03, 0.03)
        ball2_position[1] += random.uniform(-0.015, 0.02) # Rear part of the paddle

        # Create a spherical collision shape
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
        
        # Create a visual shape (optional, for better visualization)
        ball1_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 1, 1, 1])
        ball2_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 0, 0, 1.0])
        
        # Create the multi-body using the collision shape and visual shape
        ball1_body_id = p.createMultiBody(0.0041, collision_shape, ball1_visual_shape, ball1_position, paddle_orientation)
        ball2_body_id = p.createMultiBody(0.0041, collision_shape, ball2_visual_shape, ball2_position, paddle_orientation)
        
        return ball1_body_id, ball2_body_id

    def get_paddle_pose(self):
        position, orientation = p.getBasePositionAndOrientation(self.paddle_id)
        pose = position + orientation
        return pose
    
    def create_ellipses(self, n ):
        # Get the center and orientation of the paddle
        center, orientation = p.getBasePositionAndOrientation(self.paddle_id)
        center_x, center_y, center_z = center  # Assuming 3D coordinates
        initial_radius_x = self.paddle_size[0] / (n/2)
        radius_increment_x = initial_radius_x / (n/2)
        initial_radius_y = self.paddle_size[1] / (n/2)
        radius_increment_y = initial_radius_y / (n/2)
        # Convert quaternion to Euler angles to get the rotation angle
        _, _, yaw = p.getEulerFromQuaternion(orientation)

        # Array to store the IDs of the created ellipses
        ellipse_ids = []

        # Define a list of colors to rotate through
        colors = [
            [1, 0, 0, 1],  # Red
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1],  # Blue
            [1, 1, 0, 1],  # Yellow
            [1, 0, 1, 1],  # Magenta
            [0, 1, 1, 1]   # Cyan
        ]

        # Create n ellipses with increasing radii
        for i in range(n):
            radius_x = initial_radius_x + i * radius_increment_x
            radius_y = initial_radius_y + i * radius_increment_y
            color = colors[i % len(colors)]  # Rotate through the colors
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[radius_x, radius_y, 0.01],  # Ellipse dimensions
                rgbaColor=color,  # Use the rotated color
                visualFramePosition=[0, 0, 0],
                visualFrameOrientation=p.getQuaternionFromEuler([0, 0, yaw])
            )
            ellipse_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[center_x, center_y, center_z + 0.00001 * (n-i)],  # Slightly offset in z to avoid overlap
                baseOrientation=orientation
            )
            ellipse_ids.append(ellipse_id)


        # Return the array of ellipse IDs
        return ellipse_ids
    

    def is_ball_in_ellipse(self, ball_position, ellipse_center, radius_x, radius_y):
        rel_x = ball_position[0] - ellipse_center[0]
        rel_y = ball_position[1] - ellipse_center[1]
        return (rel_x**2 / radius_x**2) + (rel_y**2 / radius_y**2) <= 1

    
    def check_ball_in_ellipses(self, ball_id, n=5):
        # Get the position of the ball
        ball_position, _ = p.getBasePositionAndOrientation(ball_id)

        # Get the center and orientation of the paddle
        center, orientation = p.getBasePositionAndOrientation(self.paddle_id)
        center_x, center_y, center_z = center  # Assuming 3D coordinates

        # Array to store the results
        results = []

        # Define the initial radii and increments
        initial_radius_x = self.paddle_size[0] / (n/2)
        radius_increment_x = initial_radius_x / (n/2)
        initial_radius_y = self.paddle_size[1] / (n/2)
        radius_increment_y = initial_radius_y / (n/2)

        # Check if the ball is within each ellipse
        for i in range(n):
            radius_x = initial_radius_x + i * radius_increment_x
            radius_y = initial_radius_y + i * radius_increment_y
            ellipse_center = [center_x, center_y, center_z ]
            results.append(self.is_ball_in_ellipse(ball_position, ellipse_center, radius_x, radius_y))

        return results
    
    def update_ellipses(self):
        # Check if ellipses have been created
        if self.ellipse_ids is None:
            return

        # Get the current position and orientation of the paddle
        center, orientation = p.getBasePositionAndOrientation(self.paddle_id)
        center_x, center_y, center_z = center  # Assuming 3D coordinates

        # Update the position and orientation of each ellipse
        for i, ellipse_id in enumerate(self.ellipse_ids):
            p.resetBasePositionAndOrientation(ellipse_id, [center_x, center_y, center_z + 0.0001 * (len(self.ellipse_ids) - i)], orientation)

    def get_ball_position(self, ball_id):
        position, _ = p.getBasePositionAndOrientation(ball_id)
        return position

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

    def get_gripper_geometrical_center(self):
        """
        Returns the geometrical center of the robot's gripper. This is useful to infer if an object is within the area of the gripper.

        :return: The geometrical center (x, y, z) of the gripper
        """
        # Initialize min and max coordinates to extreme values
        overall_aabb_min = [float('inf'), float('inf'), float('inf')]
        overall_aabb_max = [float('-inf'), float('-inf'), float('-inf')]

        gripper_link_indices = [11,16]

        # Iterate through each link index in the gripper
        for link_index in gripper_link_indices:
            aabb_min, aabb_max = p.getAABB(self.robot.id, link_index)
            
            # Update the overall AABB
            for i in range(3):
                overall_aabb_min[i] = min(overall_aabb_min[i], aabb_min[i])
                overall_aabb_max[i] = max(overall_aabb_max[i], aabb_max[i])

        # Calculate the geometrical center
        geometrical_center = [(overall_aabb_max[i] + overall_aabb_min[i]) / 2 for i in range(3)]

        #print(f"Geometrical center of gripper: {geometrical_center}")

        return geometrical_center
    
