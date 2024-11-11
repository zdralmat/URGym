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
from urgym.base.utilities import rotate_quaternion, geometric_distance_reward
from urgym.base.robot import UR5Robotiq85

class Golf(Env):
    """
    Reinforcement learning environment for pushing two cubes with a robot arm.

    Observation Space:
    The observation space ha s9 items consisting of the end-effector position (x, y, z) and the position (x, y, z) of the two cubes.
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
    - Collision penalty: A penalty of -0.01 is given if the robot collides with the floor.
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
        # Observations: the stick position, position of the ball relative to the stick and position of the ball relative to the hole
        self.observation_space = Box(low=np.array([-1.0]*3 + [-1.0]*3 + [-1.0]*3), high=np.array([1.0]*3 + [1.0]*3 + [1.0]*3), dtype=np.float64)
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

        self.pitch_id = self.create_golf_pitch([0, -0.6, 0.02])
        self.stick_id = None
        self.ball_id = None
        self.positive_reward_radius = 0.1

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
                
        # Termination conditions
        if self.is_ball_in_hole():
            reward += 10
            success = True
            terminated = True
        elif not self.stick_in_fingers():
            reward -= 5
            success = False
            terminated = True
            print("Stick fell!")
        elif self.is_ball_off_pitch():
            reward -= 10
            success = False
            terminated = True
            print("Ball off pitch!")
        else:
            if self.is_floor_collision():
                reward -= 0.01
                success = False
                terminated = False
                #print("Collision with the pitch!")
            if self.reward_type == 'dense':
                # Dense reward: the distance between the ball and stick base
                ball_pos = self.get_ball_position()
                stick_pos = np.array(self.get_stick_base_position())
                distance = float(np.linalg.norm(stick_pos - np.array(ball_pos)))
                reward += geometric_distance_reward(distance, self.positive_reward_radius, 0.5) / 10

                success = False
                terminated = False
            if self.touched_with_stick(self.ball_id):
                reward += 0.1 # Reward for touching the ball
                print("Touched!")
                if self.reward_type == 'dense':
                    # Dense reward:  the distance between the ball and hole
                    ball_pos = self.get_ball_position()
                    hole_pos = self.get_hole_position()
                    distance = float(np.linalg.norm(np.array(hole_pos) - np.array(ball_pos)))
                    reward += geometric_distance_reward(distance, self.positive_reward_radius, 0.5) / 10
                success = False
                terminated = False

        info = {"is_success": success}

        if success:
            print("SUCCESS!")

        truncated = False # Managed by the environment automatically

        reward -= 0.1 # Step penalty

        #print(f"Reward: {reward:.2f}")

        return self.get_observation(), reward, terminated, truncated, info
    
    def is_floor_collision(self):
        # Check the robot
        contact_points = p.getContactPoints(bodyA=self.robot.id, bodyB=self.pitch_id)
        if contact_points:
            #print("Collision with the pitch!")
            return True
        # Check the stick
        contact_points = p.getContactPoints(bodyA=self.stick_id, bodyB=self.pitch_id)
        if contact_points:
            #print("Collision with the pitch!")
            return True
        
        return False

    def touched_with_stick(self, object_id):
        contact_points = p.getContactPoints(object_id, self.stick_id)
        if contact_points:
            return True
        return False

    def stick_in_fingers(self):
        contact_points = p.getContactPoints(self.stick_id, self.robot.id)
        fingers_indices = list(range(9, 19))
        contact_with_links = any(
            point[4] in fingers_indices or point[3] in fingers_indices
            for point in contact_points
        )
        if contact_with_links:
            return True
        return False


    def is_ball_in_hole(self, threshold=0.05):
        ball_pos = self.get_ball_position()
        hole_pos = self.get_hole_position()
        distance = np.linalg.norm(np.array(hole_pos) - np.array(ball_pos))
        return distance < threshold
    

    def is_ball_off_pitch(self):
        ball_pos = self.get_ball_position()
        if ball_pos[2] < 0:
            return True
        return False

    def get_observation(self):
        stick_position = np.array(self.get_stick_base_position())
        ball_position = np.array(self.get_ball_position())
        relative_ball_hole_position = np.array(self.get_hole_position()) - ball_position
        relative_ball_stick_position = stick_position - ball_position
        obs = np.append(stick_position, [relative_ball_stick_position, relative_ball_hole_position])
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.reset()

        if self.stick_id is not None:
            p.removeBody(self.stick_id)
        if self.ball_id is not None:
            p.removeBody(self.ball_id)
        if self.pitch_id is not None: # Recreated as it may move due to collision
            p.removeBody(self.pitch_id)
            self.pitch_id = self.create_golf_pitch([0, -0.6, 0.02])

        new_pose = [0.0, -0.60, 0.40, 0.0, 0.0, -0.70, 0.70]
        self.vertical_quaternion = new_pose[3:]

        self.robot.move_ee(new_pose, 'end')
        self.robot.open_gripper(20)
        self.wait_until_stable()

        pose = np.array(self.robot.get_joint_obs()['ee_pos'][:3])
        gripper_center = self.get_gripper_geometrical_center()
        pose[:3] = deepcopy(gripper_center)
        pose[2] -= 0.05
        self.stick_id = self.create_stick(pose)
        self.ball_id = self.create_ball()

        self.robot.close_gripper()
        self.wait_until_stable()

        return self.get_observation(), {}

    def close(self):
        p.disconnect(self.physicsClient)

    # Plain stick, does not work very well
    def _create_stick_alt(self, base_position):
        # Create a rectangular collision shape
        dimensions = np.array([0.02, 0.05, 0.2])
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=dimensions/2)
        
        # Create a visual shape (optional, for better visualization)
        visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=dimensions/2, rgbaColor=[1, 0.5, 0, 1])
        
        # Create the multi-body using the collision shape and visual shape
        base_mass = 1  # mass of the object
        stick_id = p.createMultiBody(base_mass, collision_shape, visual_shape, base_position, [0, 0, 0, 1])
        # Add friction
        p.changeDynamics(stick_id, -1, lateralFriction=10, restitution=0.1)

        return stick_id

    def create_stick(self, base_position):
        # Define dimensions of the parts of the inverted T
        vertical_part_size = [0.03, 0.03, 0.15]
        transversal_part_size = [0.1, 0.03, 0.03]
        top_disc_radius = 0.04
        top_disc_height = 0.01

        # Define the shape for the vertical part
        vertical_part_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in vertical_part_size])
        vertical_part_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[d/2 for d in vertical_part_size], rgbaColor=[1, 0.5, 0, 1], specularColor=[0, 0, 0])

        # Define the shape for the bottom transversal part
        transversal_part_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d/2 for d in transversal_part_size])
        transversal_part_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[d/2 for d in transversal_part_size], rgbaColor=[1, 0.5, 0, 1], specularColor=[0, 0, 0])

        # Define the shape for the top disc part
        top_disc_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=top_disc_radius, height=top_disc_height)
        top_disc_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=top_disc_radius, length=top_disc_height, rgbaColor=[1, 0.5, 0, 1], specularColor=[0, 0, 0])

        # Calculate the positions for the bottom transversal part and the top disc part
        bottom_transversal_part_position = [0, 0, -(vertical_part_size[2]/2 + transversal_part_size[2]/2)]
        top_disc_position = [0, 0, vertical_part_size[2]/2 + top_disc_height/2]

        # Create the combined multi-body object with the bottom transversal part and the top disc
        stick_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=vertical_part_collision,
            baseVisualShapeIndex=vertical_part_visual,
            basePosition=base_position,
            linkMasses=[1, 1],
            linkCollisionShapeIndices=[transversal_part_collision, top_disc_collision],
            linkVisualShapeIndices=[transversal_part_visual, top_disc_visual],
            linkPositions=[bottom_transversal_part_position, top_disc_position],
            linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0], [0, 0, 0]]
        )
        # Add friction
        p.changeDynamics(stick_id, -1, lateralFriction=10, restitution=0.1)

        return stick_id


    def create_ball(self, radius=0.015):
        # Calculate the ball position (centered on top of the stick)
        ball_position = [0, -0.7, 0.1]
        # Randomize the ball position a little bit
        ball_position[0] += random.uniform(-0.2, 0.2)
        ball_position[1] += random.uniform(-0.1, 0.1)

        # To test the ball in the hole event
        #ball_position = np.array(self.get_hole_position())
        #ball_position[2] += 0.04
        
        # Create a spherical collision shape
        collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
        
        # Create a visual shape (optional, for better visualization)
        visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 1, 1, 1])
        
        # Create the multi-body using the collision shape and visual shape
        ball_body_id = p.createMultiBody(0.01, collision_shape, visual_shape, ball_position)

        # Add friction
        p.changeDynamics(ball_body_id, -1, rollingFriction=10.0, restitution=0.1)
        
        return ball_body_id

    def get_stick_pose(self):
        position, orientation = p.getBasePositionAndOrientation(self.stick_id)
        pose = position + orientation
        return pose
    
    def _get_stick_base_position_alt(self):
        return self.get_stick_pose()[:3]

    def get_stick_base_position(self):
        # Get the state of the bottom transversal part (link 0)
        link_state = p.getLinkState(self.stick_id, 0)
        # The link position is the center of mass of the link
        link_position = link_state[0]
        return link_position

    def get_ball_position(self):
        position, _ = p.getBasePositionAndOrientation(self.ball_id)
        return position

    def get_hole_position(self):
        position, _ = p.getBasePositionAndOrientation(self.pitch_id)
        return position

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

    def create_golf_pitch(self, position):

        # Load the mesg
        script_dir = os.path.dirname(__file__)
        pitch_path = os.path.join(script_dir, "../meshes/golf_pitch.obj")

        # Create collision shape from the mesh
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=pitch_path)

        # Create visual shape from the mesh
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=pitch_path, rgbaColor=[0.1, 0.6, 0, 1], specularColor=[0, 0, 0])

        # Create the multi-body with the collision and visual shapes
        body_id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=position)  # Adjust the position as needed
                
        return body_id
    
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
    
