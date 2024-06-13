import time
import math
import numpy as np
from typing import Optional
import os

from gymnasium import Env
from gymnasium.spaces import Box

import pybullet as p
import pybullet_data
from urgym.utilities import YCBModels, Camera, rotate_quaternion, geometric_distance_reward, z_alignment_distance, normalize_quaternion, print_link_names_and_indices
from urgym.robot import UR5Robotiq85
import random
import traceback

class CubesGrasp(Env):

    SIMULATION_STEP_DELAY = 1 / 240.
    OBJECT_RAISE_HEIGHT = 0.2

    def __init__(self, vertical_reward = False, camera=None, render_mode='human') -> None:
        if render_mode == 'human':
            self.vis = True
        else:
            self.vis = False

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
        self.robot = UR5Robotiq85((0, 0, 0), (0, 0, 0))

        # define environment        
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        # Hide right and left menus
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        # Reorient the debug camera
        p.resetDebugVisualizerCamera(cameraDistance=2.2, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[-0.5,+0.5,0])
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

        self.vertical_reward = vertical_reward

        self.subgoals_achieved = {'approached': False, 'grasped': False}

        # Set observation and action spaces
        # Observations:
        # The subgoal achieved status (0 or 1) for two subgoals ('approached', 'grasped')
        # The end-effector position and quaternion (x, y, z, qx, qy, qz, qw) and gripper opening length in[0,1]
        # And the target cube position and quaternion (x, y, z, qx, qy, qz, qw)
        self.observation_space = Box(low=np.array([0]*2 + [-1.0]*3 + [-1]*4 + [0] + [-1.0]*3 + [-1.0]*4), high=np.array([1]*2 + [1.0]*3 + [1.0]*4 + [1] + [1.0]*3 + [1.0]*4), dtype=np.float64)
        # Actions: prob1,prob2, end-effector position and quaternion, gripper action (open/close)
        self.action_space = Box(low=np.array([0]*2 + [-1]*3 + [-1]*4 + [0]), high=np.array([1]*2 + [+1]*3 + [+1]*4 + [1]), dtype=np.float32)

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
        
        action_move_prob = action[0]
        action_gripper_prob = action[1]

        action_move_actions = action[2:-1]
        action_gripper_actions = action[-1]
        action_move_quaternion = action_move_actions[3:7]
        action_move_quaternion = normalize_quaternion(*action_move_quaternion)
        action_move_actions[3:7] = action_move_quaternion

        if action_move_prob + action_gripper_prob == 0: # Avoid the sum to be zero
            action_selected = random.choices([0, 1], weights=[0.5, 0.5], k=1)[0]
        else:
            action_selected = random.choices([0, 1], weights=[action_move_prob, action_gripper_prob], k=1)[0]
        #action_selected = np.argmax([action_move_prob, action_gripper_prob])

        if action_selected == 0:
            # Move the end effector and close
            self.robot.move_ee(action_move_actions, self.control_method)
            self.wait_until_stable()
        elif action_selected == 1:
            # Open/close the gripper
            if action_gripper_actions < 0.5:
                self.robot.open_gripper()
                self.wait_until_stable()
            else:
                self.robot.close_gripper() 
                self.wait_until_stable()

        reward += self.update_reward()
        info = {'is_success': False}
        terminated = False

        if not self.subgoals_achieved['approached']: # Never approached
            distance = self.distance_to_target(self.target_id)
            distance_reward = geometric_distance_reward(distance, 0.5, 2) / 4
            reward += distance_reward
            if self.object_approached(self.target_id):
                print("Approached!")
                self.subgoals_achieved['approached'] = True
                reward += 5
        elif self.subgoals_achieved['approached'] and not self.subgoals_achieved['grasped']: 
            if self.object_grasped(self.target_id): # Grasped for first time
                print("Grasped!")
                self.subgoals_achieved['grasped'] = True
                reward += 5
                # Additional bonus reward based on vertical alignment, if configured
                if self.vertical_reward:
                    vertical_alignment = z_alignment_distance(*self.robot.get_ee_pose()[3:])
                    reward += geometric_distance_reward(vertical_alignment, 0.15, 0.3)*2

        elif self.subgoals_achieved['approached'] and self.subgoals_achieved['grasped']: 
            if self.object_raised(self.target_id): # Raised for first time
                print("Raised!")
                reward += 10
                terminated = True
                info['is_success'] = True
        
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

        """if self.touched_with_fingers(self.cubes[0]):
            print("Touched with fingers!")
            reward += 0.1"""
        
        return reward
    
    def distance_to_target(self, target_id):
        gripper_center = self.get_gripper_geometrical_center()
        target_pos = list(self.get_cube_pose(target_id)[:3])

        distance = np.linalg.norm(np.array(gripper_center) - np.array(target_pos))
        return distance

    def object_approached(self, object_id):
        distance = self.distance_to_target(object_id)
        if distance < 0.05 and self.get_gripper_opening_length() > 0.50:
            return True
        return False
      
    def object_grasped(self, object_id):
        gripper_link_indices_left = [11,12,13]
        gripper_link_indices_right = [16,17,18]
        # Get contact points between the robot and the object
        contact_points = p.getContactPoints(bodyA=self.robot.id, bodyB=object_id)
        
        # Initialize a set to keep track of which fingers are touching the object
        touching_fingers = set()
        
        # Iterate over the contact points to check if the specified fingers are touching the object
        for point in contact_points:
            link_index = point[3]
            if link_index in gripper_link_indices_left or link_index in gripper_link_indices_right:
                touching_fingers.add(link_index)
                
            # If any part of both fingers are touching simultaneously, return True
            if any(link_index in touching_fingers for link_index in gripper_link_indices_left) and any(link_index in touching_fingers for link_index in gripper_link_indices_right):
                return True
        
        # If not all specified fingers are touching, return False
        return False


    def object_raised(self, object_id):
        position,_ = p.getBasePositionAndOrientation(object_id)
        if self.object_grasped(object_id) and position[2] > self.OBJECT_RAISE_HEIGHT: # If the object is raised
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

    def get_observation(self):
        obs_robot = dict()

        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs_robot.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None

        obs_robot.update(self.robot.get_joint_obs())

        approached = self.object_approached(self.target_id)
        grasped = self.object_grasped(self.target_id)
        if grasped: # If grasped, we also consider it as approached, altough the gripper may be almost closed
            approached = True
        obs = np.array([int(s) for s in [approached, grasped]]) # Boolean to int (0,1)
        if grasped:
            print("Ready to raise...")
        #obs = np.array([int(s) for s in list(self.subgoals_achieved.values())]) # Boolean to int (0,1) Old version based on subgoals achieved
        obs = np.append(obs, obs_robot["ee_pos"])
        obs = np.append(obs, self.get_gripper_opening_length())
        obs = np.append(obs, self.get_cube_pose(self.target_id))
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.robot.reset()
        self.create_cubes()

        self.target_id = self.cubes[0]

        #print_link_names_and_indices(self.robot.id)

        
        """# Position the end effector above the cube
        new_pose = list(self.get_cube_pose(self.target_id))
        new_pose[2] += 0.05 # A little bit higher
        new_pose[1] += 0.00 # A little bit backwards
        new_pose[3:] = [0, -0.707, 0, -0.707] # Reorient the end effector downwards
        new_pose[3:] = rotate_quaternion(new_pose[3:], math.pi/2, [0, 0, 0]) # Rotate the end effector 90 degrees
        self.robot.move_ee(new_pose, 'end')
        self.robot.open_gripper()"""

        self.robot.move_ee(self.robot.get_ee_pose(), 'end')
        self.wait_until_stable()

        self.subgoals_achieved = {'approached': False, 'grasped': False}

        return self.get_observation(), {}

    def close(self):
        p.disconnect(self.physicsClient)

    def create_cube(self, x: float, y: float, z: float, color:list=None):
        id = p.loadURDF("cube.urdf", [x, y, z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling = 0.04)
        if color != None:
            p.changeVisualShape(id, -1, rgbaColor=color)
        self.cubes.append(id)

    def create_object(self, urdf_path, x: float, y: float, z: float, scale: float , color:list=None):
        """
        Create an object in the environment.

        Args:
            urdf_path (str): The path to the URDF file of the object.
                Examples: "lego/lego.urdf", "cube.urdf", "sphere.urdf", "humanoid/humanoid.urdf", "objects/mug.urdf", "aliengo/aliengo.urdf"
            x (float): The x-coordinate of the object's initial position.
            y (float): The y-coordinate of the object's initial position.
            z (float): The z-coordinate of the object's initial position.
            scale (float): The scaling factor for the object's size.
            color (list, optional): The RGBA color values for the object's visual shape. Defaults to None.

        Returns:
            int: The unique ID of the created object in the physics simulation.
        """
        id = p.loadURDF(urdf_path, [x, y, z], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False, globalScaling = scale)
        if color != None:
            p.changeVisualShape(id, -1, rgbaColor=color)
        self.cubes.append(id)
        return id

    def create_cubes(self):
        for id in self.cubes:
            p.removeBody(id)
        self.cubes = []
        #self.create_cube(0.0, -0.7, 0.1)

        # Generate a cube within the area delimited by a minimum and maximum radius
        rmin = 0.5
        rmax = 0.8
        # Generate a random radius between r1 and r2
        r = random.uniform(rmin, rmax)
        # Generate a random angle in fron of the arm
        theta = random.uniform(-math.pi, 0)
        # Convert polar coordinates to Cartesian coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        self.create_cube(x, y, 0.1, [1,0,0,1])
        #self.create_cube(0.0, -0.2, 0.1, [1,0,0,1])
        #self.create_cube(0.1, -0.1, 0.1, [0,0,1,1])
        #self.create_cube(0.1, -0.2, 0.1, [0,1,0,1])

    def get_cube_pose(self, cube_id):
        position, orientation = p.getBasePositionAndOrientation(cube_id)
        pose = position + orientation

        return pose

    def get_gripper_real_opening_length(self):
        """
        Calculate the real opening length of the gripper.

        Returns:
            float: The real opening length of the gripper.
        """
        gripper_link_indices = [11, 16]
        # Get the positions of the gripper links
        link_state_1 = p.getLinkState(self.robot.id, gripper_link_indices[0])
        link_state_2 = p.getLinkState(self.robot.id, gripper_link_indices[1])

        # Extract the positions of the two gripper links
        pos1 = np.array(link_state_1[4])  # World position of the first link
        pos2 = np.array(link_state_2[4])  # World position of the second link

        # Calculate the distance between the two positions
        opening_length = np.linalg.norm(pos2 - pos1)
        opening_length -= 0.052  # Subtract this value to be in the range [0, 0.085]

        return opening_length
    
    def get_gripper_opening_length(self):
            """
            Returns the normalized opening length of the gripper.

            The opening length is clamped to the range indicated by the robot and then normalized
            based on the minimum and maximum opening lengths of the gripper.

            Returns:
                float: The normalized opening length of the gripper in [0,1].
            """
            value = self.get_gripper_real_opening_length()
            min_opening_length = self.robot.gripper_range[0]
            max_opening_length = self.robot.gripper_range[1]

            value = max(min_opening_length, min(max_opening_length, value))  # Clamp the value to the range [0, 0.085]

            normalized_value = (value - min_opening_length) / (max_opening_length - min_opening_length)
            return normalized_value

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
    

 
if __name__ == "__main__":
    env = CubesGrasp()
    for _ in range(50):
        obs = env.reset()
        time.sleep(3)
