#!/usr/bin/env python
""" Basic Stable-Baselines3 tester on Gymnasium environments.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Inaki Vazquez"
__email__ = "ivazquez@deusto.es"
__license__ = "GPLv3"

import argparse
import sys

from stable_baselines3 import PPO, DQN, SAC, A2C, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from ur5.envs.env_box import BoxManipulation
from ur5.envs.env_cubes import CubesManipulation
import gymnasium as gym

parser = argparse.ArgumentParser(description='Train an environment with an SB3 algorithm and saves the final policy (as well as checkpoints every 50k steps).')
parser.add_argument('-e', '--env', type=str, default="CartPole-v1", help='environment to test (e.g. CartPole-v1)')
parser.add_argument('-a', '--algo', type=str, default='PPO',
					help='algorithm to test from SB3, such as PPO (default), SAC, DQN... using default hyperparameters')
parser.add_argument('-n', '--nsteps', type=int, default=100_000, help='number of steps to train')
parser.add_argument('-r', '--recvideo', action="store_true", help='record and store video in a \"video\" directory, instead of using the screen')
parser.add_argument('-t', '--tblog', action="store_true", help='generate tensorboard logs in the \"logs\" directory')
parser.add_argument('--name', type=str, default="model", help='name of this experiment (for logs and policies)')
parser.add_argument('-v', '--visualize', action="store_true", help='visualize the training with render_mode=\'human\'')

args = parser.parse_args()

str_env = args.env
str_algo = args.algo.upper()
algo = getattr(sys.modules[__name__], str_algo) # Obtains the classname based on the string
n_steps = args.nsteps
recvideo = args.recvideo
tblog_dir = None if args.tblog==False else "./logs"
experiment_name = args.name
render_mode = 'human' if args.visualize else None

# Create environment
env = gym.make(str_env, render_mode=render_mode)

print(f"Training for {n_steps} steps with {str_algo}...")

# Instantiate the agent
model = algo('MlpPolicy', env=env, tensorboard_log=tblog_dir, verbose=True)    

# Train the agent and display a progress bar
checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./checkpoints/{experiment_name}_{str_algo}")
model.learn(total_timesteps=int(n_steps), callback=checkpoint_callback, progress_bar=True, tb_log_name=f"{experiment_name}_{str_algo}")

model.save(f"policies/{experiment_name}_{str_algo}_policy.zip")

env.close()

