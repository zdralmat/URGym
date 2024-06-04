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
import json
import numpy as np

from stable_baselines3 import PPO, DQN, SAC, A2C, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise


from urgym.algos import ActionSAC, ActionSACPolicy
import urgym.envs 
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from torch import nn


parser = argparse.ArgumentParser(description='Train an environment with an SB3 algorithm and saves the final policy (as well as checkpoints every 50k steps).')
parser.add_argument('-e', '--env', type=str, default="CartPole-v1", help='environment to test (e.g. CartPole-v1)')
parser.add_argument('-a', '--algo', type=str, default='PPO',
					help='algorithm to test from SB3, such as PPO (default), SAC, DQN... using default hyperparameters')
parser.add_argument('-n', '--nsteps', type=int, default=100_000, help='number of steps to train')
parser.add_argument('-y', '--hyperparams', type=str, default=None, help='path to json file with hyperparameters to use in the algorithm instead of the default ones')
parser.add_argument('--name', type=str, default="model", help='name of this experiment (for logs and policies)')
parser.add_argument('-v', '--visualize', action="store_true", help='visualize the training with render_mode=\'human\'')
parser.add_argument('-t', '--tblog', action="store_true", help='generate tensorboard logs in the \"logs\" directory')
parser.add_argument('-p', '--policy', type=str, default=None, help='policy to load to continue training, it will also read the replay buffer')

args = parser.parse_args()

str_env = args.env
str_algo = args.algo
algo = getattr(sys.modules[__name__], str_algo) # Obtains the classname based on the string
n_steps = args.nsteps
tblog_dir = None if args.tblog==False else "./logs"
experiment_name = args.name
render_mode = 'human' if args.visualize else None
policy_file = args.policy
hyperparams = json.load(open(args.hyperparams)) if args.hyperparams else {}

set_random_seed(42)

# Create environment
env = gym.make(str_env, render_mode=render_mode)

print(f"Training for {n_steps} steps with {str_algo}...")

# Particular configuration for ActionSAC
if str_algo == "ActionSAC":
	algo = ActionSAC
	if hyperparams.get('policy_kwargs', None) is None:
		net_arch_nodes = 256
	else:
		net_arch_nodes = hyperparams['policy_kwargs']['net_arch'][0]
	policy_kwargs = dict(
		net_arch=dict(pi=[env.action_space.shape[0]], qf=[net_arch_nodes, net_arch_nodes]),
		action_config=dict(n_actions=2, n_nodes=net_arch_nodes, layers=[(7, nn.Tanh), (1, nn.Tanh)]),
	)
	hyperparams["policy_kwargs"] = policy_kwargs
	policy_arch = ActionSACPolicy
else:
	policy_arch = "MlpPolicy"

if policy_file:
	model = algo.load(f"{policy_file}", env=env, tensorboard_log=tblog_dir, verbose=True, **hyperparams)
	replay_buffer_file = policy_file.removesuffix("_policy.zip") + "_replay_buffer.pkl"
	model.load_replay_buffer(replay_buffer_file)
	reset_tblog = False
else:
	# Noise
	"""n_actions = env.action_space.shape[-1]
	noise_std = 0.2
	action_noise_std = np.zeros(n_actions)
	action_noise_std[:2] = noise_std  # Apply noise only to the first two actions

	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise_std)"""

	# Instantiate the agent
	model = algo(policy_arch, env=env, tensorboard_log=tblog_dir, verbose=True, **hyperparams)    
	reset_tblog = True

# Train the agent and display a progress bar
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=f"./checkpoints/{experiment_name}_{str_algo}")
model.learn(total_timesteps=int(n_steps), callback=checkpoint_callback, progress_bar=True, tb_log_name=f"{experiment_name}_{str_algo}", reset_num_timesteps=reset_tblog)

model.save(f"policies/{experiment_name}_{str_algo}_policy.zip")
model.save_replay_buffer(f"policies/{experiment_name}_{str_algo}_replay_buffer.pkl")


env.close()

