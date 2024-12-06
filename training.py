import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from urgym.envs.env_two_balls_balance_v0 import TwoBallsBalance

import URGym.optuna_train as optuna_train
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


ALGORITHM = "PPO"
models_dir = f"models/{ALGORITHM}"
log_dir = "logs"

sampled_hyperparams = sample_ppo_params(trial)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the environment
env = TwoBallsBalance()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, **sampled_hyperparams)



TIMESTEPS = 10000
NUM_ITERATIONS = 10  # Adjust according to your needs

for i in range(1, NUM_ITERATIONS + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=ALGORITHM
    )
    model.save(f"{models_dir}/{TIMESTEPS * i}")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
