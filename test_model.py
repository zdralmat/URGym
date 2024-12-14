import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from urgym.envs.env_two_balls_balance_v0 import TwoBallsBalance

env = TwoBallsBalance() #replace with the two ball env
model = PPO.load("models/PPO/ model_5.zip", env=env) # put the name of the last save point



mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()