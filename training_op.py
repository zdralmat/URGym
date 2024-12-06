import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from urgym.envs.env_two_balls_balance_v0 import TwoBallsBalance

import URGym.optuna_train as optuna_train
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.env_util import make_vec_env

def objective(trial):
    #env_id = "CartPole-v1"
    #env = make_vec_env(env_id, n_envs=1)

    env = TwoBallsBalance()


    # Define the hyperparameter search space
    n_steps = trial.suggest_int('n_steps', 2048, 4096)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    n_epochs = trial.suggest_int('n_epochs', 3, 10)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        verbose=0,
    )

    model.learn(total_timesteps=10000)

    # Evaluate the model
    mean_reward = 0.0
    n_eval_episodes = 10
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            mean_reward += reward
    mean_reward /= n_eval_episodes

    return mean_reward

# Create the Optuna study
study = optuna_train.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))