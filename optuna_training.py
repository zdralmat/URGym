import optuna
from stable_baselines3 import PPO, A2C, SAC, TD3
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
#from optuna.samplers import TPESampler
#from optuna.pruners import MedianPruner
import os
from urgym.envs.env_two_balls_balance_v0 import TwoBallsBalance
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True


def objectivePPO(trial):
    env_id = "CartPole-v1"
    env = TwoBallsBalance(render_mode='train')

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

    # Set up TensorBoard logger
    tmp_path = f"./logs/optuna_trial_PPO_{trial.number}"
    os.makedirs(tmp_path, exist_ok=True)
    new_logger = configure(tmp_path, ["tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=10000, callback=TensorboardCallback())

    # Evaluate the model
    mean_reward = 0.0
    n_eval_episodes = 10
    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info, _ = env.step(action)
            mean_reward += reward
    mean_reward /= n_eval_episodes

    return mean_reward



def objectiveA2C(trial):
    env_id = "CartPole-v1"
    env = TwoBallsBalance(render_mode='train')

    # Define the hyperparameter search space
    n_steps = trial.suggest_int('n_steps', 5, 20)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)

    model = A2C(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        gae_lambda=gae_lambda,
        verbose=0,
    )

    # Set up TensorBoard logger
    tmp_path = f"./logs/optuna_trial_A2C_{trial.number}"
    os.makedirs(tmp_path, exist_ok=True)
    new_logger = configure(tmp_path, ["tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=10000, callback=TensorboardCallback())

    # Evaluate the model
    mean_reward = 0.0
    n_eval_episodes = 10
    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info, _ = env.step(action)
            mean_reward += reward
    mean_reward /= n_eval_episodes

    return mean_reward




def objectiveSAC(trial):
    env_id = "CartPole-v1"
    env = TwoBallsBalance(render_mode='train')

    # Define the hyperparameter search space
    buffer_size = trial.suggest_int('buffer_size', 10000, 1000000)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 64, 256)
    ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.1, 0.01, 0.001])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    tau = trial.suggest_float('tau', 0.005, 0.02)
    train_freq = trial.suggest_int('train_freq', 1, 10)
    gradient_steps = trial.suggest_int('gradient_steps', 1, 10)

    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        verbose=0,
    )

    # Set up TensorBoard logger
    tmp_path = f"./logs/optuna_trial_SAC_{trial.number}"
    os.makedirs(tmp_path, exist_ok=True)
    new_logger = configure(tmp_path, ["tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=10000, callback=TensorboardCallback())

    # Evaluate the model
    mean_reward = 0.0
    n_eval_episodes = 10
    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info, _ = env.step(action)
            mean_reward += reward
    mean_reward /= n_eval_episodes

    return mean_reward



def objectiveTD3(trial):
    env_id = "CartPole-v1"
    env = TwoBallsBalance(render_mode='train')
    
    # Define the hyperparameter search space
    buffer_size = trial.suggest_int('buffer_size', 10000, 1000000)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 64, 256)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    tau = trial.suggest_float('tau', 0.005, 0.02)
    train_freq = trial.suggest_int('train_freq', 1, 10)
    gradient_steps = trial.suggest_int('gradient_steps', 1, 10)
    policy_delay = trial.suggest_int('policy_delay', 1, 2)
    noise_std = trial.suggest_float('noise_std', 0.1, 0.3)

    model = TD3(
        "MlpPolicy",
        env,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        policy_delay=policy_delay,
        action_noise=None,
        verbose=0,
    )

    # Set up TensorBoard logger
    tmp_path = f"./logs/optuna_trial_TD3_{trial.number}"
    os.makedirs(tmp_path, exist_ok=True)
    new_logger = configure(tmp_path, ["tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=10000, callback=TensorboardCallback())

    # Evaluate the model
    mean_reward = 0.0
    n_eval_episodes = 10
    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info, _ = env.step(action)
            mean_reward += reward
    mean_reward /= n_eval_episodes

    return mean_reward

# Create the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objectivePPO, n_trials=100)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open("optuna_resultsPPO.txt", "w") as f:
    f.write("Best trial:\n")
    f.write("  Value: {}\n".format(trial.value))
    f.write("  Params: \n")
    for key, value in trial.params.items():
        f.write("    {}: {}\n".format(key, value))

# Create the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objectiveA2C, n_trials=100)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open("optuna_resultsA2C.txt", "w") as f:
    f.write("Best trial:\n")
    f.write("  Value: {}\n".format(trial.value))
    f.write("  Params: \n")
    for key, value in trial.params.items():
        f.write("    {}: {}\n".format(key, value))

# Create the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objectiveSAC, n_trials=100)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open("optuna_resultsSAC.txt", "w") as f:
    f.write("Best trial:\n")
    f.write("  Value: {}\n".format(trial.value))
    f.write("  Params: \n")
    for key, value in trial.params.items():
        f.write("    {}: {}\n".format(key, value))

# Create the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objectiveTD3, n_trials=100)

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open("optuna_resultsTD3.txt", "w") as f:
    f.write("Best trial:\n")
    f.write("  Value: {}\n".format(trial.value))
    f.write("  Params: \n")
    for key, value in trial.params.items():
        f.write("    {}: {}\n".format(key, value))
