import optuna
import gymnasium as gym
import json
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

import ur5.envs

# Define the SAC hyperparameters to optimize
hyperparameters = {
    'learning_rate': (1e-4, 1e-2),
    'gamma': (0.9, 0.99),
    'tau': (0.001, 0.1),
    'batch_size': [32, 64, 128, 256],
    'gradient_steps': (1, 10),
    'use_sde': (True, False),    
    'net_arch': [32, 64, 128, 256],    
}

# Define the objective function for Optuna
def objective(trial: optuna.Trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', *hyperparameters['learning_rate'], log=True)
    gamma = trial.suggest_float('gamma', *hyperparameters['gamma'])
    tau = trial.suggest_float('tau', *hyperparameters['tau'])
    batch_size = trial.suggest_categorical('batch_size', hyperparameters['batch_size'])
    gradient_steps = trial.suggest_int('gradient_steps', *hyperparameters['gradient_steps'])
    use_sde = trial.suggest_categorical('use_sde', hyperparameters['use_sde'])
    net_arch = trial.suggest_categorical('net_arch', hyperparameters['net_arch']) 

    policy_kwargs = dict(net_arch=[net_arch, net_arch])

    model = SAC('MlpPolicy', env, learning_starts=1000, gamma=gamma, tau=tau, learning_rate=learning_rate,
                batch_size=batch_size, gradient_steps=gradient_steps, use_sde=use_sde, policy_kwargs=policy_kwargs)    

    print(f"Trial {trial.number} with hyperparameters: {trial.params}")
    model.learn(total_timesteps=n_steps, progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
    return mean_reward

parser = argparse.ArgumentParser(description='Search Optuna hyperparameters.')
parser.add_argument('-e', '--env', type=str, help='environment to test (e.g. CartPole-v1)', required=True)
parser.add_argument('-t', '--trials', type=int, default=50, help='number of trials')
parser.add_argument('-n', '--nsteps', type=int, default=50_000, help='number of steps per trial')

args = parser.parse_args()

str_env = args.env
n_trials = args.trials
n_steps = args.nsteps

set_random_seed(42)

# Create environment
env = gym.make(str_env, render_mode=None)

# Set up Optuna and start the optimization
study = optuna.create_study(direction='maximize', study_name=str_env+"_sac")

print(f"Searching for the best hyperparameters in {n_trials} trials...")
study.optimize(objective, n_trials=n_trials)

# visualize the results
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_contour(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# print the result on the screen
best_trial = study.best_trial
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")
best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
print(best_trial_params)

# save the data in a JSON file
best_trial_file = open("optuna_best_trial_sac.json", "w")
best_trial_file.write(best_trial_params)
best_trial_file.close()