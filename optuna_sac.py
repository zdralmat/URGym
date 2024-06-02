import argparse
import os
import shutil
import json

import gymnasium as gym
import optuna

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import urgym.envs

# Define the SAC hyperparameters to optimize
hyperparameters = {
    'learning_rate': (1e-4, 1e-2),
    'gamma': (0.9, 0.99),
    'tau': (0.001, 0.1),
    'batch_size': [32, 64, 128, 256],
    'gradient_steps': [1, 3, 5],
    'use_sde': (True, False),    
    'net_arch_nodes': [32, 64, 128, 256],    
}

# Define the objective function for Optuna
def objective(trial: optuna.Trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', *hyperparameters['learning_rate'], log=True)
    gamma = trial.suggest_float('gamma', *hyperparameters['gamma'])
    tau = trial.suggest_float('tau', *hyperparameters['tau'])
    batch_size = trial.suggest_categorical('batch_size', hyperparameters['batch_size'])
    gradient_steps = trial.suggest_categorical('gradient_steps', hyperparameters['gradient_steps'])
    use_sde = trial.suggest_categorical('use_sde', hyperparameters['use_sde'])
    net_arch_nodes = trial.suggest_categorical('net_arch_nodes', hyperparameters['net_arch_nodes']) 

    policy_kwargs = dict(net_arch=[net_arch_nodes, net_arch_nodes])

    model = SAC('MlpPolicy', env, verbose=True, learning_starts=1000, gamma=gamma, tau=tau, learning_rate=learning_rate,
                batch_size=batch_size, gradient_steps=gradient_steps, use_sde=use_sde, policy_kwargs=policy_kwargs)    

    print(f"Trial {trial.number} with hyperparameters: {trial.params}")

    try:
        model.learn(total_timesteps=n_steps, progress_bar=True)
        print()
        print("Evaluating the model...")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print("Mean reward:", mean_reward)
        return mean_reward
    except Exception as e: # Sometimes learn can fail due to exploding gradients
        print("Skipping trial due to an error:")
        print(e)
        return float('-inf')

def create_study_dir(optuna_dir, study_dir):
    # Create the first level directory if it does not exist
    if not os.path.exists(optuna_dir):
        os.makedirs(optuna_dir)
        print(f"Directory: {optuna_dir} created.")
    
    # Full path for the study directory
    full_study_dir_path = os.path.join(optuna_dir, study_dir)
    
    # If the second level directory exists, remove it and its contents
    if os.path.exists(full_study_dir_path):
        shutil.rmtree(full_study_dir_path)
        print(f"Removed existing study directory and all its contents: {full_study_dir_path}")
    
    # Create the second level directory
    os.makedirs(full_study_dir_path)
    print(f"Created study directory: {full_study_dir_path}")


parser = argparse.ArgumentParser(description='Search Optuna hyperparameters.')
parser.add_argument('-e', '--env', type=str, help='environment to test (e.g. CartPole-v1)', required=True)
parser.add_argument('-t', '--trials', type=int, default=50, help='number of trials')
parser.add_argument('-n', '--nsteps', type=int, default=50_000, help='number of steps per trial')
parser.add_argument('-m', '--name', type=str, help='name of the study')
parser.add_argument('-c', '--continue', dest="cont", action='store_true', default=False, help='continue existing study')

args = parser.parse_args()

str_env = args.env
n_trials = args.trials
n_steps = args.nsteps
continue_study = args.cont
study_name = args.name if args.name else str_env+"_sac"
optuna_dir = f"optuna_results/"
study_dir = f"{study_name}/"
storage_file = f"sqlite:///{optuna_dir}optuna.db"

set_random_seed(42)

# Create environment
env = gym.make(str_env, render_mode=None)

# Set up Optuna and start the optimization
if not continue_study: # Delete to overwrite if it exists
    try:
        optuna.delete_study(study_name=study_name, storage=storage_file)
    except:
        pass

study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_file, load_if_exists=continue_study)

print(f"Searching for the best hyperparameters in {n_trials} trials...")
study.optimize(objective, n_trials=n_trials)

# Create the study directory
create_study_dir(optuna_dir, study_dir)

# Generate the figures of the results
full_study_dir_path = os.path.join(optuna_dir, study_dir)
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html(f"{full_study_dir_path}/optimization_history_sac.html")
fig = optuna.visualization.plot_contour(study)
fig.write_html(f"{full_study_dir_path}/contour_sac.html")
fig = optuna.visualization.plot_slice(study)
fig.write_html(f"{full_study_dir_path}/slice_sac.html")
fig = optuna.visualization.plot_param_importances(study)
fig.write_html(f"{full_study_dir_path}/param_importances_sac.html")

best_trial = study.best_trial

# Generate the policy_kwargs key before writing to file
net_arch_nodes = best_trial.params.pop('net_arch_nodes')
best_trial.params['policy_kwargs'] = {'net_arch': [net_arch_nodes, net_arch_nodes]}

# print the result on the screen
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")
best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
print(best_trial_params)

# save the data in a JSON file
best_trial_file = open(f"{full_study_dir_path}/trial_sac.json", "w")
best_trial_file.write(best_trial_params)
best_trial_file.close()

env.close

