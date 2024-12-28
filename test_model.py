import gymnasium as gym
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
from urgym.envs.env_two_balls_balance_v0 import TwoBallsBalance

#env = TwoBallsBalance() #replace with the two ball env
#model = PPO.load("models/PPO/ model_5.zip", env=env) # put the name of the last save point


def get_last_saved_model(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    if not model_files:
        return None
    model_files.sort(key=lambda f: int(f.split('_')[1][:-4]))
    last_model_path = os.path.join(models_dir, model_files[-1])
    return last_model_path

def load_model_with_new_env(model_path, new_env):
    model = PPO.load(model_path, env=new_env)
    return model

def clean_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


models_dir = "models/PPO"
log_dir = "logs/PPO"
PADDLE_MULTIPLIER = 2.6
# Create the environment
#env = TwoBallsBalance(render_mode="training")

TIMESTEPS = 10000
NUM_ITERATIONS = 10000  # Adjust according to your needs
change_step = 0 

# Instantiate the agent


last_model_path = get_last_saved_model(models_dir)
    
if last_model_path:
    change_step = int(last_model_path.split('/')[-1].split('_')[0])

    env = TwoBallsBalance(render_mode="human", paddle=PADDLE_MULTIPLIER - 0.1 * change_step)
    model = load_model_with_new_env(last_model_path, env)
    print(f"Loaded model from {last_model_path}")
else:
    env = TwoBallsBalance(render_mode="training")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)




mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()