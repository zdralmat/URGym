import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import shutil
import os
from urgym.envs.env_two_balls_balance_v0 import TwoBallsBalance


ALGORITHM = "PPO"
models_dir = f"models/{ALGORITHM}"
log_dir = "logs"
REW_TRESHOLD = 1000 #1000
PADDLE_MULTIPLIER = 2.6
CONTINUE = False
add = 0

class EpisodeDurationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeDurationCallback, self).__init__(verbose)
        self.episode_start_time = None
        self.episode_durations = []
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        self.current_rewards.append(self.locals['rewards'][0])
        if self.locals['dones'][0]:
            episode_duration = self.num_timesteps - self.episode_start_time
            self.episode_durations.append(episode_duration)
            self.episode_start_time = self.num_timesteps
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        return True

    def _on_rollout_start(self) -> None:
        self.episode_start_time = self.num_timesteps
        self.episode_rewards = []  # Reset rewards at the start of each learning segment

    def get_last_episode_duration(self):
        if self.episode_durations:
            return self.episode_durations[-1]
        else:
            return None

    def get_mean_reward(self):
        if self.episode_rewards:
            return sum(self.episode_rewards) / len(self.episode_rewards)
        else:
            return None


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






if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the environment
#env = TwoBallsBalance(render_mode="training")

TIMESTEPS = 10000
NUM_ITERATIONS = 10000  # Adjust according to your needs
change_step = 0 

# Instantiate the agent
env = None

if CONTINUE:
    last_model_path = get_last_saved_model(models_dir)
    
    if last_model_path:
        change_step = int(last_model_path.split('/')[-1].split('_')[0])
        add = int(last_model_path.split('/')[-1].split('_')[1][:-4])
        env = TwoBallsBalance(render_mode="training", paddle=PADDLE_MULTIPLIER - 0.1 * change_step)
        model = load_model_with_new_env(last_model_path, env)
        print(f"Loaded model from {last_model_path}")
    else:
        env = TwoBallsBalance(render_mode="training")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
else:
    print("the contants of the models directory are abou to be deleted")
    while True:
        print("The contents of the models directory are about to be deleted. Continue? (Y/N)")
        user_input = input().strip().upper()
        if user_input == 'Y':
            break
        elif user_input == 'N':
            print("Aborting...")
            exit()
        else:
            print("Invalid input. Please enter 'Y' to continue or 'N' to abort.")
    clean_directory(models_dir)
    env = TwoBallsBalance(render_mode="training")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

callback_durr = EpisodeDurationCallback()





for i in range(1, NUM_ITERATIONS + 1):
    
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=ALGORITHM,
        callback=callback_durr
    )
    rew = callback_durr.get_mean_reward()
    print(f"Iteration {i} - Episode drew: {rew}")
    model.save(f"{models_dir}/{change_step}_{i + add}")

    if rew and rew >= REW_TRESHOLD:
        last_model_path = get_last_saved_model(models_dir)
        mult = PADDLE_MULTIPLIER
        if PADDLE_MULTIPLIER - 0.1 * change_step > 0.5:
            mult = PADDLE_MULTIPLIER - 0.1 * change_step
            change_step += 1
        else:
            mult = 0.5
        print(f"Changing paddle multiplier to {mult}")
        if last_model_path:
            env.close()
            env = TwoBallsBalance(render_mode="training", paddle=mult)  # Example of a different environment
            model = load_model_with_new_env(last_model_path, env)
            print(f"Loaded model from {last_model_path} with new environment")


# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
