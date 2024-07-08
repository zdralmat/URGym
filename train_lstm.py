from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
import gymnasium as gym
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3 import SAC

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class LSTMActor(BasePolicy):
    """
    Actor network (policy) for SAC with an LSTM layer.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch[:-1], activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-2] if len(net_arch) > 1 else features_dim

        self.lstm = nn.LSTM(input_size=last_layer_dim, hidden_size=net_arch[-1], batch_first=True)
        last_layer_dim = net_arch[-1]

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        
        # Assuming the features input is a 3D tensor for LSTM (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(latent_pi.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        
        mean_actions = self.mu(lstm_out)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=lstm_out)
        log_std = self.log_std(lstm_out)  # type: ignore[operator]
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class LSTMSACPolicy(BasePolicy):
    """
    Custom SAC Policy class with an LSTM layer in the actor.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: LSTMActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> LSTMActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return LSTMActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode



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
if str_algo == "LSTM":
	algo = SAC
	policy_arch = LSTMSACPolicy
else:
	algo = getattr(sys.modules[__name__], str_algo) # Obtains the classname based on the string
	policy_arch = "MlpPolicy"

if policy_file:
	# No need to load the hyperparams file as they are already stored in the policy file
	model = algo.load(f"{policy_file}", env=env, tensorboard_log=tblog_dir, verbose=True)
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
