import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)


class ActionNN(nn.Module):
    def __init__(self, features_dim, n_actions=2, n_action_outputs=20, hidden_dim=64):
        super(ActionNN, self).__init__()
        self.input_dim = features_dim
        self.num_actions = n_actions
        self.num_outputs = n_action_outputs

        # Action selection subnet
        self.action_selection = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)  # Ensure that the sum of the outputs is 1
        )

        # Output subnet
        self.output_net = nn.Sequential(
            nn.Linear(features_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_action_outputs),
            nn.Tanh()  # Ensure the outputs are between -1 and 1
        )

    def forward(self, x):
        # Action selection
        action_probs = self.action_selection(x)
        selected_action = torch.argmax(action_probs, dim=-1, keepdim=True)
        
        # One-hot encoding of the selected action
        action_one_hot = torch.zeros_like(action_probs).scatter_(1, selected_action, 1)

        # Concatenate action selection with features
        combined_input = torch.cat((action_one_hot, x), dim=1)

        # Output network
        output_values = self.output_net(combined_input)
        
        # Concatenate action_probs and output_values
        concatenated_output = torch.cat((action_probs, output_values), dim=1)
        
        return concatenated_output


class AdvancedActor(Actor):
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
        action_config: Optional[Dict[str, Any]] = None,
    ):
        super(AdvancedActor, self).__init__(observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn, 
                                            use_sde, log_std_init, full_std, use_expln, clip_mean, normalize_images)
        
        if action_config is not None:
            n_actions = action_config['n_actions']
            n_action_outputs = action_config['n_action_outputs']
            hidden_dim = action_config['hidden_dim']
        else:
            n_actions = 2
            n_action_outputs = 20
            hidden_dim = 256
        
        self.latent_pi = ActionNN(features_dim=features_dim, n_actions=n_actions, n_action_outputs=n_action_outputs, hidden_dim=hidden_dim).to(self.device)
    
class ActionSACPolicy(SACPolicy):
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
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        action_config: Optional[Dict[str, Any]] = None,
    ):
        self.action_config = action_config # Here as it is used during the super() call
        super(ActionSACPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, use_sde, log_std_init, use_expln, clip_mean, features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class, optimizer_kwargs, n_critics, share_features_extractor)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> AdvancedActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor = AdvancedActor(**actor_kwargs, action_config=self.action_config).to(self.device)
        actor.mu = nn.Identity()
        return actor


class ActionSAC(SAC):
    policy: ActionSACPolicy
    actor: AdvancedActor

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):        
        
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, ent_coef, target_update_interval, target_entropy, use_sde, sde_sample_freq, use_sde_at_warmup, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)

