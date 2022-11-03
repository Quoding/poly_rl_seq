import copy

import gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import AppendBiasLayer, SlimFC, normc_initializer
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Sequence
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX

CUSTOM_MODEL_CONFIG = {"bn": False, "dropout": 0, "use_masking": True}


class RayNetwork(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # custom_model_config is a kwargs
        custom_model_config = kwargs.get("custom_model_config", {})
        use_bn = custom_model_config.get("bn", False)
        dropout_rate = custom_model_config.get("dropout", 0)
        self.use_masking = custom_model_config.get("use_masking", True)
        # List of layer sizes
        hiddens = list(model_config.get("fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        orig_space = getattr(obs_space, "original_space", obs_space)
        prev_layer_size = int(np.prod(orig_space["observations"].n))

        layers = []
        # Create layers 0 to second-last.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size
            if use_bn:
                layers.append(nn.BatchNorm1d(prev_layer_size))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

        self._hidden_layers = nn.Sequential(*layers)

        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None,
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))

        if self.use_masking:
            mask = input_dict["obs"]["action_mask"]
            obs = input_dict["obs"]["observations"]
        else:
            obs = input_dict["obs"]
            mask = 1

        # Set observations as float because of possible MultiBinary
        # MultiBinary sets 0,1 as Char (int8)
        self._hidden_out = self._hidden_layers(obs.float())
        logits = self._logits(self._hidden_out)
        logits = logits * mask

        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])

    def get_state_value(self, x):
        """For custom use / evaluation only. Is not compatible with regular RLLib workflow

        Args:
            x (torch.Tensor): Input to the VF network, state.

        Returns:
            torch.Tensor: Value of the given observation
        """
        self._hidden_layers.eval()
        x = self._hidden_layers(x)
        logits = self._logits(x)
        x = self._value_branch(x)
        self._hidden_layers.train()
        return x

    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def disable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.eval()


class PreDistRayNetwork(RayNetwork):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):

        RayNetwork.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))

        if self.use_masking:
            mask = input_dict["obs"]["action_mask"]
            obs = input_dict["obs"]["observations"]
        else:
            obs = input_dict["obs"]
            mask = 1

        # Set observations as float because of possible MultiBinary
        # MultiBinary sets 0,1 as Char (int8)
        self._hidden_out = self._hidden_layers(obs.float())
        logits = self._logits(self._hidden_out)
        ret = {"logits": logits, "mask": mask}

        return logits, []


class MaskableDQNTorchModel(DQNTorchModel, nn.Module):
    """Extension of standard TorchModelV2 to provide dueling-Q functionality."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *,
        q_hiddens: Sequence[int] = (256,),
        dueling: bool = False,
        dueling_activation: str = "relu",
        num_atoms: int = 1,
        use_noisy: bool = False,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma0: float = 0.5,
        add_layer_norm: bool = False,
    ):
        """Initialize variables of this model.
        Extra model kwargs:
            q_hiddens (Sequence[int]): List of layer-sizes after(!) the
                Advantages(A)/Value(V)-split. Hence, each of the A- and V-
                branches will have this structure of Dense layers. To define
                the NN before this A/V-split, use - as always -
                config["model"]["fcnet_hiddens"].
            dueling: Whether to build the advantage(A)/value(V) heads
                for DDQN. If True, Q-values are calculated as:
                Q = (A - mean[A]) + V. If False, raw NN output is interpreted
                as Q-values.
            dueling_activation: The activation to use for all dueling
                layers (A- and V-branch). One of "relu", "tanh", "linear".
            num_atoms: If >1, enables distributional DQN.
            use_noisy: Use noisy layers.
            v_min: Min value support for distributional DQN.
            v_max: Max value support for distributional DQN.
            sigma0 (float): Initial value of noisy layers.
            add_layer_norm: Enable layer norm (for param noise).
        """
        nn.Module.__init__(self)
        super(MaskableDQNTorchModel, self).__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            q_hiddens=q_hiddens,
            dueling=dueling,
            dueling_activation=dueling_activation,
            num_atoms=num_atoms,
            use_noisy=use_noisy,
            v_min=v_min,
            v_max=v_max,
            sigma0=sigma0,
            add_layer_norm=add_layer_norm,
        )

        custom_model_config = model_config.get("custom_model_config")
        self.n_actions = action_space.n

        hiddens = list(model_config.get("fcnet_hiddens", [128]))
        activation = model_config.get("fcnet_activation")
        orig_space = getattr(obs_space, "original_space", obs_space)
        prev_layer_size = int(np.prod(orig_space["observations"].n))

        layers = []
        # Create layers 0 to second-last.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        self._hidden_layers = nn.Sequential(*layers)

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.
        Override this in your custom model to customize the Q output head.
        Args:
            model_out: Embedding from the model layers.
        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """
        action_scores = self.advantage_module(model_out)
        # https://github.com/ray-project/ray/issues/9191
        # https://github.com/ray-project/ray/blob/master/rllib/examples/models/parametric_actions_model.py
        # seems like mask should have -inf as a value for invalid actions
        # Ones will be 0s, 0s will become -inf - this mask is additive, not multiplicative
        if self.num_atoms > 1:
            # Distributional Q-learning uses a discrete support z
            # to represent the action value distribution
            z = torch.arange(0.0, self.num_atoms, dtype=torch.float32).to(
                action_scores.device
            )

            # Compute bin edges
            z = self.v_min + z * (self.v_max - self.v_min) / float(self.num_atoms - 1)

            # Regroup atoms per action per obs:
            # ex:
            # we have 2 actions, 5 atoms each, action scores is of the shape:
            # [
            #  [1,2,3,4,6,7,8,9,10      ] (all atoms for both actions, first 5 are for 1st action, last 5 are for 2nd...)
            #  [5,6,7,8,9,10,11,12,13,14]
            # ]

            # This reshape puts it like this:
            # [
            # [
            #   [1,2,3,4,5] # First obs, atoms for action 1
            #   [6,7,8,9,10] # First obs, atoms for action 2
            # ]
            # [
            #   [5,6,7,8,9] # Second obs, atoms for action 1
            #   [10,11,12,13,14] # Second obs, atoms for action 2
            # ]
            # ]
            support_logits_per_action = torch.reshape(
                action_scores, shape=(-1, self.action_space.n, self.num_atoms)
            )

            # Transforms logits to probs
            # So every atom is now a probability mass
            support_prob_per_action = nn.functional.softmax(
                support_logits_per_action, dim=-1
            )

            action_scores = torch.sum(z * support_prob_per_action, dim=-1)
            # logits = support_logits_per_action + self.inf_mask_logits
            # probs = support_prob_per_action + self.inf_mask_logits
            # action_scores = action_scores + self.inf_mask_scores
            logits = support_logits_per_action
            probs = support_prob_per_action
            action_scores = action_scores

            return action_scores, z, logits, logits, probs
        else:
            logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
            logits = logits + inf_mask
            return action_scores, logits, logits

    def get_state_value(self, model_out):
        """Returns the state value prediction for the given state embedding."""
        # embed = model_out.get("embed")
        out = self.value_module(model_out)
        # out = self.value_module(model_out)
        return out

    def value_function(self):
        x = self.get_state_value(self.embed)
        z = torch.arange(0.0, self.num_atoms, dtype=torch.float32).to(x.device)
        z = self.v_min + z * (self.v_max - self.v_min) / float(self.num_atoms - 1)
        support_probs = nn.functional.softmax(x, dim=-1)

        mu = torch.sum(z * support_probs, dim=-1)

        std = (((mu[:, None] - z) ** 2) * support_probs).sum(dim=-1)

        return mu, std

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """We must redefine this with a custom model since we have to handle mask here. Default ModelV2 of the default DQNTorchModel would handle the forward, but wouldn't register the mask."""
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get("is_training", False))

        mask = input_dict["obs"]["action_mask"]
        self.mask = torch.clamp(torch.log(mask), FLOAT_MIN, FLOAT_MAX)
        # self.inf_mask_logits = self.inf_mask_scores[None].reshape(
        #     len(mask), self.n_actions, 1
        # )

        obs = input_dict["obs"]["observations"]
        # Set observations as float because of possible MultiBinary
        # MultiBinary sets 0,1 as Char (int8)
        embed = self._hidden_layers(obs.float())
        self.embed = embed
        return embed, []
