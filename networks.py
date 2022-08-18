import copy

import gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import AppendBiasLayer, SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch

CUSTOM_MODEL_CONFIG = {"bn": True}


class Network(nn.Module):
    def __init__(
        self,
        dim,
        n_hidden_layers,
        n_output=1,
        hidden_size=128,
        dropout_rate=None,
        batch_norm=False,
    ):
        super().__init__()
        layers = nn.ModuleList()

        layers.append(nn.Linear(dim, hidden_size))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, n_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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
        use_bn = custom_model_config.get("bn", True)

        # List of layer sizes
        hiddens = list(model_config.get("fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        prev_layer_size = int(np.prod(obs_space.shape))

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

        # Set observations as float because of possible MultiBinary
        # MultiBinary sets 0,1 as Char (int8)
        self._hidden_out = self._hidden_layers(input_dict["obs"].float())
        logits = self._logits(self._hidden_out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])

    def full_vf(self, x):
        """For custom use / evaluation only. Is not compatible with regular RLLib workflow

        Args:
            x (torch.Tensor): Input to the VF network, state.

        Returns:
            torch.Tensor: Value of the given observation
        """
        self._hidden_layers.eval()
        x = self._hidden_layers(x)
        x = self._value_branch(x)
        self._hidden_layers.train()
        return x


class NetworkDropout(nn.Module):
    def __init__(self, dim, hidden_size=100, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        return self.fc2(self.activate(self.dropout(self.fc1(x))))


class VariableNet(nn.Module):
    def __init__(self, dim, layer_widths):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim, layer_widths[0]))
        self.layers.append(nn.ReLU())

        for i in range(len(layer_widths) - 1):
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(layer_widths[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
