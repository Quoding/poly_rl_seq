from ray.rllib.agents import ppo
import ray
from environments import DEFAULT_CONFIG, PolypharmacyEnv
from ray.rllib.models import ModelCatalog
from utils import MetricsCallback
from networks import RayNetwork
import torch.nn as nn
import gym

ray.init()

ModelCatalog.register_custom_model(
    "my_torch_model",
    RayNetwork,
)
config = DEFAULT_CONFIG
config["dataset_name"] = "50_rx_100000_combis_4_patterns_3"
model_config = {"fcnet_hiddens": [10, 10, 10], "fcnet_activation": nn.ReLU}
net = RayNetwork(
    gym.spaces.Discrete(10),
    gym.spaces.Discrete(11),
    11,
    model_config=model_config,
    name="TestModel",
    custom_model_config={"use_bn": True},
)

trainer = ppo.PPOTrainer(
    env=PolypharmacyEnv,
    config={
        "framework": "torch",
        "env_config": config,
        # "num_workers": 0,
        "model": {
            "custom_model": "my_torch_model",
            "custom_model_config": {"use_bn": True},
            "fcnet_hiddens": [128],
            "fcnet_activation": "relu",
        }
        # "callbacks": MetricsCallback,
    },
)

# results = trainer.train()
# print(results)
