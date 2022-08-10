from ray.rllib.agents import ppo
import ray
from environments import DEFAULT_CONFIG, PolypharmacyEnv
from utils import MetricsCallback

ray.init()


config = DEFAULT_CONFIG
config["dataset_name"] = "50_rx_100000_combis_4_patterns_3"

trainer = ppo.PPOTrainer(
    env=PolypharmacyEnv,
    config={
        "framework": "torch",
        "env_config": config,
        "num_workers": 0,
        "callbacks": MetricsCallback,
    },
)

results = trainer.train()
print(results)
