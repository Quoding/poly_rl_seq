import logging

import gym
import ray
import torch
import torch.nn as nn
from ray.rllib.agents import ppo
from ray.rllib.evaluation import worker_set
from ray.rllib.models import ModelCatalog
from importlib import reload


from configs import DEFAULT_ENV_CONFIG, DEFAULT_REPLAY_BUFFER_CONFIG
from environments import PolypharmacyEnv
from networks import RayNetwork
from utils import TrackingCallback, compute_metrics, load_dataset
import utils

logging.basicConfig(level=logging.INFO)

ray.init()

# Set parameters via config / args
env_config = DEFAULT_ENV_CONFIG
env_config["dataset_name"] = "50_rx_100000_combis_4_patterns_3"

thresh = 1.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### METRICS STORAGE ####
jaccards = []
ratio_apps = []
ratio_found_pats = []
jaccards_alls = []
ratio_apps_alls = []
ratio_found_pats_alls = []
n_inter_alls = []
losses = []
dataset_losses = []
all_flagged_combis_idx = set()
all_flaggeds_risks = []
all_flagged_pats_idx = set()
combis, risks, pat_vecs, n_obs, n_dim = load_dataset(env_config["dataset_name"])
combis = combis.to(device)
pat_vecs = pat_vecs.to(device)

# Define true solution
true_sol_idx = torch.where(risks > thresh)[0]
true_sol = combis[true_sol_idx]
true_sol_idx = set(true_sol_idx.tolist())
n_combis_in_sol = len(true_sol_idx)

logging.info(f"There are {n_combis_in_sol} combinations in the solution set")


### SET UP NETWORK AND TRAINER ###
ModelCatalog.register_custom_model(
    "BNNetwork",
    RayNetwork,
)
trainer = ppo.PPOTrainer(
    env=PolypharmacyEnv,
    config={
        "framework": "torch",
        "env_config": env_config,
        # "num_workers": 0,
        "num_gpus": 1,
        "model": {
            "custom_model": "BNNetwork",
            "custom_model_config": {"use_bn": True},
            "fcnet_hiddens": [128],
            "fcnet_activation": "relu",
        },
        "seed": 0,
        "horizon": env_config["horizon"],
        "train_batch_size": 200,
        "callbacks": TrackingCallback,
    },
)
# Rollout fragment will be adjusted to a divider of 200

### TRAINING LOOP ###
for i in range(1):
    results = trainer.train()  # Get `train_batch_size` observations and learn on them

    with torch.no_grad():
        # Evaluate custom metrics
        # Get value function
        vf = trainer.get_policy().model.full_vf

        (metrics_dict, all_flagged_combis_idx, all_flagged_pats_idx) = compute_metrics(
            vf,
            combis,
            thresh,
            pat_vecs,
            true_sol_idx,
            1,
            all_flagged_combis_idx,
            all_flagged_pats_idx,
        )

        jaccards.append(metrics_dict["jaccard"])
        ratio_apps.append(metrics_dict["ratio_app"])
        ratio_found_pats.append(metrics_dict["percent_found_pat"])
        jaccards_alls.append(metrics_dict["jaccard_all"])
        ratio_apps_alls.append(metrics_dict["ratio_app_all"])
        ratio_found_pats_alls.append(metrics_dict["percent_found_pat_all"])
        n_inter_alls.append(metrics_dict["n_inter_all"])

        fn = lambda env: env.get_obs_states()

        observations = [
            ray.get(worker.foreach_env.remote(fn))
            for worker in trainer.workers.remote_workers()
        ]

        print(len(observations[0][0]))
