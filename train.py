import logging
import os
from itertools import chain

import gym
import ray
import torch
import torch.nn as nn
from ray.rllib.evaluation import worker_set
from ray.rllib.models import ModelCatalog

import utils
from configs import DEFAULT_ENV_CONFIG, DEFAULT_REPLAY_BUFFER_CONFIG
from environments import env_creator, load_dataset
from networks import RayNetwork
from utils import compute_metrics, parse_args, get_trainer, get_precomputed_action_mask

logging.basicConfig(level=logging.INFO)

ray.init()
args = parse_args()
get_obs_fn = lambda env: env.get_obs_states_idx()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set parameters via config / args
env_config = DEFAULT_ENV_CONFIG
env_config["dataset_name"] = args.dataset
env_config["env_name"] = args.env

combis, risks, pat_vecs, n_obs, n_dim = load_dataset(env_config["dataset_name"])
combis = combis.to(device)
pat_vecs = pat_vecs.to(device)
d1, d2 = combis.shape

masks = get_precomputed_action_mask(args.env, args.dataset, combis, device=device)

#### METRICS STORAGE ####
recalls = []
precisions = []
ratio_found_pats = []
recalls_alls = []
precisions_alls = []
ratio_found_pats_alls = []
n_inter_alls = []
losses = []
dataset_losses = []
all_flagged_combis_idx = set()
all_flaggeds_risks = []
all_flagged_pats_idx = set()


# Define true solution
true_sol_idx = torch.where(risks > args.threshold)[0]
true_sol = combis[true_sol_idx]
true_sol_idx = set(true_sol_idx.tolist())
n_combis_in_sol = len(true_sol_idx)

logging.info(f"There are {n_combis_in_sol} combinations in the solution set")


### SET UP NETWORK AND TRAINER ###
trainer = get_trainer(args, env_config, device)
### TRAINING LOOP ###
for i in range(args.iters):
    print(i)
    results = trainer.train()  # Get `train_batch_size` observations and learn on them
    with torch.no_grad():
        trainer.get_policy().config["explore"] = False
        # Evaluate custom metrics
        # Get value function
        # lr sechd - get_policy().lr_schedule = schedule class
        # alternative: trainer.save() - trainer.stop() - init a new trainer - trainer.restor(save_obj)
        # net = trainer.get_policy().model.to(device)
        (
            metrics_dict,
            all_flagged_combis_idx,
            all_flagged_pats_idx,
            estimates,
        ) = compute_metrics(
            trainer,
            combis,
            args.threshold,
            pat_vecs,
            true_sol_idx,
            all_flagged_combis_idx,
            all_flagged_pats_idx,
            env_config["env_name"],
            args.gamma,
            env_config["step_penalty"],
            torch.ones(d1, d2 + 1),
            args.n_sigmas,
            device=device,
            seen_idx="all",
        )

        # Note that metrics are computed w.r.t. "seen_idx states" (can be all, or just observed states)
        recalls.append(metrics_dict["recall"])
        precisions.append(metrics_dict["precision"])
        ratio_found_pats.append(metrics_dict["ratio_found_pat"])
        recalls_alls.append(metrics_dict["recall_all"])
        precisions_alls.append(metrics_dict["precision_all"])
        ratio_found_pats_alls.append(metrics_dict["ratio_found_pat_all"])
        n_inter_alls.append(metrics_dict["n_inter_all"])

        logging.info(
            f"trial: {i + 1}, recall: {recalls[-1]}, precision: {precisions[-1]}, ratio of patterns found: {ratio_found_pats[-1]}, n_inter: {metrics_dict['n_inter']}"
        )
        logging.info(
            f"recall all: {recalls_alls[-1]}, precision all: {precisions_alls[-1]}, ratio of patterns found all: {ratio_found_pats_alls[-1]}, n_inter all: {n_inter_alls[-1]}"
        )

        # Record observed states to avoid scouring combinatorial space later
        all_observed_states_idx = [
            ray.get(worker.foreach_env.remote(get_obs_fn))
            for worker in trainer.workers.remote_workers()
        ]
        # Unpack dimensions (1st dim is workers, 2nd is envs, 3rd are observations)
        all_observed_states_idx = list(chain(*chain(*all_observed_states_idx)))
        logging.info(
            f"Number of unique states seen: {len(set(all_observed_states_idx))}"
        )
        trainer.get_policy().config["explore"] = True

    # import matplotlib.pyplot as plt
    # import numpy as np

    # laced_idx = [None] * ((i + 1) * args.trials)
    # laced_idx[::2] = all_observed_states_idx[: (i + 1) * (args.trials // 2)]
    # laced_idx[1::2] = all_observed_states_idx[
    #     (i + 1) * (args.trials // 2) : (i + 1) * args.trials
    # ]
    # a = risks[torch.tensor(laced_idx)]
    # b = np.arange(len(a))

    # plt.scatter(b, a)
    # plt.savefig(f"viz/images/whatsplayed/{i}.png")
    # plt.clf()

    # plt.scatter(risks.cpu().numpy(), estimates.cpu().numpy())
    # plt.ylim(0, 3)
    # plt.xlim(0, 3)
    # plt.savefig(f"viz/images/pred_vs_gt/{i}.png")
    # plt.clf()
    # print("=======")

# Save metrics on disk
l = [
    "agents",
    "recalls",
    "precisions",
    "ratio_found_pats",
    "losses",
    "dataset_losses",
    "recalls_alls",
    "precisions_alls",
    "ratio_found_pats_alls",
    "n_inter_alls",
    "all_flagged_combis_idx",
    "all_flagged_risks",
    "all_observed_states_idx",
]
try:
    all_flagged_risks = risks[torch.tensor(list(all_flagged_combis_idx))]
except IndexError as e:
    logging.info("No flagged combination during the entire experiment")
    logging.info("all_flagged_risks is now an empty tensor")
    all_flagged_risks = torch.tensor([])

for item in l:
    os.makedirs(f"{args.output}/{item}/", exist_ok=True)

# torch.save(agent, f"{args.output}/agents/{args.seed}.pth")
trainer.save(f"{args.output}/agents/{args.seed}/")
torch.save(recalls, f"{args.output}/recalls/{args.seed}.pth")
torch.save(precisions, f"{args.output}/precisions/{args.seed}.pth")
torch.save(ratio_found_pats, f"{args.output}/ratio_found_pats/{args.seed}.pth")
torch.save(losses, f"{args.output}/losses/{args.seed}.pth")
torch.save(dataset_losses, f"{args.output}/dataset_losses/{args.seed}.pth")
torch.save(recalls_alls, f"{args.output}/recalls_alls/{args.seed}.pth")
torch.save(precisions_alls, f"{args.output}/precisions_alls/{args.seed}.pth")
torch.save(
    ratio_found_pats_alls, f"{args.output}/ratio_found_pats_alls/{args.seed}.pth"
)
torch.save(n_inter_alls, f"{args.output}/n_inter_alls/{args.seed}.pth")
torch.save(all_flagged_risks, f"{args.output}/all_flagged_risks/{args.seed}.pth")
torch.save(
    all_flagged_combis_idx, f"{args.output}/all_flagged_combis_idx/{args.seed}.pth"
)
torch.save(
    all_observed_states_idx, f"{args.output}/all_observed_states_idx/{args.seed}.pth"
)
