import logging
import os
from itertools import chain

import ray
import torch

from configs import DEFAULT_ENV_CONFIG
from environments import load_dataset
from utils import compute_metrics, get_trainer, parse_args

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
# combis = combis.to(device)
# pat_vecs = pat_vecs.to(device)
d1, d2 = combis.shape

# masks = get_precomputed_action_mask(args.env, args.dataset, combis, device=device)

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

### EVALUATE BEFORE TRAINING ###
logging.info("Evaluating metrics before training")
# with torch.no_grad():
#     trainer.get_policy().config["explore"] = False
#     # Evaluate custom metrics
#     (
#         metrics_dict,
#         all_flagged_combis_idx,
#         all_flagged_pats_idx,
#         estimates,
#     ) = compute_metrics(
#         trainer,
#         combis,
#         args.threshold,
#         pat_vecs,
#         true_sol_idx,
#         all_flagged_combis_idx,
#         all_flagged_pats_idx,
#         env_config["env_name"],
#         args.gamma,
#         env_config["step_penalty"],
#         torch.ones(d1, d2 + 1),
#         args.n_sigmas,
#         device=device,
#         seen_idx="all",
#     )
#     # Note that metrics are computed w.r.t. "seen_idx states" (can be all, or just observed states)
#     recalls.append(metrics_dict["recall"])
#     precisions.append(metrics_dict["precision"])
#     ratio_found_pats.append(metrics_dict["ratio_found_pat"])
#     recalls_alls.append(metrics_dict["recall_all"])
#     precisions_alls.append(metrics_dict["precision_all"])
#     ratio_found_pats_alls.append(metrics_dict["ratio_found_pat_all"])
#     n_inter_alls.append(metrics_dict["n_inter_all"])
#     trainer.get_policy().config["explore"] = True

logging.info("Starting training")

### TRAINING LOOP ###
for i in range(args.iters):
    print(i)
    # Get `train_batch_size = args.trials` observations and learn on them
    results = trainer.train()

    if ((i + 1) * args.trials) % args.eval_step == 0:
        with torch.no_grad():
            trainer.get_policy().config["explore"] = False
            trainer.save(f"{args.output}/agents_intermediate/{args.seed}/{i + 1}/")

            # Evaluate custom metrics
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
trainer.save(f"{args.output}/agents/{args.seed}/")


# https://github.com/ray-project/ray/issues/6102
