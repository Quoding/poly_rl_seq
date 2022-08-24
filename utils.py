import argparse
import json
from os.path import exists
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn.functional import softmax


# class PretrainDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

#     def __len__(self):
#         return len(self.X)


# def pretrain(X, y, vf, n_epochs=100, pessimistic=True):
#     y = y - (pessimistic * 0.5)
#     dataset = PretrainDataset(X, y)
#     # Play n_obs times without any logic just to gather experience. Sample randomly from environment.
#     dataloader = DataLoader(dataset, batchsize=512, shuffle=True)
#     optim = Adam(vf.parameters(), lr=0.01)
#     criterion = MSELoss()
#     for e in range(n_epochs):
#         for X, y in dataloader:
#             preds = vf(X)
#             loss = criterion(preds, y)
#             loss.backward()
#             optim.step()
#             optim.zero_grad()

#     return vf


def load_dataset(dataset_name, path_to_dataset="datasets"):

    dataset = pd.read_csv(f"{path_to_dataset}/combinations/{dataset_name}.csv")

    with open(f"{path_to_dataset}/patterns/{dataset_name}.json", "r") as f:
        patterns = json.load(f)
    # Remove last 3 columns that are risk, inter, dist
    combis = dataset.iloc[:, :-3]

    # Retrieve risks
    risks = dataset.iloc[:, -3]

    n_obs, n_dim = combis.shape

    pat_vecs = torch.tensor(
        [patterns[f"pattern_{i}"]["pattern"] for i in range(len(patterns))]
    )
    combis, risks = (
        torch.tensor(combis.values).float(),
        torch.tensor(risks.values).unsqueeze(1).float(),
    )

    return combis, risks, pat_vecs, n_obs, n_dim


def make_deterministic(seed=42):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Numpy
    np.random.seed(seed)

    # Built-in Python
    random.seed(seed)


def compute_jaccard(found_solution: set, true_solution: set):
    n_in_inter = 0

    intersection = found_solution & true_solution

    n_in_inter = len(intersection)

    return (
        n_in_inter / (len(found_solution) + len(true_solution) - n_in_inter),
        n_in_inter,
    )


def compute_metrics(
    net,
    combis,
    thresh,
    pat_vecs,
    true_sol_idx,
    all_flagged_combis_idx,
    all_flagged_pats_idx,
    seen_idx="all",
    step_penalty=1,
):
    """Compute metrics for combination test

    Args:
        vf_net (nn.Module): Value function neural network
        combis (torch.Tensor): all possible combinations of Rx in the dataset
        thresh (float): threshold of risk
        pat_vecs (torch.Tensor): pattern vectors used to generate dataset
        true_sol (torch.Tensor): true solution of the dataset
        n_sigmas (float): number of sigmas to consider (sigma-rule sense)
        all_flagged_combis (torch.Tensor): all previously flagged combinations
        all_flagged_pats (torch.Tensor): all previously flagged patterns

    Returns:
        tuple: tuple of metrics and updated tensors in the following order:
        metrics_dict containing:
            jaccard for current step,
            ratio_app for current step,
            percent_found_pat for current step,
            n_inter for current step,
            jaccard for all steps so far,
            ratio_app for all steps so far,
            percent_found_pat for all steps so far,
            n_inter for all steps so far,
        updated all flagged combis,
        updated all flagged pats,
    """
    if seen_idx != "all":
        seen_idx = torch.tensor(list(seen_idx))
        combis = combis[seen_idx]
    else:
        seen_idx = torch.tensor(list(range(len(combis))))

    # Parmis tous les vecteurs "existant", lesquels je trouve ? (Jaccard, ratio_app)
    action_probs = softmax(net(combis), dim=1)
    state_values = net.value_function(combis)

    sol_idx = set(seen_idx[torch.where(net(combis) > thresh)[0]].tolist())

    all_flagged_combis_idx.update(sol_idx)

    # Parmis les patrons dangereux (ground truth), combien j'en trouve tels quels
    sol_pat_idx = set(torch.where(net(pat_vecs) > thresh)[0].tolist())

    all_flagged_pats_idx.update(sol_pat_idx)

    # À quel point ma solution trouvée parmis les vecteurs du dataset est similaire à la vraie solution
    jaccard, n_inter = compute_jaccard(
        sol_idx, true_sol_idx
    )  # Jaccard for the current step

    jaccard_all, n_inter_all = compute_jaccard(
        all_flagged_combis_idx, true_sol_idx
    )  # Jaccard for all steps before + this one if we keep all previous solutions

    # Combien de patrons tels quels j'ai flag ?
    ratio_found_pat = len(sol_pat_idx) / len(pat_vecs)  # For this step
    ratio_found_pat_all = len(all_flagged_pats_idx) / len(
        pat_vecs
    )  # For all previous steps and this one

    # A quel point ma solution trouvee parmis les vecteurs du dataset est dans la vraie solution
    if len(sol_idx) == 0:
        ratio_app = float("nan")
    else:
        ratio_app = n_inter / len(sol_idx)

    if len(all_flagged_combis_idx) == 0:
        ratio_app_all = float("nan")
    else:
        ratio_app_all = n_inter_all / len(all_flagged_combis_idx)

    metrics_dict = {}

    metrics_dict["jaccard"] = jaccard
    metrics_dict["ratio_app"] = ratio_app
    metrics_dict["ratio_found_pat"] = ratio_found_pat
    metrics_dict["n_inter"] = n_inter
    metrics_dict["jaccard_all"] = jaccard_all
    metrics_dict["ratio_app_all"] = ratio_app_all
    metrics_dict["ratio_found_pat_all"] = ratio_found_pat_all
    metrics_dict["n_inter_all"] = n_inter_all

    return (
        metrics_dict,
        all_flagged_combis_idx,
        all_flagged_pats_idx,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on a given dataset")

    parser.add_argument(
        "-T", "--trials", type=int, default=200, help="Number of steps per iterations"
    )
    parser.add_argument(
        "-I", "--iters", type=int, default=100, help="Number of steps per iterations"
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="Name of dataset (located in datasets/*)"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=1.1,
        help="Good and bad action threshold",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Set random seed base for training",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=128,
        help="Width of the NN (number of neurons)",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        default=1,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="metrics/ouput/",
        help="Output directory for metrics and agents",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="randomstart",
        choices=["randomstart", "singlestart"],
        help="Environment class to use",
    )
    args = parser.parse_args()
    return args
