import json

import pandas as pd
import torch
from typing import Dict, Tuple, Any
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from os.path import exists

obs_tracking = []


class TrackingCallback(DefaultCallbacks):
    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        pass


def warmup(n_obs):
    # Play n_obs times without any logic just to gather experience. Sample randomly from environment.
    pass


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
    vf_net,
    combis,
    thresh,
    pat_vecs,
    true_sol_idx,
    n_sigmas,
    all_flagged_combis_idx,
    all_flagged_pats_idx,
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

    # Parmis tous les vecteurs "existant", lesquels je trouve ? (Jaccard, ratio_app)
    sol_idx = set(torch.where(vf_net(combis) > thresh)[0].tolist())

    all_flagged_combis_idx.update(sol_idx)

    # Parmis les patrons dangereux (ground truth), combien j'en trouve tels quels
    sol_pat_idx = set(torch.where(vf_net(pat_vecs) > thresh)[0].tolist())

    all_flagged_pats_idx.update(sol_pat_idx)

    # À quel point ma solution trouvée parmis les vecteurs du dataset est similaire à la vraie solution
    jaccard, n_inter = compute_jaccard(
        sol_idx, true_sol_idx
    )  # Jaccard for the current step

    jaccard_all, n_inter_all = compute_jaccard(
        all_flagged_combis_idx, true_sol_idx
    )  # Jaccard for all steps before + this one if we keep all previous solutions

    # Combien de patrons tels quels j'ai flag ?
    percent_found_pat = len(sol_pat_idx) / len(pat_vecs)  # For this step
    percent_found_pat_all = len(all_flagged_pats_idx) / len(
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
    metrics_dict["percent_found_pat"] = percent_found_pat
    metrics_dict["n_inter"] = n_inter
    metrics_dict["jaccard_all"] = jaccard_all
    metrics_dict["ratio_app_all"] = ratio_app_all
    metrics_dict["percent_found_pat_all"] = percent_found_pat_all
    metrics_dict["n_inter_all"] = n_inter_all

    return (
        metrics_dict,
        all_flagged_combis_idx,
        all_flagged_pats_idx,
    )


def train(trainer, n_episodes, eval_every):
    pass
