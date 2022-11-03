import argparse
from os.path import exists
from typing import Any, Dict, Tuple

import torch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.algorithms import ppo, dqn
from ray.rllib.algorithms.dqn.dqn_torch_policy import compute_q_values
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import MSELoss
from ray.tune.registry import register_env
from environments import env_creator, get_action_mask
from networks import (
    RayNetwork,
    PreDistRayNetwork,
    MaskableDQNTorchModel,
    CUSTOM_MODEL_CONFIG,
)
import numpy as np

# trainer_names_map = {"ppo": ppo.PPOTrainer, "dqn": }


def get_trainer(args, env_config, device):
    register_env("polyenv", env_creator)

    model_config = CUSTOM_MODEL_CONFIG
    if "mask" in env_config["env_name"]:
        model_config["use_masking"] = True

    # Rollout fragment will be adjusted to a divider of `train_batch_size`
    if args.agent == "ppo":
        ModelCatalog.register_custom_model("ppo_custom_net", RayNetwork)
        trainer = ppo.PPO(
            env="polyenv",
            config={
                "framework": "torch",
                "env_config": env_config,
                "num_workers": 2,
                "num_gpus": 1 if device == torch.device("cuda") else 0,
                "model": {
                    "custom_model": "ppo_custom_net",
                    "custom_model_config": model_config,
                    "fcnet_hiddens": [args.width] * args.layers,
                    "fcnet_activation": "relu",
                },
                "seed": args.seed,
                "horizon": env_config["horizon"],
                "train_batch_size": args.trials,
                "gamma": args.gamma,
                "sgd_minibatch_size": args.batchsize,
                "num_sgd_iter": args.epochs,
                "lr": args.lr,
            },
        )
    elif args.agent == "rainbow":
        ModelCatalog.register_custom_model("dqn_custom_net", MaskableDQNTorchModel)

        trainer = dqn.DQN(
            env="polyenv",
            config={
                "framework": "torch",
                "env_config": env_config,
                "num_workers": 2,
                "num_gpus": 1 if device == torch.device("cuda") else 0,
                "model": {
                    "custom_model": "dqn_custom_net",
                    "fcnet_hiddens": [args.width] * args.layers,
                    "fcnet_activation": "relu",
                },
                "seed": args.seed,
                "horizon": env_config["horizon"],
                "train_batch_size": args.trials,
                "gamma": args.gamma,
                "lr": args.lr,
                "num_atoms": 10,
                "v_min": -1,
                "v_max": 5,
                "n_step": 5,
                "noisy": True,
                "dueling": True,
                "double_q": True,
                "sigma0": 0.5,
                "hiddens": [args.width],
                "disable_env_checking": True,
                # "exploration_config": {
                #     # The Exploration class to use. In the simplest case, this is the name
                #     # (str) of any class present in the `rllib.utils.exploration` package.
                #     # You can also provide the python class directly or the full location
                #     # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
                #     # EpsilonGreedy").
                #     "type": "StochasticSampling",
                #     # Add constructor kwargs here (if any).
                # },
            },
        )

    return trainer


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


def compute_n_inter(found_solution: set, true_solution: set):

    intersection = found_solution & true_solution

    n_in_inter = len(intersection)

    return n_in_inter


def compute_metrics(
    agent,
    combis,
    thresh,
    pat_vecs,
    true_sol_idx,
    all_flagged_combis_idx,
    all_flagged_pats_idx,
    env_name,
    gamma,
    step_penalty,
    masks,
    n_sigmas,
    device,
    seen_idx="all",
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
            recall for current step,
            precision for current step,
            percent_found_pat for current step,
            n_inter for current step,
            recall for all steps so far,
            precision for all steps so far,
            percent_found_pat for all steps so far,
            n_inter for all steps so far,
        updated all flagged combis,
        updated all flagged pats,
    """
    if seen_idx != "all":
        seen_idx = torch.tensor(list(seen_idx))
        combis = combis[seen_idx]
        masks = masks[seen_idx]
    else:
        seen_idx = torch.tensor(list(range(len(combis)))).to(device)

    all_combis_estimated_reward = get_estimated_state_reward(
        agent, combis, step_penalty, env_name, gamma, masks, n_sigmas, device
    )
    all_combis_estimated_reward = all_combis_estimated_reward.cpu()
    # Parmis tous les vecteurs "existant", lesquels je trouve ? (recall, precision)
    sol_idx = set(
        seen_idx[torch.where(all_combis_estimated_reward > thresh)[0]].tolist()
    )

    all_flagged_combis_idx.update(sol_idx)

    # Parmis les patrons dangereux (ground truth), combien j'en trouve tels quels
    d1, d2 = pat_vecs.shape
    all_pats_estimated_reward = get_estimated_state_reward(
        agent,
        pat_vecs,
        step_penalty,
        env_name,
        gamma,
        torch.ones(d1, d2 + 1),
        n_sigmas,
        device,
    )

    sol_pat_idx = set(torch.where(all_pats_estimated_reward > thresh)[0].tolist())

    all_flagged_pats_idx.update(sol_pat_idx)

    # À quel point ma solution trouvée parmis les vecteurs du dataset est similaire à la vraie solution
    n_inter = compute_n_inter(sol_idx, true_sol_idx)  # recall for the current step

    n_inter_all = compute_n_inter(
        all_flagged_combis_idx, true_sol_idx
    )  # recall for all steps before + this one if we keep all previous solutions

    # Combien de patrons tels quels j'ai flag ?
    ratio_found_pat = len(sol_pat_idx) / len(pat_vecs)  # For this step
    ratio_found_pat_all = len(all_flagged_pats_idx) / len(
        pat_vecs
    )  # For all previous steps and this one

    # A quel point ma solution trouvee parmis les vecteurs du dataset est dans la vraie solution
    if len(sol_idx) == 0:
        recall = float("nan")
        precision = float("nan")
    else:
        recall = n_inter / len(true_sol_idx)
        precision = n_inter / len(sol_idx)

    if len(all_flagged_combis_idx) == 0:
        recall_all = float("nan")
        precision_all = float("nan")
    else:
        recall_all = n_inter_all / len(true_sol_idx)
        precision_all = n_inter_all / len(all_flagged_combis_idx)

    metrics_dict = {}

    metrics_dict["recall"] = recall
    metrics_dict["precision"] = precision
    metrics_dict["ratio_found_pat"] = ratio_found_pat
    metrics_dict["n_inter"] = n_inter
    metrics_dict["recall_all"] = recall_all
    metrics_dict["precision_all"] = precision_all
    metrics_dict["ratio_found_pat_all"] = ratio_found_pat_all
    metrics_dict["n_inter_all"] = n_inter_all

    return (
        metrics_dict,
        all_flagged_combis_idx,
        all_flagged_pats_idx,
        all_combis_estimated_reward,
    )


def get_estimated_state_reward(
    agent, combis, step_penalty, env_name, gamma, masks, n_sigmas, device
):
    """Compute estimates of state's reward for every combinations in `combis`
    Value function loss for PPO's critic:


    Args:
        net ([type]): [description]
        combis ([type]): [description]
        step_penalty ([type]): [description]
        env_name ([type]): [description]
        gamma ([type]): [description]
        n_sigmas ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    net = agent.get_policy().model
    d1, d2 = combis.shape

    # Expand combis to have the extra terminal state
    combis_states = torch.zeros(d1, d2 + 1)
    combis_states[:, :d2] = combis
    # combis_states = combis_states.to(device)
    # masks = masks.cpu().numpy() if type(masks) == torch.Tensor else masks
    taken_actions = []
    for i in range(d1):
        if masks is not None:
            obs = {
                "observations": combis_states[i],
                "action_mask": masks[i],
            }
        else:
            obs = {"obs": combis_states[i]}

        action = agent.compute_single_action(obs).astype("int64")
        taken_actions.append(action)
    taken_actions = torch.tensor(taken_actions)[None].to(device)
    combis_states = combis_states.to(device)
    next_states = combis_states.clone()

    if masks is not None:
        obs = {
            "obs": {
                "observations": combis_states,
                "action_mask": masks,
            }
        }
    else:
        obs = {"obs": combis_states}

    # Forward pass for state value
    net(obs)
    state_values, std_s1 = net.value_function()
    if "random" in env_name:
        # If env has a random start, then actions are like switches
        # hence the logical_not
        combi_bits = next_states.gather(1, taken_actions)
        combi_bits = torch.logical_not(combi_bits).float()
        next_states = next_states.scatter(1, taken_actions, combi_bits)
    elif env_name == "singlestart":
        # if env is single start, we can only set to one
        next_states = next_states.scatter(1, taken_actions, 1)

    # Forward pass needs to happen on policy before value is computed
    # Particular to RayNetwork taken from tutorial...
    if masks is not None:
        obs = {
            "obs": {
                "observations": next_states,
                "action_mask": torch.ones(d1, d2 + 1).to(device),
            }
        }
    else:
        obs = {"obs": next_states}

    net(obs)
    next_state_values, std_s2 = net.value_function()

    # Gives a worst case estimated reward for the state
    estimated_reward = (state_values - n_sigmas * std_s1 + step_penalty) - gamma * (
        next_state_values + n_sigmas * std_s2
    )
    return estimated_reward


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
        default="metrics/output",
        help="Output directory for metrics and agents",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="randomstart",
        choices=["randomstart", "singlestart", "maskedrandom", "maskedsingle"],
        help="Environment class to use",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Value of discount factor",
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="Minibatch size for gradient based optimizer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs during training for NN",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent",
    )

    parser.add_argument(
        "--n_sigmas",
        type=int,
        default=3,
        help="Number of sigmas to consider for confidence interval",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=200,
        help="Evaluate agent every `eval_step` steps",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rainbow",
        choices=["rainbow", "ppo"],
        help="Agent to use",
    )
    args = parser.parse_args()
    return args


def get_precomputed_action_mask(env_name, dataset_name, combis, device):
    if "mask" in env_name:
        print("Mask usage detected")
        if exists(f"datasets/masks/{dataset_name}.pth"):
            print("Loading precomputed masks")
            masks = torch.load(f"datasets/masks/{dataset_name}.pth")
        else:
            print("No precomputed masks found. Computing them right now.")
            d1, d2 = combis.shape
            masks = torch.zeros(d1, d2 + 1)
            combis_states = torch.zeros(d1, d2 + 1)
            combis_states[:, :d2] = combis
            combis_states = combis_states.to(device)
            for i in range(len(combis_states)):
                masks[i] = torch.from_numpy(get_action_mask(combis_states[i], combis))

            masks = masks.cpu().numpy()
            torch.save(masks, f"datasets/masks/{dataset_name}.pth")
        return masks
    return None
