import argparse
from os.path import exists
from typing import Any, Dict, Tuple

import torch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import NoPreprocessor
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.agents import ppo
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import MSELoss
from ray.tune.registry import register_env
from environments import env_creator, get_action_mask
from networks import RayNetwork, CUSTOM_MODEL_CONFIG


def get_trainer(
    args,
    env_config,
):
    register_env("polyenv", env_creator)
    ModelCatalog.register_custom_model("custom_net", RayNetwork)
    ModelCatalog.register_custom_preprocessor("noprep", NoPreprocessor)

    model_config = CUSTOM_MODEL_CONFIG
    if "mask" in env_config["env_name"]:
        model_config["use_masking"] = True

    # Rollout fragment will be adjusted to a divider of `train_batch_size`
    trainer = ppo.PPOTrainer(
        env="polyenv",
        config={
            "framework": "torch",
            "env_config": env_config,
            # "num_workers": 0,
            "num_gpus": 0,
            "model": {
                "custom_model": "custom_net",
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
            "use_gae": True,  # Try to get value function directly
            "lr_schedule": None,
            "lambda": 1,  # GAE estimation parameter
            "lr": args.lr,
            "preprocessor_pref": None,
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
    env_name,
    gamma,
    step_penalty,
    masks,
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

    all_combis_estimated_reward = get_estimated_state_reward(
        net, combis, step_penalty, env_name, gamma, masks, device
    )

    # Parmis tous les vecteurs "existant", lesquels je trouve ? (Jaccard, ratio_app)
    sol_idx = set(
        seen_idx[torch.where(all_combis_estimated_reward > thresh)[0]].tolist()
    )

    all_flagged_combis_idx.update(sol_idx)

    # Parmis les patrons dangereux (ground truth), combien j'en trouve tels quels
    all_pats_estimated_reward = get_estimated_state_reward(
        net, pat_vecs, step_penalty, env_name, gamma, 1, device
    )

    sol_pat_idx = set(torch.where(all_pats_estimated_reward > thresh)[0].tolist())

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
        all_combis_estimated_reward,
    )


def get_estimated_state_reward(
    net, combis, step_penalty, env_name, gamma, masks, device
):
    """Compute estimates of state's reward for every combinations in `combis`
    Value function loss for PPO's critic:


            rollout[Postprocessing.ADVANTAGES] = (
                discounted_returns - rollout[SampleBatch.VF_PREDS]
            )
            train_batch[Postprocessing.VALUE_TARGETS] = rollout[Postprocessing.ADVANTAGES] + rollout[SampleBatch.VF_PREDS]

            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)



    Args:
        net ([type]): [description]
        combis ([type]): [description]
        step_penalty ([type]): [description]
        env_name ([type]): [description]
        gamma ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    d1, d2 = combis.shape
    # Expand combis to have the extra terminal state
    combis_states = torch.zeros(d1, d2 + 1)
    combis_states[:, :d2] = combis
    combis_states = combis_states.to(device)

    if masks is not None:
        obs = {"obs": {"observations": combis_states, "action_mask": masks}}
    else:
        obs = {"obs": combis_states}

    # Forward pass for state value
    logits, _ = net(obs)
    state_values = net.value_function()
    # state_values = trainer..value_function()  # (100000, 51)

    # Get next states and get their value to extract estimated reward
    taken_actions = torch.argmax(logits, dim=1).unsqueeze(0)  # (100000, 1)
    next_states = combis_states.clone()

    if env_name == "randomstart":
        combi_bits = next_states.gather(1, taken_actions)
        combi_bits = torch.logical_not(combi_bits).float()
        next_states = next_states.scatter(1, taken_actions, combi_bits)
    elif env_name == "singlestart":
        # if env is single start, we can only set to one
        next_states = next_states.scatter(1, taken_actions, 1)

    # Forward pass needs to happen on policy before value is computed
    # Particular to RayNetwork taken from tutorial...
    if masks is not None:
        obs = {"obs": {"observations": next_states, "action_mask": 1}}
    else:
        obs = {"obs": next_states}

    net(obs)
    next_state_values = net.value_function()

    estimated_reward = (state_values + step_penalty) - gamma * next_state_values
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
        default="metrics/ouput/",
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
                masks[i] = get_action_mask(combis_states[i], d2, combis)

            torch.save(masks, f"datasets/masks/{dataset_name}.pth")
        return masks
    return None
