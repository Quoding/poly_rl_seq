import gym
import torch
import random
import numpy as np
import pandas as pd
import json

# Value from literature
MAX_N_RX = 15

REASON_TO_REWARD_MAP = {
    "end_action": 0,
    "max_rx": 0,  # Set to 0 as to not discourage exploration in "high number of Rx" spaces
    "end_action_no_end": -1,
}


class PolypharmacyEnv(gym.Env):
    def __init__(self, config):
        # Setup variables with configuration
        self.dataset_name = config["dataset_name"]
        self.min_n_rx = config["min_n_rx"]
        self.horizon = config["horizon"]
        self.step_penalty = config["step_penalty"]

        noise = config["noise"]

        self.mean_noise = torch.tensor([0.0])
        self.std_noise = torch.tensor([noise])

        self.combis, self.risks, self.pat_vecs, self.n_obs, self.n_dim = load_dataset(
            self.dataset_name
        )

        # Possible Rx IDs to select. Add 1 to n_dim so we have a special "exit" action available to the agent
        self.action_space = gym.spaces.Discrete(self.n_dim + 1)

        # Full set of possible combinations + terminal state
        self.observation_space = gym.spaces.MultiBinary(self.n_dim + 1)

        # State equivalent
        self.current_state = None
        self.last_reward = 0

        self.step_count = 0

        self.END_ACTION = self.n_dim

        self.all_observed_states = []
        self.all_observed_states_idx = []

    def reset(self):
        idx = torch.randint(0, len(self.combis), size=(1,))[0]
        self.current_state = torch.zeros(self.n_dim + 1)
        self.current_state[: self.n_dim] = torch.tensor(self.combis[idx]).clone()
        self.record_state(idx.item())
        self.step_count = 0

        return self.current_state

    def get_obs_states_idx(self):
        return self.all_observed_states_idx

    def get_obs_states(self):
        return self.all_observed_states

    def record_state(self, idx):
        self.all_observed_states_idx.append(idx)
        self.all_observed_states.append(self.current_state.clone())

    def _is_done(self, action):
        state_sum = self.current_state.sum()

        # Agent decided to stop and he is allowed to.
        if action == self.END_ACTION:
            if state_sum >= self.min_n_rx:
                return {"done": True, "reason": "end_action"}
            else:
                return {"done": False, "reason": "end_action_no_end"}
        # If agent has the maximum number of Rx to consider Polypharmacy
        elif state_sum > MAX_N_RX:
            return {"done": True, "reason": "max_rx"}
        # Agent played the max amount of steps
        elif self.step_count >= self.horizon:
            return {"done": True, "reason": "max_steps"}

        else:
            return {"done": False, "reason": "not_done"}

    def _bind_state_to_dataset(self):
        """1-NN search of the current state in dataset vectors

        Returns:
            torch.Tensor: nearest neighbor of `vec` in `set_existing_vecs`
        """
        combi = self.current_state[:-1]
        dists = torch.norm(combi - self.combis, dim=1, p=1)
        knn_idx = dists.topk(1, largest=False).indices[0]
        new_state = self.combis[knn_idx].clone()
        same_state = (new_state == self.current_state[:-1]).all()

        self.current_state[:-1] = new_state
        self.record_state(knn_idx.item())
        return knn_idx, same_state

    def step(self, action):
        done_dict = self._is_done(action)
        done = done_dict["done"]
        reason = done_dict["reason"]

        # Update state if action wasn't to end the episode
        if reason != "end_action_no_end":
            self.current_state[action] = (self.current_state[action].bool() ^ 1).float()

        # If we have a shortcut to the reward, take it
        if reason in REASON_TO_REWARD_MAP.keys():
            reward = REASON_TO_REWARD_MAP[reason]

        # If agent's state isn't polypharmacy, give a step penalty reward
        elif self.current_state.sum() < self.min_n_rx:
            reward = -self.step_penalty
        else:
            knn_idx, same_state = self._bind_state_to_dataset()
            reward_noise = torch.normal(self.mean_noise, self.std_noise)
            reward = self.risks[knn_idx] + reward_noise - self.step_penalty
            reward = reward.item()
        self.step_count += 1

        return self.current_state, reward, done, done_dict


class SingleStartPolypharmacyEnv(PolypharmacyEnv):
    def __init__(self, config):
        super.__init__(config)

    def reset(self):
        self.current_state = torch.zeros(self.n_dim + 1).float()
        self.step_count = 0

        return self.current_state

    def step(self, action):
        done_dict = self._is_done(action)
        done = done_dict["done"]
        reason = done_dict["reason"]

        # Update state if action wasn't to end the episode
        if reason != "end_action_no_end":
            self.current_state[action] = 1

        # If we have a shortcut to the reward, take it.
        if reason in REASON_TO_REWARD_MAP.keys():
            reward = REASON_TO_REWARD_MAP[reason]
        else:
            knn_idx = self._bind_state_to_dataset()
            reward_noise = torch.normal(self.mean_noise, self.std_noise)
            reward = self.risks[knn_idx] + reward_noise - self.step_penalty
            reward = reward.item()

        self.step_count += 1

        return self.current_state, reward, done, done_dict


class MaskedPolypharmacyEnv(PolypharmacyEnv):
    def __init__(self, config):
        """Init fn takes the same args as PolypharmacyEnv"""
        super().__init__(config)

        # Redefine observation space
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.MultiBinary(
                    self.action_space.n,
                ),
                "observations": self.observation_space,
            }
        )
        # Get combis as set to do quick membership lookups
        self.combis_set = map(tuple, tuple(self.combis.tolist()))

    def reset(self):
        idx = torch.randint(0, len(self.combis), size=(1,))[0]
        self.current_state = torch.zeros(self.n_dim + 1)
        self.current_state[: self.n_dim] = self.combis[idx].clone()
        self.record_state(idx.item())
        self.step_count = 0

        mask = get_action_mask(self.current_state, self.combis)
        obs = {"action_mask": mask, "observations": self.current_state}
        # self.previous_mask_avail = np.where(mask == 1)[0]

        return obs

    def reset_at_state_mask(self, state, mask):
        self.current_state = state
        obs = {"action_mask": mask, "observation": self.current_state}
        # self.previous_mask_avail = np.where(mask == 1)[0]

        return obs

    def update_combis(self, new_combis):
        self.combis = new_combis
        self.combis_set = map(tuple, tuple(self.combis.tolist()))

    def _find_current_state_idx(self):
        # TODO Perhaps change this so it is more efficient rather than compute a L1 dist everytime... Closest will ALWAYS be 0 distance since this is a parametric action env
        idx, _ = self._bind_state_to_dataset()
        return idx

    def step(self, action):
        done_dict = self._is_done(action)
        done = done_dict["done"]
        reason = done_dict["reason"]

        # assert action in self.previous_mask_avail

        # Update state if action wasn't to end the episode
        if reason != "end_action_no_end":
            self.current_state[action] = (self.current_state[action].bool() ^ 1).float()

        # If we have a shortcut to the reward, take it
        if reason in REASON_TO_REWARD_MAP.keys():
            reward = REASON_TO_REWARD_MAP[reason]

        # If agent's state isn't polypharmacy, give a step penalty reward
        elif self.current_state.sum() < self.min_n_rx:
            reward = -self.step_penalty
        else:
            idx = self._find_current_state_idx()
            reward_noise = torch.normal(self.mean_noise, self.std_noise)
            reward = self.risks[idx] + reward_noise - self.step_penalty
            reward = reward.item()

        self.step_count += 1
        mask = get_action_mask(self.current_state, self.combis)
        obs = {"action_mask": mask, "observations": self.current_state.cpu().numpy()}
        # self.previous_mask_avail = np.where(mask == 1)[0]

        return obs, reward, done, done_dict


def env_creator(env_config):
    env_name = env_config["env_name"]
    if env_name == "randomstart":
        return PolypharmacyEnv(env_config)
    elif env_name == "singlestart":
        return SingleStartPolypharmacyEnv(env_config)
    elif env_name == "maskedrandom":
        return MaskedPolypharmacyEnv(env_config)
    else:
        raise Exception("Env name is not mapped to an existing env class")


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


def get_action_mask(state, combis):
    """Creates a mask based on current state

    Args:
        state (torch.Tensor): current state
        n_dim (int): number of dimensions for dataset (Number of actions, excluding "end" action)
        combis (torch.Tensor): tensor containing all possible combinations currently available in dataset

    Returns:
        torch.Tensor: action mask for current state (additive mask)
    """
    # Get rx part
    state = state[:-1]
    n_dim = len(state)
    dists = torch.norm(state - combis, dim=1, p=1)
    possible_next_states_idx = torch.where(dists == 1)[0]
    possible_next_states = combis[possible_next_states_idx]

    # Dimension 0 contains next states, dimension 1 contains location of different bit (actions)
    available_actions_idx = (
        torch.where((possible_next_states - state) != 0)[1].cpu().numpy()
    )

    # Create mask
    mask = np.zeros((n_dim + 1,), dtype=np.float32)
    mask[available_actions_idx] = 1
    mask[-1] = 1  # Always let the agent play the "end" action

    return mask
