import gym
from utils import load_dataset
import torch
import random

# Value from literature
MAX_N_RX = 15

REASON_TO_REWARD_MAP = {
    "end_action": 0,
    "max_rx": 0,  # Set to 0 as to not discourage exploration in "high number of Rx" spaces
    "end_action_no_end": -1,
}

DEFAULT_CONFIG = {
    "dataset_name": "set it yourself",
    "noise": 0,
    "min_n_rx": 5,
    "horizon": 20,
    "step_penalty": 1,
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

        # Full set of possible combinations
        self.observation_space = gym.spaces.MultiBinary(self.n_dim)

        # State equivalent
        self.current_state = None
        self.last_reward = 0

        self.step_count = 0

        self.END_ACTION = self.n_dim

    def reset(self):
        # Sample an random number of Rx to set active
        number_of_rx = random.randint(self.min_n_rx, MAX_N_RX)
        # Sample `number_of_rx` integers. These are the Rx IDs to activate.
        begin_rx_idx = torch.randint(self.n_dim, size=(number_of_rx,))
        # Put the previously sampled combination in vector format
        observation_combi = torch.zeros(self.n_dim)
        observation_combi[begin_rx_idx] = 1

        self.current_state = observation_combi
        self.step_count = 0

        print(number_of_rx)

        return self.current_state

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
        dists = torch.norm(self.current_state - self.combis, dim=1, p=1)
        knn_idx = dists.topk(1, largest=False).indices[0]
        self.current_state = self.combis[knn_idx]
        return knn_idx

    def step(self, action):
        done_dict = self._is_done(action)
        done = done_dict["done"]
        reason = done_dict["reason"]

        # Update state if action wasn't to end the episode
        if action != self.END_ACTION:
            self.current_state[action] = (self.current_state[action].bool() ^ 1).float()

        # If we have a shortcut to the reward, take it.
        if reason in REASON_TO_REWARD_MAP.keys():
            reward = REASON_TO_REWARD_MAP[reason]
        else:
            knn_idx = self._bind_state_to_dataset()
            reward_noise = torch.normal(self.mean_noise, self.std_noise)
            reward = self.risks[knn_idx] + reward_noise - self.step_penalty
            reward = reward.item()

        return self.current_state, reward, done, done_dict
