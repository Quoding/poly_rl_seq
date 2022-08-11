import json

import pandas as pd
import torch
from typing import Dict, Tuple
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class MetricsCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.user_data["testmetric"] = []
        episode.custom_metrics["testmetric"] = []
        pass

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        pass

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        pass

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        pass

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        pass

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        pass

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs,
    ):
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
