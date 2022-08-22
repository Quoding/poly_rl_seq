DEFAULT_ENV_CONFIG = {
    "dataset_name": "set it yourself",
    "noise": 0,
    "min_n_rx": 5,
    "horizon": 20,
    "step_penalty": 1,
    "env_name": "randomstart",
}

DEFAULT_REPLAY_BUFFER_CONFIG = {
    # The ReplayBuffer class to use. Any class that obeys the
    # ReplayBuffer API can be used here. In the simplest case, this is the
    # name (str) of any class present in the `rllib.utils.replay_buffers`
    # package. You can also provide the python class directly or the
    # full location of your class (e.g.
    # "ray.rllib.utils.replay_buffers.replay_buffer.ReplayBuffer").
    "type": "MultiAgentPrioritizedReplayBuffer",
    # The capacity of units that can be stored in one ReplayBuffer
    # instance before eviction.
    "capacity": 30000,
    # Specifies how experiences are stored. Either 'sequences' or
    # 'timesteps'.
    "storage_unit": "timesteps",
    # Add constructor kwargs here (if any).
    "replay_burn_in": 10000,
    "prioritized_replay_alpha": 0.5,
}
