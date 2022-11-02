from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import gymnasium as gym

from helpers.constants import DEFAULT_RANDOM_SEED

ENV_META: Dict[str, Dict[str, Any]] = {
    "Blackjack-v1": {
        "action_map": {
            0: "stick",
            1: "hit",
        },
        "params": {"natural": False, "sab": True},
    },
    "Taxi-v3": {
        "map_shape": (5, -5),
        "action_map": {
            0: "south",
            1: "north",
            2: "east",
            3: "west",
            4: "pickup",
            5: "drop off",
        },
        "params": {},
    },
    "CliffWalking-v0": {
        "map_shape": (4, 12),
        "action_map": {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        },
        "params": {},
    },
    "FrozenLake-v1": {
        "map_shape": (4, 4),
        "action_map": {
            0: "left",
            1: "down",
            2: "right",
            3: "up",
        },
        "params": {
            "map_name": "4x4",
            "is_slippery": False,
            "desc": ["SFFF", "FHFH", "FFFH", "HFFG"],
        },
    },
}


def get_env(env_name: str, render_mode: str = None):
    env = gym.make(
        env_name,
        **ENV_META[env_name].get("params", {}),
        render_mode=render_mode,
    )
    env.action_space.seed(DEFAULT_RANDOM_SEED)

    return env


def get_env_name(env):
    return env.unwrapped.spec.id


def get_env_state_dims(env) -> Union[int, Tuple[int, ...]]:
    if isinstance(env.observation_space, gym.spaces.tuple.Tuple):
        return tuple(d.n for d in env.observation_space)

    return env.observation_space.n


def get_env_action_dims(env) -> int:
    return env.action_space.n


def get_env_shape(env):
    env_name = get_env_name(env)

    state_dims = get_env_state_dims(env)
    if isinstance(state_dims, tuple):
        return state_dims

    return ENV_META[env_name].get("map_shape", (-1, -1))


def get_env_action_name_map(env) -> Dict[int, str]:
    env_name = get_env_name(env)
    return ENV_META[env_name]["action_map"]


def get_env_report_functions(env):
    env_name = get_env_name(env).split("-v")[0]

    # Make this a registry dict in constants?
    if env_name in ["CliffWalking", "FrozenLake"]:
        from helpers.environment.gridworlds import print_policy, plot_stats

        return print_policy, plot_stats

    elif env_name in ["Blackjack"]:
        from helpers.environment.blackjack import print_policy, plot_stats

        return print_policy, plot_stats
