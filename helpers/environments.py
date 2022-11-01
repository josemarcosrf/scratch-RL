from typing import Any
from typing import Dict
from typing import Tuple

import gymnasium as gym

ENV_META: Dict[str, Dict[str, Any]] = {
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
    return gym.make(
        env_name,
        **ENV_META[env_name].get("params", {}),
        render_mode=render_mode,
    )


def get_env_name(env):
    return env.unwrapped.spec.id


def get_env_state_dims(env) -> Tuple[int, ...]:
    if isinstance(env.observation_space, gym.spaces.tuple.Tuple):
        return tuple(d.n for d in env.observation_space)

    return (env.observation_space.n,)


def get_env_action_dims(env) -> Tuple[int]:
    return (env.action_space.n,)


def get_env_map_shape(env):
    env_name = get_env_name(env)
    return ENV_META[env_name]["map_shape"]


def get_env_action_name_map(env) -> Dict[int, str]:
    env_name = get_env_name(env)
    return ENV_META[env_name]["action_map"]
