from typing import Any
from typing import Dict

import gymnasium as gym

ENV_META: Dict[str, Dict[str, Any]] = {
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


def get_env_map_shape(env):
    env_name = get_env_name(env)
    return ENV_META[env_name]["map_shape"]


def get_env_action_name_map(env) -> Dict[int, str]:
    env_name = get_env_name(env)
    return ENV_META[env_name]["action_map"]
