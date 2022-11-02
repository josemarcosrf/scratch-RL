import abc
import warnings
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from helpers.environment import get_env_action_dims
from helpers.environment import get_env_action_name_map
from helpers.environment import get_env_shape
from helpers.environment import get_env_state_dims
from helpers.plotting import plot_heatmap
from helpers.plotting import plot_line
from helpers.plotting import plot_vector_field


class TabularAgent(metaclass=abc.ABCMeta):
    def __init__(self, env):

        """Initializes a SARSA Tabular agent for the given environment.

        For simplicity we are assuming the following:
         - actions range from 0 to n_actions
         - There's no step-size scheduling (remains constant)
        """
        self.env = env
        self.n_states = get_env_state_dims(env)
        self.n_actions = get_env_action_dims(env)
        self.action_map = get_env_action_name_map(env)
        self.env_shape = get_env_shape(env)
        # Initialize the Q table
        # Initialize the Q table
        if isinstance(self.n_states, tuple):
            size = (*self.n_states, self.n_actions)
        else:
            size = (self.n_states, self.n_actions)

        self.Q = np.random.uniform(0, 1, size=size)
        # self.Q[-1, :] = 0

    @abc.abstractmethod
    def learn(
        self, num_episodes: int, max_ep_steps: int, *args, **kwargs
    ) -> Dict[str, Any]:
        return {}
