import abc
import warnings
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from helpers.environments import get_env_action_dims
from helpers.environments import get_env_action_name_map
from helpers.environments import get_env_shape
from helpers.environments import get_env_state_dims
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
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
    ) -> Dict[str, Any]:
        return {}

    def print_policy(self, stats: Dict[str, Any]):

        if self.Q.ndim > 3:
            warnings.warn(f"Skipping printing the policy table. Q dims: {self.Q.ndim}")
            return

        q = self.Q.reshape(*self.env_shape, self.n_actions)
        action_func = np.vectorize(lambda x: self.action_map[x])
        table = [action_func(np.argmax(row, axis=-1)) for row in q]

        h, w = self.env_shape[:2]
        for i in range(h):
            for j in range(w):
                if sum(stats["visits"][i * w + j]) == 0:
                    table[i][j] = "x"

        print(tabulate(table, tablefmt="simple_grid"))

    def plot_stats(self, stats):
        fig, axis = plt.subplots(1, 3, figsize=(20, 10))
        plot_line(stats["ep_rewards"], title="Episode rewards", ax=axis[0])
        plot_line(stats["ep_length"], title="Episode length", ax=axis[1])

        if len(self.env_shape) == 2:
            state_visits = np.sum(stats["visits"], axis=-1).reshape(self.env_shape)
            plot_heatmap(state_visits, title="state visits", ax=axis[2])

        plt.show()
