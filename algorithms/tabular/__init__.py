from typing import Any
from typing import Dict

import numpy as np
from tabulate import tabulate

from helpers.environments import get_env_action_name_map
from helpers.environments import get_env_map_shape
from helpers.plotting import plot_vector_field


class TabularAgent:
    def __init__(self, env):

        """Initializes a SARSA Tabular agent for the given environment.

        For simplicity we are assuming the following:
         - actions range from 0 to n_actions
         - There's no step-size scheduling (remains constant)
        """
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.env = env
        self.action_map = get_env_action_name_map(env)
        self.h, self.w = get_env_map_shape(env)
        # self.Q = np.zeros((self.n_states, self.n_actions))
        self.Q = np.random.uniform(0, 1, size=(self.n_states, self.n_actions))
        self.Q[-1, :] = 0

    def print_policy(self, stats: Dict[str, Any]):
        q = self.Q.reshape(self.h, self.w, self.n_actions)
        action_func = np.vectorize(lambda x: self.action_map[x])
        table = [action_func(np.argmax(row, axis=-1)) for row in q]

        for i in range(self.h):
            for j in range(self.w):
                if sum(stats["visits"][i * self.w + j]) == 0:
                    table[i][j] = "x"

        print(tabulate(table, tablefmt="simple_grid"))

    def _plot_q(self, save_fpath: str = None):
        """Transforms the Q table into a vector field representation of
        the agents action preferences.
        NOTE: numpy and pyplot have swapped axis!
        NOTE: The plot is flipped on the Y axis (axis=0 for numpy)
            pyplot     -->  indexes (0, 0) at the bottom-left
            numpy/gym  -->  indexes (0, 0) at the top-left
        Args:
            h (int): Grid world height
            w (int): Grid world width
        """
        q = self.Q.reshape(self.h, self.w, self.n_actions)

        # Q-values are negative so we need to take the opposite dir as vectors
        # NOTE: This directions are only correct for the Cliff env!
        # RIGHT - LEFT
        u = q[:, :, 1] - q[:, :, 3]
        u /= np.maximum(1, np.maximum(q[:, :, 1], q[:, :, 3]))
        # UP - DOWN
        v = q[:, :, 0] - q[:, :, 2]
        v /= np.maximum(1, np.maximum(q[:, :, 0], q[:, :, 2]))

        plot_vector_field(u, v, save_fpath=save_fpath)
