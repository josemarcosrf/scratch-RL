from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from helpers.plotting import plot_heatmap
from helpers.plotting import plot_line


def print_policy(agent, stats: Dict[str, Any], *args, **kwargs):

    q = agent.Q.reshape(*agent.env_shape, agent.n_actions)
    action_func = np.vectorize(lambda x: agent.action_map[x])
    table = [action_func(np.argmax(row, axis=-1)) for row in q]

    h, w = agent.env_shape[:2]
    for i in range(h):
        for j in range(w):
            if sum(stats["visits"][i * w + j]) == 0:
                table[i][j] = "x"

    print(tabulate(table, tablefmt="simple_grid"))


def plot_stats(agent, stats, *args, **kwargs):
    fig, axis = plt.subplots(1, 3, figsize=(20, 10))
    plot_line(stats["ep_rewards"], title="Episode rewards", ax=axis[0])
    plot_line(stats["ep_length"], title="Episode length", ax=axis[1])

    state_visits = np.sum(stats["visits"], axis=-1).reshape(agent.env_shape)
    plot_heatmap(state_visits, title="state visits", ax=axis[2])

    plt.show()
