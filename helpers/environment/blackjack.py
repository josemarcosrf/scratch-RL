from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def print_episode(episode: List[Tuple[Any, ...]], action_map: Dict[int, str]):
    rows = []
    for (state, action, reward) in episode:
        psum, dcard, ace = state
        summary = (psum, dcard, ace, action_map[action], reward)
        rows.append(tuple(map(str, summary)))

    print(
        tabulate(
            rows,
            headers=["Player sum", "Dealer's card", "usable ACE", "action", "reward"],
            tablefmt="simple_grid",
        )
    )


def print_policy(agent, *args, **kwargs):
    # NOTE: # We crop the policy for player card sum > 21
    for usable_ace in [True, False]:
        policy = np.argmax(agent.Q[:, :, int(usable_ace), :], axis=-1)
        action_func = np.vectorize(lambda x: agent.action_map[x])
        table = [action_func(row) for row in policy[1:21]]

        print("Usable ACE:" if usable_ace else "No usable ACE:")
        print(tabulate(table, tablefmt="simple_grid"))


def plot_stats(agent, stats, *args, **kwargs):
    def plot_heatmap_by_ace(
        M: np.array, fig_title: str, block: bool = True, policy: bool = False
    ):
        if policy:
            ace_m = np.argmax(M[:, :, 1, :], axis=-1)
            no_ace_m = np.argmax(M[:, :, 0, :], axis=-1)
        else:
            ace_m = np.sum(M[:, :, 1, :], axis=-1)
            no_ace_m = np.sum(M[:, :, 0, :], axis=-1)

        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        fig.suptitle(fig_title, fontsize=16)

        am = ax[0].imshow(ace_m)
        ax[0].set_title("Usable ACE")
        ax[0].invert_yaxis()
        ax[0].set_xlabel("Dealer's hand")
        ax[0].set_ylabel("Player's sum")
        fig.colorbar(am, ax=ax[0])
        nam = ax[1].imshow(no_ace_m)
        ax[1].set_title("No usable ACE")
        ax[1].invert_yaxis()
        ax[1].set_xlabel("Dealer's hand")
        ax[1].set_ylabel("Player's sum")
        fig.colorbar(nam, ax=ax[1])

        plt.show(block=block)

    # Plot Q-value
    plot_heatmap_by_ace(agent.Q, "Q-value", block=False)
    # Plot the policy
    plot_heatmap_by_ace(agent.Q, "Policy from Q", block=False, policy=True)
    # Returns
    returns = stats["returns"]
    plot_heatmap_by_ace(returns, "Total Returns", block=False)
    # Visits
    visits = stats["visits"]
    plot_heatmap_by_ace(visits, "State Visits")
