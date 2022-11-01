import logging
import random
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from algorithms.tabular import TabularAgent
from helpers import init_logger
from helpers.cli import get_cli_parser
from helpers.constants import DEFAULT_RANDOM_SEED
from helpers.environments import get_env
from helpers.plotting import plot_heatmap
from helpers.plotting import plot_line


logger = logging.getLogger(__name__)


class TabularSARSA(TabularAgent):
    def __init__(self, env):

        """Initializes a SARSA Tabular agent for the given environment.

        For simplicity we are assuming the following:
         - actions range from 0 to n_actions
         - There's no step-size scheduling (remains constant)
        """
        super().__init__(env)

    def run_policy(self, state: int) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon

        Args:
            state (int): agent state
        """
        if random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions))

        return np.argmax(self.Q[state, :])

    def observe(self, s: int, a: int, r: float, next_s: int, next_a: int) -> int:
        """Here is where the Q-update happens

        Args:
            s (int): current state
            a (int): current action
            r (float): reward
            next_s (int): next state (usually denoted as: s')
            next_a (int): next action (usually denoted as: a')
        """
        self.Q[s, a] += self.alpha * (
            r + self.gamma * self.Q[next_s, next_a] - self.Q[s, a]
        )
        return next_a

    def train(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
    ) -> Dict[str, Any]:
        """Implements the On-policy TD Control algorithm 'Tabular SARSA'

        Args:
            num_episodes (int): max number of episodes
            max_ep_steps (int): max number of steps per episode
            discount (float): discount factor (gamma)
            epsilon (float): probability of taking a random action (epsilon-greedy)
            step_size (float): learning step size (alpha)
        """
        logger.info("Start training")
        self.alpha = step_size
        self.gamma = discount
        self.epsilon = epsilon

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "visits": np.zeros((self.n_states, self.n_actions)),
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # Init S & chose A from S using policy derived from Q
            state, _ = self.env.reset()
            action = self.run_policy(state)

            for i in range(max_ep_steps):
                # Take action A, observe S' and R
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Chose A' from S' using policy derived from Q
                next_action = self.run_policy(next_state)
                action = self.observe(state, action, reward, next_state, next_action)

                if terminated or truncated:
                    break

                # Collect some stats
                stats["ep_length"][ep_i] = i
                stats["ep_rewards"][ep_i] += reward
                stats["visits"][state, action] += 1

                state = next_state

        # Print the policy over the map
        self.env.close()
        self.print_policy(stats)
        # self._plot_q()

        return stats


if __name__ == "__main__":

    args = get_cli_parser("SARSA-learning options").parse_args()

    init_logger(level=args.log_level, my_logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)
    env.action_space.seed(DEFAULT_RANDOM_SEED)

    logger.info("Initializing agent")
    agent = TabularSARSA(env)
    stats = agent.train(
        num_episodes=args.num_episodes,
        max_ep_steps=args.num_steps,
        step_size=args.step_size,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
    )

    fig, axis = plt.subplots(1, 3, figsize=(20, 10))
    plot_line(stats["ep_rewards"], title="Episode rewards", ax=axis[0])
    plot_line(stats["ep_length"], title="Episode length", ax=axis[1])

    state_visits = np.sum(stats["visits"], axis=-1).reshape(agent.h, agent.w)
    logger.debug(state_visits)
    plot_heatmap(state_visits, title="state visits", ax=axis[2])

    plt.show()
