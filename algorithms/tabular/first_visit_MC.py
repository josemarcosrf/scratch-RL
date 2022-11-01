import logging
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm

from algorithms import State
from algorithms import state_as_ints
from algorithms.tabular import TabularAgent
from helpers import init_logger
from helpers.cli import get_cli_parser
from helpers.constants import DEFAULT_RANDOM_SEED
from helpers.environments import get_env


logger = logging.getLogger(__name__)


class TabularMonteCarlo(TabularAgent):
    def __init__(self, env):
        """Initializes a tabular first-visit Monte Carlo agent for the given environment.

        For simplicity we are assuming the following:
         - actions range from 0 to n_actions
        """
        super().__init__(env)
        self.G = np.zeros(self.Q.shape)  # the rewards accumulator
        logger.debug(f"Q has shape: {self.Q.shape}")

    @state_as_ints
    def run_policy(self, state: State) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon
        Args:
            state (int): agent state
        """
        if random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions))

        return np.argmax(self.Q[state][:])

    def generate_episode(self, max_ep_steps: int):
        # Init S & chose A from S using policy derived from Q
        state, _ = self.env.reset()
        action = self.run_policy(state)

        episode = []
        for _ in range(max_ep_steps):
            # Take action A, observe S' and R
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            if terminated or truncated:
                break

            episode.append((state, action, reward))

            next_action = self.run_policy(next_state)
            action = next_action
            state = next_state

        return episode

    def observe(self, episodes: List[Tuple[Any, ...]]):
        @state_as_ints
        def add_to_returns(state, action, reward):
            returns[state][action] += reward
            visits[state][action] += 1

        returns = np.zeros(self.Q.shape)
        visits = np.zeros(self.Q.shape)

        for (s, a, r) in episodes:
            add_to_returns(s, a, r)

        self.Q = returns / visits  # TODO: divide only visited (state, action) pairs

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
    ) -> Dict[str, Any]:

        logger.info("Start learning")
        self.alpha = step_size
        self.gamma = discount
        self.epsilon = epsilon

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "visits": None,
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # 1. Generate an episode following the currrent policy
            episode = self.generate_episode(max_ep_steps)

            # 2. Observe the episode returns for eac (s, a)
            self.observe(episode)

            # Collect some stats
            stats["ep_length"][ep_i] = len(episode)
            stats["ep_rewards"][ep_i] = sum(r for (_, _, r) in episode)

        self.env.close()

        return stats


if __name__ == "__main__":

    args = get_cli_parser("Monte Carlo learning options").parse_args()

    init_logger(level=args.log_level, my_logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)
    env.action_space.seed(DEFAULT_RANDOM_SEED)

    logger.info("Initializing agent")
    agent = TabularMonteCarlo(env)
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.num_steps,
        step_size=args.step_size,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
    )

    agent.plot_stats(stats)
