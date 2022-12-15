import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from algorithms import State
from algorithms import state_as_ints
from algorithms.tabular import TabularAgent
from helpers.cli import get_cli_parser
from helpers.environment import get_env
from helpers.environment import get_env_report_functions
from helpers.logio import init_logger

# mypy: ignore-errors


class TabularMonteCarlo(TabularAgent):
    """Tabular first-visit Monte Carlo agent for episodic environments.

    For simplicity we are assuming the following:
        - actions range from 0 to n_actions
    """

    def __init__(self, env):
        super().__init__(env)
        self.Q = np.zeros(self.Q.shape)
        self.G = np.zeros(self.Q.shape)  # rewards accumulator
        self.visits = np.zeros(self.Q.shape)  # visit counter (to compute averages)
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
            episode.append((state, action, reward))

            if terminated or truncated:
                break

            next_action = self.run_policy(next_state)
            action = next_action
            state = next_state

        return episode

    def observe(self, episode: List[Tuple[Any, ...]]):
        @state_as_ints
        def add_to_returns(state, action, reward):
            self.G[state][action] += reward
            self.visits[state][action] += 1

        # To guarantee we only count first-visits
        seen_states: Dict[Tuple[Any, ...], bool] = {}
        for (s, a, r) in episode:
            if not seen_states.get((s, a), False):
                add_to_returns(s, a, r)
                seen_states[(s, a)] = True

        self.Q = np.where(self.visits > 0, self.G / self.visits, self.Q)

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        epsilon: float,
    ) -> Dict[str, Any]:

        logger.info("Start learning")
        self.epsilon = epsilon

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "visits": np.zeros(self.Q.shape),
            "returns": np.zeros(self.Q.shape),
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # 1. Generate an episode following the currrent policy
            episode = self.generate_episode(max_ep_steps)

            # 2. Observe the episode returns for eac (s, a)
            self.observe(episode)

            # Collect some stats
            stats["visits"] = self.visits.copy()
            stats["returns"] = self.G.copy()
            stats["ep_length"][ep_i] = len(episode)
            stats["ep_rewards"][ep_i] = sum(r for (_, _, r) in episode)

        self.env.close()

        return stats


if __name__ == "__main__":

    args = get_cli_parser("Monte Carlo learning options").parse_args()

    init_logger(level=args.log_level, logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)

    logger.info("Initializing agent")
    agent = TabularMonteCarlo(env)
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.num_steps,
        epsilon=args.explore_probability,
    )

    print_policy, plot_stats = get_env_report_functions(env)
    print_policy(agent, stats)
    plot_stats(agent, stats)
