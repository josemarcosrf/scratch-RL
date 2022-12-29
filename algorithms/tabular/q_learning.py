import random
from typing import Any
from typing import Dict

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from algorithms import fix_state
from algorithms import State
from algorithms import state_as_ints
from algorithms.tabular import TabularAgent
from helpers.cli import get_cli_parser
from helpers.environment import get_env
from helpers.environment import get_env_report_functions
from helpers.logio import init_logger

# mypy: ignore-errors


class TabularQLearning(TabularAgent):
    """On-policy TD Control Q-learning Tabular agent for episodic environments.

    For simplicity we are assuming the following:
        - actions range from 0 to n_actions
        - There's no step-size scheduling (remains constant)
    """

    def __init__(self, env):
        super().__init__(env)
        logger.debug(f"Q has shape: {self.Q.shape}")

    @state_as_ints
    def run_policy(self, state: State) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon"""
        if random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions))

        return np.argmax(self.Q[state][:])

    @state_as_ints
    def observe(self, s: State, a: int, r: float, next_s: State) -> None:
        """Here is where the Q-update happens"""
        self.Q[s][a] += self.alpha * (
            r + self.gamma * np.max(self.Q[next_s][:], axis=-1) - self.Q[s][a]
        )

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
    ) -> Dict[str, Any]:  # type: ignore
        """Implements the Off-policy TD Control learning algorithm learning
        'Tabular Q-learning'

        Args:
            num_episodes (int): max number of episodes
            max_ep_steps (int): max number of steps per episode
            discount (float): discount factor (gamma)
            epsilon (float): probability of taking a random action (epsilon-greedy)
            step_size (float): learning step size (alpha)
        """
        logger.info("Start learning")
        self.alpha = step_size
        self.gamma = discount
        self.epsilon = epsilon

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "visits": np.zeros(self.Q.shape),
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # Init S
            state, _ = self.env.reset()

            for i in range(max_ep_steps):
                # Chose A from S
                action = self.run_policy(state)

                # Take action A, observe S' and R
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.observe(state, action, reward, next_state)

                if terminated or truncated:
                    break

                # Collect some stats
                stats["ep_length"][ep_i] = i
                stats["ep_rewards"][ep_i] += reward
                stats["visits"][fix_state(state)][action] += 1

                state = next_state

        # Done!
        self.env.close()

        return stats


if __name__ == "__main__":

    args = get_cli_parser("Q-learning options").parse_args()

    init_logger(level=args.log_level, logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)

    logger.info("Initializing agent")
    agent = TabularQLearning(env)
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.max_episode_steps,
        step_size=args.step_size,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
    )

    print_policy, plot_stats = get_env_report_functions(env)
    print_policy(agent, stats)
    plot_stats(agent, stats)
