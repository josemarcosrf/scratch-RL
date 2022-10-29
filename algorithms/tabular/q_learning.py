import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from tqdm.auto import tqdm

from helpers import init_logger
from helpers.cli import get_cli_parser
from helpers.constants import DEFAULT_RANDOM_SEED
from helpers.environments import get_env
from helpers.environments import get_env_action_name_map
from helpers.environments import get_env_map_shape
from helpers.plotting import plot_line


logger = logging.getLogger(__name__)


class TabularQLearning:
    def __init__(self, env):

        """Initializes a Q-learning Tabular agent for the given environment.

        For simplicity we are assuming the following:
         - actions range from 0 to n_actions
         - There's no step-size scheduling (remains constant)
        """
        self.stats = {}
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.env = env
        self.action_map = get_env_action_name_map(env)
        self.h, self.w = get_env_map_shape(env)
        # self.Q = np.zeros((self.n_states, self.n_actions))
        self.Q = np.random.uniform(0, 1, size=(self.n_states, self.n_actions))
        self.Q[-1, :] = 0

    def run_policy(self, state: int) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon

        Args:
            state (int): agent state
        """
        if random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions))

        return np.argmax(self.Q[state, :])

    def observe(self, s: int, a: int, r: float, next_s: int) -> None:
        """Here is where the Q-update happens
        Args:
            s (int): current state
            a (int): current action
            r (float): reward
            next_s (int): next state (usually denoted as: s')
        """
        self.Q[s, a] += self.alpha * (
            r + self.gamma * np.max(self.Q[next_s, :], axis=-1) - self.Q[s, a]
        )

    def train(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
    ) -> None:
        """Implements the Off-policy TD Control algorithm 'Tabular Q-learning'

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

        self.stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "visits": np.zeros((self.n_states, self.n_actions)),
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # Init S & chose A from S using policy derived from Q
            state, _ = self.env.reset()

            for i in range(max_ep_steps):
                action = self.run_policy(state)

                # Take action A, observe S' and R
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Chose A' from S' using policy derived from Q
                self.observe(state, action, reward, next_state)
                if terminated or truncated:
                    break

                # Collect some stats
                self.stats["ep_length"][ep_i] = i
                self.stats["ep_rewards"][ep_i] += reward
                self.stats["visits"][state, action] += 1

                state = next_state

        # Print the policy over the map
        self.env.close()
        self.print_policy()

    def print_policy(self):
        q = self.Q.reshape(self.h, self.w, self.n_actions)
        action_func = np.vectorize(lambda x: self.action_map[x])
        table = [action_func(np.argmax(row, axis=-1)) for row in q]

        for i in range(self.h):
            for j in range(self.w):
                if sum(self.stats["visits"][i * self.w + j]) == 0:
                    table[i][j] = "x"
        print(tabulate(table, tablefmt="simple_grid"))


if __name__ == "__main__":

    args = get_cli_parser("Q-learning options").parse_args()

    init_logger(level=args.log_level, my_logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)
    env.action_space.seed(DEFAULT_RANDOM_SEED)

    logger.info("Initializing agent")
    agent = TabularQLearning(env)
    agent.train(
        num_episodes=args.num_episodes,
        max_ep_steps=args.num_steps,
        step_size=0.5,
        discount=0.95,
        epsilon=0.1,
    )

    fig, axis = plt.subplots(1, 2, figsize=(15, 10))
    plot_line(agent.stats["ep_rewards"], title="Episode rewards", ax=axis[0])
    plot_line(agent.stats["ep_length"], title="Episode length", ax=axis[1])

    plt.show()
