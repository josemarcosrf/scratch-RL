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
from helpers.plotting import plot_vector_field


logger = logging.getLogger(__name__)


class TabularSARSA:
    def __init__(self, env):

        """Initializes a SARSA Tabular agent for the given environment.

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
    ) -> None:
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
                self.stats["ep_length"][ep_i] = i
                self.stats["ep_rewards"][ep_i] += reward
                self.stats["visits"][state, action] += 1

                state = next_state

        # Print the policy over the map
        self.env.close()
        self.print_policy()
        # self._plot_q()

    def print_policy(self):
        q = self.Q.reshape(self.h, self.w, self.n_actions)
        action_func = np.vectorize(lambda x: self.action_map[x])
        table = [action_func(np.argmax(row, axis=-1)) for row in q]

        for i in range(self.h):
            for j in range(self.w):
                if sum(self.stats["visits"][i * self.w + j]) == 0:
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


if __name__ == "__main__":

    args = get_cli_parser("SARSA-learning options").parse_args()

    init_logger(level=args.log_level, my_logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)
    env.action_space.seed(DEFAULT_RANDOM_SEED)

    logger.info("Initializing agent")
    agent = TabularSARSA(env)
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
