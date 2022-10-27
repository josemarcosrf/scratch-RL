import gym

import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from common.constants import DEFAULT_RANDOM_SEED


class TabularSARSA:

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        step_size: float,
        discount: float,
        epsilon: float,
        *args,
        **kwargs,
    ):
        """Initializes a SARSA Tabular agent.

        For simplicity we are assuming the following:
         - actions range from 0 to n_actions
         - There's no step-size scheduling (remains constant)

        Args:
            n_states (int): number of environment states
            n_actions (int): number of possible actions
            step_size (float): learning step size
            discount (float): future reward discount factor
            epsilon (float): e-greedy parameter
            *args: Description
            **kwargs: Description
        """

        self.ns = n_states
        self.na = n_actions
        self.alpha = step_size
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def run_policy(self, state: int) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon

        Args:
            state (int): agent state
        """
        if random.random() > self.epsilon:
            return np.random.choice(range(self.na))

        return np.argmax(self.Q[state, :])

    def observe(self, s: int, a: int, r: float, next_s: int, next_a: int) -> int:
        """Here is where the Q-update happens

        Args:
            r (float): Description
            new_state (int): Description
        """
        self.Q[s, a] += self.alpha * (
            r + self.gamma * self.Q[next_s, next_a] - self.Q[s, a]
        )
        return next_a

    def plot_q(self, h: int, w: int):
        """Summary

        NOTE: numpy and pyplot have swapped axis!
        NOTE: The plot is flipped on the Y axis (axis=0 for numpy)
            pyplot     -->  indexes (0, 0) at the bottom-left
            numpy/gym  -->  indexes (0, 0) at the top-left

        Args:
            h (int): Grid world height
            w (int): Grid world width
        """

        # Meshgrid
        x, y = np.meshgrid(np.arange(0, w, 1.0), np.arange(0, h, 1.0))

        ax = plt.axes()
        ax.set_xticks(x[0])
        ax.set_yticks([l[0] for l in y])

        q = self.Q.reshape(h, w, self.na).copy()

        # RIGHT - LEFT
        u = q[:, :, 1] - q[:, :, 3]
        u /= np.maximum(1, np.maximum(q[:, :, 1], q[:, :, 3]))
        # UP - DOWN
        v = q[:, :, 0] - q[:, :, 2]
        v /= np.maximum(1, np.maximum(q[:, :, 0], q[:, :, 2]))

        # Plotting Vector Field with QUIVER
        # Note the swapped axis
        plt.quiver(x + 0.5, y + 0.5, v, -u, color="g")
        plt.title("Q table")

        # Setting x, y boundary limits
        plt.xlim(0, w)
        plt.ylim(h, 0)  # flipped y-axis

        # Show plot with grid
        plt.grid()
        plt.show()


if __name__ == "__main__":

    NUM_EPISODES = 500
    MAX_EP_STEPS = 500

    env = gym.make("CliffWalking-v0", render_mode=None)  # "human"

    agent = TabularSARSA(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        step_size=0.05,
        discount=0.95,
        epsilon=0.1,
    )

    env.action_space.seed(DEFAULT_RANDOM_SEED)
    state, info = env.reset(seed=DEFAULT_RANDOM_SEED)

    episode_iter = tqdm(range(NUM_EPISODES))
    for ep_i in episode_iter:
        episode_iter.set_description(f"Episode: {ep_i}")

        # Init S & chose A from S using policy derived from Q
        state, info = env.reset()
        action = agent.run_policy(state)

        for i in range(MAX_EP_STEPS):
            # Take action A, observe S' and R
            next_state, reward, terminated, truncated, info = env.step(action)

            # Chose A' from S' using policy derived from Q
            next_action = agent.run_policy(next_state)
            action = agent.observe(state, action, reward, next_state, next_action)

            if terminated or truncated:
                state, info = env.reset()
            else:
                state = next_state

    # Plot the action preference
    agent.plot_q(4, 12)

    env.close()
