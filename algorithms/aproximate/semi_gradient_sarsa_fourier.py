import random
from itertools import product
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from algorithms import State
from helpers.cli import get_cli_parser
from helpers.environment import get_env
from helpers.environment import get_env_action_dims
from helpers.environment import get_env_action_name_map
from helpers.environment import get_env_shape
from helpers.environment import get_env_state_dims
from helpers.features.fourier import FourierBasis
from helpers.features.tile_coding import IHT
from helpers.features.tile_coding import tiles
from helpers.logio import init_logger

# mypy: ignore-errors

# TODO: Implement the exact same but with the tested fourier aproximator
# TODO: Implement experience reply buffer instead


def plot_stats(stats):
    _, ax = plt.subplots(1, 3)
    # Episode steps
    ax[0].set_title("Episode length")
    ax[0].plot(stats["ep_length"])
    # Total episode rewards
    ax[1].set_title("Episode rewards")
    ax[1].plot(stats["ep_rewards"])
    # Loss over training
    ax[2].set_title("Loss over time")
    ax[2].plot(stats["loss"])
    plt.show()


class FourierLinearQFunction:
    def __init__(self, obs_size: int = 3, n_params: int = 20):
        # Generate a random set of axis frequency vectors.
        c = np.random.randint(size=(n_params + 1, obs_size), low=0, high=n_params)
        self.weights = np.ones(n_params + 1)
        self.features = FourierBasis(1, n_params, c)

    def __call__(self, s: State, a: int) -> Any:
        x = np.array((*s, a))
        return np.dot(self.weights, self.features.encode(x))

    def update(self, s: State, a: int, delta: float):
        x = np.array((*s, a))
        derivate_val = self.features.encode(x)
        self.weights += delta * derivate_val
        self.weights /= np.max(self.weights)  # clipping


class SemiGradientSARSA:
    """On-policy control n-step Semi-gradient SARSA with function approximation
    for episodic environents.

    For simplicity we are assuming the following:
        - actions range from 0 to n_actions
        - There's no step-size scheduling (remains constant)
    """

    def __init__(self, env):
        super().__init__()
        # Environment
        box = env.observation_space
        self.env = env
        self.boundaries = list(zip(box.low, box.high))
        self.n_states = get_env_state_dims(env)
        self.n_actions = get_env_action_dims(env)
        self.action_map = get_env_action_name_map(env)
        self.env_shape = get_env_shape(env)
        # Q-value approximate function + Fourier featurizer
        self.Q = FourierLinearQFunction()

    def normalize_state(self, s: State) -> State:
        s_norm = []
        for d, b in zip(s, self.boundaries):
            rang = b[1] - b[0]
            s_norm.append((d - b[0]) / rang)

        return tuple(s_norm)

    def normalize_action(self, action: int) -> float:
        return action / self.n_actions

    def plot_q_function(self, resolution: int = 50):
        # Generate X, Y coordinates
        ranges = []
        box = self.env.observation_space
        if box.bounded_above.any() and box.bounded_below.any():
            for (l, h) in zip(box.low, box.high):
                step_size = (h - l) / resolution
                ranges.append(np.arange(l, h, step_size))

        x, y = np.meshgrid(*ranges)
        # Compute the Q-values and the get max over actions as the State value
        states = list(product(*ranges))
        z = []
        for state in states:
            s = self.normalize_state(state)
            z.append(
                max(self.Q(s, self.normalize_action(a)) for a in range(self.n_actions))
            )

        z = np.array(z).reshape(x.shape)

        # Plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="viridis", edgecolor="green")
        ax.set_title("Q-value function")
        plt.show()

    def run_policy(self, state: State, ep) -> int:
        """Run the current policy. In this case e-greedy"""
        if random.random() < self.epsilon[ep]:
            return np.random.choice(range(self.n_actions))

        s = self.normalize_state(state)
        return np.argmax(
            [self.Q(s, self.normalize_action(a)) for a in range(self.n_actions)]
        )

    def observe(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        terminated: bool,
    ):
        """Here is where the Q-update happens

        Args:
            state (State): current state
            action (int): current action
            reward (float): reward
            next_state (State): next state (usually denoted as: s')
            next_action (int): next action (usually denoted as: a')
        """
        s = self.normalize_state(state)
        a = self.normalize_action(action)
        if terminated:
            td = self.alpha * (reward - self.Q(s, a))
        else:
            ns = self.normalize_state(next_state)
            na = self.normalize_action(next_action)
            td = self.alpha * (reward + self.gamma * self.Q(ns, na) - self.Q(s, a))

        self.Q.update(s, a, delta=td)

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
        n_step: int = 1,
    ) -> Dict[str, Any]:
        """Implements the On-policy TD Control algorithm 'n-step Semi Gradient SARSA'

        Args:
            num_episodes (int): max number of episodes
            max_ep_steps (int): max number of steps per episode
            discount (float): discount factor (gamma)
            epsilon (float): probability of taking a random action (epsilon-greedy)
            n_step (int): n-step return update target
        """
        logger.info("Start learning")
        self.alpha = step_size
        self.gamma = discount
        # linearly decaying exploration probability
        self.epsilon = np.arange(0, epsilon, epsilon / num_episodes)[::-1]

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # Init S & chose A from S using policy derived from Q
            state, _ = self.env.reset()
            action = self.run_policy(state, ep_i)

            for step in range(max_ep_steps):
                # Take action A, observe S' and R
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Chose A' from S' using policy derived from Q
                next_action = self.run_policy(next_state, ep_i)

                # Let the agent observe and update the Q-function
                self.observe(state, action, reward, next_state, next_action, terminated)

                # Collect some stats
                stats["ep_length"][ep_i] = step
                stats["ep_rewards"][ep_i] += reward

                if terminated or truncated:
                    if terminated:
                        logger.notice(f"terminated! | last R:{reward}")
                    break

                state = next_state

            # TODO: Remove
            if ep_i % 1000 == 0:
                self.plot_q_function()

            # Average loss over episode steps
            ep_r = stats["ep_rewards"][ep_i]
            ep_steps = stats["ep_length"][ep_i]
            logger.debug(f"Episode: {ep_i} -> R:{ep_r} ({ep_steps} steps)")

        # Done!
        self.env.close()

        return stats


if __name__ == "__main__":

    args = get_cli_parser("SARSA-learning options").parse_args()

    init_logger(level=args.log_level, logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)

    logger.info("Initializing agent")
    agent = SemiGradientSARSA(env)
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.max_episode_steps,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
        step_size=args.step_size,
    )

    plot_stats(stats)
