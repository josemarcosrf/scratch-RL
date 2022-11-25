import argparse
import logging
import sys
from typing import Any
from typing import Callable
from typing import Dict

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from algorithms import State
from helpers.environment.random_1D_walk import Random1DWalk
from helpers.features.fourier import FourierBasis
from helpers.logio import init_logger
from helpers.plotting import plot_line

init_logger(level=logging.DEBUG, logger=logger)


class FourierLinearValueFunction:
    def __init__(self, domain_range: int, n_params: int = 5):
        # Generate a random set of axis frequency vectors.
        c = np.arange(n_params + 1)
        self.weights = np.ones(n_params + 1)
        self.features = FourierBasis(domain_range, n_params, c)

    def __call__(self, s: State) -> Any:
        return np.dot(self.weights, self.features.encode(s))

    def update(self, s: State, delta: float):
        derivate_val = self.features.encode(s)
        self.weights += delta * derivate_val


class SemiGradientTD:
    r"""This class implements:
    'Semi-gradient TD(0) for estimating v ≈ v_{\pi} (Chapter 9. - page 164)
    from Sutton and Barto's book 'Reinforcement Learning: An Introduction'
    """

    def __init__(self, env, value_func: FourierLinearValueFunction, policy: Callable):
        self.env = env
        self.V = value_func
        self.policy = policy

    def run_policy(self, state: State) -> int:
        return self.policy(state)

    def update(self, s: State, r: float, next_s: State) -> None:
        delta = self.alpha * (r + self.gamma * self.V(next_s) - self.V(s))

        # logger.debug(f"UPDATING for state {s} -> Delta: {delta}")
        self.V.update(s, delta)

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
    ) -> Dict[str, Any]:

        self.alpha = step_size
        self.epsilon = epsilon
        self.gamma = discount

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "visits": np.zeros(self.env.state_space.n),
            "returns": np.zeros(self.env.state_space.n),
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
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update(state, reward, next_state)

                if terminated or truncated:
                    break

                # Collect some stats
                stats["ep_length"][ep_i] = i
                stats["ep_rewards"][ep_i] += reward
                stats["visits"][state] += 1

                state = next_state

        # Print the policy over the map
        self.env.close()

        return stats


def compute_true_value_function(env):
    v_left = [-1 * (0.5**i) for i in range(env.state_space.n)]
    v_right = [-1 * v for v in v_left[::-1]]

    return np.array(v_left) + np.array(v_right)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-size", required=True, type=int, help="Size of 1D random walk world"
    )
    parser.add_argument(
        "--fourier-basis", type=int, default=5, help="Number of Fourier basis"
    )
    return parser


if __name__ == "__main__":
    """Implementation of:
    'Figure 9.5: Fourier basis vs polynomials on the 1000-state random walk'
    from Sutton and Barto's book:
    'Reinforcement Learning: An Introduction'
    """

    args = cli().parse_args()

    # Create the environment
    env = Random1DWalk(size=args.env_size)

    # Define a policy for the 1000-step random walk world
    def policy(_) -> int:
        # No matter what state we are in, for this environment policy
        # as we go left / right randomly with equal probability
        return list(env.ACTIONS.keys())[np.random.binomial(1, 0.5)]

    # compute the real value function
    true_v = compute_true_value_function(env)

    # Create a linear function aproximation
    value_func = FourierLinearValueFunction(args.env_size, n_params=args.fourier_basis)

    sg_td = SemiGradientTD(env, value_func, policy)
    sg_td.learn(
        num_episodes=1000,
        max_ep_steps=2000,
        epsilon=0.5,
        discount=0.95,
        step_size=0.5,
    )

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(true_v)
    ax[0].set_title("True Value Function")
    ax[1].plot([value_func(s) for s in range(env.state_space.n)])
    ax[1].set_title("Fourier Value Function")
    plt.legend()
    plt.show()
