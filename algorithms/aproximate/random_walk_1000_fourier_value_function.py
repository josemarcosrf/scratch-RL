import logging
from typing import Any
from typing import Callable
from typing import Dict

import numpy as np
from tqdm.auto import tqdm

from algorithms import State
from helpers.environment.random_1D_walk import Random1DWalk
from helpers.features.fourier import FourierBasis
from helpers.io import init_logger
from helpers.plotting import plot_line

logger = logging.getLogger(__name__)
init_logger(level=logging.DEBUG, my_logger=logger)


class FourierLinearValueFunction:
    def __init__(self, interval_len: int, n_params: int = 5):
        # Generate a random set of axis frequency vectors.
        c = np.arange(0, n_params + 1)
        self.weights = np.zeros(n_params + 1)
        self.features = FourierBasis(interval_len, n_params, c)

    def __call__(self, s: State) -> Any:
        return np.dot(self.weights, self.features.encode(s))

    def update(self, s: State, delta: float):
        derivate_val = self.__call__(s)
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
        return self.policy(state, self.epsilon)

    def update(self, s: State, r: float, next_s: State) -> None:
        delta = self.alpha * (r + self.gamma * self.V(next_s) - self.V(s))
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
                stats["visits"][state][action] += 1

                state = next_state

        # Print the policy over the map
        self.env.close()

        return stats


def compute_true_value_function(
    env, gamma: float, iterations: int = 1000, min_delta: float = 1e-2
):
    V = np.zeros(env.state_space.n)

    # Repeat for as many iterations
    for i in tqdm(range(iterations)):
        # old_v = V.copy()

        # starting at each state...
        for init_state in range(env.state_space.n):
            # ... and continuing until we reach completion
            state = init_state
            env.reset(state)  # hack the initial state

            for action in env.ACTIONS.keys():
                next_state, reward, terminated, _, _ = env.step(action)

                mark_terminal = "(T)" if terminated else "   "
                update_reward = reward if terminated else reward + gamma * V[next_state]
                logger.debug(
                    f"s:{state:^2} -> a:{action:^2} -> s':{next_state:^2} "
                    f"{mark_terminal} | r: {reward:^2} "
                    f" ==> V[{state}] += {update_reward}"
                )

                # Update the Value function
                if terminated:
                    logger.debug("-----------------------")
                    V[state] += reward
                    break

                V[state] += reward + gamma * V[next_state]
                state = next_state

            logger.debug(f"ITER {i} -> V: {V}")
            input("...")

        # # Early stop if the improvement is less than min_delta
        # delta = np.abs(np.sum(V - old_v))
        # if delta < min_delta:
        #     break

    return V


if __name__ == "__main__":
    """Implementation of:
    'Figure 9.5: Fourier basis vs polynomials on the 1000-state random walk'
    from Sutton and Barto's book:
    'Reinforcement Learning: An Introduction'
    """

    # Create the environment
    env_size = 10
    env = Random1DWalk(size=10)

    # Define a policy for the 1000-step random walk world
    def policy(_) -> int:
        # No matter what state we are in, for this environment policy
        # as we go left / right randomly with equal probability
        return list(env.ACTIONS.keys())[np.random.binomial(1, 0.5)]

    # compute the real value function
    real_v = compute_true_value_function(env, 0.95, iterations=10)

    from matplotlib import pyplot as plt

    plt.plot(real_v)
    plt.show()

    # # Create a linear function aproximation
    # value_func = FourierLinearValueFunction(env_size)

    # sg_td = SemiGradientTD(env, value_func, policy)
    # sg_td.learn(
    #     num_episodes=1000,
    #     max_ep_steps=2000,
    #     epsilon=0.5,
    #     discount=0.95,
    #     step_size=0.5,
    # )
