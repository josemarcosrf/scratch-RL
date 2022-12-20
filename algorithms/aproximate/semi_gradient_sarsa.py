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
from helpers.features.tile_coding import IHT
from helpers.features.tile_coding import tiles
from helpers.logio import init_logger
from helpers.models import QNetwork

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


class SemiGradientSARSA:
    """On-policy control n-step Semi-gradient SARSA with function approximation
    for episodic environents.

    For simplicity we are assuming the following:
        - actions range from 0 to n_actions
        - There's no step-size scheduling (remains constant)
    """

    def __init__(
        self, env, step_size: float, tiling_size: int = 2**10, num_tilings: int = 8
    ):
        super().__init__()
        # Environment
        self.env = env
        self.n_states = get_env_state_dims(env)
        self.n_actions = get_env_action_dims(env)
        self.action_map = get_env_action_name_map(env)
        self.env_shape = get_env_shape(env)
        # State featurizer
        box = env.observation_space
        self.tiling_size = tiling_size
        self.num_tilings = num_tilings
        self.boundaries = list(zip(box.low, box.high))
        self.iht = IHT(self.tiling_size)
        # Q-value approximate function
        # Note: instead of having Q: SxA -> R this implements Q: S -> R^|A|
        # This is: The network takes a given state S and outputs scores
        # for each of the actions
        # TODO: Make it featurizer aware
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.Q = QNetwork(
            input_size=self.tiling_size,
            output_size=self.n_actions,
            lr=step_size / self.num_tilings,
        ).to(self.device)
        logger.notice(f"Running Q-Network in '{self.device}' device")

    def plot_q_function(self):
        # Generate X, Y coordinates
        ranges = []
        box = self.env.observation_space
        if box.bounded_above.any() and box.bounded_below.any():
            for (l, h) in zip(box.low, box.high):
                ranges.append(np.arange(l, h, (h - l) / self.num_tilings))

        x, y = np.meshgrid(*ranges)
        # Compute the Q-values
        states = list(product(*ranges))
        with torch.no_grad():
            z = (
                self.Q(self.featurize(states))
                .cpu()
                .numpy()
                .max(axis=-1)
                .reshape(x.shape)
            )
        # Plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap="viridis", edgecolor="green")
        ax.set_title("Q-value function")
        plt.show()

    @staticmethod
    def poly_featurize(states: Union[State, List[State]]) -> torch.FloatTensor:
        """Sort of polinomial features. i.e: [x, y, x * y, x**2, y**2]"""

        def _feat(state):
            _state = list(state)
            return torch.FloatTensor(
                _state + [s1 * s2 for s1 in _state for s2 in _state]
            )

        if isinstance(states, list):
            return torch.vstack([_feat(s) for s in states])

        return _feat(states)

    def tile_featurize(self, states: Union[State, List[State]]) -> torch.FloatTensor:
        def _feat(state):
            indices = tiles(
                self.iht,
                self.num_tilings,
                [
                    self.num_tilings * s / (b[1] - b[0])
                    for s, b in zip(state, self.boundaries)
                ],
            )
            # 1-hot encoding
            x = np.zeros(self.tiling_size)
            x[indices] = 1
            return torch.FloatTensor(x)

        if isinstance(states, list):
            return torch.vstack([_feat(s) for s in states])

        return _feat(states)

    def run_policy(self, state: State) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon

        Args:
            state (int): agent state
        """
        if random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions))

        with torch.no_grad():
            return np.argmax(self.Q(self.featurize(state)).cpu()).numpy().item()

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
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
        self.gamma = discount
        self.epsilon = epsilon
        self.featurize = self.tile_featurize  # set tiling as encoding of states

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "loss": [],
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # Init S & chose A from S using policy derived from Q
            state, _ = self.env.reset()
            action = self.run_policy(state)

            ep_loss = []
            q_values = []
            targets = []
            for i in range(max_ep_steps):
                # Take action A, observe S' and R
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Chose A' from S' using policy derived from Q
                next_action = self.run_policy(next_state)

                # NOTE: As we modified the Q-function to output scores for all actions
                # and we want to use the gradients from pytorch we modify the update
                # rule by reframing it as a supervised target for backprop using MSELoss
                q_s = self.Q(self.featurize(state))
                q_ns = self.Q(self.featurize(next_state))
                g = reward + self.gamma * q_ns
                target = q_s.detach().clone()
                target[action] = g[next_action]

                # TODO: Implement a better batching.
                # For now: Accumulate for batch update of the Q-network
                q_values.append(q_s)
                targets.append(target)

                # Collect some stats
                stats["ep_length"][ep_i] = i
                stats["ep_rewards"][ep_i] += reward

                if terminated or truncated:
                    logger.info(
                        f"terminated={terminated} | "
                        f"truncated={truncated} | last R:{reward}"
                    )
                    break

                state = next_state

            # Update Q-network
            ep_loss = self.Q.update(torch.vstack(q_values), torch.vstack(targets))

            # # TODO: Remove
            # if ep_i % 1000 == 0:
            #     self.plot_q_function()

            # Average loss over episode steps
            stats["loss"].append(np.mean(ep_loss))
            ep_r = stats["ep_rewards"][ep_i]
            ep_steps = stats["ep_length"][ep_i]
            logger.debug(
                f"Episode: {ep_i} -> R:{ep_r} "
                f"[loss: {ep_loss:.4f}] ({ep_steps} steps)"
            )

        # Print the policy over the map
        self.env.close()

        return stats


if __name__ == "__main__":

    # reference implementations:
    # - https://medium.com/swlh/learning-with-deep-sarsa-openai-gym-c9a470d027a
    # - https://ai.stackexchange.com/questions/35717/how-to-perform-the-back-propagation-step-in-semi-gradient-sarsa-using-a-deep-neu
    # - https://stackoverflow.com/questions/45377404/episodic-semi-gradient-sarsa-with-neural-network
    # - https://github.com/self-supervisor/SARSA-Mountain-Car-Sutton-and-Barto
    # - https://gist.github.com/neilslater/28004397a544f97b2ff03d25d4ddae52
    args = get_cli_parser("SARSA-learning options").parse_args()

    init_logger(level=args.log_level, logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)

    logger.info("Initializing agent")
    agent = SemiGradientSARSA(
        env,
        step_size=args.step_size,
    )
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.num_steps,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
    )

    plot_stats(stats)
