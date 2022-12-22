import random
from itertools import product
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
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


def re_timewrap(env, max_steps: int):
    from gymnasium.wrappers.time_limit import TimeLimit

    env.spec.max_episode_steps = max_steps
    return TimeLimit(env.unwrapped, max_episode_steps=max_steps)


def plot_stats(stats):
    _, ax = plt.subplots(1, 3)
    # Episode steps
    ax[0].set_title("Episode length")
    ax[0].plot(stats["ep_length"])
    # Total episode rewards
    ax[1].set_title("Episode rewards")
    ax[1].plot(stats["ep_rewards"])
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
        self.featurize = self.tile_featurize
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
        # Prepare for interactive plotting
        plt.ion()
        plt.show()
        self.fig = plt.figure(figsize=(8, 6))

    def plot_q_function(self, block: bool = False):
        # Generate X, Y coordinates
        ranges = []
        box = self.env.observation_space
        if box.bounded_above.any() and box.bounded_below.any():
            for (l, h) in zip(box.low, box.high):
                step_size = (h - l) / self.num_tilings / 10
                ranges.append(np.arange(l, h, step_size))
        x, y = np.meshgrid(*ranges)

        # Compute the Q-values
        states = list(product(*ranges))
        with torch.no_grad():
            q = self.Q(self.featurize(states)).cpu().numpy()

        # Max over actions as the State value
        z = q.max(axis=-1).reshape(x.shape)

        # Colorize the plot based on the chose action
        # FIXME: These colors are only valid for MountainCar!
        rgb = [(1, 0, 0, 0.3), (0.3, 0.3, 0.3, 0.3), (0, 0, 1, 0.3)]
        colors = np.array([rgb[i] for i in q.argmax(axis=-1)]).reshape((*x.shape, 4))

        # Plot
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        ax.set_title("Q-value function")
        ax.plot_surface(x, y, z, cmap="viridis", edgecolor="green", facecolors=colors)
        if block:
            plt.ioff()
            plt.show()
        else:
            plt.ion()
            plt.draw()
            plt.pause(0.001)

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

    def run_policy(self, state: State, step) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon

        Args:
            state (int): agent state
        """
        if random.random() < self.epsilon[step]:
            return np.random.choice(range(self.n_actions))

        with torch.no_grad():
            return np.argmax(self.Q(self.featurize(state)).cpu()).numpy().item()

    def observe(self, n_step_buffer: List[Tuple[Any]]):
        """Here is where the Q-update happens.

        NOTE: As we modified the Q-function to output scores for all actions
        and we want to use the gradients from pytorch we modify the update
        rule by reframing it as a supervised target for backprop using MSELoss

        Args:
            state (State): current state
            action (int): current action
            reward (float): reward
            next_state (State): next state (usually denoted as: s')
            next_action (int): next action (usually denoted as: a')
        """
        # FIXME: Adapt to a terminal state observation!
        n = len(n_step_buffer)
        state, action, _, _, _ = n_step_buffer[0]
        next_state, next_action, _, _, _ = n_step_buffer[-1]
        q_ns = self.Q(self.featurize(next_state))
        # Compute G_t
        g = (
            sum(self.gamma**i * r for i, (_, _, r, _, _) in enumerate(n_step_buffer))
            + self.gamma**n * q_ns[next_action]
        )
        # Reframe as a supervised update
        q_s = self.Q(self.featurize(state))
        target = q_s.detach().clone()
        target[action] = g

        # Backprop
        return self.Q.update(q_s, target)

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        n: int = 4,
    ) -> Dict[str, Any]:
        """Implements the On-policy TD Control algorithm 'n-step Semi Gradient SARSA'

        Args:
            num_episodes (int): max number of episodes
            max_ep_steps (int): max number of steps per episode
            discount (float): discount factor (gamma)
            epsilon (float): probability of taking a random action (epsilon-greedy)
            n (int): n-step return update target
        """
        logger.info("Start learning")
        self.gamma = discount
        # # linearly decaying exploration probability
        # self.epsilon = np.arange(0, epsilon, epsilon / num_episodes)[::-1]
        self.epsilon = np.ones(num_episodes) * epsilon

        stats = {
            "ep_length": np.zeros(num_episodes),
            "ep_rewards": np.zeros(num_episodes),
            "ep_loss": np.zeros(num_episodes),
        }

        episode_iter = tqdm(range(num_episodes))
        for ep_i in episode_iter:
            episode_iter.set_description(f"Episode: {ep_i}")

            # Init S & chose A from S using policy derived from Q
            state, _ = self.env.reset()
            action = self.run_policy(state, ep_i)

            ep_loss = []
            n_step_buffer = []
            for t in range(max_ep_steps):

                if state[0] > 0.3:
                    logger.notice(f"👀 this far! {state}")

                # Take action A, observe S' and R
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Chose A' from S' using policy derived from Q
                next_action = self.run_policy(next_state, ep_i)

                if len(n_step_buffer) >= n:
                    # Update Q-network
                    loss = self.observe(n_step_buffer)
                    stats["ep_loss"][ep_i] += loss
                    n_step_buffer = []
                else:
                    # Let the agent observe and update the Q-function
                    n_step_buffer.append(
                        (state, action, reward, next_state, next_action)
                    )

                # Collect some stats
                stats["ep_length"][ep_i] = t
                stats["ep_rewards"][ep_i] += reward

                if terminated or truncated:
                    if terminated:
                        logger.notice(f"terminated! | last R:{reward}")
                    break

                state = next_state

            # TODO: Remove
            if ep_i % 50 == 0:
                self.plot_q_function()

            # TODO: Remove
            ep_r = stats["ep_rewards"][ep_i]
            ep_steps = stats["ep_length"][ep_i]
            ep_loss = stats["ep_loss"][ep_i]
            logger.debug(
                f"Episode: {ep_i} -> R:{ep_r} "
                f"[loss: {ep_loss:.4f}] ({ep_steps} steps)"
            )

        # Done!
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

    # FIXME: Remove time re-wrapping
    env = re_timewrap(env, args.max_episode_steps)
    logger.debug(f" > env max steps: {env.spec.max_episode_steps}")

    logger.info("Initializing agent")
    agent = SemiGradientSARSA(
        env,
        step_size=args.step_size,
    )
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.max_episode_steps,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
        n=4,
    )
    agent.plot_q_function(block=True)
    plot_stats(stats)

    import pickle

    with open("sarsa-mountain-car.pkl", "wb") as f:
        pickle.dump(agent, f)
