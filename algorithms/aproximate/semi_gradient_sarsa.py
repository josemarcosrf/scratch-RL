import random
from typing import Any
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm

from algorithms import State
from helpers.cli import get_cli_parser
from helpers.environment import get_env
from helpers.environment import get_env_action_dims
from helpers.environment import get_env_action_name_map
from helpers.environment import get_env_shape
from helpers.environment import get_env_state_dims
from helpers.logio import init_logger

# mypy: ignore-errors


class SemiGradientSARSA:
    """On-policy control n-step Semi-gradient SARSA with function approximation
    for episodic environents.

    For simplicity we are assuming the following:
        - actions range from 0 to n_actions
        - There's no step-size scheduling (remains constant)
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.n_states = get_env_state_dims(env)
        self.n_actions = get_env_action_dims(env)
        self.action_map = get_env_action_name_map(env)
        self.env_shape = get_env_shape(env)
        # Init a Q-value approximate function
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.Q = QNetwork().to(self.device)
        logger.notice(f"Running Q-Network in '{self.device}' device")

    def run_policy(self, state: State) -> int:
        """Run the current policy. In this case e-greedy with constant epsilon

        Args:
            state (int): agent state
        """
        if random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions))

        with torch.no_grad():
            return np.argmax(self.Q(state).cpu()).numpy().item()

    def observe(
        self, s: State, a: int, r: float, next_s: State, next_a: int, terminal: bool
    ) -> int:
        """Here is where the Q-update happens

        Args:
            s (State): current state
            a (int): current action
            r (float): reward
            next_s (State): next state (usually denoted as: s')
            next_a (int): next action (usually denoted as: a')
        """
        if terminal:
            # Update without the estimate of the next state (as it is terminal)
            delta_q = self.Q(s)
        else:
            # Update with the estimate delta d = q(s',a';W) - q(s,a;W)
            delta_q = self.Q(next_s) - self.Q(s)

        # TODO: Perform update with the TD-error
        td = self.alpha * (r + self.gamma * delta_q)
        loss = self.Q.update(s, td)

        return next_a, loss

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
            step_size (float): learning step size (alpha)
            n_step (int): n-step return update target
        """
        logger.info("Start learning")
        self.alpha = step_size
        self.gamma = discount
        self.epsilon = epsilon

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

            for i in range(max_ep_steps):
                # Take action A, observe S' and R
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Chose A' from S' using policy derived from Q
                next_action = self.run_policy(next_state)
                action, loss = self.observe(
                    state, action, reward, next_state, next_action, terminated
                )

                if terminated or truncated:
                    break

                # Collect some stats
                stats["ep_length"][ep_i] = i
                stats["ep_rewards"][ep_i] += reward
                stats["loss"].append(loss)

                state = next_state

        # Print the policy over the map
        self.env.close()

        return stats


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the neural network
        # TODO: Figure the input size from the env state snd action-space
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 3)
        # Define a loss function
        self.criterion = nn.MSELoss()
        # Define an optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, s: State):
        device = next(self.parameters()).device.type
        x = self.featurize(s).to(device)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    @staticmethod
    def featurize(state: State) -> np.array:
        # polinomial features. i.e: [x, y, x * y, x**2, y**2]
        _state = list(state)
        return torch.FloatTensor(_state + [s1 * s2 for s1 in _state for s2 in _state])

    def update(self, s: State, delta: float) -> float:
        pred = self.forward(s)
        loss = self.criterion(pred, delta)
        # Zero gradients, perform a backward pass, and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy().item()


if __name__ == "__main__":

    # reference implementations:
    # - https://github.com/self-supervisor/SARSA-Mountain-Car-Sutton-and-Barto
    args = get_cli_parser("SARSA-learning options").parse_args()

    init_logger(level=args.log_level, logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)

    logger.info("Initializing agent")
    agent = SemiGradientSARSA(env)
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.num_steps,
        step_size=args.step_size,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
    )

    # print_policy, plot_stats = get_env_report_functions(env)
    # print_policy(agent, stats)
    # plot_stats(agent, stats)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)

    ax[0].set_title("Episode length")
    ax[0].plot(stats["ep_length"])

    ax[1].set_title("Episode rewards")
    ax[1].plot(stats["ep_rewards"])

    ax[2].set_title("Loss over time")
    ax[2].plot(stats["loss"])
    plt.legend()
    plt.show()
