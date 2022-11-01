import logging
import random
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from algorithms.tabular import TabularAgent
from helpers import init_logger
from helpers.cli import get_cli_parser
from helpers.constants import DEFAULT_RANDOM_SEED
from helpers.environments import get_env
from helpers.plotting import plot_stats


logger = logging.getLogger(__name__)


class TabularMonteCarlo(TabularAgent):
    def __init__(self, env):
        """Initializes a tabular first-visit Monte Carlo agent for the given environment.

        For simplicity we are assuming the following:
         - actions range from 0 to n_actions
        """
        super().__init__(env)
        logger.debug(f"Q has shape: {self.Q.shape}")

    def learn(
        self,
        num_episodes: int,
        max_ep_steps: int,
        discount: float,
        epsilon: float,
        step_size: float,
    ) -> Dict[str, Any]:
        return {}


if __name__ == "__main__":

    args = get_cli_parser("Monte Carlo learning options").parse_args()

    init_logger(level=args.log_level, my_logger=logger)

    logger.info("Initializing environment")
    env = get_env(args.env_name, render_mode=args.render_mode)
    env.action_space.seed(DEFAULT_RANDOM_SEED)

    logger.info("Initializing agent")
    agent = TabularMonteCarlo(env)
    stats = agent.learn(
        num_episodes=args.num_episodes,
        max_ep_steps=args.num_steps,
        step_size=args.step_size,
        discount=args.discount_factor,
        epsilon=args.explore_probability,
    )

    plot_stats(stats, agent.env_shape)
