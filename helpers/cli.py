import argparse

from helpers.constants import DEFAULT_DISCOUNT_FACTOR
from helpers.constants import DEFAULT_ENV_RENDER_MODES
from helpers.constants import DEFAULT_EXPLORE_PROBABILITY
from helpers.constants import DEFAULT_MAX_EP_STEPS
from helpers.constants import DEFAULT_NUM_EPISODES
from helpers.constants import DEFAULT_STEP_SIZE
from helpers.environments import ENV_META


def make_logging_options(parser: argparse.ArgumentParser):
    parser.add_argument_group("Logging options")
    parser.add_argument(
        "--log-level",
        "-v",
        default="INFO",
        help="Sets the logging level",
    )


def make_env_options(parser: argparse.ArgumentParser):
    parser.add_argument_group("Environment options")
    parser.add_argument(
        "--env-name",
        "-e",
        required=True,
        choices=ENV_META.keys(),
        help="The gymnasium environment name",
    )
    parser.add_argument(
        "--render-mode",
        choices=DEFAULT_ENV_RENDER_MODES,
        help="Environment rendering mode",
    )


def make_train_options(parser: argparse.ArgumentParser):
    parser.add_argument_group("Training options")
    parser.add_argument(
        "--num-episodes",
        "-ne",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="The max number of training episodes",
    )
    parser.add_argument(
        "--num-steps",
        "-ns",
        type=int,
        default=DEFAULT_MAX_EP_STEPS,
        help="The max number of training steps per episode",
    )
    parser.add_argument(
        "--step-size",
        "-st",
        type=float,
        default=DEFAULT_STEP_SIZE,
        help="Training step size parameter (alpha)",
    )
    parser.add_argument(
        "--discount-factor",
        "-df",
        type=float,
        default=DEFAULT_DISCOUNT_FACTOR,
        help="Reward discount factor (gamma)",
    )
    parser.add_argument(
        "--explore-probability",
        "-ep",
        type=float,
        default=DEFAULT_EXPLORE_PROBABILITY,
        help="Probability of taking a random action (epsilon)",
    )


def get_cli_parser(desc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(desc)
    make_env_options(parser)
    make_train_options(parser)
    make_logging_options(parser)

    return parser
