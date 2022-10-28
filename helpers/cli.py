import argparse

from helpers.constants import DEEFAULT_MAX_EP_STEPS
from helpers.constants import DEEFAULT_NUM_EPISODES
from helpers.constants import DEFAULT_ENV_RENDER_MODES
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
        default=DEEFAULT_NUM_EPISODES,
        help="The max number of training episodes",
    )
    parser.add_argument(
        "--num-steps",
        "-ns",
        type=int,
        default=DEEFAULT_MAX_EP_STEPS,
        help="The max number of training steps per episode",
    )


def get_cli_parser(desc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(desc)
    make_env_options(parser)
    make_train_options(parser)
    make_logging_options(parser)

    return parser
