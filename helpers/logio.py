import os
import sys
from functools import partialmethod
from typing import Optional

from tqdm.auto import tqdm


LOGURU_FMT = (
    "<dim>{time:YYYY-MM-DD HH:mm:ss}</dim> <level>{level:<8}</level> | "
    "<blue>{name:^15}:{function:^15}:{line:>3}</blue> - {message}"
)


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.write(msg, end="")

    @classmethod
    def flush(_):
        sys.stdout.flush()


def init_logger(
    logger,
    level,
    fmt: Optional[str] = None,
    logdir: Optional[str] = None,
    filename: Optional[str] = None,
    rotation_minutes: Optional[int] = 60,
    retention_days: Optional[int] = 14,
    serialize: Optional[bool] = False,
):
    _fmt = fmt or LOGURU_FMT

    # if not TqdmStream, we should use: 'sys.stderr'
    handlers = [
        {"sink": TqdmStream(), "format": _fmt, "colorize": True, "level": level},
    ]
    # logging to file
    if logdir and filename:
        handlers.append(
            {
                "sink": os.path.join(logdir, filename),
                "rotation": f"{rotation_minutes} minute",
                "retention": f"{retention_days} days",
                "serialize": serialize,
            }
        )

    config = {"handlers": handlers}
    logger.configure(**config)

    # Add a 'NOTICE' level
    if not hasattr(logger.__class__, "notice"):
        logger.level("NOTICE", no=25, icon="🤖", color="<cyan><bold>")
        logger.__class__.notice = partialmethod(logger.__class__.log, "NOTICE")
