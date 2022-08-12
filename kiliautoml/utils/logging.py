import sys
from typing import Set

from loguru import logger

from kiliautoml.utils.type import VerboseLevelT


def set_kili_logging(verbose: VerboseLevelT):
    logger.remove()
    logger.add(
        sys.stderr,
        level=verbose,
        colorize=True,
        format="<yellow>KiliAutoML</yellow> {level} <level>{message}</level>",
    )


class OneTimeLogger:
    messages_already_printed: Set[str] = set()

    def __call__(self, msg: str) -> None:
        """If the first argument in the print is a deja-vu string, do not print"""

        if msg not in self.messages_already_printed:
            self.messages_already_printed.add(msg)
            logger.info(msg)
        else:
            # Already printed
            pass


one_time_logger = OneTimeLogger()
