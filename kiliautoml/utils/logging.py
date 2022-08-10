import logging
from typing import Any, Dict, Set

from termcolor import colored
from typing_extensions import Literal

VerboseLevelT = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def set_logging(verbose: VerboseLevelT):
    map_level: Dict[VerboseLevelT, Any] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig(filename="automl.log", level=map_level[verbose])


def kili_print(*args, **kwargs) -> None:
    print(colored("kili:", "yellow", attrs=["bold"]), *args, **kwargs)


class OneTimePrinter:
    messages_already_printed: Set[str] = set()

    def __call__(self, *args, **kwargs) -> None:
        """If the first argument in the print is a deja-vu string, do not print"""

        if args and isinstance(args[0], str):
            if args[0] not in self.messages_already_printed:
                self.messages_already_printed.add(args[0])
                kili_print(*args, **kwargs)
            else:
                # Already printed
                pass
        else:
            kili_print(*args, **kwargs)


class KiliLogger:
    def info(self, msg):
        ...
