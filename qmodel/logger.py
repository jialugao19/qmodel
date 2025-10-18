import logging
import sys

def _standard_handler():
    GREEN = "\033[32m"
    BLUE = "\033[94m"   # 36 for loguru
    STRONG = "\033[1m"
    RESET = "\033[0m"

    fmt = logging.Formatter(f"{GREEN}%(asctime)s.%(msecs)03d{RESET} | {STRONG}%(levelname)-7s{RESET} | {BLUE}%(filename)s{RESET}:{BLUE}%(funcName)s{RESET}:{BLUE}%(lineno)d{RESET} - %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG)
    return sh


class MyLogger:
    def __init__(self, name=""):
        self._logger_name = name if name else None
        self._logger = logging.getLogger(self._logger_name)

    def __getattr__(self, name):
        # Prevent infinite recursion by not intercepting internal attributes
        if name in {"_logger", "_logger_name", "_get_logger"}:
            raise AttributeError(f"{name} is not accessible via __getattr__")
        return getattr(self._get_logger(), name)

    def _get_logger(self):
        # Defensive lookup to ensure _logger is always valid
        if not hasattr(self, "_logger") or self._logger is None:
            self._logger = logging.getLogger(self._logger_name)
        return self._logger

    def __getstate__(self):
        # Ensure pickling only stores logger name (not actual logger instance)
        return {'_logger_name': self._logger_name}

    def __setstate__(self, state):
        # On unpickling, restore logger from name
        self._logger_name = state['_logger_name']
        self._logger = logging.getLogger(self._logger_name)

    def add_file_log(self, log_file: str):
        add_file_log(self, log_file)


def _init_logger(logger: MyLogger):
    sh = _standard_handler()

    logger.handlers.clear()
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    logger.info("Logger initialized")

def _is_initialized(logger: MyLogger) -> bool:
    if len(logger.handlers) != 1:
        return False
    handler = logger.handlers[0]
    if handler.formatter is None:
        return False
    fmt_string = handler.formatter._fmt

    return (fmt_string is not None) and fmt_string.startswith("\033")

def add_file_log(logger: MyLogger, log_file: str):
    # set output to file
    fh = logging.FileHandler(log_file)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(filename)s:%(funcName)s:%(lineno)d - %(message)s")

    fh.setFormatter(fmt)
    logger.addHandler(fh)


# create a default logger
# wrapping by MyLogger is necessary to ensure it will be recreated in subprocesses
# simply calling logging.getLogger will not work
logger = MyLogger()

if not _is_initialized(logger):
    _init_logger(logger)
