import logging
import time


class UTCFormatter(logging.Formatter):
    converter = time.gmtime  # type: ignore


def get_base_logging_handler():
    handler = logging.StreamHandler()
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    handler.setFormatter(UTCFormatter(fmt=format, datefmt=date_format))
    return handler


def get_file_logging_handler(filename):
    handler = logging.FileHandler(filename)
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    handler.setFormatter(UTCFormatter(fmt=format, datefmt=date_format))
    return handler


def get_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, handlers=[get_base_logging_handler()])
    return logger
