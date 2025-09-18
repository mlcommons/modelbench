import logging
import time


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    handler.setFormatter(UTCFormatter(fmt=format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    return logger
