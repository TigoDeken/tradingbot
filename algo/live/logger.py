import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "live", "logs")


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s UTC  %(levelname)-8s  %(name)s  %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    fmt.converter = __import__("time").gmtime  # UTC timestamps

    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "engine.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    setup_logger()
    return logging.getLogger(name)
