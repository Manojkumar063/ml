import logging
import os
from config import config

os.makedirs(os.path.dirname(config.log_path), exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        fh = logging.FileHandler(config.log_path)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
