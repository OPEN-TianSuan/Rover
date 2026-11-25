"""Configure logging for the Rover namespace on import."""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

LOGGER_NAME = "rover"


@lru_cache(maxsize=1)
def _get_log_level() -> int:
    level_str = os.getenv("ROVER_LOG_LEVEL", "info").lower().strip()
    level_map = {
        "0": logging.CRITICAL + 1,
        "none": logging.CRITICAL + 1,
        "off": logging.CRITICAL + 1,
        "false": logging.CRITICAL + 1,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    return level_map.get(level_str, logging.INFO)


logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(_get_log_level())
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
