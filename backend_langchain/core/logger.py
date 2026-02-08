import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PACKAGE_ROOT / "output" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_level(value: Optional[str]) -> int:
    if not value:
        return logging.DEBUG
    return getattr(logging, value.upper(), logging.DEBUG)


def setup_logger(name: str) -> logging.Logger:
    """Configure a namespaced logger with console + file handlers."""
    logger = logging.getLogger(name)
    target_level = _resolve_level(os.environ.get("LOG_LEVEL"))
    logger.setLevel(target_level)

    if not getattr(logger, "_setup_done", False):
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_name = LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(file_name, encoding="utf-8")
        file_handler.setLevel(target_level)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(target_level)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False
        logger._setup_done = True

    return logger

