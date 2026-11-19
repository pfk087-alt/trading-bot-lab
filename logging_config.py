"""Configuracion centralizada de logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_file: str | Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configura logging de consola y archivo con rotacion."""
    logger = logging.getLogger("trading_bot_lab")
    if getattr(logger, "_tbl_configured", False):
        return logger

    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._tbl_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(name: str = "trading_bot_lab") -> logging.Logger:
    return logging.getLogger(name if name.startswith("trading_bot_lab") else f"trading_bot_lab.{name}")
