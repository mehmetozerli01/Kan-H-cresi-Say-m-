"""Merkezi loglama yapılandırması — terminal + dosya."""

import logging
import os

from config import LOG_DIR, LOG_FILE

_LOGGING_CONFIGURED = False


def setup_logging(log_file: str = LOG_FILE) -> None:
    """Terminale ve pipeline.log dosyasına log yazar."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    log_dir = os.path.dirname(log_file) or LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Modül bazlı logger döndürür."""
    setup_logging()
    return logging.getLogger(name)
