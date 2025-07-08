import logging
import os
from logging.handlers import SocketHandler, TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_STREAM = os.getenv("LOG_TO_STREAM", "true").lower() == "true"
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/neurotrace.log")
LOG_TO_NETWORK = os.getenv("LOG_TO_NETWORK", "false").lower() == "true"
LOG_HOST = os.getenv("LOG_HOST", "localhost")
LOG_PORT = int(os.getenv("LOG_PORT", 9020))


def get_logger(name: str) -> logging.Logger:
    """
    Central logger factory for Neurotrace with support for:
    - Stream logs
    - Daily rotating file logs
    - Optional network logging via sockets
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False  # prevent duplicate logs

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

        if LOG_TO_STREAM:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if LOG_TO_FILE:
            Path(LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
            file_handler = TimedRotatingFileHandler(LOG_FILE_PATH, when="midnight", backupCount=7)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if LOG_TO_NETWORK:
            try:
                socket_handler = SocketHandler(LOG_HOST, LOG_PORT)
                socket_handler.setFormatter(formatter)
                logger.addHandler(socket_handler)
            except Exception as e:
                logger.warning(f"Could not attach network log handler: {e}")

    return logger
