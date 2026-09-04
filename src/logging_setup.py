"""Logging setup: one application logger, configured once at the composition root.

Core modules log through :func:`get_logger` — ``"docqa.<module>"`` child
loggers — without caring how logging is configured. The composition root
calls :func:`configure_logging` once at startup: INFO reaches the console,
DEBUG reaches a log file under the data directory, so startup problems,
Chroma misbehavior, and model-call failures leave a trace.

Third-party libraries (chromadb, streamlit, …) keep their own separate
logging namespaces and are untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER_NAME = "docqa"
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class _NoTracebackInfoFormatter(logging.Formatter):
    """Formatter that keeps INFO lines compact while showing tracebacks on errors."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            return super().format(record)
        record_exc_info = record.exc_info
        record.exc_info = None
        try:
            return super().format(record)
        finally:
            record.exc_info = record_exc_info


def get_logger(name: str) -> logging.Logger:
    """Return the application logger for ``name`` (a ``docqa.<name>`` child logger)."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def configure_logging(log_dir: Path, console_level: int = logging.INFO) -> None:
    """Configure the application logger: ``console_level`` to the console, DEBUG to ``log_dir / "doc-qa.log"``.

    Streamlit reruns the app module on every interaction, so this is
    idempotent: handlers are replaced, never duplicated. The file handler is
    opened lazily (``delay=True``) so an unused log file is never created.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(_NoTracebackInfoFormatter(_LOG_FORMAT))
    logger.addHandler(console)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / "doc-qa.log", encoding="utf-8", delay=True
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(file_handler)
