"""Console diagnostics. Never log notes, prompts, model output or embeddings by default."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os
import sys
from uuid import uuid4

REQUEST_ID: ContextVar[str] = ContextVar("notes_request_id", default="-")
LOGGER_NAME = "notes"


def new_request_id() -> str:
    return uuid4().hex[:12]


def request_id() -> str:
    return REQUEST_ID.get()


@contextmanager
def trace_context(identifier: str):
    token = REQUEST_ID.set(identifier)
    try:
        yield
    finally:
        REQUEST_ID.reset(token)


def allow_content_preview() -> bool:
    """Explicit opt-in only: malformed upstream lines may contain private text."""
    return os.getenv("NOTES_LOG_PREVIEW", "0").strip().lower() in {"1", "true", "yes"}


def safe_preview(value: str, limit: int = 120) -> str:
    if not allow_content_preview():
        return "[hidden; set NOTES_LOG_PREVIEW=1 to show raw excerpts]"
    # ascii() escapes Cyrillic and control characters; this also works in Git Bash.
    return ascii(value[:limit] + ("..." if len(value) > limit else ""))


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID.get()
        return True


def configure_logging() -> None:
    """Idempotent console handler independent of Uvicorn's logging config."""
    logger = logging.getLogger(LOGGER_NAME)
    level_name = os.getenv("NOTES_LOG_LEVEL", "INFO").strip().upper()
    if level_name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level_name = "INFO"
    logger.setLevel(getattr(logging, level_name))
    logger.propagate = False
    if not any(getattr(handler, "_notes_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._notes_handler = True
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-7s | req=%(request_id)s | "
            "%(name)s | %(message)s", datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.info("diagnostics.ready level=%s previews=%s pid=%s",
                level_name, allow_content_preview(), os.getpid())
