"""
logging_config.py — Structured Logging Setup (Prompt P22)
=========================================================

Configures `structlog` for the EcoPackAI application.
It provides JSON formatting in production and colored console
formatting for development.
"""

import logging
import sys
from typing import Any, Dict

import structlog

from src.settings import get_settings


def configure_logging() -> None:
    """Configure structured logging application-wide."""
    settings = get_settings()

    # Define common processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Environment-specific formatting
    if settings.is_production:
        formatter = structlog.processors.JSONRenderer()
    else:
        formatter = structlog.dev.ConsoleRenderer(colors=True)

    # Configure stdlib logging
    log_level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Reconfigure the root logger's handlers to use the structlog formatter
    formatter_wrapper = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            formatter,
        ],
    )
    for handler in logging.root.handlers:
        handler.setFormatter(formatter_wrapper)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)
