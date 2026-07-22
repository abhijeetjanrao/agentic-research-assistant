"""
Centralized logging setup.

Why loguru instead of stdlib logging:
    In a multi-agent system, logs are your main debugging tool: which agent
    ran, in what order, with what inputs/outputs. Loguru gives structured,
    readable logs with almost no boilerplate, and makes it trivial to route
    logs to a file in production and stdout in development -- important
    later for AWS deployment, where stdout is what CloudWatch picks up.

Usage:
    from app.logging_config import setup_logging, logger
    setup_logging()
    logger.info("Manager agent started session {session_id}", session_id=123)
"""

import sys

from loguru import logger

from app.config import get_settings


def setup_logging() -> None:
    """Configure loguru sinks. Call this once, at app startup."""
    settings = get_settings()

    logger.remove()  # remove loguru's default handler so we control format

    # Console sink -- human-readable, colorized, good for local dev
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        enqueue=True,  # thread/async-safe, important since FastAPI + agents run concurrently
    )

    # File sink -- rotates daily, keeps 14 days, useful once deployed
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="00:00",
        retention="14 days",
        enqueue=True,
        backtrace=True,
        diagnose=settings.debug,  # only show variable values in tracebacks during dev
    )


__all__ = ["setup_logging", "logger"]
