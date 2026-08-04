"""Loguru configuration. Call setup_logging() once at process startup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# rag/logs/ — sibling of src/, eval_runs/, etc. Gitignored: runtime output,
# not source.
_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"


def setup_logging(level: str = "INFO", service_name: str = "app") -> None:
    """Call once at process startup (main scripts and FastAPI lifespan).

    Every start writes to both stderr (as before) and a dated file under
    rag/logs/<service_name>_YYYY-MM-DD.log — one file per calendar day,
    persisted across restarts. This is what to check after an incident
    instead of `podman logs`/`journalctl` (which need container/VM access
    the Drupal team doesn't have, and which `journalctl --user` can't even
    read on the GPU VM due to a permissions gap).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        _LOGS_DIR / f"{service_name}_{{time:YYYY-MM-DD}}.log",
        level=level,
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    )
