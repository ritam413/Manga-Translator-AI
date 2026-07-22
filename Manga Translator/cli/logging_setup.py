"""
cli/logging_setup.py — Configures file + stderr logging for the CLI.

Call `setup_logging(project_dir, log_level)` once at CLI startup.
All existing `logging.getLogger("workers.*")` / `"stages.*"` / etc.
calls automatically write to the log file with zero changes to existing code.
"""

from __future__ import annotations

import logging
import os

from cli.config.defaults import log_dir, log_path


def setup_logging(project_dir: str, log_level: str = "INFO") -> str:
    """
    Attach a FileHandler to the root logger so all module loggers write to
    .manga-translator/logs/run.log in addition to stderr.

    Args:
        project_dir: Project root directory.
        log_level:   Logging level string (DEBUG | INFO | WARNING | ERROR).

    Returns:
        Absolute path to the log file.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    _log_dir = log_dir(project_dir)
    _log_file = log_path(project_dir)
    os.makedirs(_log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler — append mode so successive runs accumulate
    file_handler = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is called twice
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(_log_file):
            return _log_file  # already configured

    root_logger.addHandler(file_handler)
    return _log_file
