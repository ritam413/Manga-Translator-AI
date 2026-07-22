"""
cli/lock.py — PID-based lock file to prevent concurrent pipeline runs.

Usage:
    from cli.lock import acquire_lock, release_lock

    lock_file = acquire_lock(project_dir)  # raises RuntimeError if locked
    try:
        ...
    finally:
        release_lock(lock_file)

acquire_lock() also registers an atexit handler so the lock is released
automatically on normal interpreter exit.
"""

from __future__ import annotations

import atexit
import os
import logging

from cli.config.defaults import lock_path

logger = logging.getLogger("cli.lock")


def _is_process_alive(pid: int) -> bool:
    """
    Return True if a process with the given PID is currently running.

    Windows does not support os.kill(pid, 0) the same way POSIX does.
    We use os.kill with signal 0; on Windows this raises OSError with
    errno.EPERM if the process exists but we lack permission, or
    errno.ESRCH if it does not exist.
    """
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def acquire_lock(project_dir: str) -> str:
    """
    Write a PID lock file and return its path.

    If a lock file already exists and its PID belongs to a live process,
    raises RuntimeError to prevent concurrent runs.

    If the lock belongs to a dead process (crash recovery), the stale lock
    is silently removed and a new one is written.

    Args:
        project_dir: Project root directory.

    Returns:
        Absolute path to the lock file (for passing to release_lock()).

    Raises:
        RuntimeError: If another live manga process is already running.
    """
    _lock_path = lock_path(project_dir)
    os.makedirs(os.path.dirname(_lock_path), exist_ok=True)

    if os.path.isfile(_lock_path):
        try:
            with open(_lock_path, "r", encoding="utf-8") as f:
                existing_pid = int(f.read().strip())
            if _is_process_alive(existing_pid):
                raise RuntimeError(
                    f"Another manga process (PID {existing_pid}) is already "
                    f"running in this project. To force-clear the lock: "
                    f"delete {_lock_path}"
                )
            else:
                logger.warning(
                    "[lock] Stale lock found (PID %d is dead). Removing.", existing_pid
                )
                os.remove(_lock_path)
        except (ValueError, OSError) as exc:
            logger.warning("[lock] Could not read existing lock file: %s", exc)
            try:
                os.remove(_lock_path)
            except OSError:
                pass

    with open(_lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    atexit.register(_atexit_release, _lock_path)
    logger.debug("[lock] Acquired lock at %s (PID=%d)", _lock_path, os.getpid())
    return _lock_path


def release_lock(lock_file_path: str) -> None:
    """
    Release the lock by removing the lock file.

    Idempotent — safe to call even if the lock was already released.
    """
    try:
        if os.path.isfile(lock_file_path):
            os.remove(lock_file_path)
            logger.debug("[lock] Released lock: %s", lock_file_path)
    except OSError as exc:
        logger.warning("[lock] Could not release lock %s: %s", lock_file_path, exc)


def _atexit_release(lock_file_path: str) -> None:
    """atexit hook — silently release lock on interpreter exit."""
    try:
        if os.path.isfile(lock_file_path):
            os.remove(lock_file_path)
    except OSError:
        pass
