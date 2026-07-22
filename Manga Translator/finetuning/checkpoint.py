"""Atomic checkpoint load/save for dataset extraction."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from finetuning.models import ExtractionState

logger = logging.getLogger(__name__)


def load_checkpoint(path: Path) -> ExtractionState:
    if not path.exists():
        return ExtractionState()

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt checkpoint file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Corrupt checkpoint file {path}: expected JSON object")

    return ExtractionState.from_dict(data)


def save_checkpoint(state: ExtractionState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(tmp_fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)
        logger.debug("Checkpoint saved to %s", path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def reset_checkpoint(path: Path) -> None:
    if path.exists():
        path.unlink()
