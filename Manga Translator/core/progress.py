"""
core/progress.py — Unified progress bar utility wrapping Rich.
"""

from __future__ import annotations
import sys
from typing import Any, Iterable, Optional, TypeVar
from rich.progress import (
    Progress as RichProgress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TaskID,
)
from rich.console import Console

T = TypeVar("T")


class MangaProgress:
    """
    Unified reusable progress bar context manager wrapping rich.progress.Progress.
    Provides a standardized look and feel for CLI commands and ML pipeline scripts.
    """
    def __init__(
        self,
        console: Optional[Console] = None,
        transient: bool = False,
        disable: bool = False,
    ) -> None:
        self.console = console or Console(stderr=False)
        self.transient = transient
        self.disable = disable
        
        self.progress = RichProgress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=self.transient,
            disable=self.disable or not self.console.is_terminal,
        )

    def __enter__(self) -> MangaProgress:
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def add_task(
        self,
        description: str,
        total: int = 0,
        visible: bool = True,
    ) -> TaskID:
        return self.progress.add_task(description, total=total, visible=visible)

    def update(self, task_id: TaskID, **kwargs: Any) -> None:
        self.progress.update(task_id, **kwargs)

    def advance(self, task_id: TaskID, advance: float = 1) -> None:
        self.progress.advance(task_id, advance)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Safely print text to the console without corrupting progress rendering."""
        self.progress.console.print(*args, **kwargs)

    def track(
        self,
        sequence: Iterable[T],
        description: str = "Processing",
        total: Optional[int] = None,
    ) -> Iterable[T]:
        """Wrap an iterable and automatically advance a progress task."""
        if total is None:
            try:
                total = len(sequence)  # type: ignore
            except (TypeError, AttributeError):
                total = 0

        task_id = self.add_task(description, total=total)
        for item in sequence:
            yield item
            self.advance(task_id)
