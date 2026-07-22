"""
cli/commands/ocr_cmd.py — `manga ocr`

Wraps OcrWorker (Stage 7 — MangaOCR) with a Rich spinner.
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console

from cli.commands._shared import (
    setup, load_engine, build_controller,
    build_page_filter, print_stage_report,
)
from cli.worker_factory import build_ocr_worker
from cli.lock import acquire_lock, release_lock

console = Console()


def ocr(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-run even for pages where Phase A is already complete.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", metavar="N",
        help="Process only the first N pages (sorted numerically).",
    ),
    pages: Optional[List[str]] = typer.Option(
        None, "--page", metavar="FILENAME",
        help="Process specific page(s) by filename. Repeatable.",
    ),
) -> None:
    """
    Run MangaOCR on each detected text region (Phase A, Stage 7).

    Reads text_regions[].box (populated by [bold cyan]manga detect[/]),
    crops each region from the source image, and writes raw_text.
    """
    abs_dir, config = setup(project_dir)
    lock_file = acquire_lock(abs_dir)
    try:
        engine = load_engine(abs_dir, config)
        page_filter = build_page_filter(engine.manifest, limit, pages)

        worker = build_ocr_worker(abs_dir, config)
        controller = build_controller(
            engine=engine,
            worker=worker,
            stage_display_name="Stage 7: ocr",
            marks_phase=None,
            checks_phase="a",
            force=force,
            page_filter=page_filter,
        )

        from core.progress import MangaProgress

        manifest = engine.manifest
        pages_to_run = [p for p in manifest.pages.values() if controller._should_process(p)]

        with MangaProgress(console=console) as mp:
            task_id = mp.add_task(
                "[bold green]Running MangaOCR…",
                total=len(pages_to_run),
            )
            controller.on_page_done = lambda filename, success: mp.advance(task_id)
            report = controller.run()

        print_stage_report(report, "OCR (MangaOCR)")
    finally:
        release_lock(lock_file)
