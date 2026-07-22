"""
cli/commands/detect_cmd.py — `manga detect`

Wraps TextDetectorWorker (Stage 6) with a Rich spinner.
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console

from cli.commands._shared import (
    setup, load_engine, build_controller,
    build_page_filter, print_stage_report, abort,
)
from cli.worker_factory import build_text_detector
from cli.lock import acquire_lock, release_lock

console = Console()


def detect(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-run even for pages where Phase A is already complete.",
    ),
    no_gpu: bool = typer.Option(
        False, "--no-gpu",
        help="Disable GPU (overrides config). Useful for debugging.",
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
    Detect text regions in each page using EasyOCR (Phase A, Stage 6).

    Populates text_regions[].box for every page in the manifest.
    Skip pages where Phase A is already complete unless [bold]--force[/] is used.
    """
    abs_dir, config = setup(project_dir)

    if no_gpu:
        # Temporarily override GPU flag without mutating the saved config
        from cli.config.models import CliConfig
        config = CliConfig.model_validate({
            **config.model_dump(),
            "pipeline": {**config.pipeline.model_dump(), "gpu": False},
        })

    lock_file = acquire_lock(abs_dir)
    try:
        engine = load_engine(abs_dir, config)
        page_filter = build_page_filter(engine.manifest, limit, pages)

        worker = build_text_detector(abs_dir, config)
        controller = build_controller(
            engine=engine,
            worker=worker,
            stage_display_name="Stage 6: detect-text",
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
                "[bold green]Running Text Detector (EasyOCR)…",
                total=len(pages_to_run),
            )
            controller.on_page_done = lambda filename, success: mp.advance(task_id)
            report = controller.run()

        print_stage_report(report, "Text Detection")
    finally:
        release_lock(lock_file)
