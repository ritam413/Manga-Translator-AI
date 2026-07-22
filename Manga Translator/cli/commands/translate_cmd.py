"""
cli/commands/translate_cmd.py — `manga translate`

Wraps TranslationWorker (Stage 8) with a Rich spinner.
Provides --prefer-online flag to override config.
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console

from cli.commands._shared import (
    setup, load_engine, build_controller,
    build_page_filter, print_stage_report,
)
from cli.worker_factory import build_translation_worker
from cli.lock import acquire_lock, release_lock

console = Console()


def translate(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-translate pages where Phase A is already complete.",
    ),
    prefer_online: bool = typer.Option(
        False, "--prefer-online",
        help=(
            "Use Google Translate (online) as the primary backend instead of "
            "Argos (offline). Requires internet access."
        ),
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
    Translate Japanese OCR text to English (Phase A, Stage 8).

    Backend priority (default): Argos offline → Google Translate fallback.
    Use [bold]--prefer-online[/] to reverse the order.

    After this stage, [bold]phase_a_completed = True[/] for every processed page.
    """
    abs_dir, config = setup(project_dir)
    lock_file = acquire_lock(abs_dir)
    try:
        engine = load_engine(abs_dir, config)
        page_filter = build_page_filter(engine.manifest, limit, pages)

        worker = build_translation_worker(config, prefer_online=prefer_online, project_dir=abs_dir)
        controller = build_controller(
            engine=engine,
            worker=worker,
            stage_display_name="Stage 8: translate",
            marks_phase="a",
            checks_phase="a",
            force=force,
            page_filter=page_filter,
        )

        backend_note = "Google Translate (online)" if prefer_online else "Argos (offline, with Google fallback)"
        from core.progress import MangaProgress

        manifest = engine.manifest
        pages_to_run = [p for p in manifest.pages.values() if controller._should_process(p)]

        with MangaProgress(console=console) as mp:
            task_id = mp.add_task(
                f"[bold green]Translating (backend: {backend_note})…",
                total=len(pages_to_run),
            )
            controller.on_page_done = lambda filename, success: mp.advance(task_id)
            report = controller.run()

        print_stage_report(report, "Translation")
    finally:
        release_lock(lock_file)
