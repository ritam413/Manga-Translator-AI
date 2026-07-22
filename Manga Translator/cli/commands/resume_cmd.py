"""
cli/commands/resume_cmd.py — `manga resume`

Runs all 5 pipeline stages in order, skipping pages already complete.
Equivalent to `python main.py execute all` with force=False.

Also accepts --from / --to to run a sub-range of stages.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from cli.commands._shared import (
    setup, load_engine, build_controller,
    print_stage_report, abort,
)
from cli.worker_factory import (
    build_text_detector,
    build_ocr_worker,
    build_translation_worker,
    build_inpainter,
    build_typesetter,
)
from cli.lock import acquire_lock, release_lock
from cli.plugin_registry import PluginRegistry
from cli.config.defaults import plugin_dir

console = Console()


def resume(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    from_stage: Optional[str] = typer.Option(
        None, "--from", metavar="STAGE",
        help="Start from this stage (inclusive). e.g. detect-text, ocr, translate, inpaint, typeset",
    ),
    to_stage: Optional[str] = typer.Option(
        None, "--to", metavar="STAGE",
        help="Stop after this stage (inclusive).",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-run stages even for pages already marked complete.",
    ),
    model_version: str = typer.Option(
        "v2", "--model-version",
        help="Model version to use (v1: single-word/column-based, v2: merged-bubble).",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit",
        help="Process only the first N pages (sorted numerically).",
    ),
    max_words_per_line: Optional[int] = typer.Option(
        None, "--max-words-per-line",
        help="Maximum number of words per typeset line.",
    ),
) -> None:
    """
    Resume the pipeline from where it left off.

    Runs all 5 stages in order ([bold]detect-text → ocr → translate → inpaint → typeset[/bold]),
    automatically skipping pages where each phase is already complete.

    Use [bold]--from[/bold] / [bold]--to[/bold] to run a subset of stages.
    Use [bold]--force[/bold] to re-process already-complete pages.
    """
    abs_dir, config = setup(project_dir)
    
    if max_words_per_line is not None:
        config.pipeline.max_words_per_line = max_words_per_line
        
    lock_file = acquire_lock(abs_dir)
    try:
        engine = load_engine(abs_dir, config)

        # ── Resolve stage range via PluginRegistry ─────────────────────
        registry = PluginRegistry(plugin_dir=plugin_dir(abs_dir))

        if from_stage or to_stage:
            _from = from_stage or registry.list_names()[0]
            _to = to_stage or registry.list_names()[-1]
            try:
                registry.validate_range(_from, _to)
                stages = registry.get_range(_from, _to)
            except ValueError as exc:
                abort(
                    f"{exc}\n\nValid stage names: {', '.join(registry.list_names())}"
                )
        else:
            stages = registry.list_all()

        if not stages:
            abort("No stages to run.")

        # ── Handle page filtering ──────────────────────────────────────────
        page_filter = None
        if limit is not None:
            def _sort_key_numeric(filename: str) -> tuple:
                import re
                m = re.match(r'^(\d+)', filename)
                return (int(m.group(1)), filename) if m else (float('inf'), filename)
            all_filenames = sorted(engine.manifest.pages.keys(), key=_sort_key_numeric)
            page_filter = all_filenames[:limit]

        # ── Build (worker, controller) for each stage ──────────────────
        stage_plans = _build_stage_plans(stages, abs_dir, config, engine, force, model_version, page_filter)

        # ── Execute with global pipeline progress bar ──────────────────
        console.print()
        all_reports = []

        from core.progress import MangaProgress

        with MangaProgress(console=console) as mp:
            pipeline_task = mp.add_task(
                "[bold]Pipeline[/bold]", total=len(stage_plans)
            )
            page_task = mp.add_task(
                "[dim]Pages[/dim]", total=0, visible=False
            )

            for stage_name, display_name, controller in stage_plans:
                # Count how many pages will actually run for this stage
                manifest = engine.manifest
                pages_to_run = [p for p in manifest.pages.values() if controller._should_process(p)]
                
                # Reset and show page progress for the current stage
                mp.update(
                    page_task,
                    description=f"[dim]{display_name} pages[/dim]",
                    total=len(pages_to_run),
                    completed=0,
                    visible=True,
                )
                
                # Assign callback to advance page progress task
                controller.on_page_done = lambda filename, success: mp.advance(page_task)

                mp.update(
                    pipeline_task,
                    description=f"[bold]Pipeline[/bold]  [cyan]{display_name}[/cyan]",
                )
                report = controller.run()
                all_reports.append((display_name, report))
                mp.advance(pipeline_task)

        # ── Print per-stage summaries ──────────────────────────────────
        console.print()
        for stage_display, report in all_reports:
            print_stage_report(report, stage_display)

        # ── Overall exit code ──────────────────────────────────────────
        total_failed = sum(len(r.failed) for _, r in all_reports)
        if total_failed > 0:
            console.print(
                f"\n[bold yellow]⚠[/]  Pipeline complete with {total_failed} failure(s). "
                "See details above and check logs."
            )
            raise typer.Exit(code=1)
        else:
            console.print("\n[bold green]✓[/] Pipeline complete. All pages processed.")

    finally:
        release_lock(lock_file)


# ── Stage plan builder ─────────────────────────────────────────────────────

# Maps stage name → (marks_phase, checks_phase)
_PHASE_MAP: dict[str, tuple[Optional[str], Optional[str]]] = {
    "detect-text": (None, "a"),
    "ocr":         (None, "a"),
    "translate":   ("a",  "a"),
    "inpaint":     (None, "b"),
    "typeset":     ("b",  "b"),
}


def _build_stage_plans(stages, abs_dir, config, engine, force, model_version: str, page_filter: Optional[list[str]] = None):
    """
    Build a list of (stage_name, display_name, StageController) tuples
    for each stage in the registry tuple format.
    """
    plans = []
    for stage_id, stage_name, stage_desc, worker_class, phase in stages:
        marks, checks = _PHASE_MAP.get(stage_name, (None, phase))

        worker = _build_worker_for_stage(stage_name, abs_dir, config, worker_class, model_version)
        display_name = f"Stage {stage_id}: {stage_name}"

        controller = build_controller(
            engine=engine,
            worker=worker,
            stage_display_name=display_name,
            marks_phase=marks,
            checks_phase=checks,
            force=force,
            page_filter=page_filter,
        )
        plans.append((stage_name, display_name, controller))
    return plans


def _build_worker_for_stage(stage_name: str, abs_dir: str, config, worker_class, model_version: str):
    """Dispatch to the correct worker factory function by stage name."""
    if stage_name == "detect-text":
        return build_text_detector(abs_dir, config, model_version=model_version)
    elif stage_name == "ocr":
        return build_ocr_worker(abs_dir, config, model_version=model_version)
    elif stage_name == "translate":
        return build_translation_worker(config, project_dir=abs_dir, model_version=model_version)
    elif stage_name == "inpaint":
        return build_inpainter(abs_dir, config)
    elif stage_name == "typeset":
        return build_typesetter(abs_dir, config, model_version=model_version)
    else:
        # Plugin stage — instantiate worker_class with no args
        # Plugin authors must ensure their workers have a no-arg __init__
        # or override this via their own CLI command.
        raise NotImplementedError(
            f"Plugin stage {stage_name!r} cannot be auto-instantiated by resume. "
            "Create a dedicated command in your plugin."
        )
