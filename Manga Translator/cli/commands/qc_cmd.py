"""
cli/commands/qc_cmd.py — `manga qc`

Runs QualityChecker directly on the manifest (no AI model, no StageController).
Removes regions that fail heuristic filters and reports what was removed.
"""

from __future__ import annotations

import copy

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

from cli.commands._shared import setup, load_engine, abort
from cli.lock import acquire_lock, release_lock

console = Console()


def qc(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be removed without modifying the manifest.",
    ),
) -> None:
    """
    Run quality-control heuristics on all text regions in the manifest.

    Applies six filters:
      - Bounding box size (too small/large)
      - Aspect ratio (extreme shapes)
      - OCR confidence threshold
      - Text density
      - Duplicate region detection
      - Garbage text detection (non-Japanese patterns)

    Regions that fail are [bold red]removed from the manifest[/bold red]
    unless [bold]--dry-run[/bold] is specified.
    """
    abs_dir, config = setup(project_dir)
    lock_file = acquire_lock(abs_dir)
    try:
        engine = load_engine(abs_dir, config)
        manifest = engine.manifest

        from quality.checker import QualityChecker
        checker = QualityChecker()

        total_before = sum(len(p.text_regions) for p in manifest.pages.values())
        removed_total = 0
        flagged_total = 0

        # ── Run QC per page ────────────────────────────────────────────
        from core.progress import MangaProgress

        with MangaProgress(console=console) as mp:
            pages = list(manifest.pages.values())
            for page in mp.track(pages, description="Running quality checks"):
                if dry_run:
                    # Work on a deep copy so we don't modify the actual page
                    page_copy = copy.deepcopy(page)
                    report = checker.check_page(page_copy)
                else:
                    report = checker.check_page(page)
                    engine.update_page(page.image_filename, page)

                removed_total += report.total_removed
                flagged_total += report.total_flagged

        if not dry_run:
            engine.save_manifest(backup=True)

        total_after = total_before - removed_total

        # ── Summary table ──────────────────────────────────────────────
        table = Table(
            title="[bold]Quality Control Results[/bold]"
            + (" [dim](dry run — no changes saved)[/dim]" if dry_run else ""),
            box=rich_box.ROUNDED,
            show_header=True,
            header_style="bold dim",
        )
        table.add_column("Metric", style="dim")
        table.add_column("Count", justify="right")

        table.add_row("Regions before QC", str(total_before))
        table.add_row("[red]Removed[/red]", f"[red]{removed_total}[/red]")
        table.add_row("[yellow]Flagged for review[/yellow]", f"[yellow]{flagged_total}[/yellow]")
        table.add_row("[green]Remaining[/green]", f"[green]{total_after}[/green]")

        console.print()
        console.print(table)

        if dry_run:
            console.print(
                "\n[dim]Re-run without [bold]--dry-run[/bold] to apply changes.[/dim]"
            )
        else:
            console.print(
                f"\n[bold green]✓[/] Manifest updated: "
                f"{removed_total} region(s) removed, {total_after} remaining."
            )
    finally:
        release_lock(lock_file)
