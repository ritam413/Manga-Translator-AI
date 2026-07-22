"""
cli/commands/scan_cmd.py — `manga scan` and `manga plan` (alias)

Scans input directory → creates/updates project_manifest.json.
Replaces the old `python main.py plan` verb.
"""

from __future__ import annotations

import os
from typing import Optional

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

from cli.commands._shared import setup, abort

console = Console()


def _scan_impl(
    project_dir: str,
    input_subdir: str,
    reset: bool = False,
) -> None:
    """Shared implementation for `scan` and `plan`."""
    abs_dir, config = setup(project_dir)

    # Override input_subdir if explicitly passed
    effective_input_subdir = input_subdir or config.pipeline.input_subdir

    from core.manifest import ManifestEngine

    engine = ManifestEngine(abs_dir, input_subdir=effective_input_subdir)

    if reset:
        if os.path.exists(engine.manifest_path):
            os.remove(engine.manifest_path)
            console.print(f"[dim]Deleted existing manifest: {engine.manifest_path}[/dim]")
        backup_path = engine.manifest_path + ".bak"
        if os.path.exists(backup_path):
            os.remove(backup_path)
            console.print(f"[dim]Deleted existing manifest backup: {backup_path}[/dim]")

    manifest = engine.load_or_create(
        project_name=config.general.project_name,
        source_lang=config.translation.source_language,
        target_lang=config.translation.target_language,
    )

    from core.progress import MangaProgress

    with MangaProgress(console=console) as mp:
        task_id = mp.add_task("[bold green]Scanning input directory…", total=100)
        def on_progress(current: int, total: int):
            mp.update(task_id, total=total, completed=current)
        added, missing = engine.sync_with_disk(on_progress=on_progress)

    engine.save_manifest(backup=True)

    # ── Results table ──────────────────────────────────────────────────
    table = Table(
        title="[bold]Scan Results[/bold]",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")

    total = len(manifest.pages)
    phase_a_done = sum(1 for p in manifest.pages.values() if p.phase_a_completed)
    phase_b_done = sum(1 for p in manifest.pages.values() if p.phase_b_completed)

    table.add_row("Total pages in manifest", str(total))
    table.add_row("[green]+[/green] Newly added", f"[green]{len(added)}[/green]")
    table.add_row(
        "[yellow]⚠[/yellow]  Missing from disk",
        f"[yellow]{len(missing)}[/yellow]" if missing else "0",
    )
    table.add_row("Phase A complete", f"{phase_a_done}/{total}")
    table.add_row("Phase B complete", f"{phase_b_done}/{total}")

    console.print()
    console.print(table)
    console.print(
        f"\n[bold green]✓[/] Manifest saved: [cyan]{engine.manifest_path}[/]"
    )

    if missing:
        console.print(
            f"\n[yellow]⚠[/]  {len(missing)} page(s) in manifest but missing from disk:"
        )
        for fn in missing[:5]:
            console.print(f"  [yellow]✗[/yellow] {fn}")
        if len(missing) > 5:
            console.print(f"  [dim]… and {len(missing) - 5} more[/dim]")

    console.print(
        f"\n  Next: [bold cyan]manga status --project-dir {project_dir}[/]"
    )


def scan(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    input_subdir: str = typer.Option(
        "input", "--input-subdir",
        help="Name of the subdirectory containing source images.",
    ),
    reset: bool = typer.Option(
        False, "--reset",
        help="Reset and completely overwrite/recreate the project manifest file.",
    ),
) -> None:
    """
    Scan the input directory and create/update the project manifest.

    Equivalent to the old [bold]python main.py plan[/bold] command.
    Run this first when starting a new project.
    """
    _scan_impl(project_dir, input_subdir, reset=reset)


def plan(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    input_subdir: str = typer.Option(
        "input", "--input-subdir",
        help="Name of the subdirectory containing source images.",
    ),
    reset: bool = typer.Option(
        False, "--reset",
        help="Reset and completely overwrite/recreate the project manifest file.",
    ),
) -> None:
    """
    Alias for [bold cyan]manga scan[/bold]. Scan and update the project manifest.
    """
    _scan_impl(project_dir, input_subdir, reset=reset)
