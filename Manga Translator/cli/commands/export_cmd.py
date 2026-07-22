"""
cli/commands/export_cmd.py — `manga export`

Packages Phase B output images into a .cbz (Comic Book ZIP) or .zip archive.
Skips pages with phase_b_completed=False unless --include-incomplete is set.
"""

from __future__ import annotations

import os
import zipfile
from datetime import datetime
from typing import Optional

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn

from cli.commands._shared import setup, load_engine, abort

console = Console()

_FORMAT_CHOICES = ("cbz", "zip")


def export(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help=(
            "Output archive path. Defaults to "
            "{project_dir}/{project_name}.cbz (or .zip)."
        ),
    ),
    fmt: str = typer.Option(
        "cbz", "--format", "-f", metavar="cbz|zip",
        help="Archive format. 'cbz' is standard for manga viewers.",
    ),
    include_incomplete: bool = typer.Option(
        False, "--include-incomplete",
        help="Include pages where phase_b_completed=False (they may be blank).",
    ),
) -> None:
    """
    Export translated manga pages to a .cbz or .zip archive.

    Packages all [bold]output/[/bold] images where Phase B is complete
    into a single archive suitable for any CBZ-compatible manga reader.

    Pages with phase_b_completed=False are skipped with a warning unless
    [bold]--include-incomplete[/bold] is used.
    """
    if fmt not in _FORMAT_CHOICES:
        abort(f"Invalid format {fmt!r}. Choose: cbz or zip")

    abs_dir, config = setup(project_dir)
    engine = load_engine(abs_dir, config)
    manifest = engine.manifest

    output_subdir = os.path.join(abs_dir, config.pipeline.output_subdir)

    # ── Collect pages ──────────────────────────────────────────────────
    all_pages = sorted(manifest.pages.values(), key=lambda p: p.image_filename)
    complete_pages = [p for p in all_pages if p.phase_b_completed]
    skipped_pages = [p for p in all_pages if not p.phase_b_completed]

    if include_incomplete:
        pages_to_export = all_pages
    else:
        pages_to_export = complete_pages

    if not pages_to_export:
        abort(
            "No pages are ready to export (phase_b_completed=False for all pages). "
            "Run [bold cyan]manga resume[/] to complete the pipeline first."
        )

    # ── Resolve output path ────────────────────────────────────────────
    if output is None:
        safe_name = manifest.project_name.replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d")
        output = os.path.join(abs_dir, f"{safe_name}_{ts}.{fmt}")

    # Warn about skipped pages
    if skipped_pages and not include_incomplete:
        console.print(
            f"\n[yellow]⚠[/yellow]  {len(skipped_pages)} page(s) skipped "
            "(phase_b_completed=False). Use [bold]--include-incomplete[/bold] to force."
        )

    # ── Build archive ──────────────────────────────────────────────────
    tmp_output = output + ".tmp"
    n_added = 0
    n_missing = 0
    missing_files: list[str] = []

    from core.progress import MangaProgress

    with MangaProgress(console=console) as mp:
        task = mp.add_task(
            f"[cyan]Packaging {fmt.upper()}…", total=len(pages_to_export)
        )
        with zipfile.ZipFile(tmp_output, "w", zipfile.ZIP_DEFLATED) as zf:
            for page in pages_to_export:
                img_path = os.path.join(output_subdir, page.image_filename)
                if not os.path.isfile(img_path):
                    missing_files.append(page.image_filename)
                    n_missing += 1
                else:
                    zf.write(img_path, arcname=page.image_filename)
                    n_added += 1
                mp.advance(task)

    if n_added == 0:
        os.remove(tmp_output)
        abort("No output images found on disk. Has Phase B completed?")

    os.replace(tmp_output, output)

    # ── Summary ────────────────────────────────────────────────────────
    size_mb = os.path.getsize(output) / 1024 / 1024
    table = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value")
    table.add_row("Archive", f"[cyan]{output}[/cyan]")
    table.add_row("Format", fmt.upper())
    table.add_row("Pages included", f"[green]{n_added}[/green]")
    if n_missing:
        table.add_row("Missing on disk", f"[yellow]{n_missing}[/yellow]")
    table.add_row("Size", f"{size_mb:.1f} MB")

    console.print()
    console.print(table)
    console.print(f"\n[bold green]✓[/] Export complete.")
