"""
cli/commands/manifest_cmd.py — `manga manifest` sub-commands

Sub-app: manifest show | validate | backup
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Optional

import typer
from rich import box as rich_box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.commands._shared import setup, load_engine

console = Console()

manifest_app = typer.Typer(
    name="manifest",
    help="Inspect and manage the project manifest.",
    no_args_is_help=True,
)


@manifest_app.command("show")
def manifest_show(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    page: Optional[str] = typer.Option(
        None, "--page",
        help="Show detail for a single page (by filename).",
    ),
) -> None:
    """
    Display a summary of the project manifest as a Rich table.

    Use [bold]--page FILENAME[/bold] to see region-level detail for one page.
    """
    abs_dir, config = setup(project_dir)
    engine = load_engine(abs_dir, config)
    manifest = engine.manifest

    if page:
        _show_page_detail(manifest, page, abs_dir)
        return

    pages = list(manifest.pages.values())
    total = len(pages)

    # ── Header ─────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold]{manifest.project_name}[/bold]  ·  {abs_dir}\n"
            f"[dim]{manifest.source_language} → {manifest.target_language}  |  "
            f"{total} pages[/dim]",
            title="[bold cyan]Manifest Summary[/bold cyan]",
            border_style="cyan",
        )
    )

    # ── Pages table ────────────────────────────────────────────────────
    table = Table(
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Regions", justify="right")
    table.add_column("Phase A", justify="center")
    table.add_column("Phase B", justify="center")

    for p in sorted(pages, key=lambda x: x.image_filename):
        regions = len(p.text_regions)
        table.add_row(
            p.image_filename,
            f"{p.width}×{p.height}",
            str(regions),
            "[green]✓[/green]" if p.phase_a_completed else "[dim]–[/dim]",
            "[magenta]✓[/magenta]" if p.phase_b_completed else "[dim]–[/dim]",
        )

    console.print(table)


@manifest_app.command("validate")
def manifest_validate(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
) -> None:
    """
    Validate the manifest by checking all listed pages exist on disk.

    Reports missing files and any pages with zero text regions.
    """
    abs_dir, config = setup(project_dir)
    engine = load_engine(abs_dir, config)
    manifest = engine.manifest

    input_dir = os.path.join(abs_dir, config.pipeline.input_subdir)
    missing: list[str] = []
    no_regions: list[str] = []

    from core.progress import MangaProgress

    with MangaProgress(console=console) as mp:
        pages = list(manifest.pages.items())
        for fn, page in mp.track(pages, description="Validating manifest against disk"):
            img_path = os.path.join(input_dir, fn)
            if not os.path.isfile(img_path):
                missing.append(fn)
            if len(page.text_regions) == 0 and page.phase_a_completed:
                no_regions.append(fn)

    ok = len(manifest.pages) - len(missing)
    console.print()

    if missing:
        console.print(f"[red]✗[/red] {len(missing)} page(s) missing from disk:")
        for fn in missing:
            console.print(f"   [red]{fn}[/red]")
    else:
        console.print("[green]✓[/green] All manifest pages are present on disk.")

    if no_regions:
        console.print(
            f"\n[yellow]⚠[/yellow]  {len(no_regions)} page(s) have phase_a_completed=True "
            "but zero text regions:"
        )
        for fn in no_regions[:5]:
            console.print(f"   [yellow]{fn}[/yellow]")
        if len(no_regions) > 5:
            console.print(f"   [dim]… and {len(no_regions) - 5} more[/dim]")

    console.print(
        f"\n  Manifest: {ok}/{len(manifest.pages)} pages valid"
        f" | Path: [dim]{engine.manifest_path}[/dim]"
    )


@manifest_app.command("backup")
def manifest_backup(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    tag: str = typer.Option(
        "", "--tag", "-t",
        help="Optional label appended to the backup filename.",
    ),
) -> None:
    """
    Create a timestamped backup of the current project manifest.

    Output: [dim]project_manifest.YYYYMMDD_HHMMSS[_tag].json.bak[/dim]
    """
    abs_dir, config = setup(project_dir)
    engine = load_engine(abs_dir, config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    backup_name = f"project_manifest.{ts}{suffix}.json.bak"
    backup_path = os.path.join(abs_dir, backup_name)

    shutil.copy2(engine.manifest_path, backup_path)
    console.print(f"\n[bold green]✓[/] Backup created: [cyan]{backup_name}[/]")


# ── Internal helpers ───────────────────────────────────────────────────────

def _show_page_detail(manifest, filename: str, abs_dir: str) -> None:
    """Print region-level detail for a single page."""
    page = manifest.pages.get(filename)
    if page is None:
        console.print(
            f"[red]✗[/red] Page not found in manifest: [bold]{filename}[/bold]"
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]{filename}[/bold]  {page.width}×{page.height}\n"
            f"Phase A: {'[green]complete[/green]' if page.phase_a_completed else '[dim]pending[/dim]'}  |  "
            f"Phase B: {'[magenta]complete[/magenta]' if page.phase_b_completed else '[dim]pending[/dim]'}",
            title="[bold cyan]Page Detail[/bold cyan]",
            border_style="cyan",
        )
    )

    if not page.text_regions:
        console.print("[dim]No text regions.[/dim]")
        return

    table = Table(box=rich_box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Box (x1,y1,x2,y2)", style="cyan")
    table.add_column("Raw text")
    table.add_column("Translation")
    table.add_column("Review", justify="center")

    for r in page.text_regions:
        b = r.box
        table.add_row(
            r.id,
            f"{b.x_min},{b.y_min},{b.x_max},{b.y_max}",
            r.raw_text[:40] + ("…" if len(r.raw_text) > 40 else ""),
            r.translated_text[:40] + ("…" if len(r.translated_text) > 40 else ""),
            "[yellow]⚠[/yellow]" if r.needs_review else "",
        )

    console.print(table)
