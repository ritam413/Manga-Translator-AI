"""
cli/commands/status_cmd.py — `manga status`

Reads the manifest and displays a Rich progress overview:
  - Per-phase completion counts
  - Overall pipeline progress bars
  - Needs-review flag summary
"""

from __future__ import annotations

import typer
from rich import box as rich_box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
)
from rich.table import Table

from cli.commands._shared import setup, load_engine

console = Console()


def status(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
) -> None:
    """
    Show the current pipeline completion status for a project.

    Reads the project manifest and displays:
      - Total page count
      - Phase A progress (text detection → OCR → translation)
      - Phase B progress (inpainting → typesetting)
      - Pages flagged for human review
    """
    abs_dir, config = setup(project_dir)
    engine = load_engine(abs_dir, config)
    manifest = engine.manifest

    pages = list(manifest.pages.values())
    total = len(pages)

    if total == 0:
        console.print(
            Panel(
                "No pages in manifest. Run [bold cyan]manga scan[/] first.",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=0)

    phase_a_done = sum(1 for p in pages if p.phase_a_completed)
    phase_b_done = sum(1 for p in pages if p.phase_b_completed)
    needs_review = sum(
        1
        for p in pages
        for r in p.text_regions
        if r.needs_review
    )
    total_regions = sum(len(p.text_regions) for p in pages)

    # ── Header panel ───────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold]{manifest.project_name}[/bold]  ·  "
            f"[dim]{abs_dir}[/dim]\n"
            f"[dim]{config.translation.source_language} → "
            f"{config.translation.target_language}[/dim]",
            title="[bold cyan]Manga Translator — Status[/bold cyan]",
            border_style="cyan",
        )
    )

    # ── Stats table ────────────────────────────────────────────────────
    table = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Total pages", str(total))
    table.add_row("Total text regions", str(total_regions))
    table.add_row(
        "Needs human review",
        f"[yellow]{needs_review}[/yellow]" if needs_review else "0",
    )

    console.print(table)

    # ── Progress bars ──────────────────────────────────────────────────
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
        expand=False,
    ) as progress:
        progress.add_task(
            "[cyan]Phase A [dim](detect → ocr → translate)[/dim]",
            completed=phase_a_done,
            total=total,
        )
        progress.add_task(
            "[magenta]Phase B [dim](inpaint → typeset)   [/dim]",
            completed=phase_b_done,
            total=total,
        )

    # ── Overall completion ─────────────────────────────────────────────
    pct_a = phase_a_done / total * 100 if total else 0
    pct_b = phase_b_done / total * 100 if total else 0

    console.print(
        f"\n  Phase A: [bold cyan]{phase_a_done}/{total}[/] pages complete "
        f"([cyan]{pct_a:.1f}%[/])"
    )
    console.print(
        f"  Phase B: [bold magenta]{phase_b_done}/{total}[/] pages complete "
        f"([magenta]{pct_b:.1f}%[/])"
    )

    if phase_b_done == total:
        console.print(
            "\n[bold green]🎉 Pipeline complete![/]  "
            "Run [bold cyan]manga export[/] to package the output images."
        )
    elif phase_a_done == total:
        console.print(
            "\n[yellow]Phase A complete.[/]  "
            "Run [bold cyan]manga resume[/] to run Phase B (inpaint + typeset)."
        )
    else:
        console.print(
            "\n[dim]Run [bold cyan]manga resume[/] to continue the pipeline.[/dim]"
        )
    console.print()
