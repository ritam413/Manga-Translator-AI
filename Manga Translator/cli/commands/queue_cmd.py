"""
cli/commands/queue_cmd.py — `manga queue`

Sub-app: queue list

Reads the manifest and shows a table of pages with pending work.
This is a read-only inspector — it does not enqueue or run anything.
"""

from __future__ import annotations

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

from cli.commands._shared import setup, load_engine

console = Console()

queue_app = typer.Typer(
    name="queue",
    help="Inspect pending pipeline work (read-only manifest view).",
    no_args_is_help=True,
)


@queue_app.command("list")
def queue_list(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    phase: str = typer.Option(
        "all", "--phase",
        help="Filter: 'a' (Phase A pending), 'b' (Phase B pending), 'all'.",
    ),
    limit: int = typer.Option(
        50, "--limit", "-n", metavar="N",
        help="Maximum rows to display (default 50).",
    ),
) -> None:
    """
    List pages with pending pipeline work.

    Reads the manifest and displays all pages that have not yet completed
    one or both pipeline phases. Use [bold]--phase a|b|all[/bold] to filter.
    """
    abs_dir, config = setup(project_dir)
    engine = load_engine(abs_dir, config)
    manifest = engine.manifest

    if phase not in ("a", "b", "all"):
        console.print(
            "[red]✗[/red] --phase must be one of: a, b, all"
        )
        raise typer.Exit(code=1)

    pending = [
        p for p in manifest.pages.values()
        if (phase in ("a", "all") and not p.phase_a_completed)
        or (phase in ("b", "all") and not p.phase_b_completed)
    ]

    total = len(manifest.pages)
    console.print(
        f"\n[bold]Pending work[/bold]  |  Project: [cyan]{abs_dir}[/cyan]  "
        f"|  Filter: phase={phase}\n"
    )

    if not pending:
        console.print("[bold green]✓[/] No pending work — pipeline is complete!")
        raise typer.Exit(code=0)

    table = Table(
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        caption=f"Showing {min(len(pending), limit)} of {len(pending)} pending pages (total: {total})",
    )
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Phase A", justify="center")
    table.add_column("Phase B", justify="center")
    table.add_column("Regions", justify="right")

    for p in sorted(pending, key=lambda x: x.image_filename)[:limit]:
        table.add_row(
            p.image_filename,
            "[green]✓[/green]" if p.phase_a_completed else "[red]pending[/red]",
            "[magenta]✓[/magenta]" if p.phase_b_completed else "[red]pending[/red]",
            str(len(p.text_regions)),
        )

    console.print(table)
    console.print(
        f"\n  Run [bold cyan]manga resume --project-dir {project_dir}[/] "
        "to process pending pages."
    )
