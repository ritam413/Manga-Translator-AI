"""
cli/commands/init_cmd.py — `manga init`

Creates the project workspace structure without scanning images.
Idempotent: safe to run on an existing project (never overwrites config).
"""

from __future__ import annotations

import os

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

from cli.config.defaults import (
    config_dir,
    config_path,
    log_dir,
    plugin_dir,
    DEFAULT_CONFIG_TOML,
)

console = Console()

PROJECT_DIR_HELP = "Project root directory to initialise (default: current directory)."
REINIT_HELP = "Overwrite existing config.toml with defaults (destructive)."


def init(
    project_dir: str = typer.Argument(".", help=PROJECT_DIR_HELP),
    reinit: bool = typer.Option(False, "--reinit", help=REINIT_HELP),
) -> None:
    """
    Initialise a manga translator project workspace.

    Creates the following layout (does NOT scan for images):

    \b
      project_dir/
      ├── input/                      ← place source manga pages here
      ├── output/                     ← translated pages written here
      └── .manga-translator/
          ├── config.toml             ← edit to customise pipeline settings
          ├── logs/                   ← run.log written here
          └── plugins/                ← drop plugin .py files here

    Run [bold cyan]manga scan[/] after init to populate the manifest.
    """
    abs_dir = os.path.abspath(project_dir)

    created: list[str] = []
    skipped: list[str] = []

    # ── Create directories ─────────────────────────────────────────────
    for subdir in [
        os.path.join(abs_dir, "input"),
        os.path.join(abs_dir, "output"),
        config_dir(abs_dir),
        log_dir(abs_dir),
        plugin_dir(abs_dir),
    ]:
        if not os.path.isdir(subdir):
            os.makedirs(subdir, exist_ok=True)
            created.append(os.path.relpath(subdir, abs_dir))
        else:
            skipped.append(os.path.relpath(subdir, abs_dir))

    # ── Write config.toml ──────────────────────────────────────────────
    cfg_path = config_path(abs_dir)
    if not os.path.isfile(cfg_path) or reinit:
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        action = "overwritten" if reinit and os.path.isfile(cfg_path) else "created"
        created.append(os.path.relpath(cfg_path, abs_dir))
    else:
        skipped.append(os.path.relpath(cfg_path, abs_dir))

    # ── Summary table ──────────────────────────────────────────────────
    table = Table(
        title=f"[bold]Initialised:[/bold] {abs_dir}",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Path", style="cyan")
    table.add_column("Status", justify="right")

    for path in created:
        table.add_row(path, "[green]created[/green]")
    for path in skipped:
        table.add_row(path, "[dim]already exists[/dim]")

    console.print()
    console.print(table)
    console.print()
    console.print("[bold green]✓[/] Workspace ready.")
    console.print(
        f"  Next: [bold cyan]manga scan --project-dir {project_dir}[/]"
        " to populate the manifest."
    )
