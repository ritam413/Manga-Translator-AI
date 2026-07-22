"""
cli/commands/_shared.py — Internal helpers shared across all command modules.

Never import from this module in tests — it is an implementation detail.
All public API is exposed through the individual command modules.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box as rich_box

from cli.config.models import CliConfig, load_config
from cli.config.defaults import config_path, plugin_dir
from cli.logging_setup import setup_logging
from cli.plugin_registry import PluginRegistry

# ── Shared Rich console ───────────────────────────────────────────────────

console = Console(stderr=False)
err_console = Console(stderr=True)


# ── Project resolution ────────────────────────────────────────────────────

def resolve_project_dir(project_dir: str) -> str:
    """Resolve and validate the project directory."""
    abs_dir = os.path.abspath(project_dir)
    if not os.path.isdir(abs_dir):
        err_console.print(
            f"[bold red]Error:[/] Project directory does not exist: {abs_dir}"
        )
        raise typer.Exit(code=1)
    return abs_dir


def load_engine(project_dir: str, config: CliConfig):
    """
    Load ManifestEngine with the project manifest.

    Raises typer.Exit(1) if the manifest is not found (directs user to
    run `manga scan` first).
    """
    from core.manifest import ManifestEngine

    engine = ManifestEngine(
        project_dir,
        input_subdir=config.pipeline.input_subdir,
    )
    if not os.path.exists(engine.manifest_path):
        err_console.print(
            Panel(
                f"[bold red]No manifest found.[/]\n\n"
                f"Run [bold cyan]manga scan --project-dir {project_dir}[/] first "
                f"to scan your input images and create the manifest.",
                title="[red]Manifest Missing[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
    engine.load_or_create(
        project_name=config.general.project_name,
        source_lang=config.translation.source_language,
        target_lang=config.translation.target_language,
    )
    return engine


def setup(project_dir: str) -> Tuple[str, CliConfig]:
    """
    Resolve project_dir, load config, and configure logging.

    Call at the start of every command that does pipeline work.
    Returns (abs_project_dir, config).
    """
    abs_dir = resolve_project_dir(project_dir)
    config = load_config(abs_dir)
    setup_logging(abs_dir, config.general.log_level)
    return abs_dir, config


def make_plugin_registry(project_dir: str) -> PluginRegistry:
    """Return a PluginRegistry with any local plugins loaded."""
    return PluginRegistry(plugin_dir=plugin_dir(project_dir))


# ── Build StageController ─────────────────────────────────────────────────

def build_controller(
    engine,
    worker,
    stage_display_name: str,
    marks_phase: Optional[str],
    checks_phase: Optional[str],
    force: bool,
    page_filter: Optional[List[str]],
    run_quality_check: bool = False,
):
    """
    Instantiate a StageController with standard CLI defaults.

    This is the single place that wires engine + worker → controller
    in the CLI layer.
    """
    from stages.controller import StageController

    return StageController(
        engine=engine,
        worker=worker,
        stage_name=stage_display_name,
        marks_phase=marks_phase,
        checks_phase=checks_phase,
        run_quality_check=run_quality_check,
        page_filter=page_filter,
        force=force,
        flush_vram=True,
    )


# ── Page filter helpers ───────────────────────────────────────────────────

def build_page_filter(
    manifest,
    limit: Optional[int],
    pages: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Build the page_filter list from --limit / --pages flags.

    Returns None (= all pages) if neither flag is set.
    """
    if pages:
        return list(pages)
    if limit is not None:
        all_files = sorted(manifest.pages.keys(), key=_sort_key_numeric)
        return all_files[:limit]
    return None


def _sort_key_numeric(filename: str) -> tuple:
    """Natural sort: numeric prefix first."""
    m = re.match(r"^(\d+)", filename)
    return (int(m.group(1)), filename) if m else (float("inf"), filename)


# ── Rich output helpers ───────────────────────────────────────────────────

def print_stage_report(report, stage_name: str) -> None:
    """Print a StageReport as a Rich summary table."""
    n_ok = len(report.succeeded)
    n_fail = len(report.failed)
    n_skip = len(report.skipped)
    total = report.total_pages

    status_color = "green" if n_fail == 0 else "red"
    status_text = "PASSED" if n_fail == 0 else "FAILED"

    table = Table(
        title=f"[bold]{stage_name}[/bold]",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")

    table.add_row("Total pages", str(total))
    table.add_row("[green]Succeeded[/green]", f"[green]{n_ok}[/green]")
    table.add_row("[yellow]Skipped[/yellow]", f"[yellow]{n_skip}[/yellow]")
    table.add_row("[red]Failed[/red]", f"[red]{n_fail}[/red]")

    console.print()
    console.print(table)

    if n_fail > 0:
        console.print(
            f"\n[bold {status_color}]{status_text}[/]  "
            f"{n_fail} page(s) failed. Check logs for details."
        )
        for fn in report.failed[:10]:
            console.print(f"  [red]✗[/red] {fn}")
        if n_fail > 10:
            console.print(f"  [dim]... and {n_fail - 10} more[/dim]")
    else:
        console.print(f"\n[bold {status_color}]{status_text}[/]  All pages processed.\n")


def abort(message: str, exit_code: int = 1) -> None:
    """Print an error panel and exit."""
    err_console.print(
        Panel(f"[bold red]{message}[/]", title="[red]Error[/red]", border_style="red")
    )
    raise typer.Exit(code=exit_code)


def success(message: str) -> None:
    """Print a success panel."""
    console.print(
        Panel(f"[bold green]{message}[/]", border_style="green")
    )
