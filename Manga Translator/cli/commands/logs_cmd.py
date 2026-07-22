"""
cli/commands/logs_cmd.py — `manga logs`

Sub-app: logs show | logs tail | logs clear

Manages .manga-translator/logs/run.log
"""

from __future__ import annotations

import collections
import os
import time

import typer
from rich.console import Console

from cli.commands._shared import setup
from cli.config.defaults import log_path

console = Console()

logs_app = typer.Typer(
    name="logs",
    help="View and manage the pipeline run log.",
    no_args_is_help=True,
)


def _ensure_log(log_file: str) -> bool:
    """Return True if log file exists and is non-empty."""
    return os.path.isfile(log_file) and os.path.getsize(log_file) > 0


@logs_app.command("show")
def logs_show(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    lines: int = typer.Option(
        100, "--lines", "-n",
        help="Number of lines to show (most recent).",
    ),
    level: str = typer.Option(
        "all", "--level", "-l",
        help="Filter by log level: all | debug | info | warning | error",
    ),
) -> None:
    """
    Print the last N lines of the run log.
    """
    abs_dir, _ = setup(project_dir)
    log_file = log_path(abs_dir)

    if not _ensure_log(log_file):
        console.print(
            f"[dim]No log file yet at {log_file}.\n"
            "Run a pipeline command first to generate logs.[/dim]"
        )
        raise typer.Exit(code=0)

    level_filter = level.upper() if level != "all" else None

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        recent = collections.deque(f, maxlen=lines)

    filtered = [
        ln for ln in recent
        if level_filter is None or f"— {level_filter} —" in ln
    ]

    console.print(
        f"[dim]── {log_file} (last {len(filtered)} lines) ──[/dim]\n"
    )

    for ln in filtered:
        _print_log_line(ln.rstrip())


@logs_app.command("tail")
def logs_tail(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    level: str = typer.Option(
        "all", "--level", "-l",
        help="Filter by log level: all | debug | info | warning | error",
    ),
) -> None:
    """
    Follow the log file in real time (Ctrl-C to stop).
    """
    abs_dir, _ = setup(project_dir)
    log_file = log_path(abs_dir)

    if not os.path.isfile(log_file):
        # Create empty log file so we can tail it
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        open(log_file, "w").close()

    level_filter = level.upper() if level != "all" else None
    console.print(f"[dim]Tailing {log_file} (Ctrl-C to stop)…[/dim]\n")

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            # Seek to end
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    if level_filter is None or f"— {level_filter} —" in line:
                        _print_log_line(line.rstrip())
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


@logs_app.command("clear")
def logs_clear(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    confirm: bool = typer.Option(
        False, "--confirm",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """
    Clear (truncate) the run log file.
    """
    abs_dir, _ = setup(project_dir)
    log_file = log_path(abs_dir)

    if not os.path.isfile(log_file):
        console.print("[dim]No log file to clear.[/dim]")
        raise typer.Exit(code=0)

    size_kb = os.path.getsize(log_file) / 1024
    if not confirm:
        confirmed = typer.confirm(
            f"Clear log file ({size_kb:.1f} KB)? This cannot be undone."
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(code=0)

    open(log_file, "w").close()  # truncate
    console.print(f"[bold green]✓[/] Log cleared: [cyan]{log_file}[/]")


# ── Rich log line formatter ────────────────────────────────────────────────

def _print_log_line(line: str) -> None:
    """Colour-code a log line by its level."""
    if "— ERROR —" in line or "— CRITICAL —" in line:
        console.print(f"[red]{line}[/red]")
    elif "— WARNING —" in line:
        console.print(f"[yellow]{line}[/yellow]")
    elif "— DEBUG —" in line:
        console.print(f"[dim]{line}[/dim]")
    else:
        console.print(line)
