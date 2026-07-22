"""
cli/commands/config_cmd.py — `manga config`

Sub-app: config show | config set | config reset

Manages .manga-translator/config.toml via the CliConfig Pydantic model.
"""

from __future__ import annotations

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

from cli.commands._shared import setup
from cli.config.models import load_config, save_config
from cli.config.defaults import config_path, DEFAULT_CONFIG_TOML

console = Console()

config_app = typer.Typer(
    name="config",
    help="Manage CLI configuration (stored in .manga-translator/config.toml).",
    no_args_is_help=True,
)


@config_app.command("show")
def config_show(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
) -> None:
    """
    Display all current configuration values.
    """
    abs_dir, config = setup(project_dir)
    cfg_path = config_path(abs_dir)

    table = Table(
        title=f"[bold]Config[/bold]  [dim]{cfg_path}[/dim]",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")
    table.add_column("Type", style="dim")

    for section, field, value in config.all_keys():
        dot_key = f"{section}.{field}"
        table.add_row(dot_key, str(value), type(value).__name__)

    console.print()
    console.print(table)
    console.print(
        "\n  Edit: [dim]manga config set <section.key> <value>[/dim]"
    )


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ..., help="Config key in dot notation, e.g. [bold]pipeline.gpu[/bold]"
    ),
    value: str = typer.Argument(
        ..., help="New value (type-coerced automatically)."
    ),
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
) -> None:
    """
    Set a single config value and persist to config.toml.

    \b
    Examples:
      manga config set pipeline.gpu false
      manga config set pipeline.inpaint_radius 5
      manga config set general.log_level DEBUG
      manga config set pipeline.prefer_offline false
    """
    abs_dir, config = setup(project_dir)

    try:
        old_value = config.get_value(key)
        updated_config = config.set_value(key, value)
        save_config(updated_config, abs_dir)
        new_value = updated_config.get_value(key)

        console.print(f"\n[bold green]✓[/] Updated [cyan]{key}[/cyan]")
        console.print(f"   [dim]Old:[/dim] {old_value!r}  →  [dim]New:[/dim] [bold]{new_value!r}[/bold]")
    except KeyError as exc:
        console.print(f"[red]✗[/red] {exc}")
        console.print(
            "\n  Available keys (run [bold cyan]manga config show[/] to see all values):"
        )
        for s, f, _ in config.all_keys():
            console.print(f"   [dim]{s}.{f}[/dim]")
        raise typer.Exit(code=1)
    except (ValueError, TypeError) as exc:
        console.print(f"[red]✗[/red] Invalid value: {exc}")
        raise typer.Exit(code=1)


@config_app.command("reset")
def config_reset(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
    confirm: bool = typer.Option(
        False, "--confirm",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """
    Reset config.toml to factory defaults (destructive).

    The current config.toml will be overwritten. Make a backup first with
    [bold cyan]manga manifest backup[/bold cyan] if needed.
    """
    abs_dir, _ = setup(project_dir)

    if not confirm:
        confirmed = typer.confirm(
            "Reset config.toml to defaults? This cannot be undone."
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(code=0)

    cfg_path = config_path(abs_dir)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(DEFAULT_CONFIG_TOML)

    console.print(f"\n[bold green]✓[/] Config reset to defaults: [cyan]{cfg_path}[/]")
