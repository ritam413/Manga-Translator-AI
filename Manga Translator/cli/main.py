"""
cli/main.py — Root Typer application for the manga-translator CLI.

Entry point: `manga` console script (configured in pyproject.toml).

Install:
    pip install -e .          # installs `manga` console script

Usage:
    manga --help
    manga init .
    manga scan --project-dir /path/to/project
    manga status
    manga detect --force
    manga ocr
    manga translate --prefer-online
    manga qc --dry-run
    manga manifest show
    manga manifest validate
    manga queue list
    manga worker status
    manga export --format cbz
    manga config show
    manga config set pipeline.gpu false
    manga logs tail
    manga doctor
    manga resume --from translate --to typeset

Architecture notes:
  - Every sub-app is registered via app.add_typer() (no global singletons).
  - All pipeline internals (core/, workers/, stages/) are unchanged.
  - The PluginRegistry subclasses StageRegistry to add plugin support.
  - Progress bars are stage-level (Option C) — no StageController subclassing.
"""

from __future__ import annotations

import typer

# ── Sub-apps (each has their own Typer() instance) ────────────────────────
from cli.commands.manifest_cmd import manifest_app
from cli.commands.queue_cmd import queue_app
from cli.commands.worker_cmd import worker_app
from cli.commands.config_cmd import config_app
from cli.commands.logs_cmd import logs_app

# ── Top-level command functions ───────────────────────────────────────────
from cli.commands.init_cmd import init
from cli.commands.scan_cmd import scan, plan
from cli.commands.status_cmd import status
from cli.commands.detect_cmd import detect
from cli.commands.ocr_cmd import ocr
from cli.commands.translate_cmd import translate
from cli.commands.qc_cmd import qc
from cli.commands.export_cmd import export
from cli.commands.doctor_cmd import doctor
from cli.commands.resume_cmd import resume

# ── Root app ──────────────────────────────────────────────────────────────

app = typer.Typer(
    name="manga",
    help=(
        "[bold cyan]manga-translator[/bold cyan] — Production-grade manga translation pipeline.\n\n"
        "Behaves like [bold]git[/bold] / [bold]docker[/bold]: each sub-command is a "
        "self-contained operation. Run [bold]manga COMMAND --help[/bold] for details.\n\n"
        "[dim]Powered by: Typer · Rich · Pydantic · EasyOCR · MangaOCR · Argos · OpenCV[/dim]"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
    pretty_exceptions_show_locals=False,
)

# ── Register sub-apps ─────────────────────────────────────────────────────
app.add_typer(manifest_app, name="manifest", help="Inspect and manage the project manifest.")
app.add_typer(queue_app,    name="queue",    help="Show pending pipeline work (read-only).")
app.add_typer(worker_app,   name="worker",   help="Show available AI worker backends.")
app.add_typer(config_app,   name="config",   help="Manage CLI configuration (config.toml).")
app.add_typer(logs_app,     name="logs",     help="View and manage the pipeline run log.")

# ── Register top-level commands ───────────────────────────────────────────
app.command("init",      help="Initialise project workspace (dirs + config)."   )(init)
app.command("scan",      help="Scan input dir and create/update manifest."       )(scan)
app.command("plan",      help="Alias for [bold cyan]scan[/bold cyan]."           )(plan)
app.command("status",    help="Show pipeline completion status."                 )(status)
app.command("detect",    help="Run EasyOCR text detection (Stage 6)."           )(detect)
app.command("ocr",       help="Run MangaOCR on detected regions (Stage 7)."     )(ocr)
app.command("translate", help="Translate Japanese → English (Stage 8)."         )(translate)
app.command("qc",        help="Run quality-control heuristics on manifest."      )(qc)
app.command("export",    help="Export output images to CBZ/ZIP archive."        )(export)
app.command("doctor",    help="Run full system health diagnostics."              )(doctor)
app.command("resume",    help="Continue pipeline from current manifest state."   )(resume)

# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
