"""
cli/commands/worker_cmd.py — `manga worker`

Sub-app: worker status

Checks which AI backends and heavy dependencies are importable/available.
This is an inspection command — it does not start or stop any processes.
"""

from __future__ import annotations

import os
import sys

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

from cli.commands._shared import setup

console = Console()

worker_app = typer.Typer(
    name="worker",
    help="Inspect available AI worker backends (read-only).",
    no_args_is_help=True,
)


@worker_app.command("status")
def worker_status(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
) -> None:
    """
    Show which AI backends and dependencies are available on this machine.

    Checks EasyOCR, MangaOCR, Argos Translate, Google Translate, and OpenCV.
    This is equivalent to [bold]manga doctor --backends-only[/bold].
    """
    abs_dir, config = setup(project_dir)

    table = Table(
        title="[bold]Worker Backend Status[/bold]",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Backend", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Notes", style="dim")

    checks = [
        ("EasyOCR",          "OCR / Detection",  _check_easyocr),
        ("MangaOCR",         "OCR (Japanese)",   _check_manga_ocr),
        ("Argos Translate",  "Translation (offline)", _check_argos),
        ("Google Translate", "Translation (online)",  _check_google_translate),
        ("OpenCV (cv2)",     "Inpainting",        _check_opencv),
        ("Pillow (PIL)",     "Typesetting",       _check_pillow),
        ("ONNX Runtime GPU", "GPU inference",     _check_onnxruntime),
    ]

    from core.progress import MangaProgress

    with MangaProgress(console=console) as mp:
        for name, kind, checker in mp.track(checks, description="Checking backends"):
            ok, note = checker()
            status_cell = "[green]✓ available[/green]" if ok else "[red]✗ unavailable[/red]"
            table.add_row(name, kind, status_cell, note)

    console.print()
    console.print(table)

    # Translation config
    pref = "Argos (offline)" if config.pipeline.prefer_offline else "Google (online)"
    console.print(f"\n  [dim]Configured primary translation backend:[/dim] {pref}")
    console.print(
        f"  [dim]GPU:[/dim] {'[green]enabled[/green]' if config.pipeline.gpu else '[yellow]disabled[/yellow]'}"
    )
    console.print(
        "\n  [dim]Run [bold cyan]manga doctor[/bold cyan] for full system diagnostics.[/dim]"
    )


# ── Backend checkers (each returns (ok: bool, note: str)) ─────────────────

def _check_easyocr() -> tuple[bool, str]:
    try:
        import easyocr  # noqa: F401
        return True, ""
    except ImportError as exc:
        return False, str(exc)


def _check_manga_ocr() -> tuple[bool, str]:
    try:
        import manga_ocr  # noqa: F401
        return True, ""
    except ImportError as exc:
        return False, str(exc)


def _check_argos() -> tuple[bool, str]:
    try:
        import argostranslate.translate  # noqa: F401
        return True, ""
    except ImportError as exc:
        return False, str(exc)


def _check_google_translate() -> tuple[bool, str]:
    try:
        from deep_translator import GoogleTranslator  # noqa: F401
        return True, "requires internet"
    except ImportError as exc:
        return False, str(exc)


def _check_opencv() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        return True, f"v{cv2.__version__}"
    except ImportError as exc:
        return False, str(exc)


def _check_pillow() -> tuple[bool, str]:
    try:
        from PIL import Image  # noqa: F401
        import PIL
        return True, f"v{PIL.__version__}"
    except ImportError as exc:
        return False, str(exc)


def _check_onnxruntime() -> tuple[bool, str]:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu = "CUDAExecutionProvider" in providers
        return True, "CUDA available" if gpu else "CPU only"
    except ImportError as exc:
        return False, str(exc)
