"""
cli/commands/doctor_cmd.py — `manga doctor`

Full system diagnostics. Checks GPU, dependencies, assets, and permissions.
Outputs a colour-coded Rich table — green=OK, yellow=warning, red=fail.
"""

from __future__ import annotations

import os
import sys

import typer
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

from cli.commands._shared import setup
from cli.config.defaults import config_path

console = Console()

_OK = "[bold green]✓  OK[/bold green]"
_WARN = "[bold yellow]⚠  WARN[/bold yellow]"
_FAIL = "[bold red]✗  FAIL[/bold red]"


def _ok(note: str = "") -> tuple[str, str]:
    return _OK, note


def _warn(note: str) -> tuple[str, str]:
    return _WARN, note


def _fail(note: str) -> tuple[str, str]:
    return _FAIL, note


def doctor(
    project_dir: str = typer.Option(
        ".", "--project-dir", "-p", envvar="MANGA_PROJECT_DIR",
        help="Project root directory.",
    ),
) -> None:
    """
    Run a full system health check and print a diagnostic report.

    Checks:
      - Python version (≥ 3.10)
      - GPU / CUDA availability
      - Heavy AI dependencies (EasyOCR, MangaOCR, Argos, cv2)
      - Required assets (Bangers font)
      - Directory permissions
      - CLI config file
      - Manifest file integrity
    """
    abs_dir, config = setup(project_dir)

    table = Table(
        title="[bold cyan]Manga Translator — Doctor[/bold cyan]",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Details", style="dim")

    checks: list[tuple[str, tuple[str, str]]] = []

    from core.progress import MangaProgress

    with MangaProgress(console=console) as mp:
        task_id = mp.add_task("[bold green]Running diagnostics…", total=15)
        check_fns = [
            ("Python version",           _chk_python),
            ("GPU / CUDA",               _chk_gpu),
            ("ONNX Runtime GPU",         _chk_onnx),
            ("EasyOCR",                  lambda: _chk_import("easyocr")),
            ("MangaOCR",                 lambda: _chk_import("manga_ocr")),
            ("Argos Translate",          lambda: _chk_import("argostranslate.translate")),
            ("Google Translate client",  lambda: _chk_import("deep_translator")),
            ("OpenCV (cv2)",             lambda: _chk_import("cv2")),
            ("Pillow (PIL)",             lambda: _chk_import("PIL")),
            ("Bangers font",             lambda: _chk_font(abs_dir, config.pipeline.font_path)),
            ("input/ directory",         lambda: _chk_dir_readable(os.path.join(abs_dir, config.pipeline.input_subdir))),
            ("output/ directory",        lambda: _chk_dir_writable(os.path.join(abs_dir, config.pipeline.output_subdir))),
            (".manga-translator/ dir",   lambda: _chk_dir_writable(os.path.join(abs_dir, ".manga-translator"))),
            ("config.toml",              lambda: _chk_config(abs_dir)),
            ("project_manifest.json",    lambda: _chk_manifest(abs_dir)),
        ]
        checks = []
        for name, fn in check_fns:
            checks.append((name, fn()))
            mp.advance(task_id)

    n_fail = sum(1 for _, (s, _) in checks if "FAIL" in s)
    n_warn = sum(1 for _, (s, _) in checks if "WARN" in s)

    for name, (status, note) in checks:
        table.add_row(name, status, note)

    console.print()
    console.print(table)

    if n_fail == 0 and n_warn == 0:
        console.print("\n[bold green]✓ All checks passed. System is ready.[/bold green]")
    elif n_fail == 0:
        console.print(f"\n[bold yellow]⚠ {n_warn} warning(s). Pipeline should still work.[/bold yellow]")
    else:
        console.print(
            f"\n[bold red]✗ {n_fail} check(s) failed. "
            "Fix the issues above before running the pipeline.[/bold red]"
        )
        raise typer.Exit(code=1)


# ── Check functions ────────────────────────────────────────────────────────

def _chk_python() -> tuple[str, str]:
    v = sys.version_info
    note = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        return _ok(note)
    return _fail(f"Python ≥ 3.10 required, got {note}")


def _chk_gpu() -> tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return _ok(f"{name} ({vram:.1f} GB VRAM)")
        return _warn("CUDA not available — AI stages will run on CPU (very slow)")
    except ImportError:
        return _warn("torch not installed — cannot check GPU")


def _chk_onnx() -> tuple[str, str]:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return _ok(f"v{ort.__version__} (CUDA)")
        return _warn(f"v{ort.__version__} (CPU only)")
    except ImportError:
        return _fail("onnxruntime not installed")


def _chk_import(module_name: str) -> tuple[str, str]:
    try:
        mod = __import__(module_name)
        ver = getattr(mod, "__version__", "")
        return _ok(f"v{ver}" if ver else "")
    except ImportError as exc:
        return _fail(str(exc))


def _chk_font(project_dir: str, font_path: str) -> tuple[str, str]:
    resolved = font_path if os.path.isabs(font_path) else os.path.join(project_dir, font_path)
    if os.path.isfile(resolved):
        return _ok(os.path.basename(resolved))
    return _warn(f"Not found at {resolved}. TypesetterWorker will use fallback font.")


def _chk_dir_readable(dir_path: str) -> tuple[str, str]:
    if not os.path.isdir(dir_path):
        return _warn(f"Does not exist: {dir_path}")
    if os.access(dir_path, os.R_OK):
        n = len(os.listdir(dir_path))
        return _ok(f"{n} items")
    return _fail(f"Not readable: {dir_path}")


def _chk_dir_writable(dir_path: str) -> tuple[str, str]:
    os.makedirs(dir_path, exist_ok=True)
    if os.access(dir_path, os.W_OK):
        return _ok(dir_path)
    return _fail(f"Not writable: {dir_path}")


def _chk_config(project_dir: str) -> tuple[str, str]:
    cfg = config_path(project_dir)
    if os.path.isfile(cfg):
        try:
            from cli.config.models import CliConfig
            CliConfig.from_toml(cfg)
            return _ok(cfg)
        except Exception as exc:
            return _warn(f"Parse error: {exc}")
    return _warn(f"Not found — run [bold cyan]manga init[/bold cyan] to create it")


def _chk_manifest(project_dir: str) -> tuple[str, str]:
    manifest_file = os.path.join(project_dir, "project_manifest.json")
    if not os.path.isfile(manifest_file):
        return _warn("Not found — run [bold cyan]manga scan[/bold cyan] to create it")
    try:
        from core.models import ProjectManifest
        m = ProjectManifest.load_from_file(manifest_file)
        return _ok(f"{len(m.pages)} pages")
    except Exception as exc:
        return _fail(f"Corrupt manifest: {exc}")
