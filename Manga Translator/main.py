#!/usr/bin/env python3
"""
main.py — Manga Translator Pipeline CLI

Double-stage CLI with two primary verbs:
  - `plan`:    Scan input directory and create/update project manifest.
  - `execute`: Run one or more numbered stages against the manifest.
  - `stages`:  List all available pipeline stages.

Examples:
    python main.py plan --project-dir /path/to/project
    python main.py execute --stage translate
    python main.py execute --from ocr --to typeset
    python main.py execute all
    python main.py stages

Architecture:
  - Argument parsing via argparse with subcommands
  - Stage registry centralizes stage definitions
  - StageController orchestrates worker execution
  - ManifestEngine handles project I/O
  - No hardcoded stage numbers in CLI code
"""

import argparse
import os
import sys
import logging
from typing import Optional, List

from core.manifest import ManifestEngine
from stages.controller import StageController
from stages.registry import StageRegistry
from workers import (
    TextDetectorWorker, OcrWorker, TranslationWorker,
    InpainterWorker, TypesetterWorker, VramManager,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_plan(args: argparse.Namespace) -> int:
    """
    Subcommand: `plan`

    Scans the input directory for image files and creates or updates the
    project manifest (project_manifest.json).

    Args:
        args: Parsed arguments with attributes:
            - project_dir: Project root directory
            - input_subdir: Input subdirectory name (default "input")

    Returns:
        0 on success, 1 on error.
    """
    try:
        project_dir = args.project_dir
        input_subdir = args.input_subdir

        logger.info(
            "[CLI] Planning manifest for project: %s (input_subdir=%s)",
            project_dir,
            input_subdir,
        )

        # Initialize manifest engine
        engine = ManifestEngine(project_dir, input_subdir=input_subdir)

        # Load or create manifest
        manifest = engine.load_or_create(
            project_name=os.path.basename(project_dir),
            source_lang="ja",
            target_lang="en",
        )

        # Scan disk for images
        added, missing = engine.sync_with_disk()
        logger.info(
            "[CLI] Manifest sync: +%d added, -%d missing (total: %d pages)",
            len(added),
            len(missing),
            len(manifest.pages),
        )

        # Save manifest
        engine.save_manifest(backup=True)
        logger.info("[CLI] Manifest created/updated: %s", engine.manifest_path)

        print(f"Manifest ready: {len(manifest.pages)} pages")
        return 0

    except Exception as exc:
        logger.error("[CLI] Plan failed: %s", exc, exc_info=True)
        print(f"Plan failed: {exc}", file=sys.stderr)
        return 1


def _sort_key_numeric(filename: str) -> tuple:
    """Sort helper: splits filename into (numeric_prefix, rest) for natural ordering."""
    import re
    m = re.match(r'^(\d+)', filename)
    return (int(m.group(1)), filename) if m else (float('inf'), filename)


def cmd_execute(args: argparse.Namespace) -> int:
    """
    Subcommand: `execute`

    Runs one or more stages against the project manifest.

    Supports three modes:
      1. Single stage:      --stage NAME
      2. Range of stages:   --from NAME --to NAME
      3. All stages:        'all' argument

    Args:
        args: Parsed arguments with attributes:
            - project_dir: Project root directory
            - input_subdir: Input subdirectory name
            - stage: Single stage name (or None)
            - from_stage: Start of range (or None)
            - to_stage: End of range (or None)
            - all_stages: Boolean, True if "all" was passed
            - force: Boolean, force re-run already-complete pages

    Returns:
        0 on success, 1 on error.
    """
    try:
        
        project_dir = args.project_dir
        input_subdir = args.input_subdir
        force = args.force
        limit = args.limit        # int or None
        pages_arg = args.pages    # list[str] or None

        model_version = getattr(args, "model_version", None)
        if model_version is None:
            if sys.stdin.isatty():
                try:
                    ans = input("Select model version (1: v1 (single-word), 2: v2 (merged-bubble) [default]): ").strip()
                    if ans == "1" or ans.lower() == "v1":
                        model_version = "v1"
                    else:
                        model_version = "v2"
                except Exception:
                    model_version = "v2"
            else:
                model_version = "v2"

        logger.info("[CLI] Executing stages for project: %s (model version: %s)", project_dir, model_version)

        # Load manifest
        engine = ManifestEngine(project_dir, input_subdir=input_subdir)
        if os.path.exists(engine.manifest_path):
            manifest = engine.load_or_create(project_name=os.path.basename(project_dir))
        else:
            raise RuntimeError(
                f"Manifest not found at {engine.manifest_path}. Run `plan` first."
            )
        if manifest is None:
            raise RuntimeError(
                f"Manifest not found at {engine.manifest_path}. "
                "Run `plan` first."
            )

        # ── Build page_filter from --pages / --limit ────────────────────────
        page_filter: Optional[List[str]] = None

        if pages_arg:
            # Explicit list of filenames
            page_filter = list(pages_arg)
            logger.info("[CLI] Page filter: %d explicit pages", len(page_filter))
        elif limit is not None:
            # First N pages sorted numerically by filename
            all_filenames = sorted(manifest.pages.keys(), key=_sort_key_numeric)
            page_filter = all_filenames[:limit]
            logger.info(
                "[CLI] Page filter: first %d of %d pages (--limit)",
                len(page_filter),
                len(all_filenames),
            )

        # Determine which stages to run
        registry = StageRegistry()
        stages_to_run: List[tuple] = []

        if args.all_stages:
            stages_to_run = registry.list_all()
        elif args.stage:
            try:
                stage_id, worker_class, phase = registry.get_by_name(args.stage)
                # Reconstruct the full tuple
                for sid, name, desc, wc, ph in registry.list_all():
                    if sid == stage_id:
                        stages_to_run = [(sid, name, desc, wc, ph)]
                        break
            except ValueError as exc:
                logger.error("[CLI] Invalid stage: %s", exc)
                print(f"Invalid stage: {exc}", file=sys.stderr)
                print(f"Available stages: {', '.join(registry.list_names())}")
                return 1
        elif args.from_stage and args.to_stage:
            try:
                stages_to_run = registry.get_range(args.from_stage, args.to_stage)
            except ValueError as exc:
                logger.error("[CLI] Invalid range: %s", exc)
                print(f"Invalid range: {exc}", file=sys.stderr)
                print(f"Available stages: {', '.join(registry.list_names())}")
                return 1
        else:
            raise RuntimeError(
                "Must specify --stage NAME, --from X --to Y, or use 'all'"
            )

        # Execute stages
        results = []
        from core.progress import MangaProgress

        with MangaProgress() as mp:
            pipeline_task = mp.add_task(
                "[bold]Pipeline[/bold]", total=len(stages_to_run)
            )
            page_task = mp.add_task(
                "[dim]Pages[/dim]", total=0, visible=False
            )

            for stage_id, stage_name, stage_desc, worker_class, phase in stages_to_run:
                logger.info(
                    "[CLI] Executing stage %d: %s — %s",
                    stage_id,
                    stage_name,
                    stage_desc,
                )

                try:
                    # Instantiate worker with appropriate parameters
                    from cli.config.models import load_config
                    config = load_config(project_dir)
                    if args.max_words_per_line is not None:
                        config.pipeline.max_words_per_line = args.max_words_per_line
                    if getattr(args, "translator_backend", None) is not None:
                        config.translation.backend = args.translator_backend
                    if getattr(args, "ollama_url", None) is not None:
                        config.translation.ollama_url = args.ollama_url
                    if getattr(args, "ollama_model", None) is not None:
                        config.translation.ollama_model = args.ollama_model

                    from cli.worker_factory import (
                        build_text_detector, build_ocr_worker, build_translation_worker,
                        build_inpainter, build_typesetter
                    )
                    if worker_class == TextDetectorWorker:
                        worker = build_text_detector(project_dir, config, model_version=model_version)
                    elif worker_class == OcrWorker:
                        worker = build_ocr_worker(project_dir, config, model_version=model_version)
                    elif worker_class == TranslationWorker:
                        worker = build_translation_worker(config, project_dir=project_dir, model_version=model_version)
                    elif worker_class == InpainterWorker:
                        worker = build_inpainter(project_dir, config)
                    elif worker_class == TypesetterWorker:
                        worker = build_typesetter(project_dir, config, model_version=model_version)
                    else:
                        raise RuntimeError(f"Unknown worker class: {worker_class}")

                    controller = StageController(
                        engine=engine,
                        worker=worker,
                        stage_name=f"Stage {stage_id}: {stage_name}",
                        marks_phase=phase if stage_name in ("translate", "typeset") else None,
                        checks_phase=phase,
                        run_quality_check=False,
                        page_filter=page_filter,
                        force=force,
                        flush_vram=True,
                    )

                    # Count how many pages will actually run for this stage
                    pages_to_run = [p for p in engine.manifest.pages.values() if controller._should_process(p)]

                    mp.update(
                        page_task,
                        description=f"[dim]Stage {stage_id} Pages[/dim]",
                        total=len(pages_to_run),
                        completed=0,
                        visible=True,
                    )

                    controller.on_page_done = lambda filename, success: mp.advance(page_task)

                    mp.update(
                        pipeline_task,
                        description=f"[bold]Pipeline[/bold]  [cyan]Stage {stage_id}: {stage_name}[/cyan]",
                    )

                    report = controller.run()

                    logger.info(
                        "[CLI] Stage %d complete: %d passed, %d failed, %d skipped",
                        stage_id,
                        len(report.succeeded),
                        len(report.failed),
                        len(report.skipped),
                    )

                    results.append((stage_name, report))
                    mp.advance(pipeline_task)

                    if len(report.failed) > 0:
                        logger.warning(
                            "[CLI] Stage %d had failures; continuing to next stage",
                            stage_id,
                        )

                except Exception as exc:
                    logger.error(
                        "[CLI] Stage %d (%s) failed: %s",
                        stage_id,
                        stage_name,
                        exc,
                        exc_info=True,
                    )
                    print(f"Stage {stage_name} failed: {exc}", file=sys.stderr)
                    return 1

        # Print summary
        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        for stage_name, report in results:
            status = "PASS" if len(report.failed) == 0 else "FAIL"
            print(
                f"{status:6s}  {stage_name:30s}  "
                f"{len(report.succeeded):3d} passed, {len(report.failed):3d} failed, {len(report.skipped):3d} skipped"
            )
        print("=" * 70)

        return 0

    except Exception as exc:
        logger.error("[CLI] Execute failed: %s", exc, exc_info=True)
        print(f"Execute failed: {exc}", file=sys.stderr)
        return 1


def cmd_stages(args: argparse.Namespace) -> int:
    """
    Subcommand: `stages`

    Lists all available pipeline stages with their IDs and descriptions.

    Args:
        args: Parsed arguments (minimal).

    Returns:
        0 on success.
    """
    registry = StageRegistry()
    stages = registry.list_all()

    print("\nAvailable Pipeline Stages:")
    print("=" * 80)
    print(f"{'ID':<3} {'Name':<20} {'Description':<50} {'Phase':<5}")
    print("-" * 80)

    for stage_id, stage_name, description, _, phase in stages:
        phase_label = "Read" if phase == "a" else "Write"
        # Truncate description if too long
        desc_trunc = (
            description if len(description) <= 48
            else description[:45] + "…"
        )
        print(f"{stage_id:<3} {stage_name:<20} {desc_trunc:<50} {phase_label:<5}")

    print("=" * 80)
    print("\nUsage examples:")
    print("  python main.py execute --stage translate")
    print("  python main.py execute --from ocr --to typeset")
    print("  python main.py execute all")
    print()

    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    """
    Subcommand: `finetune`
    
    Reads user corrections from a JSONL file, merges them with the base layout dataset,
    and runs the training script to retrain the layout Random Forest regressor model.
    """
    import json
    import subprocess
    from pathlib import Path
    
    try:
        model_version = args.model_version
        root_dir = Path(__file__).resolve().parent
        
        corrections_path = Path(args.use_corrections)
        if not corrections_path.is_absolute():
            corrections_path = root_dir / corrections_path
            
        base_dataset_path = root_dir / "training_data" / f"dataset_{model_version}.jsonl"
        combined_path = root_dir / "training_data" / f"combined_dataset_{model_version}.jsonl"
        
        logger.info(
            "[CLI] Starting layout model training. Version=%s, Corrections=%s",
            model_version,
            corrections_path,
        )
        
        # 1. Load corrections
        corrections_map = {}
        if corrections_path.is_file():
            logger.info("[CLI] Reading corrections from: %s", corrections_path)
            with open(corrections_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line.strip())
                        manga_id = row.get("manga_id", "default")
                        page = row.get("page", "")
                        region_idx = row.get("region_index", 0)
                        corrections_map[(manga_id, page, region_idx)] = row
                    except Exception as e:
                        logger.warning("[CLI] Failed to parse correction line: %s", e)
        else:
            logger.info("[CLI] No corrections file found at %s. Will train on base dataset only.", corrections_path)
            
        # 2. Load base dataset and merge overrides
        merged_rows = []
        base_count = 0
        override_count = 0
        
        if base_dataset_path.is_file():
            logger.info("[CLI] Reading base dataset from: %s", base_dataset_path)
            with open(base_dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line.strip())
                        manga_id = row.get("manga_id", "default")
                        page = row.get("page", "")
                        region_idx = row.get("region_index", 0)
                        
                        key = (manga_id, page, region_idx)
                        if key in corrections_map:
                            merged_rows.append(corrections_map.pop(key))
                            override_count += 1
                        else:
                            merged_rows.append(row)
                        base_count += 1
                    except Exception as e:
                        logger.warning("[CLI] Failed to parse base dataset line: %s", e)
        else:
            logger.info("[CLI] Base dataset file not found at %s. Creating new dataset from corrections.", base_dataset_path)
            
        # 3. Add any remaining new correction regions (e.g. newly drawn bubbles)
        new_bubble_count = len(corrections_map)
        for row in corrections_map.values():
            merged_rows.append(row)
            
        logger.info(
            "[CLI] Merge statistics: Base rows=%d, Overridden=%d, Newly added=%d, Total training rows=%d",
            base_count,
            override_count,
            new_bubble_count,
            len(merged_rows),
        )
        
        # 4. Save combined dataset
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w", encoding="utf-8") as f:
            for row in merged_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
        # 5. Determine the python executable to use
        python_exe = sys.executable
        venv_python_win = root_dir / "venv" / "Scripts" / "python.exe"
        venv_python_unix = root_dir / "venv" / "bin" / "python"
        if venv_python_win.exists():
            python_exe = str(venv_python_win)
        elif venv_python_unix.exists():
            python_exe = str(venv_python_unix)
            
        # 6. Execute scripts/train_layout_model.py
        train_script = root_dir / "scripts" / "train_layout_model.py"
        cmd = [
            python_exe,
            "-u",
            str(train_script),
            "--model-version", model_version,
            "--dataset-file", str(combined_path),
            "--model-out", str(root_dir / "assets" / "models" / f"layout_model_{model_version}.pkl"),
        ]
        
        logger.info("[CLI] Executing: %s", " ".join(cmd))
        
        # Run subprocess directly so output goes to stdout/stderr in real-time
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(root_dir))
        return result.returncode
        
    except Exception as exc:
        logger.error("[CLI] Finetune failed: %s", exc, exc_info=True)
        print(f"Finetune failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="manga-translator",
        description="Translate Japanese manga pages to English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py plan --project-dir /path/to/project
  python main.py execute --stage translate
  python main.py execute --from ocr --to typeset
  python main.py execute all
  python main.py stages
        """,
    )

    # Global arguments
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--input-subdir",
        default="input",
        help="Input subdirectory name (default: 'input')",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Subcommand: finetune ──────────────────────────────────────────────────
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Finetune/train layout model using user corrections and base dataset",
    )
    finetune_parser.add_argument(
        "--use-corrections",
        default="training_data/user_corrections.jsonl",
        help="Path to user corrections JSONL file",
    )
    finetune_parser.add_argument(
        "--model-version",
        choices=["v1", "v2"],
        default="v2",
        help="Model version to retrain (default: v2)",
    )
    finetune_parser.set_defaults(func=cmd_finetune)

    # ── Subcommand: plan ──────────────────────────────────────────────────────
    plan_parser = subparsers.add_parser(
        "plan",
        help="Scan input directory and create/update project manifest",
    )
    plan_parser.set_defaults(func=cmd_plan)

    # ── Subcommand: execute ──────────────────────────────────────────────────
    execute_parser = subparsers.add_parser(
        "execute",
        help="Run one or more pipeline stages",
    )
    execute_parser.set_defaults(func=cmd_execute)

    # Stage selection (mutually exclusive)
    stage_group = execute_parser.add_mutually_exclusive_group(required=False)
    stage_group.add_argument(
        "--stage",
        type=str,
        help="Run a single named stage (e.g., 'translate')",
    )
    stage_group.add_argument(
        "--from-stage",
        dest="from_stage",
        type=str,
        help="Start of stage range (e.g., 'ocr')",
    )

    execute_parser.add_argument(
        "--to-stage",
        dest="to_stage",
        type=str,
        help="End of stage range (e.g., 'typeset')",
    )

    execute_parser.add_argument(
        "all_stages",
        nargs="?",
        const=True,
        default=False,
        help="Run all stages (pass 'all' as positional argument)",
    )

    execute_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-run of already-complete pages",
    )

    execute_parser.add_argument(
        "--model-version",
        choices=["v1", "v2"],
        default=None,
        help="Model version to use (v1: single-word/column-based, v2: merged-bubble)",
    )

    execute_parser.add_argument(
        "--max-words-per-line",
        type=int,
        default=None,
        help="Maximum number of words per typeset line (overrides config setting)",
    )

    execute_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N pages (sorted numerically). e.g. --limit 10",
    )

    execute_parser.add_argument(
        "--pages",
        nargs="+",
        default=None,
        metavar="FILENAME",
        help="Process only the named pages. e.g. --pages 1.jpg 2.jpg 5.jpg",
    )

    execute_parser.add_argument(
        "--translator-backend",
        dest="translator_backend",
        choices=["auto", "ollama", "tm", "google", "argos"],
        default=None,
        help="Translation backend to use (auto, ollama, tm, google, argos)",
    )

    execute_parser.add_argument(
        "--ollama-url",
        dest="ollama_url",
        type=str,
        default=None,
        help="Ollama server base URL (e.g. http://localhost:11434)",
    )

    execute_parser.add_argument(
        "--ollama-model",
        dest="ollama_model",
        type=str,
        default=None,
        help="Ollama model tag to use for translation",
    )

    # ── Subcommand: stages ───────────────────────────────────────────────────
    stages_parser = subparsers.add_parser(
        "stages",
        help="List all available pipeline stages",
    )
    stages_parser.set_defaults(func=cmd_stages)

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "execute":
        # Validate stage selection
        has_single = args.stage is not None
        has_from = args.from_stage is not None
        has_to = args.to_stage is not None
        has_all = args.all_stages

        if not (has_single or (has_from and has_to) or has_all):
            execute_parser.print_help()
            print(
                "\nError: Must specify --stage NAME, --from X --to Y, or pass 'all'",
                file=sys.stderr,
            )
            return 1

        # Validate mutually exclusive page-selection flags
        if args.limit is not None and args.pages is not None:
            print(
                "Error: --limit and --pages are mutually exclusive.",
                file=sys.stderr,
            )
            return 1

        if args.limit is not None and args.limit < 1:
            print("Error: --limit must be a positive integer.", file=sys.stderr)
            return 1

        # Validate range if both --from and --to are specified
        if has_from and has_to:
            registry = StageRegistry()
            try:
                registry.validate_range(args.from_stage, args.to_stage)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                print(f"Available stages: {', '.join(registry.list_names())}")
                return 1

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
