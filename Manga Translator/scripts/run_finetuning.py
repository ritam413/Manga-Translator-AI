#!/usr/bin/env python3
"""Master orchestration script for the Manga Translator ML Finetuning Pipeline.

Runs resolution alignment, dataset extraction, layout model training, and
translation memory database compilation sequentially for a given model version.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_finetuning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the entire Manga Translator finetuning pipeline."
    )
    # Orchestrator & General args
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Target model version ('v1' or 'v2'). Default: v2",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="training_data/raw",
        help="Directory containing raw Japanese manga images (default: training_data/raw)",
    )
    parser.add_argument(
        "--eng-dir",
        type=str,
        default="training_data/en",
        help="Directory containing translated English manga images (default: training_data/en)",
    )

    # Extractor args
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete existing checkpoint before starting dataset extraction (appends to dataset)",
    )
    parser.add_argument(
        "--reset-checkpoint-reset",
        action="store_true",
        help="Delete existing checkpoint and overwrite/delete output JSONL dataset (overwrite fresh)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Number of image pairs to process before saving a checkpoint (default: 50)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU to match JP and EN text regions (default: 0.3)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Use CPU only for EasyOCR text detection",
    )
    parser.add_argument(
        "--no-manga-ocr",
        action="store_true",
        help="Skip manga-ocr JP transcription (faster debug run)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging in sub-scripts",
    )

    # Training args
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum typesetting confidence threshold for layout training (default: 0.5)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV hyperparameter optimization during layout training",
    )

    # Translation Memory args
    parser.add_argument(
        "--vocab-threshold",
        type=float,
        default=0.3,
        help="Co-occurrence threshold for terminology extraction (default: 0.3)",
    )

    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    """Run a command as a subprocess, streaming its output."""
    logger.info("Executing command: %s", " ".join(cmd))
    try:
        # We stream output directly to console
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Command failed with exit status %d", exc.returncode)
        sys.exit(exc.returncode)


def main() -> int:
    args = parse_args()
    version = args.model_version
    root_dir = Path(__file__).resolve().parent.parent

    logger.info("======================================================================")
    logger.info("Starting Manga Translator ML Finetuning Pipeline (Version: %s)", version)
    logger.info("======================================================================")

    # Determine the python executable to use. Prefer the workspace venv if it exists.
    python_exe = sys.executable
    venv_python_win = root_dir / "venv" / "Scripts" / "python.exe"
    venv_python_unix = root_dir / "venv" / "bin" / "python"
    if venv_python_win.exists():
        python_exe = str(venv_python_win)
    elif venv_python_unix.exists():
        python_exe = str(venv_python_unix)

    # 1. Align Image Resolutions
    logger.info("\n--- Step 1: Aligning English image resolutions with raw Japanese images ---")
    align_cmd = [
        python_exe,
        str(root_dir / "scripts" / "align_image_resolutions.py"),
        "--raw-dir",
        args.raw_dir,
        "--eng-dir",
        args.eng_dir,
    ]
    run_command(align_cmd)

    # 2. Run Dataset Extractor
    logger.info("\n--- Step 2: Extracting dataset from paired manga pages ---")
    extractor_cmd = [
        python_exe,
        str(root_dir / "scripts" / "dataset_extractor.py"),
        "--model-version",
        version,
        "--raw-dir",
        args.raw_dir,
        "--eng-dir",
        args.eng_dir,
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--iou-threshold",
        str(args.iou_threshold),
    ]
    if args.reset_checkpoint:
        extractor_cmd.append("--reset-checkpoint")
    if args.reset_checkpoint_reset:
        extractor_cmd.append("--reset-checkpoint-reset")
    if args.no_gpu:
        extractor_cmd.append("--no-gpu")
    if args.no_manga_ocr:
        extractor_cmd.append("--no-manga-ocr")
    if args.verbose:
        extractor_cmd.append("--verbose")
    run_command(extractor_cmd)

    # 3. Train Layout Optimization Model
    logger.info("\n--- Step 3: Training Layout Optimization Model ---")
    train_cmd = [
        python_exe,
        str(root_dir / "scripts" / "train_layout_model.py"),
        "--model-version",
        version,
        "--min-confidence",
        str(args.min_confidence),
    ]
    if args.tune:
        train_cmd.append("--tune")
    run_command(train_cmd)

    # 4. Build Translation Memory Database
    logger.info("\n--- Step 4: Compiling Translation Memory Database ---")
    tm_cmd = [
        python_exe,
        str(root_dir / "scripts" / "build_translation_memory.py"),
        "--model-version",
        version,
        "--vocab-threshold",
        str(args.vocab_threshold),
    ]
    run_command(tm_cmd)

    logger.info("======================================================================")
    logger.info("ML Finetuning Pipeline completed successfully!")
    logger.info("======================================================================")
    logger.info("Generated Artifacts:")
    logger.info("  1. Dataset:             training_data/dataset_%s.jsonl", version)
    logger.info("  2. Inpaint Masks:       training_data/masks_%s/", version)
    logger.info("  3. Layout Model:        assets/models/layout_model_%s.pkl", version)
    logger.info("  4. Translation Memory:  assets/models/translation_memory_%s.json", version)
    logger.info("")
    logger.info("To run the main pipeline using these versioned models:")
    logger.info("  python main.py execute all --project-dir <dir> --model-version %s", version)
    logger.info("======================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
