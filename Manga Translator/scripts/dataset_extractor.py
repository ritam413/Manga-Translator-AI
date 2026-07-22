"""Thin CLI entrypoint for the finetuning dataset extractor."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finetuning.extractor import DatasetExtractor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset Extractor for Manga Translator Finetuning"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="raw(jap)",
        help="Directory containing raw Japanese manga images",
    )
    parser.add_argument(
        "--eng-dir",
        type=str,
        default="eng",
        help="Directory containing translated English manga images",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Model version to generate training data for ('v1' or 'v2')",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Checkpoint tracking file",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Number of image pairs to process before saving a checkpoint",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU to match JP and EN text regions",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete existing checkpoint before starting",
    )
    parser.add_argument(
        "--reset-checkpoint-reset",
        action="store_true",
        help="Delete existing checkpoint and output JSONL dataset (overwrite)",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default=None,
        help="Directory for per-region inpaint mask PNGs",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU for EasyOCR (default)",
    )
    gpu_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="Use CPU only for EasyOCR",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-manga-ocr",
        action="store_true",
        help="Skip manga-ocr JP transcription (faster, CPU debug mode)",
    )
    parser.add_argument(
        "--ocr-crop-margin",
        type=int,
        default=4,
        help="Pixel margin added around jp_box before manga-ocr crop (default: 4)",
    )

    args = parser.parse_args()
    gpu = not args.no_gpu
    model_version = args.model_version

    output_file = args.output_file
    if output_file is None:
        output_file = f"training_data/dataset_{model_version}.jsonl"

    checkpoint_file = args.checkpoint_file
    if checkpoint_file is None:
        checkpoint_file = f"training_data/extraction_state_{model_version}.json"

    masks_dir = args.masks_dir
    if masks_dir is None:
        masks_dir = f"training_data/masks_{model_version}"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    extractor = DatasetExtractor(
        raw_dir=args.raw_dir,
        eng_dir=args.eng_dir,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        checkpoint_interval=args.checkpoint_interval,
        reset=args.reset_checkpoint,
        gpu=gpu,
        iou_threshold=args.iou_threshold,
        masks_dir=masks_dir,
        use_manga_ocr=not args.no_manga_ocr,
        ocr_crop_margin_px=args.ocr_crop_margin,
        model_version=model_version,
        reset_checkpoint_reset=args.reset_checkpoint_reset,
    )

    result = extractor.extract()
    print(
        f"Done — processed: {result['processed']}, "
        f"skipped: {result['skipped']}, failed: {result['failed']}"
    )
    if result["processed"] > 0:
        return 0
    return 1 if result["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
