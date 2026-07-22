#!/usr/bin/env python3
"""Parse dataset.jsonl and build a Translation Memory (TM) Database."""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow importing finetuning modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from finetuning.translation_memory import TranslationMemoryDB, TranslationMemoryEntry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_translation_memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Translation Memory Database from dataset.jsonl."
    )
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Model version target ('v1' or 'v2')",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Path to the dataset.jsonl file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save the serialized Translation Memory Database (.json)",
    )
    parser.add_argument(
        "--vocab-threshold",
        type=float,
        default=0.3,
        help="Co-occurrence threshold for terminology extraction (default: 0.3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_version = args.model_version

    dataset_file = args.dataset_file
    if dataset_file is None:
        dataset_file = f"training_data/dataset_{model_version}.jsonl"

    output_file = args.output_file
    if output_file is None:
        output_file = f"assets/models/translation_memory_{model_version}.json"

    dataset_path = Path(dataset_file)
    output_path = Path(output_file)

    if not dataset_path.exists():
        logger.error("Dataset file not found at: %s", dataset_path)
        return 1

    db = TranslationMemoryDB()
    rows_processed = 0
    skipped_empty = 0

    logger.info("Reading dataset from: %s", dataset_path)
    
    # Count total lines for progress bar
    total_lines = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    from core.progress import MangaProgress
    mp = MangaProgress()
    mp.__enter__()
    task_id = mp.add_task("Reading dataset", total=total_lines)

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                mp.advance(task_id)
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    row = json.loads(line_str)
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d is not valid JSON. Skipping. Error: %s", line_num, exc)
                    continue

                # 1. Text extraction & fallbacks
                jp_text = row.get("jp_ocr_text")
                if jp_text is None or str(jp_text).strip() == "":
                    jp_text = row.get("jp_text_detected", "")

                en_text = row.get("en_text_detected", "")

                jp_text = str(jp_text).strip()
                en_text = str(en_text).strip()

                if not jp_text or not en_text:
                    skipped_empty += 1
                    continue

                manga_id = row.get("manga_id", "default")
                page_number = int(row.get("page_number", 0))
                story_position = str(row.get("story_position", ""))
                region_index = int(row.get("region_index", 0))

                # 2. Add entry to database
                entry = TranslationMemoryEntry(
                    jp_text=jp_text,
                    en_text=en_text,
                    page_number=page_number,
                    story_position=story_position,
                    region_index=region_index,
                )
                db.add_entry(manga_id, entry)
                rows_processed += 1
    finally:
        mp.__exit__(None, None, None)

    logger.info("Processed %d valid translation pairs (skipped %d empty)", rows_processed, skipped_empty)

    if rows_processed == 0:
        logger.error("No valid translation pairs found. Aborting.")
        return 1

    # 3. Build bilingual vocabularies
    logger.info("Extracting terminology dictionaries with threshold %.2f...", args.vocab_threshold)
    db.build_vocabularies(args.vocab_threshold)

    # 4. Serialize database to output JSON file
    try:
        db.save(output_path)
        logger.info("Translation Memory successfully saved to: %s", output_path)
    except Exception as exc:
        logger.error("Failed to save Translation Memory: %s", exc)
        return 1

    # Print summary statistics
    logger.info("--- Database Statistics ---")
    logger.info("Total Mangas: %d", len(db.manga_entries))
    for m_id, entries in db.manga_entries.items():
        vocab_size = len(db.manga_vocab.get(m_id, {}))
        logger.info("  Manga: '%s' -> %d entries, %d vocab mappings", m_id, len(entries), vocab_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
