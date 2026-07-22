#!/usr/bin/env python3
"""Script to align English image resolutions with Japanese raw images to prevent dimension mismatches."""

import argparse
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("align_image_resolutions")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize English translated images to match raw Japanese page dimensions."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="training_data/raw",
        help="Directory containing raw Japanese manga images",
    )
    parser.add_argument(
        "--eng-dir",
        type=str,
        default="training_data/en",
        help="Directory containing translated English manga images",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    eng_dir = Path(args.eng_dir)
    
    if not raw_dir.exists():
        logger.error("Raw directory not found: %s", raw_dir)
        return 1
    if not eng_dir.exists():
        logger.error("English directory not found: %s", eng_dir)
        return 1

    # Find pairs the exact same way as dataset_extractor
    pairs = []
    seen = set()

    for ext in IMAGE_EXTENSIONS:
        # Flat layout
        for raw_path in raw_dir.glob(f"*{ext}"):
            if not raw_path.is_file():
                continue
            img_name = raw_path.name
            eng_path = eng_dir / img_name
            pair_id = f"default/{img_name}"
            if eng_path.exists() and pair_id not in seen:
                pairs.append((raw_path, eng_path))
                seen.add(pair_id)

        # Nested layout (manga_name/img_name)
        for raw_path in raw_dir.glob(f"*/*{ext}"):
            if not raw_path.is_file():
                continue
            manga_name = raw_path.parent.name
            img_name = raw_path.name
            eng_path = eng_dir / manga_name / img_name
            pair_id = f"{manga_name}/{img_name}"
            if eng_path.exists() and pair_id not in seen:
                pairs.append((raw_path, eng_path))
                seen.add(pair_id)

    logger.info("Found %d image pairs to check.", len(pairs))
    aligned_count = 0
    resized_count = 0
    
    from core.progress import MangaProgress

    with MangaProgress() as mp:
        for raw_path, eng_path in mp.track(pairs, description="Aligning image resolutions"):
            try:
                with Image.open(raw_path) as raw_img, Image.open(eng_path) as eng_img:
                    raw_size = raw_img.size
                    eng_size = eng_img.size
                    
                    if raw_size == eng_size:
                        aligned_count += 1
                        continue
                    
                    # Resize English image to match Raw image
                    mp.print(
                        f"Dimension mismatch for {raw_path.parent.name}/{raw_path.name}: Raw {raw_size} vs Eng {eng_size}. Resizing..."
                    )
                    logger.info(
                        "Dimension mismatch for %s/%s: Raw %s vs Eng %s. Resizing English image...",
                        raw_path.parent.name, raw_path.name, raw_size, eng_size
                    )
                    
                    # Load fully in memory so we can close the file context and overwrite
                    eng_resized = eng_img.resize(raw_size, Image.Resampling.LANCZOS)
                    
                # Overwrite the English image
                eng_resized.save(eng_path)
                resized_count += 1
                
            except Exception as exc:
                logger.error("Failed to align %s/%s: %s", raw_path.parent.name, raw_path.name, exc)

    logger.info("Check complete. Already aligned: %d, Resized to match: %d", aligned_count, resized_count)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
