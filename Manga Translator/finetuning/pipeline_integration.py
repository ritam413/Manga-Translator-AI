"""finetuning/pipeline_integration.py — Shared layer bridging finetuning artifacts and main pipeline."""

from __future__ import annotations
import os
import pickle
import re
import unicodedata
import numpy as np
from pathlib import Path
from typing import Any, Tuple

from core.models import BoundingBox, ProjectManifest
from finetuning.translation_memory import TranslationMemoryDB


def load_translation_memory(path: str | Path) -> TranslationMemoryDB | None:
    """Load TranslationMemoryDB from JSON path; return None if missing/disabled."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        db = TranslationMemoryDB()
        db.load(p)
        return db
    except Exception:
        return None


def load_layout_model(path: str | Path) -> Any | None:
    """Load Random Forest layout model from pickle path; return None if missing/disabled."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None


def build_story_position(manga_id: str, page_number: int, region_index: int) -> str:
    """Format story position identifier consistently with Phase 6 schema."""
    return f"{manga_id}::{page_number:04d}::{region_index:02d}"


def derive_page_order(manifest: ProjectManifest) -> dict[str, int]:
    """Port numeric-prefix sort mapping from page filenames in manifest to 1-based page number."""
    from finetuning.page_order import _sort_key
    filenames = list(manifest.pages.keys())
    sorted_filenames = sorted(filenames, key=_sort_key)
    return {filename: idx for idx, filename in enumerate(sorted_filenames, 1)}


def strip_notations(text: str) -> str:
    """Normalize and strip brackets, furigana guides, and common punctuation."""
    import unicodedata
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = re.sub(r'[\(\)（）「」『』。、，．？！?!_~-]', '', normalized)
    return cleaned.strip()


SMALL_TO_LARGE = {
    'ぁ': 'あ', 'ぃ': 'い', 'ぅ': 'う', 'ぇ': 'え', 'ぉ': 'お',
    'っ': 'つ', 'ゃ': 'や', 'ゅ': 'ゆ', 'ょ': 'よ', 'ゎ': 'わ',
    'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ',
    'ッ': 'ツ', 'ャ': 'ヤ', 'ュ': 'ユ', 'ョ': 'ヨ', 'ヮ': 'ワ',
}


def katakana_to_hiragana(text: str) -> str:
    """Convert Katakana characters in text to Hiragana."""
    res = []
    for c in text:
        code = ord(c)
        if 12449 <= code <= 12534:
            res.append(chr(code - 96))
        else:
            res.append(c)
    return "".join(res)


def normalize_kana_size(text: str) -> str:
    """Normalize small Hiragana/Katakana characters to standard size."""
    return "".join(SMALL_TO_LARGE.get(c, c) for c in text)


def normalize_for_matching(text: str) -> str:
    """Apply comprehensive normalization for spelling matching."""
    text = strip_notations(text)
    text = katakana_to_hiragana(text)
    text = normalize_kana_size(text)
    # Normalize common phonetic representations
    text = text.replace("王", "おう")
    text = text.replace("オウ", "おう")
    text = text.replace("おう", "お")
    text = text.replace("おお", "お")
    text = text.replace("おー", "お")
    text = text.replace("こにちは", "こんにちは")
    return text


def correct_ocr_with_vocab(db: TranslationMemoryDB | None, manga_id: str, raw_text: str) -> str:
    """Perform fuzzy token substitution of OCR typos against normalized vocab keys."""
    if not db or not manga_id or manga_id not in db.manga_vocab:
        return raw_text
    vocab = db.manga_vocab[manga_id]
    if not vocab:
        return raw_text

    # Merge Kanji and Katakana into single tokens, ignoring Hiragana particles as separators
    jp_regex = re.compile(r'[\u30a0-\u30ff\u30fc\u4e00-\u9fff]+')
    from nltk.metrics import edit_distance

    output = []
    last_idx = 0
    for match in jp_regex.finditer(raw_text):
        token = match.group()
        start, end = match.span()
        output.append(raw_text[last_idx:start])

        norm_token = normalize_for_matching(token)
        if not norm_token:
            output.append(token)
            last_idx = end
            continue

        # Check if the token (normalized) exists in vocabulary (normalized)
        best_key = None
        best_sim = 0.0
        
        # Check for exact matches on un-normalized or normalized values first
        found_exact = False
        for vocab_key in vocab.keys():
            if token == vocab_key:
                best_key = vocab_key
                best_sim = 1.0
                found_exact = True
                break
            norm_key = normalize_for_matching(vocab_key)
            if norm_token == norm_key:
                best_key = vocab_key
                best_sim = 1.0
                found_exact = True
                break

        if not found_exact:
            # Fuzzy match against keys to find canonical Japanese word
            for vocab_key in vocab.keys():
                norm_key = normalize_for_matching(vocab_key)
                if not norm_key:
                    continue
                dist = edit_distance(norm_token, norm_key)
                max_len = max(len(norm_token), len(norm_key))
                sim = 1.0 - (dist / max_len) if max_len > 0 else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_key = vocab_key

        if best_key and best_sim >= 0.8:
            output.append(best_key)
        else:
            output.append(token)

        last_idx = end
    output.append(raw_text[last_idx:])
    return "".join(output)


def build_context_prefix(db: TranslationMemoryDB | None, manga_id: str, story_position: str, limit: int = 5) -> str:
    """Retrieve preceding JP sentences and join with '。' separator."""
    if not db or not manga_id or not story_position:
        return ""
    context_list = db.get_context(manga_id, story_position, limit=limit)
    if not context_list:
        return ""
    return "。".join(jp for jp, en in context_list if jp)


def prepend_context(prefix: str, text: str) -> str:
    """Glue context prefix to target text with a distinctive delimiter."""
    if not prefix:
        return text
    return f"{prefix}\n---\n{text}"


def extract_layout_features(box: BoundingBox, jp_text: str) -> np.ndarray:
    """Extract standard 6-feature vector matching train script."""
    w = box.x_max - box.x_min
    h = box.y_max - box.y_min
    area = w * h
    aspect_ratio = w / h if h > 0 else 0.0
    char_count = len(jp_text.strip())
    char_density = char_count / area if area > 0 else 0.0
    return np.array([w, h, area, aspect_ratio, float(char_count), char_density], dtype=np.float32)


def build_jp_ink_inpaint_mask(page_gray: np.ndarray, box: BoundingBox) -> Tuple[np.ndarray, float]:
    """Runtime contour mask on JP crop returning full-page mask and confidence score."""
    from finetuning.typesetting_features import get_text_ink_mask
    
    try:
        closed, _, crop_origin = get_text_ink_mask(page_gray, box, crop_margin_px=2)
    except Exception:
        return np.zeros_like(page_gray), 0.0

    crop_h, crop_w = closed.shape
    crop_area = crop_h * crop_w
    fg_pixels = np.count_nonzero(closed)

    confidence = 1.0
    if fg_pixels == 0:
        confidence = 0.0
    else:
        ratio = fg_pixels / crop_area
        if ratio > 0.8:
            confidence -= 0.8
        elif ratio < 0.005:
            confidence -= 0.6

    confidence = max(0.0, min(1.0, confidence))

    # Build full-page mask
    h, w = page_gray.shape
    mask_full = np.zeros((h, w), dtype=np.uint8)

    # Paste crop mask onto full-page mask
    x_offset, y_offset = crop_origin
    y_end = min(h, y_offset + crop_h)
    x_end = min(w, x_offset + crop_w)
    
    mask_full[y_offset:y_end, x_offset:x_end] = closed[:y_end-y_offset, :x_end-x_offset]

    return mask_full, confidence
