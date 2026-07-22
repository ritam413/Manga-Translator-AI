# Finetuning Next Steps — Agent Handoff

> **Read only this file to implement the current phase.**  
> Supplementary references are listed at the bottom; do not explore the whole repo unless blocked.

---

## 1. Project snapshot

**Goal:** Build a training dataset from paired manga pages (Japanese raw + human-translated English) to finetune layout and translation helpers for the main pipeline.

**Hardware:** NVIDIA GTX 1050ti (4GB VRAM). GPU models must be loaded once per batch, then unloaded.

**Entry point:** `python scripts/dataset_extractor.py` (script-only; no `manga` CLI command for finetuning).

**Image layouts (both supported):**
- Nested: `raw(jap)/manga_name/1.jpg` ↔ `eng/manga_name/1.jpg`
- Flat: `raw(jap)/1.jpg` ↔ `eng/1.jpg` (manga_id = `"default"`)

---

## 2. What is already done

### Phase 1 — Environment (complete)
- `training_data/`, `assets/models/`, `requirements-finetuning.txt` exist

### Phase 2 — Incremental extractor shell (complete)
| File | Purpose |
|------|---------|
| [`finetuning/extractor.py`](../finetuning/extractor.py) | `DatasetExtractor` — pair loop, checkpointing, JSONL append |
| [`finetuning/checkpoint.py`](../finetuning/checkpoint.py) | Atomic checkpoint load/save/reset |
| [`finetuning/models.py`](../finetuning/models.py) | `ExtractionState`, `ExtractionRecord`, `DetectedRegion` |
| [`scripts/dataset_extractor.py`](../scripts/dataset_extractor.py) | CLI entry point |

### Phase 3 — Geometric matching (complete)
| File | Purpose |
|------|---------|
| [`finetuning/geometry.py`](../finetuning/geometry.py) | `calculate_iou`, `match_regions` (greedy matcher) |
| [`finetuning/detection.py`](../finetuning/detection.py) | `detect_text_regions` — EasyOCR wrapper |
| [`finetuning/extractor.py`](../finetuning/extractor.py) | `process_pair()` detects JP+EN boxes, matches via IoU, writes matched JSONL rows |

**Phase 3 JSONL row schema (one row per matched region):**
```json
{
  "pair_id": "manga1/1.png",
  "manga_id": "manga1",
  "page": "1",
  "region_index": 0,
  "status": "matched",
  "iou": 0.82,
  "jp_box": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 80},
  "en_box": {"x_min": 12, "y_min": 22, "x_max": 98, "y_max": 78},
  "jp_text_detected": "こんにちは",
  "en_text_detected": "HELLO",
  "raw_path": "...",
  "eng_path": "..."
}
```

**CLI flags:** `--gpu` / `--no-gpu`, `--iou-threshold` (default 0.3), `--checkpoint-interval`, `--reset-checkpoint`

**Tests:** 25 passing in `tests/test_geometry.py`, `tests/test_detection.py`, `tests/test_dataset_extractor.py`

### Phase 4 — Typesetting Feature Extraction (complete)
| File | Purpose |
|------|---------|
| [`finetuning/typesetting_features.py`](../finetuning/typesetting_features.py) | `extract_typesetting_features` — font size, line spacing, padding, wrap shape |
| [`finetuning/models.py`](../finetuning/models.py) | `TypesettingFeatures` dataclass |
| [`finetuning/extractor.py`](../finetuning/extractor.py) | Merges typesetting targets into JSONL rows |

**Phase 4 Enriched JSONL row:**
Fields added: `font_size`, `padding`, `line_spacing`, `wrap_shape`, `line_count`, `typesetting_confidence`.

### Phase 5 — Inpaint Mask Extraction (complete)
| File | Purpose |
|------|---------|
| [`finetuning/mask_extraction.py`](../finetuning/mask_extraction.py) | Extracts cropped per-region inpaint masks using image differencing and EN ink subtraction |
| [`finetuning/geometry.py`](../finetuning/geometry.py) | Added `union_box` helper to compute bounds containing both JP and EN regions |

**Phase 5 JSONL additions:**
Fields added: `mask_path`, `mask_roi_box`, `mask_confidence`, `mask_pixel_count`. PNG files saved to `training_data/masks/`.

### Phase 6 — Story Context & Translation Extraction (complete)
| File | Purpose |
|------|---------|
| [`finetuning/manga_ocr_extraction.py`](../finetuning/manga_ocr_extraction.py) | `MangaOcrExtractor` — lazy load/unload, crop+OCR per jp_box, NFKC normalization |
| [`finetuning/page_order.py`](../finetuning/page_order.py) | `PageOrderTracker` — numeric-prefix sort → 1-based `page_number` per manga |
| [`finetuning/models.py`](../finetuning/models.py) | `JpOcrResult` and `StoryContextMetadata` dataclasses |
| [`finetuning/extractor.py`](../finetuning/extractor.py) | MangaOCR co-loaded with EasyOCR per batch; per-region JP OCR with fallback |

**Phase 6 JSONL additions:**
Fields added: `jp_ocr_text`, `jp_ocr_confidence`, `page_number`, `story_position`.

**Tests:** 72 passing (10 manga-ocr + 6 page-order + 3 extractor + 14 layout-model-training tests).

### Phase 7 — Layout Optimization Model Training (complete)
| File | Purpose |
|------|---------|
| [`scripts/train_layout_model.py`](../scripts/train_layout_model.py) | Standalone training script to load JSONL, engineer features, clip outliers, tune hyperparameters, and serialize Random Forest model |
| [`tests/test_train_layout_model.py`](../tests/test_train_layout_model.py) | Unit tests (14 passed) |

### Phase 8 — Translation Memory Database Construction (complete)
| File | Purpose |
|------|---------|
| [`finetuning/translation_memory.py`](../finetuning/translation_memory.py) | `TranslationMemoryDB` & `TranslationMemoryEntry` — exact/fuzzy matching, context tracking, vocabulary builder |
| [`scripts/build_translation_memory.py`](../scripts/build_translation_memory.py) | Standalone builder script to compile TM database |
| [`tests/test_build_translation_memory.py`](../tests/test_build_translation_memory.py) | Unit tests (9 passed) |

---

## 3. Completed: Phase 9 & Phase 10 — Integration into Pipeline

### 3.1 Objective

Integrate the Translation Memory (TM) Database and context tracking mechanism into Phase A (OCR and Translator workers) and the layout Random Forest regressor and contour inpainting mask extraction into Phase B (Typesetter and Inpainter workers) of the main translation pipeline.

### 3.2 Status

- [x] TM database successfully loads in Stage 7 and Stage 8 workers.
- [x] Stage 7 implements spelling corrections based on TM vocabulary.
- [x] Stage 8 implements TM exact and fuzzy translation lookup bypassing.
- [x] Stage 8 utilizes preceding context (up to 5 sentences) for the translation call.
- [x] Stage 9 implements contour inpainting with heuristic confidence fallback.
- [x] Stage 10 implements Layout Model dynamic fit sizing constraints.
- [x] All 64 integration tests and 266 unit tests pass cleanly.

---

## 4. Commands cheat sheet

```powershell
# Use project venv
venv\Scripts\python.exe scripts/dataset_extractor.py --reset-checkpoint --no-gpu --checkpoint-interval 2

# Run finetuning tests
venv\Scripts\python.exe -m pytest tests/test_geometry.py tests/test_detection.py tests/test_dataset_extractor.py -v

# Inspect output
Get-Content training_data/dataset.jsonl
```

---

## 5. Out of scope (current phase)

- Modifying Phase B stages (Inpaint mask & layout model typesetting - Phase 10)
- `manga extract` CLI command
- New pip dependencies (NLTK and scikit-learn are already in requirements-finetuning.txt)

---

## 6. Optional references

| File | When to read |
|------|--------------|
| [`finetuning_context.md`](finetuning_context.md) | Roadmap status |
| [`implementation_plan_finetuning.md`](implementation_plan_finetuning.md) | Full 10-phase architecture |
| [`library_documentation_finetuning.md`](library_documentation_finetuning.md) | sklearn/nltk usage (later phases) |
| [`workers/typesetter.py`](../workers/typesetter.py) | Reference for how main pipeline sizes text |
