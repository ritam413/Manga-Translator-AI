# Finetuning Context Tracker

This document tracks the status of the machine learning extraction and finetuning upgrades.
*Note: Do not update `project_context.md` or `next_steps.md` until Phase 9 (Integration) begins.*

## Current Status
- **Current Active Phase:** Complete ✅
- **Overall Completion:** 100%

## The 10-Phase Roadmap
- [x] **Phase 1:** Environment & Dependency Setup
- [x] **Phase 2:** Incremental Dataset Extractor Architecture
- [x] **Phase 3:** Geometric Matching & Alignment Logic
- [x] **Phase 4:** Typesetting Feature Extraction
- [x] **Phase 5:** Inpaint Mask Extraction
- [x] **Phase 6:** Story Context & Translation Extraction
- [x] **Phase 7:** Layout Optimization Model Training
- [x] **Phase 8:** Translation Memory Database Construction
- [x] **Phase 9:** Integration into Pipeline Phase A (OCR & Translation)
- [x] **Phase 10:** Integration into Pipeline Phase B (Inpainting & Typesetting)

## Phase 3 Deliverables (complete)

| Item | Location |
|------|----------|
| IoU + greedy matcher | `finetuning/geometry.py` |
| EasyOCR detection wrapper | `finetuning/detection.py` |
| `DetectedRegion` model | `finetuning/models.py` |
| Matched JSONL rows in `process_pair` | `finetuning/extractor.py` |
| CLI flags `--gpu`/`--no-gpu`, `--iou-threshold` | `scripts/dataset_extractor.py` |
| Unit tests | `tests/test_geometry.py`, `tests/test_detection.py`, `tests/test_dataset_extractor.py` (25 tests) |

## Phase 4 Deliverables (complete)

| Item | Location |
|------|----------|
| `TypesettingFeatures` dataclass | `finetuning/models.py` |
| Typesetting features extractor | `finetuning/typesetting_features.py` |
| Integrated typesetting rows | `finetuning/extractor.py` |
| Unit tests | `tests/test_typesetting_features.py` (5 tests) |

## Phase 5 Deliverables (complete)

| Item | Location |
|------|----------|
| `union_box` helper | `finetuning/geometry.py` |
| Mask extraction core | `finetuning/mask_extraction.py` |
| Extractor integration | `finetuning/extractor.py` |
| `masks_dir` CLI flag | `scripts/dataset_extractor.py` |
| Unit tests | `tests/test_mask_extraction.py` (7 tests) |

## Phase 6 Deliverables (complete)

| Item | Location |
|------|----------|
| `JpOcrResult` dataclass | `finetuning/models.py` |
| `StoryContextMetadata` dataclass | `finetuning/models.py` |
| `MangaOcrExtractor` (load/unload/extract_text lifecycle) | `finetuning/manga_ocr_extraction.py` |
| `PageOrderTracker` (numeric-prefix sort) | `finetuning/page_order.py` |
| Extractor integration (co-load lifecycle, `raw_pil`, Phase 6 row fields) | `finetuning/extractor.py` |
| CLI flags `--no-manga-ocr`, `--ocr-crop-margin` | `scripts/dataset_extractor.py` |
| Unit tests (10 manga-ocr + 6 page-order + 3 extractor) | `tests/test_manga_ocr_extraction.py`, `tests/test_page_order.py`, `tests/test_dataset_extractor.py` |

## Phase 7 Deliverables (complete)

| Item | Location |
|------|----------|
| Standalone training script | `scripts/train_layout_model.py` |
| Random Forest model artifact | `assets/models/layout_model.pkl` |
| CLI flags `--dataset-file`, `--model-out`, `--min-confidence`, `--tune` | `scripts/train_layout_model.py` |
| Unit tests (14 tests) | `tests/test_train_layout_model.py` |

## Phase 8 Deliverables (complete)

| Item | Location |
|------|----------|
| Core translation memory logic | `finetuning/translation_memory.py` |
| Standalone database builder script | `scripts/build_translation_memory.py` |
| Compiled TM database | `assets/models/translation_memory.json` |
| Unit tests (9 tests) | `tests/test_build_translation_memory.py` |

## Phase 9 Deliverables (complete)

| Item | Location |
|------|----------|
| TM loaded in OCR / Translate | `workers/ocr_worker.py`, `workers/translator.py` |
| Exact/fuzzy bypass | `workers/translator.py` |
| Context prepend for Argos | `workers/translator.py` |
| OCR Vocab Correction Heuristic | `workers/ocr_worker.py` |
| Pipeline Integration Layer | `finetuning/pipeline_integration.py` |
| Configuration Extensions | `cli/config/models.py`, `core/models.py` |
| Integration tests (~40 tests) | `tests/integration/test_integration_tm_ocr.py`, `tests/integration/test_integration_tm_translate.py` |

## Phase 10 Deliverables (complete)

| Item | Location |
|------|----------|
| Layout RF prediction bounds | `workers/typesetter.py` |
| Contour inpaint mask | `workers/inpainter.py` |
| Heuristic confidence fallback | `workers/inpainter.py` |
| Integration tests (~24 tests) | `tests/integration/test_integration_contour_inpaint.py`, `tests/integration/test_integration_layout_typeset.py`, `tests/integration/test_integration_e2e_regression.py` |

## Execution Notes
- Hardware Target: NVIDIA GTX 1050ti (4GB VRAM).
- Checkpointing: Extraction chunked into micro-batches of 50 images for crash recovery.
- EasyOCR reader reloaded every `checkpoint_interval` pairs to manage VRAM.
- Dataset Location: `training_data/dataset.jsonl` (one row per matched region)
- Checkpoint File: `training_data/extraction_state.json`
- Model Artifacts: `assets/models/`
- Run: `python scripts/dataset_extractor.py --reset-checkpoint --no-gpu`
