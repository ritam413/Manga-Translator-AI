# Manga Translator — Finetuning Implementation Plan

This document outlines the 10-phase architecture for building a machine learning dataset from 6,000 reference images and training lightweight local models to upgrade the manga translation pipeline.

## The 10-Phase Finetuning Roadmap

### Phase 1: Environment & Dependency Setup
- Identify and install required libraries (e.g., `scikit-learn` for layout models, `nltk` or `fast_align` for translation memory).
- Update `library_documentation_finetuning.md`.
- Set up directory structures (`training_data/`, `assets/models/`).

### Phase 2: Incremental Dataset Extractor Architecture
- Build the core `DatasetExtractor` class.
- Implement the micro-batching system (`--checkpoint-interval 50`) to save progress frequently and minimize RAM usage.
- Implement checkpointing to resume extraction safely.

### Phase 3: Geometric Matching & Alignment Logic
- Load paired images (`raw(jap)/manga1/1.jpg` and `eng/manga1/1.jpg`).
- Run `TextDetectorWorker` on the Japanese image.
- Run `easyocr` on the English image.
- Develop geometric overlap logic (IoU) to map the English text bounding box back to the corresponding Japanese speech bubble.

### Phase 4: Typesetting Feature Extraction
- Analyze the English text bounding box extracted in Phase 3.
- Calculate professional typesetting parameters:
  - Estimated font size
  - Inner margin/padding from the bubble edge
  - Line spacing and text wrapping shape (rectangular vs elliptical).
- Append these parameters as target labels in `dataset.jsonl` (JSON Lines format).

### Phase 5: Inpaint Mask Extraction
- Perform image differencing between the raw Japanese page and the final English page around the text bubble regions.
- Extract the exact pixel mask showing what the human typesetter erased.
- Save these masks to `training_data/masks/` to teach Stage 9 how to erase non-destructively.

### Phase 6: Story Context & Translation Extraction
- Run `manga-ocr` on the raw Japanese bubble.
- Map it to the translated English text.
- Tag the extracted pair with `manga_id` and sequential `page_number` to preserve story continuity.
- Append to `dataset.jsonl` row by row.

### Phase 7: Layout Optimization Model Training
- Build `scripts/train_layout_model.py`.
- Train a lightweight machine learning model (e.g., Random Forest via `scikit-learn`) using `dataset.jsonl`.
- **Input:** Japanese Bubble Width, Height, Area, Character Count.
- **Output:** Predicted English Font Size, Margins, Line Spacing.
- Save weights to `assets/models/layout_model.pkl`.

### Phase 8: Translation Memory Database Construction
- Build `scripts/build_translation_memory.py`.
- Parse `dataset.jsonl` line by line to extract `Japanese <-> English` pairs.
- Build a fast, local dictionary grouped by `manga_id` to memorize character names and vocabulary.
- Build a prefix-context mechanism to pass the last 5 sentences to ArgosTranslate.

### Phase 9: Integration into Pipeline Phase A (Stages 7 & 8)
- *Note: Updates `project_context.md` only once this phase begins.*
- **Stage 7 (OCR):** Add OCR fallback using Translation Memory phrase matching.
- **Stage 8 (Translator):** Integrate the Translation Memory lookup and prefix-context feeding before calling `argostranslate`.

### Phase 10: Integration into Pipeline Phase B (Stages 9 & 10)
- *Note: Updates `project_context.md` only once this phase begins.*
- **Stage 9 (Inpainter):** Use the rules learned from Phase 5 masks to do contour-based inpainting instead of rigid bounding boxes.
- **Stage 10 (Typesetter):** Load `layout_model.pkl` to dynamically predict font sizes and apply elliptical wrapping.

---
**Status:** Phase 1 complete. Phase 2 in progress (extractor shell hardening). See `finetuning_context.md` for live status.
